"""Deterministic 3D transformations of the VCG: rotation, scaling, and their composition.

Conventions (project research, R3):

* angles in **degrees**; rotations are right-handed about the axis (``x``, ``y`` or ``z``);
* ``axes_order`` is the order in which the three elementary rotations are applied — with
  ``axes_order=(a1, a2, a3)`` and ``angles_deg=(θ1, θ2, θ3)``,
  ``R = R_a3(θ3) · R_a2(θ2) · R_a1(θ1)`` (the first axis is applied first);
* ``order="rotate_then_scale"`` (default) gives ``M = S · R`` — the ``D S R D⁻¹`` of Gopal et al.
  (2021) — and ``"scale_then_rotate"`` gives ``M = R · S``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .dipole import apply_matrix_ecg
from .matrices import VCG_AXES

ORDERS: tuple[str, str] = ("rotate_then_scale", "scale_then_rotate")

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _rot(axis: str, theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _as_triplet(values, name: str) -> tuple[float, float, float]:
    try:
        t = tuple(float(v) for v in values)
    except TypeError:
        raise ValueError(f"{name} must be a sequence of 3 numbers, got {values!r}") from None
    if len(t) != 3:
        raise ValueError(f"{name} must have exactly 3 values, got {len(t)}: {values!r}")
    if any(math.isnan(v) or math.isinf(v) for v in t):
        raise ValueError(f"{name} must be finite, got {values!r}")
    return t  # type: ignore[return-value]


def _validate_angles(angles) -> tuple[float, float, float]:
    t = _as_triplet(angles, "angles_deg")
    for v in t:
        if not -180.0 <= v <= 180.0:
            raise ValueError(f"angles_deg must lie in [-180, 180] degrees, got {angles!r}")
    return t


def _validate_scales(scales) -> tuple[float, float, float]:
    t = _as_triplet(scales, "scales")
    for v in t:
        if v <= 0.0:
            raise ValueError(f"scales must be strictly positive, got {scales!r}")
    return t


def _validate_axes_order(axes_order) -> tuple[str, str, str]:
    t = tuple(axes_order)
    if len(t) != 3 or set(t) != set(VCG_AXES):
        raise ValueError(
            f"axes_order must be a permutation of {VCG_AXES}, got {axes_order!r}"
        )
    return t  # type: ignore[return-value]


def _validate_order(order: str) -> str:
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    return order


@dataclass(frozen=True)
class Transform:
    """A deterministic rotation + scaling of the VCG.

    Attributes
    ----------
    angles_deg
        Rotation angles (degrees, in [-180, 180]) about the axes listed in ``axes_order``.
    axes_order
        Permutation of ``("x", "y", "z")``: order in which the elementary rotations apply.
    scales
        Strictly positive scale factors along x, y, z (in that fixed order, independent of
        ``axes_order``).
    order
        ``"rotate_then_scale"`` (``M = S·R``, default) or ``"scale_then_rotate"`` (``M = R·S``).
    """

    angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axes_order: tuple[str, str, str] = ("x", "y", "z")
    scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
    order: str = "rotate_then_scale"

    def __post_init__(self) -> None:
        object.__setattr__(self, "angles_deg", _validate_angles(self.angles_deg))
        object.__setattr__(self, "axes_order", _validate_axes_order(self.axes_order))
        object.__setattr__(self, "scales", _validate_scales(self.scales))
        object.__setattr__(self, "order", _validate_order(self.order))

    # --- matrices -------------------------------------------------------------------------------

    def rotation_matrix(self) -> np.ndarray:
        """(3 × 3) orthogonal matrix, det = 1: ``R_a3(θ3) · R_a2(θ2) · R_a1(θ1)``."""
        R = np.eye(3)
        for axis, deg in zip(self.axes_order, self.angles_deg, strict=True):
            R = _rot(axis, math.radians(deg)) @ R
        return R

    def scale_matrix(self) -> np.ndarray:
        """(3 × 3) diagonal matrix ``diag(s_x, s_y, s_z)``."""
        return np.diag(np.asarray(self.scales, dtype=np.float64))

    def matrix(self) -> np.ndarray:
        """The full VCG-space operator: ``S·R`` or ``R·S`` depending on ``order``."""
        R, S = self.rotation_matrix(), self.scale_matrix()
        return S @ R if self.order == "rotate_then_scale" else R @ S

    def is_identity(self) -> bool:
        """True iff all angles are 0 and all scales are 1 (then ``matrix()`` is exactly ``I``)."""
        return all(a == 0.0 for a in self.angles_deg) and all(s == 1.0 for s in self.scales)

    def compose(self, other: Transform) -> np.ndarray:
        """Matrix of ``self ∘ other`` (apply ``other`` first). Returned as a matrix: the product of
        two ``S·R`` operators is generally not itself an ``S·R`` — use ``apply_matrix_ecg``."""
        return self.matrix() @ other.matrix()

    # --- application ----------------------------------------------------------------------------

    def apply_vcg(self, vcg, *, leads_axis: int = -2):
        """Apply to a VCG ``(..., 3, T)`` (axes dimension at ``leads_axis``)."""
        v = np.asarray(vcg)
        axis = leads_axis % v.ndim
        moved = v if axis == v.ndim - 2 else np.moveaxis(v, axis, v.ndim - 2)
        if moved.shape[-2] != 3:
            raise ValueError(f"apply_vcg: expected 3 axes along dim -2, got {tuple(v.shape)}")
        out = np.einsum("ij,...jt->...it", self.matrix(), moved.astype(np.float64, copy=False))
        if np.issubdtype(v.dtype, np.floating) and v.dtype != np.float64:
            out = out.astype(v.dtype, copy=False)
        return out if axis == v.ndim - 2 else np.moveaxis(out, v.ndim - 2, axis)

    def apply_ecg(
        self,
        ecg,
        leads: str | Sequence[str],
        *,
        mode: str = "residual",
        method: str = "dower",
        leads_axis: int = -2,
    ):
        """Apply to an ECG through the VCG; see ``apply_matrix_ecg`` for ``mode`` and ``method``."""
        return apply_matrix_ecg(
            ecg, self.matrix(), leads, mode=mode, method=method, leads_axis=leads_axis
        )

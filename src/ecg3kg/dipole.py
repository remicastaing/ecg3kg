"""ECG ↔ VCG conversion and application of a 3 × 3 matrix to an ECG through the VCG.

Shapes: an ECG is ``(..., n_leads, T)`` (``leads_axis=-2``, any other position through
``leads_axis``); a VCG is ``(..., 3, T)``. Any leading dimensions are a batch. Computations run in
float64; the output has the dtype and shape of the input.

Two modes for ``apply_matrix_ecg`` (see the project research, R2):

* ``"residual"`` (default): ``x_out = x + D (M − I) D⁻¹ x`` on the 8 independent leads. Exact
  identity when ``M = I``; equal to the projected form on any purely dipolar signal (``x = D v``);
  keeps untouched the part of a real ECG that the dipole model does not explain.
* ``"project"``: ``x_out = D M D⁻¹ x`` — the formula of Gopal et al. (2021). A projection: even
  ``M = I`` alters a real ECG (it removes the non-dipolar residual).

In both modes the 4 derived leads (III, aVR, aVL, aVF) are recomputed from the transformed I and II.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .leads import derivation_matrix, move_leads_axis, resolve_leads, selection_matrix
from .matrices import DOWER, RECONSTRUCTIONS

MODES: tuple[str, ...] = ("residual", "project")


def _reconstruction(method: str) -> np.ndarray:
    try:
        return RECONSTRUCTIONS[method]
    except KeyError:
        raise ValueError(
            f"unknown reconstruction method {method!r}; expected one of {sorted(RECONSTRUCTIONS)}"
        ) from None


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def _check_leads_dim(x, n_expected: int, what: str) -> None:
    if x.shape[-2] != n_expected:
        raise ValueError(
            f"{what}: expected {n_expected} along the leads axis, got shape {tuple(x.shape)}"
        )


def _lead_matrix(ecg8_matrix: np.ndarray, order: Sequence[str]) -> np.ndarray:
    """Lift an (8 × 8) operator on independent leads to an (n × n) operator on ``order``.

    ``G @ A @ S`` selects the 8 independent leads, applies ``A``, and re-derives every lead.
    """
    return derivation_matrix(order) @ ecg8_matrix @ selection_matrix(order)


def ecg_to_vcg(ecg, leads: str | Sequence[str], *, method: str = "dower", leads_axis: int = -2):
    """Reconstruct the VCG ``(..., 3, T)`` from an ECG ``(..., n_leads, T)``.

    Only the 8 independent leads (I, II, V1–V6) are used; derived leads are ignored.
    ``method`` is ``"dower"`` (pseudo-inverse of Dower, default) or ``"kors"``.
    """
    order = resolve_leads(leads)
    K = _reconstruction(method)
    x, _ = move_leads_axis(np.asarray(ecg), leads_axis)
    _check_leads_dim(x, len(order), "ecg_to_vcg")
    A = K @ selection_matrix(order)  # (3 × n)
    out = np.einsum("ij,...jt->...it", A, x.astype(np.float64, copy=False))
    return _cast_like(out, x)


def _cast_like(out: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Give ``out`` the floating dtype of ``like`` (integer inputs stay float64)."""
    if np.issubdtype(like.dtype, np.floating) and like.dtype != np.float64:
        return out.astype(like.dtype, copy=False)
    return out


def vcg_to_ecg(vcg, leads: str | Sequence[str], *, leads_axis: int = -2):
    """Project a VCG ``(..., 3, T)`` onto the leads of ``leads`` (8 or 12), derived leads exact."""
    order = resolve_leads(leads)
    v = np.asarray(vcg)
    if v.shape[-2] != 3:
        raise ValueError(f"vcg_to_ecg: expected 3 axes along dim -2, got shape {tuple(v.shape)}")
    A = derivation_matrix(order) @ DOWER  # (n × 3)
    out = _cast_like(np.einsum("ij,...jt->...it", A, v.astype(np.float64, copy=False)), v)
    axis = leads_axis % out.ndim
    return out if axis == out.ndim - 2 else np.moveaxis(out, out.ndim - 2, axis)


def _application_matrix(matrix, order: Sequence[str], mode: str, method: str) -> np.ndarray:
    """(n × n) matrix implementing ``apply_matrix_ecg`` for a given lead order, mode and method."""
    M = np.asarray(matrix, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"matrix must be 3 × 3, got {M.shape}")
    K = _reconstruction(method)
    if mode == "project":
        A8 = DOWER @ M @ K
        return _lead_matrix(A8, order)
    # residual: x + D (M − I) K x  on the 8 independent leads, then re-derive.
    A8 = np.eye(8) + DOWER @ (M - np.eye(3)) @ K
    return _lead_matrix(A8, order)


def apply_matrix_ecg(
    ecg,
    matrix,
    leads: str | Sequence[str],
    *,
    mode: str = "residual",
    method: str = "dower",
    leads_axis: int = -2,
):
    """Apply a 3 × 3 VCG-space matrix ``M`` to an ECG and return an ECG of the same shape/dtype.

    See the module docstring for the two ``mode`` values. ``method`` selects the reconstruction
    (``"dower"`` or ``"kors"``). Derived leads are recomputed exactly.
    """
    _check_mode(mode)
    order = resolve_leads(leads)
    x_in = np.asarray(ecg)
    x, restore = move_leads_axis(x_in, leads_axis)
    _check_leads_dim(x, len(order), "apply_matrix_ecg")
    A = _application_matrix(matrix, order, mode, method)
    out = np.einsum("ij,...jt->...it", A, x.astype(np.float64, copy=False))
    if np.issubdtype(x_in.dtype, np.floating):
        out = out.astype(x_in.dtype, copy=False)
    return restore(out)

"""Random augmentation policy: the law of Gopal et al. (2021), reproducible by construction.

Per ECG (batch element), ``RandomVCGAugment.draw`` samples — **always in this order**, whatever the
flags, so that a given generator state always yields the same draws (research R5):

1. ``applied ~ Bernoulli(p)``;
2. a permutation of the axes (uniform over the 6 orders);
3. three angles ``~ U[-r, r]`` (degrees);
4. three scale factors ``~ U[1, s]``;
5. three inversions ``~ Bernoulli(1/2)`` (``s_i ← 1 / s_i``).

Randomness comes only from the ``rng`` the caller passes (a ``numpy.random.Generator`` or an integer
seed). No global state is ever touched. An element with ``applied=False`` is returned **unchanged,
bit for bit** — it never goes through the projection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .dipole import MODES, apply_matrix_ecg
from .geometry import Transform
from .leads import move_leads_axis, resolve_leads
from .matrices import RECONSTRUCTIONS, VCG_AXES

_PERMUTATIONS: tuple[tuple[str, str, str], ...] = (
    ("x", "y", "z"),
    ("x", "z", "y"),
    ("y", "x", "z"),
    ("y", "z", "x"),
    ("z", "x", "y"),
    ("z", "y", "x"),
)


def _coerce_rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError(
        f"rng must be a numpy.random.Generator or an integer seed, got {type(rng).__name__}"
    )


@dataclass(frozen=True)
class Draw:
    """What was drawn for one ECG: whether the policy applied, and the exact transformation."""

    applied: bool
    transform: Transform


@dataclass(frozen=True)
class RandomVCGAugment:
    """Random rotation + scaling of the VCG, with the sampling law of 3KG.

    Parameters
    ----------
    max_rotation_deg
        r: each angle is drawn uniformly in [-r, r] degrees. Paper's best: 45.
    max_scale
        s ≥ 1: each factor is drawn uniformly in [1, s], then inverted with probability 1/2.
        Paper's best: 1.5.
    p
        Probability of applying the transformation to a given ECG (1 = always).
    rotate, scale
        Disable one component (angles forced to 0 / scales forced to 1). Draws are still consumed.
    mode, method
        Passed to ``apply_matrix_ecg`` (``"residual"``/``"project"``, ``"dower"``/``"kors"``).
    """

    max_rotation_deg: float = 45.0
    max_scale: float = 1.5
    p: float = 1.0
    rotate: bool = True
    scale: bool = True
    mode: str = "residual"
    method: str = "dower"

    def __post_init__(self) -> None:
        r, s, p = float(self.max_rotation_deg), float(self.max_scale), float(self.p)
        if not 0.0 <= r <= 180.0:
            raise ValueError(
                f"max_rotation_deg must lie in [0, 180], got {self.max_rotation_deg!r}"
            )
        if not s >= 1.0:
            raise ValueError(f"max_scale must be >= 1, got {self.max_scale!r}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must lie in [0, 1], got {self.p!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.method not in RECONSTRUCTIONS:
            raise ValueError(
                f"method must be one of {sorted(RECONSTRUCTIONS)}, got {self.method!r}"
            )
        object.__setattr__(self, "max_rotation_deg", r)
        object.__setattr__(self, "max_scale", s)
        object.__setattr__(self, "p", p)

    # --- sampling -------------------------------------------------------------------------------

    def draw_one(self, rng: np.random.Generator) -> Draw:
        """Sample the parameters for one ECG (draw order is part of the contract)."""
        applied = bool(rng.random() < self.p)
        perm = _PERMUTATIONS[int(rng.integers(len(_PERMUTATIONS)))]
        angles = rng.uniform(-self.max_rotation_deg, self.max_rotation_deg, size=3)
        scales = rng.uniform(1.0, self.max_scale, size=3)
        invert = rng.random(3) < 0.5
        if not applied:
            return Draw(False, Transform())
        if not self.rotate:
            angles = np.zeros(3)
            perm = VCG_AXES
        if not self.scale:
            scales = np.ones(3)
        else:
            scales = np.where(invert, 1.0 / scales, scales)
        return Draw(
            True,
            Transform(
                angles_deg=tuple(float(a) for a in angles),
                axes_order=perm,
                scales=tuple(float(s) for s in scales),
            ),
        )

    def draw(self, rng, n: int = 1) -> list[Draw]:
        """Sample ``n`` independent draws from ``rng`` (Generator or seed)."""
        g = _coerce_rng(rng)
        return [self.draw_one(g) for _ in range(int(n))]

    # --- application ----------------------------------------------------------------------------

    def __call__(
        self,
        ecg,
        leads: str | Sequence[str],
        rng,
        *,
        leads_axis: int = -2,
        return_params: bool = False,
    ):
        """Augment an ECG ``(n_leads, T)`` or a batch ``(..., n_leads, T)``: one draw per element.

        Returns the augmented array (same type, dtype, shape), or ``(array, draws)`` when
        ``return_params`` is true — ``draws`` is a flat list aligned with the batch elements
        (row-major over the leading dimensions).
        """
        order = resolve_leads(leads)
        x_in = np.asarray(ecg)
        x, restore = move_leads_axis(x_in, leads_axis)
        if x.shape[-2] != len(order):
            raise ValueError(
                f"expected {len(order)} leads along the leads axis, got shape {tuple(x_in.shape)}"
            )
        lead_shape = x.shape[-2:]
        flat = x.reshape(-1, *lead_shape)  # (B, n_leads, T)
        draws = self.draw(rng, flat.shape[0])
        out = flat.copy()  # not-applied elements stay bit-exact copies of the input
        for i, d in enumerate(draws):
            if d.applied:
                out[i] = apply_matrix_ecg(
                    flat[i], d.transform.matrix(), order, mode=self.mode, method=self.method
                )
        out = restore(out.reshape(x.shape))
        return (out, draws) if return_params else out

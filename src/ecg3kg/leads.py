"""Lead orders: named presets, validation, and the selection / derivation matrices.

An ECG array only makes sense with a declared lead order. Nothing in this package guesses it: every
public function takes ``leads`` — a preset name or an explicit sequence of lead names — and refuses
anything it cannot validate.

The 12 standard leads carry 8 independent signals (I, II, V1–V6). The 4 derived leads follow from
I and II exactly:

    III = II − I,   aVR = −(I + II) / 2,   aVL = I − II / 2,   aVF = II − I / 2.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .matrices import INDEPENDENT_LEADS

ALL_LEADS: frozenset[str] = frozenset(
    {"I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"}
)

LEAD_PRESETS: dict[str, tuple[str, ...]] = {
    # Native order of MIMIC-IV-ECG WFDB records (note aVF before aVL).
    "mimic": ("I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"),
    # The most common order (PTB-XL, wfdb defaults): aVL before aVF.
    "standard": ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"),
    # The 8 independent leads only, in canonical order.
    "independent8": INDEPENDENT_LEADS,
}

# Coefficients of each derived lead on (I, II).
_DERIVED: dict[str, tuple[float, float]] = {
    "III": (-1.0, 1.0),
    "aVR": (-0.5, -0.5),
    "aVL": (1.0, -0.5),
    "aVF": (-0.5, 1.0),
}


def resolve_leads(leads: str | Sequence[str]) -> tuple[str, ...]:
    """Validate a lead order and return it as a tuple of names.

    ``leads`` is a preset name (``"mimic"``, ``"standard"``, ``"independent8"``) or a sequence of
    lead names. The order must contain no duplicate, only known names, and all 8 independent leads
    (I, II, V1–V6). Raises ``ValueError`` otherwise; never guesses.
    """
    if isinstance(leads, str):
        try:
            return LEAD_PRESETS[leads]
        except KeyError:
            raise ValueError(
                f"unknown lead preset {leads!r}; known presets: {sorted(LEAD_PRESETS)}. "
                "Pass an explicit sequence of lead names otherwise."
            ) from None
    if leads is None:
        raise ValueError("`leads` is required: pass a preset name or a sequence of lead names")
    order = tuple(str(name) for name in leads)
    if not order:
        raise ValueError("`leads` is empty")
    unknown = [n for n in order if n not in ALL_LEADS]
    if unknown:
        raise ValueError(f"unknown lead name(s) {unknown}; allowed: {sorted(ALL_LEADS)}")
    if len(set(order)) != len(order):
        raise ValueError(f"duplicate lead name(s) in {order}")
    missing = [n for n in INDEPENDENT_LEADS if n not in order]
    if missing:
        raise ValueError(
            f"lead order {order} lacks independent lead(s) {missing}: I, II and V1–V6 are all "
            "required to reconstruct the VCG"
        )
    return order


def selection_matrix(order: Sequence[str]) -> np.ndarray:
    """(8 × n) matrix picking I, II, V1–V6 — in canonical order — out of ``order``."""
    order = tuple(order)
    S = np.zeros((len(INDEPENDENT_LEADS), len(order)), dtype=np.float64)
    for row, name in enumerate(INDEPENDENT_LEADS):
        S[row, order.index(name)] = 1.0
    return S


def derivation_matrix(order: Sequence[str]) -> np.ndarray:
    """(n × 8) matrix rebuilding every lead of ``order`` from the canonical 8 independent leads."""
    order = tuple(order)
    G = np.zeros((len(order), len(INDEPENDENT_LEADS)), dtype=np.float64)
    col = {name: k for k, name in enumerate(INDEPENDENT_LEADS)}
    for row, name in enumerate(order):
        if name in col:
            G[row, col[name]] = 1.0
        else:
            c1, c2 = _DERIVED[name]
            G[row, col["I"]] = c1
            G[row, col["II"]] = c2
    return G


def move_leads_axis(x, leads_axis: int) -> tuple[np.ndarray, Callable]:
    """Bring the leads axis to position −2 (shape ``(..., n_leads, T)``) and return the inverse.

    Works for numpy arrays (``np.moveaxis``) and torch tensors (``Tensor.movedim``). Returns
    ``(moved, restore)`` where ``restore(y)`` puts the axis back where it came from.
    """
    ndim = x.ndim
    if ndim < 2:
        raise ValueError(f"an ECG needs at least 2 dimensions (leads, time); got shape {x.shape}")
    axis = leads_axis % ndim
    target = ndim - 2
    if axis == target:
        return x, lambda y: y
    if hasattr(x, "movedim"):  # torch.Tensor
        return x.movedim(axis, target), lambda y: y.movedim(target, axis)
    return np.moveaxis(x, axis, target), lambda y: np.moveaxis(y, target, axis)

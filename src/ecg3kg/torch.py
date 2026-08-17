"""Optional torch adapter: the same transformations on ``torch.Tensor`` inputs (extra ``[torch]``).

Same semantics as the numpy core — same matrices, same modes, same draws (sampling stays on the
numpy ``Generator``, so there is a single determinism to test). Input and output share device,
dtype and shape. Computations run in float64 on the input's device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised through the test with sys.modules patch
    raise ImportError(
        "ecg3kg.torch needs PyTorch: pip install ecg3kg[torch] (or install torch yourself)"
    ) from exc

from .augment import RandomVCGAugment
from .dipole import MODES, _application_matrix
from .geometry import Transform
from .leads import move_leads_axis, resolve_leads


def apply_matrix_ecg(
    ecg: torch.Tensor,
    matrix,
    leads: str | Sequence[str],
    *,
    mode: str = "residual",
    method: str = "dower",
    leads_axis: int = -2,
) -> torch.Tensor:
    """Tensor counterpart of :func:`ecg3kg.apply_matrix_ecg`."""
    if not isinstance(ecg, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(ecg).__name__}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    order = resolve_leads(leads)
    x, restore = move_leads_axis(ecg, leads_axis)
    if x.shape[-2] != len(order):
        raise ValueError(
            f"expected {len(order)} leads along the leads axis, got shape {tuple(ecg.shape)}"
        )
    A = _application_matrix(np.asarray(matrix, dtype=np.float64), order, mode, method)
    A_t = torch.as_tensor(A, dtype=torch.float64, device=ecg.device)
    out = torch.einsum("ij,...jt->...it", A_t, x.to(torch.float64))
    if ecg.dtype.is_floating_point:
        out = out.to(ecg.dtype)
    return restore(out)


def apply_ecg(
    ecg: torch.Tensor,
    transform: Transform,
    leads: str | Sequence[str],
    *,
    mode: str = "residual",
    method: str = "dower",
    leads_axis: int = -2,
) -> torch.Tensor:
    """Apply a :class:`ecg3kg.Transform` to a tensor ECG."""
    return apply_matrix_ecg(
        ecg, transform.matrix(), leads, mode=mode, method=method, leads_axis=leads_axis
    )


@dataclass(frozen=True)
class RandomVCGAugmentTorch(RandomVCGAugment):
    """:class:`ecg3kg.RandomVCGAugment` for tensors. Draws come from the numpy ``rng`` as usual."""

    def __call__(
        self,
        ecg: torch.Tensor,
        leads: str | Sequence[str],
        rng,
        *,
        leads_axis: int = -2,
        return_params: bool = False,
    ):
        if not isinstance(ecg, torch.Tensor):
            raise TypeError(f"expected a torch.Tensor, got {type(ecg).__name__}")
        order = resolve_leads(leads)
        x, restore = move_leads_axis(ecg, leads_axis)
        if x.shape[-2] != len(order):
            raise ValueError(
                f"expected {len(order)} leads along the leads axis, got shape {tuple(ecg.shape)}"
            )
        lead_shape = x.shape[-2:]
        flat = x.reshape(-1, *lead_shape)
        draws = self.draw(rng, flat.shape[0])
        out = flat.clone()
        for i, d in enumerate(draws):
            if d.applied:
                out[i] = apply_matrix_ecg(
                    flat[i], d.transform.matrix(), order, mode=self.mode, method=self.method
                )
        out = restore(out.reshape(x.shape))
        return (out, draws) if return_params else out

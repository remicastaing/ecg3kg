"""The optional torch adapter gives the same numbers as the numpy core."""

import sys

import numpy as np
import pytest

from ecg3kg import RandomVCGAugment, Transform, apply_matrix_ecg

torch = pytest.importorskip("torch")

import ecg3kg.torch as e3t  # noqa: E402


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_torch_same_result_as_numpy(ecg_standard, dtype):
    batch = np.stack([ecg_standard, 0.7 * ecg_standard]).astype(dtype)
    t = Transform(angles_deg=(20, -35, 50), scales=(1.3, 0.8, 1.1))
    ref = t.apply_ecg(batch, leads="standard")
    out = e3t.apply_ecg(torch.from_numpy(batch), t, leads="standard")
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6 if dtype == np.float32 else 1e-12)
    out2 = e3t.apply_matrix_ecg(torch.from_numpy(batch), t.matrix(), leads="standard")
    np.testing.assert_allclose(out2.numpy(), apply_matrix_ecg(batch, t.matrix(), leads="standard"),
                               rtol=1e-6 if dtype == np.float32 else 1e-12)


def test_torch_device_and_dtype_preserved(ecg_standard):
    x = torch.from_numpy(ecg_standard.astype(np.float32))
    out = e3t.apply_ecg(x, Transform(angles_deg=(10, 0, 0)), leads="standard")
    assert out.dtype == torch.float32 and out.device == x.device and out.shape == x.shape
    # leads_axis on tensors too.
    xt = x.T.contiguous()
    out_t = e3t.apply_ecg(xt, Transform(angles_deg=(10, 0, 0)), leads="standard", leads_axis=-1)
    assert out_t.shape == xt.shape
    np.testing.assert_allclose(out_t.T.numpy(), out.numpy(), rtol=1e-6)


def test_random_augment_torch_matches_numpy(ecg_standard, rng):
    batch = np.repeat(ecg_standard[None], 5, axis=0) * rng.uniform(0.5, 1.5, size=(5, 1, 1))
    aug_np = RandomVCGAugment(p=0.8)
    aug_t = e3t.RandomVCGAugmentTorch(p=0.8)
    ref, d_np = aug_np(batch, leads="standard", rng=0, return_params=True)
    out, d_t = aug_t(torch.from_numpy(batch), leads="standard", rng=0, return_params=True)
    assert isinstance(out, torch.Tensor)
    assert d_np == d_t
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-12, atol=1e-12)
    # Not-applied elements are bit-exact copies of the input tensor.
    for i, d in enumerate(d_t):
        if not d.applied:
            assert torch.equal(out[i], torch.from_numpy(batch)[i])


def test_import_error_message_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "ecg3kg.torch", raising=False)
    with pytest.raises(ImportError, match="pip install ecg3kg\\[torch\\]"):
        import importlib

        importlib.import_module("ecg3kg.torch")

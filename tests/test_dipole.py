"""ECG ↔ VCG conversion and the two application modes (residual / project)."""

import numpy as np
import pytest

from ecg3kg import (
    DOWER,
    INVERSE_DOWER,
    KORS,
    apply_matrix_ecg,
    ecg_to_vcg,
    resolve_leads,
    vcg_to_ecg,
)
from ecg3kg.leads import derivation_matrix, selection_matrix

STD = resolve_leads("standard")


def _idx(order, name):
    return order.index(name)


def _some_matrix():
    # A non-trivial 3×3 matrix (rotation about z by 30° then anisotropic scale).
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    S = np.diag([1.2, 0.8, 1.1])
    return S @ R


def test_vcg_roundtrip(vcg_synthetic):
    ecg = vcg_to_ecg(vcg_synthetic, leads="standard")
    assert ecg.shape == (12, vcg_synthetic.shape[-1])
    back = ecg_to_vcg(ecg, leads="standard", method="dower")
    np.testing.assert_allclose(back, vcg_synthetic, rtol=1e-6, atol=1e-9)


def test_kors_reconstruction(vcg_synthetic):
    ecg = vcg_to_ecg(vcg_synthetic, leads="standard")
    kors = ecg_to_vcg(ecg, leads="standard", method="kors")
    assert kors.shape == (3, vcg_synthetic.shape[-1])
    assert np.isfinite(kors).all()
    assert np.linalg.matrix_rank(KORS @ DOWER) == 3
    # Kors is not an inverse of Dower: only a loose, axis-wise agreement is expected (the fixture
    # is a synthetic loop, not a physiological one; ~0.83 on z).
    for axis in range(3):
        r = np.corrcoef(kors[axis], vcg_synthetic[axis])[0, 1]
        assert r > 0.75, f"axis {axis}: corr={r:.3f}"
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(kors, vcg_synthetic, rtol=1e-6)


def test_derived_leads_exact(vcg_synthetic, ecg_standard):
    for out in (vcg_to_ecg(vcg_synthetic, leads="standard"),
                apply_matrix_ecg(ecg_standard, _some_matrix(), leads="standard")):
        i, ii = out[_idx(STD, "I")], out[_idx(STD, "II")]
        np.testing.assert_allclose(out[_idx(STD, "III")], ii - i, atol=1e-12)
        np.testing.assert_allclose(out[_idx(STD, "aVR")], -(i + ii) / 2, atol=1e-12)
        np.testing.assert_allclose(out[_idx(STD, "aVL")], i - ii / 2, atol=1e-12)
        np.testing.assert_allclose(out[_idx(STD, "aVF")], ii - i / 2, atol=1e-12)


def test_project_mode_idempotent(ecg_with_residual):
    once = apply_matrix_ecg(ecg_with_residual, np.eye(3), leads="standard", mode="project")
    twice = apply_matrix_ecg(once, np.eye(3), leads="standard", mode="project")
    np.testing.assert_allclose(twice, once, rtol=1e-12, atol=1e-12)
    # The projection removes the residual: output differs from input.
    assert np.abs(once - ecg_with_residual).max() > 1e-3


def test_residual_mode_identity_on_residual_signal(ecg_with_residual):
    out = apply_matrix_ecg(ecg_with_residual, np.eye(3), leads="standard", mode="residual")
    np.testing.assert_array_equal(out, ecg_with_residual)


def test_zero_transform_on_inconsistent_12_leads(ecg_standard):
    inconsistent = ecg_standard.copy()
    inconsistent[_idx(STD, "III")] += 1e-3  # 1 µV (signals in mV)
    out = apply_matrix_ecg(inconsistent, np.eye(3), leads="standard", mode="residual")
    S = selection_matrix(STD)
    np.testing.assert_array_equal(S @ out, S @ inconsistent)  # 8 independent leads: bit-exact
    i, ii = out[_idx(STD, "I")], out[_idx(STD, "II")]
    np.testing.assert_allclose(out[_idx(STD, "III")], ii - i, atol=1e-12)  # re-derived
    assert np.abs(out[_idx(STD, "III")] - inconsistent[_idx(STD, "III")]).max() > 5e-4


def test_modes_agree_on_dipolar_signal(vcg_synthetic, ecg_standard):
    M = _some_matrix()
    expected = derivation_matrix(STD) @ (DOWER @ (M @ vcg_synthetic))
    for mode in ("residual", "project"):
        out = apply_matrix_ecg(ecg_standard, M, leads="standard", mode=mode, method="dower")
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-9)


def test_leads_axis(ecg_standard):
    M = _some_matrix()
    ref = apply_matrix_ecg(ecg_standard, M, leads="standard")
    out = apply_matrix_ecg(ecg_standard.T.copy(), M, leads="standard", leads_axis=-1)
    assert out.shape == ecg_standard.T.shape
    np.testing.assert_allclose(out.T, ref, rtol=1e-12)


def test_dtype_and_shape_preserved(ecg_standard):
    batch = np.stack([ecg_standard, 2 * ecg_standard, -ecg_standard]).astype(np.float32)
    out = apply_matrix_ecg(batch, _some_matrix(), leads="standard")
    assert out.shape == batch.shape
    assert out.dtype == np.float32
    assert isinstance(out, np.ndarray)
    # Batch elements are processed independently.
    single = apply_matrix_ecg(batch[1], _some_matrix(), leads="standard")
    np.testing.assert_allclose(out[1], single, rtol=1e-6)


def test_nonfinite_propagates(ecg_standard):
    bad = ecg_standard.copy()
    bad[_idx(STD, "V3"), 10] = np.nan
    out = apply_matrix_ecg(bad, _some_matrix(), leads="standard")
    assert np.isnan(out[:, 10]).any()
    assert np.isfinite(out[:, :10]).all()


def test_ecg_to_vcg_uses_only_independent_leads(ecg_standard):
    """Derived leads may be garbage: the reconstruction ignores them."""
    garbled = ecg_standard.copy()
    for name in ("III", "aVR", "aVL", "aVF"):
        garbled[_idx(STD, name)] = 999.0
    np.testing.assert_array_equal(
        ecg_to_vcg(garbled, leads="standard"), ecg_to_vcg(ecg_standard, leads="standard")
    )


def test_inverse_dower_kills_residual(ecg_with_residual, ecg_standard):
    """The residual fixture is indeed invisible to the reconstruction."""
    a = ecg_to_vcg(ecg_with_residual, leads="standard")
    b = ecg_to_vcg(ecg_standard, leads="standard")
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert INVERSE_DOWER.shape == (3, 8)

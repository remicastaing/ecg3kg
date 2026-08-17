"""Random augmentation policy: law of the draws, determinism, independence, no-op guarantees."""

import numpy as np
import pytest
from scipy import stats

from ecg3kg import Draw, RandomVCGAugment, Transform, resolve_leads
from ecg3kg.leads import selection_matrix

STD = resolve_leads("standard")


def _batch(ecg, n, rng):
    """n distinct copies (scaled) of the same ECG, shape (n, 12, T)."""
    factors = rng.uniform(0.5, 1.5, size=(n, 1, 1))
    return np.repeat(ecg[None], n, axis=0) * factors


def _inconsistent_batch(ecg, n, rng):
    b = _batch(ecg, n, rng)
    b[:, STD.index("III")] += 1e-3  # 1 µV off: derived leads no longer exact
    return b


def test_seed_determinism(ecg_standard, rng):
    batch = _batch(ecg_standard, 6, rng)
    aug = RandomVCGAugment()
    out1, d1 = aug(batch, leads="standard", rng=np.random.default_rng(7), return_params=True)
    out2, d2 = aug(batch, leads="standard", rng=np.random.default_rng(7), return_params=True)
    np.testing.assert_array_equal(out1, out2)
    assert d1 == d2


def test_different_seeds_differ(ecg_standard, rng):
    batch = _batch(ecg_standard, 6, rng)
    aug = RandomVCGAugment()
    out1 = aug(batch, leads="standard", rng=np.random.default_rng(1))
    out2 = aug(batch, leads="standard", rng=np.random.default_rng(2))
    assert np.abs(out1 - out2).max() > 1e-3


def test_independent_draws_in_batch(ecg_standard, rng):
    batch = _batch(ecg_standard, 8, rng)
    _, draws = RandomVCGAugment()(batch, leads="standard", rng=3, return_params=True)
    transforms = [d.transform for d in draws]
    assert len(set(transforms)) == 8


def test_draw_distribution():
    n = 1000
    aug = RandomVCGAugment(max_rotation_deg=45.0, max_scale=1.5, p=1.0)
    draws = aug.draw(np.random.default_rng(0), n)
    assert len(draws) == n and all(d.applied for d in draws)
    angles = np.array([d.transform.angles_deg for d in draws])  # (n, 3)
    scales = np.array([d.transform.scales for d in draws])
    for axis in range(3):
        p = stats.kstest(angles[:, axis], stats.uniform(loc=-45, scale=90).cdf).pvalue
        assert p > 0.01, f"angles axis {axis}: KS p={p:.4f}"
    frac_below_one = (scales < 1.0).mean()
    assert 0.45 <= frac_below_one <= 0.55, frac_below_one
    assert scales.min() >= 1 / 1.5 - 1e-12 and scales.max() <= 1.5 + 1e-12
    # Axes permutation ≈ uniform over the 6 orders.
    perms = [d.transform.axes_order for d in draws]
    counts = np.array([perms.count(p) for p in sorted(set(perms))])
    assert len(counts) == 6
    assert stats.chisquare(counts).pvalue > 0.01


def test_p_zero_is_identity(ecg_standard, rng):
    batch = _inconsistent_batch(ecg_standard, 5, rng)
    out, draws = RandomVCGAugment(p=0.0)(batch, leads="standard", rng=0, return_params=True)
    np.testing.assert_array_equal(out, batch)  # bit-exact, even with inconsistent derived leads
    assert all(not d.applied for d in draws)
    assert all(d.transform.is_identity() for d in draws)


def test_not_applied_elements_untouched(ecg_standard, rng):
    batch = _inconsistent_batch(ecg_standard, 40, rng)
    out, draws = RandomVCGAugment(p=0.5)(batch, leads="standard", rng=11, return_params=True)
    applied = np.array([d.applied for d in draws])
    assert 5 < applied.sum() < 35
    np.testing.assert_array_equal(out[~applied], batch[~applied])
    S = selection_matrix(STD)
    for i in np.flatnonzero(applied):
        assert np.abs(S @ out[i] - S @ batch[i]).max() > 1e-6


def test_p_partial():
    draws = RandomVCGAugment(p=0.3).draw(np.random.default_rng(5), 1000)
    frac = np.mean([d.applied for d in draws])
    assert 0.25 <= frac <= 0.35, frac


def test_rotate_only_scale_only():
    rot_only = RandomVCGAugment(scale=False).draw(np.random.default_rng(0), 50)
    assert all(d.transform.scales == (1.0, 1.0, 1.0) for d in rot_only)
    assert any(d.transform.angles_deg != (0.0, 0.0, 0.0) for d in rot_only)
    scale_only = RandomVCGAugment(rotate=False).draw(np.random.default_rng(0), 50)
    assert all(d.transform.angles_deg == (0.0, 0.0, 0.0) for d in scale_only)
    assert any(d.transform.scales != (1.0, 1.0, 1.0) for d in scale_only)


def test_return_params_alignment(ecg_standard, rng):
    batch = _batch(ecg_standard, 6, rng)
    out, draws = RandomVCGAugment()(batch, leads="standard", rng=4, return_params=True)
    assert len(draws) == 6
    for i, d in enumerate(draws):
        assert isinstance(d, Draw)
        np.testing.assert_array_equal(out[i], d.transform.apply_ecg(batch[i], leads="standard"))


def test_draw_order_stable():
    """Frozen draws for seed 0: any change in the draw order or law changes these values."""
    draws = RandomVCGAugment(max_rotation_deg=45.0, max_scale=1.5, p=0.9).draw(
        np.random.default_rng(0), 3
    )
    got = [
        (d.applied, d.transform.axes_order, tuple(round(a, 6) for a in d.transform.angles_deg),
         tuple(round(s, 6) for s in d.transform.scales))
        for d in draws
    ]
    assert got == REFERENCE_DRAWS_SEED_0, got


# Recorded on 2026-08-17 with the implementation of T023 and frozen since; see research R5 for the
# draw order. If this test fails, the draw order or law changed — that is a breaking change.
REFERENCE_DRAWS_SEED_0 = [
    (True, ("y", "z", "x"), (-41.312383, -43.512513, 28.194322), (1.456378, 1.303318, 1.364748)),
    (True, ("x", "z", "y"), (32.166385, -41.977298, 20.66899), (0.919263, 0.698524, 0.786949)),
    (True, ("x", "y", "z"), (13.247056, 10.38466, -10.46902), (1.498605, 1.490418, 0.744729)),
]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rotation_deg": -1.0},
        {"max_rotation_deg": 181.0},
        {"max_scale": 0.9},
        {"p": -0.1},
        {"p": 1.5},
        {"mode": "nope"},
        {"method": "nope"},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        RandomVCGAugment(**kwargs)


def test_int_seed_accepted(ecg_standard):
    a = RandomVCGAugment()(ecg_standard, leads="standard", rng=123)
    b = RandomVCGAugment()(ecg_standard, leads="standard", rng=np.random.default_rng(123))
    np.testing.assert_array_equal(a, b)


def test_single_ecg_and_batch_agree(ecg_standard):
    """One draw per batch element; a single (12, T) ECG counts as a batch of one."""
    aug = RandomVCGAugment()
    single = aug(ecg_standard, leads="standard", rng=9)
    batch = aug(ecg_standard[None], leads="standard", rng=9)
    assert single.shape == ecg_standard.shape and batch.shape == (1, *ecg_standard.shape)
    np.testing.assert_array_equal(batch[0], single)


def test_output_dtype_and_leads_axis(ecg_standard):
    aug = RandomVCGAugment()
    x = ecg_standard.T.astype(np.float32).copy()  # (T, 12)
    out = aug(x, leads="standard", rng=0, leads_axis=-1)
    assert out.shape == x.shape and out.dtype == np.float32
    ref = aug(ecg_standard.astype(np.float32), leads="standard", rng=0)
    np.testing.assert_allclose(out.T, ref, rtol=1e-6)


def test_transform_reproducible_from_draw(ecg_standard):
    """A Draw is enough to replay a transformation exactly."""
    aug = RandomVCGAugment()
    out, (d,) = aug(ecg_standard, leads="standard", rng=42, return_params=True)
    replay = Transform(**{k: getattr(d.transform, k) for k in ("angles_deg", "axes_order",
                                                             "scales", "order")})
    np.testing.assert_array_equal(replay.apply_ecg(ecg_standard, leads="standard"), out)

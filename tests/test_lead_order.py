"""The same transformation gives the same physics whatever the declared lead order."""

import numpy as np

from ecg3kg import Transform, resolve_leads, vcg_to_ecg


def _reorder(ecg, src, dst):
    return ecg[[src.index(name) for name in dst]]


def test_lead_order_invariance(ecg_mimic, ecg_standard):
    mimic, std = resolve_leads("mimic"), resolve_leads("standard")
    t = Transform(angles_deg=(25, -40, 10), scales=(1.3, 0.8, 1.1))
    out_mimic = t.apply_ecg(ecg_mimic, leads="mimic")
    out_std = t.apply_ecg(ecg_standard, leads="standard")
    np.testing.assert_allclose(_reorder(out_mimic, mimic, std), out_std, rtol=1e-12, atol=1e-12)


def test_independent8_in_out(ecg_independent8, ecg_standard, vcg_synthetic):
    t = Transform(angles_deg=(25, -40, 10), scales=(1.3, 0.8, 1.1))
    out8 = t.apply_ecg(ecg_independent8, leads="independent8")
    assert out8.shape == ecg_independent8.shape == (8, vcg_synthetic.shape[-1])
    out12 = t.apply_ecg(ecg_standard, leads="standard")
    assert out12.shape == (12, vcg_synthetic.shape[-1])
    # The 8 independent leads agree between the two calls.
    std, ind8 = resolve_leads("standard"), resolve_leads("independent8")
    np.testing.assert_allclose(_reorder(out12, std, ind8), out8, rtol=1e-12, atol=1e-12)
    # vcg_to_ecg respects the requested order size.
    assert vcg_to_ecg(vcg_synthetic, leads="independent8").shape == (8, vcg_synthetic.shape[-1])
    assert vcg_to_ecg(vcg_synthetic, leads="mimic").shape == (12, vcg_synthetic.shape[-1])

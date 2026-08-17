"""Lead orders: presets, validation, selection and derivation matrices."""

import numpy as np
import pytest

from ecg3kg import leads as L


def test_presets():
    assert L.resolve_leads("mimic") == (
        "I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6",
    )
    assert L.resolve_leads("standard") == (
        "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
    )
    assert L.resolve_leads("independent8") == ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


def test_free_sequence_accepted():
    order = ["V6", "V5", "V4", "V3", "V2", "V1", "II", "I", "aVF"]
    assert L.resolve_leads(order) == tuple(order)


@pytest.mark.parametrize(
    "bad",
    [
        "MIMIC-IV",  # unknown preset
        ["I", "II", "V1", "V2", "V3", "V4", "V5", "V5"],  # duplicate
        ["I", "II", "V1", "V2", "V3", "V4", "V5"],  # V6 missing
        ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6", "V7"],  # unknown name
        [],
        None,
    ],
)
def test_invalid_orders_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        L.resolve_leads(bad)


def test_selection_matrix_extracts_independent_leads_in_canonical_order():
    order = L.resolve_leads("mimic")
    S = L.selection_matrix(order)
    assert S.shape == (8, 12)
    x = np.arange(12.0)  # lead k has value k
    picked = S @ x
    expected = [order.index(name) for name in L.INDEPENDENT_LEADS]
    np.testing.assert_array_equal(picked, expected)


def test_derivation_matrix_reconstructs_derived_leads_exactly():
    order = L.resolve_leads("standard")
    G = L.derivation_matrix(order)
    assert G.shape == (12, 8)
    rng = np.random.default_rng(0)
    ecg8 = rng.normal(size=8)  # I, II, V1..V6
    ecg12 = G @ ecg8
    i, ii = ecg8[0], ecg8[1]
    idx = {n: k for k, n in enumerate(order)}
    assert ecg12[idx["I"]] == i
    assert ecg12[idx["II"]] == ii
    np.testing.assert_allclose(ecg12[idx["III"]], ii - i, rtol=0, atol=1e-15)
    np.testing.assert_allclose(ecg12[idx["aVR"]], -(i + ii) / 2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(ecg12[idx["aVL"]], i - ii / 2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(ecg12[idx["aVF"]], ii - i / 2, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(ecg12[[idx[f"V{k}"] for k in range(1, 7)]], ecg8[2:])


def test_selection_then_derivation_is_identity_on_consistent_ecg():
    order = L.resolve_leads("mimic")
    S, G = L.selection_matrix(order), L.derivation_matrix(order)
    ecg8 = np.random.default_rng(1).normal(size=8)
    ecg12 = G @ ecg8
    np.testing.assert_allclose(G @ (S @ ecg12), ecg12, atol=1e-15)


def test_move_leads_axis_roundtrip():
    x = np.random.default_rng(2).normal(size=(4, 500, 12))
    moved, restore = L.move_leads_axis(x, -1)
    assert moved.shape == (4, 12, 500)
    np.testing.assert_array_equal(restore(moved), x)

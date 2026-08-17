"""Projection matrices: shapes, rank, pseudo-inverse consistency, published values, sensitivity."""

import numpy as np
import pytest

from ecg3kg import matrices as m

# Inverse Dower as published (Edenbrandt & Pahlm 1988; reproduced in Jaros et al. 2019, Table 1,
# and in Clifford's idowerT.m). Rows X, Y, Z; columns V1..V6, I, II.
INVERSE_DOWER_PUBLISHED_V1_TO_II = np.array(
    [
        [-0.172, -0.074, 0.122, 0.231, 0.239, 0.194, 0.156, -0.010],
        [0.057, -0.019, -0.106, -0.022, 0.041, 0.048, -0.227, 0.887],
        [-0.229, -0.310, -0.246, -0.063, 0.055, 0.108, 0.022, 0.102],
    ]
)
# Column permutation from (V1..V6, I, II) to the package's canonical (I, II, V1..V6).
_PUB_TO_CANONICAL = [6, 7, 0, 1, 2, 3, 4, 5]


def test_dower_shape_and_rank():
    assert m.DOWER.shape == (8, 3)
    assert np.linalg.matrix_rank(m.DOWER) == 3
    assert m.DOWER.dtype == np.float64
    assert not m.DOWER.flags.writeable


def test_inverse_dower_is_left_inverse():
    assert m.INVERSE_DOWER.shape == (3, 8)
    np.testing.assert_allclose(m.INVERSE_DOWER @ m.DOWER, np.eye(3), atol=1e-10)


def test_pinv_matches_published_inverse_dower():
    published = INVERSE_DOWER_PUBLISHED_V1_TO_II[:, _PUB_TO_CANONICAL]
    np.testing.assert_allclose(m.INVERSE_DOWER, published, atol=1e-2)
    # Tighter than the spec asks: the published table is a rounding of pinv(D).
    assert np.abs(m.INVERSE_DOWER - published).max() < 1e-3


def test_kors_shape():
    assert m.KORS.shape == (3, 8)
    assert m.KORS.dtype == np.float64
    assert not m.KORS.flags.writeable
    # Kors is a regression matrix, not an inverse of Dower — but K·D must be well conditioned.
    kd = m.KORS @ m.DOWER
    assert np.linalg.matrix_rank(kd) == 3


def test_independent_leads_order():
    assert m.INDEPENDENT_LEADS == ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


@pytest.mark.parametrize("row,col", [(0, 0), (4, 2), (7, 1)])
def test_matrix_sensitivity(row, col):
    """Altering a single Dower coefficient breaks the match with the published inverse."""
    perturbed = m.DOWER.copy()
    perturbed[row, col] += 0.05
    pinv = np.linalg.pinv(perturbed)
    published = INVERSE_DOWER_PUBLISHED_V1_TO_II[:, _PUB_TO_CANONICAL]
    assert np.abs(pinv - published).max() > 1e-3

"""Synthetic fixtures: a known VCG and its Dower projections in the three lead orders."""

import numpy as np
import pytest

from ecg3kg import DOWER, INVERSE_DOWER, resolve_leads
from ecg3kg.leads import derivation_matrix

T = 500


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def vcg_synthetic():
    """A 3 × T loop with non-zero components on all three axes."""
    t = np.linspace(0.0, 2.0 * np.pi, T)
    x = np.sin(t) + 0.3 * np.sin(3 * t)
    y = 0.8 * np.cos(t)
    z = 0.5 * np.sin(2 * t) + 0.2
    return np.stack([x, y, z]).astype(np.float64)


def _project(vcg, preset):
    order = resolve_leads(preset)
    return derivation_matrix(order) @ (DOWER @ vcg)


@pytest.fixture
def ecg_standard(vcg_synthetic):
    return _project(vcg_synthetic, "standard")


@pytest.fixture
def ecg_mimic(vcg_synthetic):
    return _project(vcg_synthetic, "mimic")


@pytest.fixture
def ecg_independent8(vcg_synthetic):
    return _project(vcg_synthetic, "independent8")


@pytest.fixture
def ecg_with_residual(vcg_synthetic, rng):
    """Standard-order ECG plus a component outside the image of DOWER (a null vector of pinv(D)).

    Such a residual is invisible to the VCG reconstruction: INVERSE_DOWER @ residual8 == 0. It is
    what a real ECG carries beyond the dipole model. Derived leads stay consistent.
    """
    # Null space of INVERSE_DOWER (3 × 8): 5-dimensional in R^8.
    _, _, vt = np.linalg.svd(INVERSE_DOWER)
    null = vt[3:]  # (5, 8), rows span the null space
    residual8 = null.T @ rng.normal(size=5)  # a random null-space vector, shape (8,)
    profile = np.sin(np.linspace(0, 6 * np.pi, T))
    residual8 = 0.1 * residual8[:, None] * profile[None, :]  # (8, T)
    order = resolve_leads("standard")
    ecg8 = DOWER @ vcg_synthetic  # (8, T), canonical order
    return derivation_matrix(order) @ (ecg8 + residual8)  # derived once: exactly consistent

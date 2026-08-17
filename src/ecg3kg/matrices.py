"""Projection matrices between the 12-lead ECG and the vectorcardiogram (VCG).

The 12-lead ECG carries only 8 independent signals: I, II and V1–V6 (III, aVR, aVL, aVF are linear
combinations of I and II). The VCG represents the cardiac dipole on three orthogonal axes X, Y, Z.

* ``DOWER`` (8 × 3) maps a VCG to the 8 independent leads: ``ecg8 = DOWER @ vcg``. Coefficients from
  Dower et al. (1980), based on Frank's torso model, as reproduced in G. D. Clifford's ``idowerT.m``
  (ecgtools, 2006) and in Vondrak & Penhaker 2022 (Front. Physiol. 13:856590, Table 3).
* ``INVERSE_DOWER`` (3 × 8) is the Moore–Penrose pseudo-inverse of ``DOWER`` (Edenbrandt & Pahlm
  1988). It is *computed* here rather than typed in; the published table (Jaros, Martinek & Danys
  2019, Sensors 19(14):3072, Table 1) is what the test-suite checks it against.
* ``KORS`` (3 × 8) is the regression matrix of Kors et al. (1990), an empirical alternative
  reconstruction; values from Jaros et al. 2019, Table 2. It is *not* an inverse of ``DOWER``.

All matrices use the canonical lead order ``INDEPENDENT_LEADS`` = (I, II, V1, V2, V3, V4, V5, V6)
for their lead dimension, and (X, Y, Z) for the VCG dimension. Verified against sources on
2026-08-17 (see the project spec, research R1).
"""

from __future__ import annotations

import numpy as np

INDEPENDENT_LEADS: tuple[str, ...] = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
"""Canonical order of the 8 independent leads used by every matrix in this module."""

VCG_AXES: tuple[str, str, str] = ("x", "y", "z")


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.flags.writeable = False
    return a


DOWER: np.ndarray = _frozen(
    [
        # X        Y        Z
        [0.632, -0.235, 0.059],  # I
        [0.235, 1.066, -0.132],  # II
        [-0.515, 0.157, -0.917],  # V1
        [0.044, 0.164, -1.387],  # V2
        [0.882, 0.098, -1.277],  # V3
        [1.213, 0.127, -0.601],  # V4
        [1.125, 0.127, -0.086],  # V5
        [0.831, 0.076, 0.230],  # V6
    ]
)
"""Dower matrix D (8 × 3): ``ecg8 = D @ vcg``, leads in ``INDEPENDENT_LEADS`` order."""

INVERSE_DOWER: np.ndarray = _frozen(np.linalg.pinv(DOWER))
"""Inverse Dower matrix (3 × 8) = pinv(D): ``vcg = INVERSE_DOWER @ ecg8``."""

KORS: np.ndarray = _frozen(
    [
        # I      II     V1     V2     V3     V4     V5     V6
        [0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],  # X
        [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],  # Y
        [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31],  # Z
    ]
)
"""Kors regression matrix (3 × 8): ``vcg = KORS @ ecg8``, leads in ``INDEPENDENT_LEADS`` order."""

RECONSTRUCTIONS: dict[str, np.ndarray] = {"dower": INVERSE_DOWER, "kors": KORS}
"""Available ECG → VCG reconstructions, by name."""

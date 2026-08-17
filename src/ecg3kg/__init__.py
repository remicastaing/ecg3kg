"""ecg3kg — 3D augmentation of 12-lead ECGs through the vectorcardiogram (3KG).

The twelve leads of an ECG are projections of one 3D cardiac dipole (the vectorcardiogram, VCG).
This package reconstructs that dipole from the 8 independent leads, rotates and scales it, and
projects it back — the augmentation of Gopal et al., *3KG*, ML4H 2021.
"""

from .leads import LEAD_PRESETS, resolve_leads
from .matrices import DOWER, INDEPENDENT_LEADS, INVERSE_DOWER, KORS

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DOWER",
    "INDEPENDENT_LEADS",
    "INVERSE_DOWER",
    "KORS",
    "LEAD_PRESETS",
    "resolve_leads",
]

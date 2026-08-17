"""Governance: the public repository must never track a signal, embedding or data file.

The package is developed alongside a project whose data (PhysioNet / MIMIC-IV) is under a data use
agreement. Tests and examples use synthetic signals only. This test fails if git tracks a file with
a data-like extension, or if a source file mentions MIMIC-specific paths.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_EXT = re.compile(
    r"\.(parquet|npy|npz|dat|hea|mat|csv|h5|hdf5|pkl|pickle|feather|arrow)$", re.I
)
FORBIDDEN_TEXT = ("DATA_ROOT", "physionet.org/files/mimic", "mimic-iv-ecg/1.0")


def _tracked_files():
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    return [line for line in out.splitlines() if line]


def test_no_data_files_tracked():
    offenders = [f for f in _tracked_files() if DATA_EXT.search(f)]
    assert not offenders, f"data-like files tracked by git: {offenders}"


def test_no_data_paths_in_sources():
    offenders = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "examples").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN_TEXT:
            if needle in text:
                offenders.append((str(path.relative_to(ROOT)), needle))
    assert not offenders, offenders

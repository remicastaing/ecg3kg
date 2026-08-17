"""Before/after figure for the README, on a synthetic ECG (no patient data involved).

Run:  python examples/plot_before_after.py [out.png]
Needs matplotlib (not a dependency of the package): pip install matplotlib
"""

from __future__ import annotations

import sys

import numpy as np

import ecg3kg


def synthetic_vcg(fs: int = 500, seconds: float = 2.0, hr_bpm: float = 72.0) -> np.ndarray:
    """A crude but recognisable dipole loop: P, QRS and T lobes, at 72 bpm. Shape (3, T)."""
    t = np.arange(int(fs * seconds)) / fs
    period = 60.0 / hr_bpm
    phase = (t % period) / period  # 0..1 within each beat
    v = np.zeros((3, t.size))

    def lobe(center, width, amp_xyz):
        g = np.exp(-0.5 * ((phase - center) / width) ** 2)
        for k in range(3):
            v[k] += amp_xyz[k] * g

    lobe(0.15, 0.030, (0.10, 0.12, 0.02))  # P
    lobe(0.30, 0.008, (-0.20, 0.10, 0.25))  # Q (small, forward)
    lobe(0.32, 0.012, (1.20, 0.90, -0.60))  # R (big, left-inferior-anterior)
    lobe(0.345, 0.010, (-0.35, 0.10, 0.45))  # S
    lobe(0.55, 0.060, (0.30, 0.25, -0.15))  # T
    return v


def main(out_path: str = "examples/out/before_after.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = 500
    vcg = synthetic_vcg(fs=fs)
    leads = "standard"
    ecg = ecg3kg.vcg_to_ecg(vcg, leads=leads)  # (12, T)
    t = ecg3kg.Transform(angles_deg=(30.0, 15.0, 0.0), scales=(1.2, 1.0, 0.9))
    aug = t.apply_ecg(ecg, leads=leads)

    order = ecg3kg.resolve_leads(leads)
    show = ["I", "II", "V1", "V5"]
    time = np.arange(ecg.shape[-1]) / fs
    fig, axes = plt.subplots(len(show), 1, figsize=(9, 6), sharex=True)
    for ax, name in zip(axes, show, strict=True):
        k = order.index(name)
        ax.plot(time, ecg[k], color="0.35", lw=1.2, label="before")
        ax.plot(time, aug[k], color="tab:red", lw=1.2, label="after")
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        "ecg3kg — dipole rotated (30°, 15°, 0°) and scaled (1.2, 1.0, 0.9): "
        "all leads move together (synthetic ECG)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])

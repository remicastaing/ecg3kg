# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## Unreleased

## 0.1.0 — 2026-08-17

First release: an independent implementation of the geometric augmentation of Gopal et al., *3KG* (ML4H 2021).

### Added

- ECG ↔ VCG conversion: `ecg_to_vcg` (Dower pseudo-inverse by default, Kors regression as `method="kors"`), `vcg_to_ecg` (derived leads III, aVR, aVL, aVF recomputed exactly).
- Projection matrices `DOWER`, `INVERSE_DOWER` (= `pinv(DOWER)`, checked against the published coefficients), `KORS`; coefficients verified against Jaros, Martinek & Danys 2019 (Sensors 19(14):3072, Tables 1–2) and G. D. Clifford's `idowerT.m` (ecgtools, 2006).
- `Transform`: deterministic rotation (three angles, any axes order) and per-axis scaling of the VCG, composed as `S·R` (default) or `R·S`; `apply_vcg`, `apply_ecg`, `matrix`, `compose`, `is_identity`.
- `apply_matrix_ecg` with two modes: `"residual"` (default; `x + D(M − I)D⁻¹x`, exact identity at `M = I`, keeps the non-dipolar residual) and `"project"` (`D M D⁻¹ x`, the paper's formula).
- `RandomVCGAugment`: the sampling law of the paper (angles ~ U[−r, r], axes order uniform, scales ~ U[1, s] inverted with probability ½), probability of application `p`, `rotate`/`scale` switches, one independent draw per batch element, generator injected (never global), draws returned as `Draw` objects. Elements not applied are returned bit for bit.
- Lead orders: presets `"mimic"`, `"standard"`, `"independent8"` or any explicit list containing the 8 independent leads; `leads_axis` for arrays with the leads on another axis. The order is mandatory — never guessed.
- Optional torch adapter `ecg3kg.torch` (`apply_ecg`, `apply_matrix_ecg`, `RandomVCGAugmentTorch`), same numbers as the numpy core, same draws.
- Test-suite of ~80 property tests (identity, 3 × 120° = identity, +90° x sends y → z, per-axis scale, non-commutativity, lead-order invariance, exact derived leads, VCG round-trip, draw distribution, determinism, independence, matrix sensitivity, torch = numpy), CI on Python 3.10–3.13, `CITATION.cff`, MIT license.

### Notes

- Release check from the GitHub tag (2026-08-17): fresh venv + `pip install "git+https://github.com/remicastaing/ecg3kg@v0.1.0"` + README example fetched from the tag = **5 s**; CI green on Python 3.10–3.13.
- Local rehearsal of the release checklist (2026-08-17, Apple M-series, before the GitHub push): fresh venv + install from the git repository + README example + clone + full test-suite = **17 s** end to end; the test-suite itself runs in **0.4 s** warm (12.7 s on the very first run of a fresh venv, which is numpy/scipy byte-compilation, not the tests); `import ecg3kg` takes 27 ms and does not import torch. To be re-measured from the GitHub tag once pushed.
- Package name `ecg3kg` checked on PyPI on 2026-08-17: not taken (HTTP 404 on `https://pypi.org/pypi/ecg3kg/json`). PyPI publication is prepared (`publish.yml`, trusted publishing) but not performed with this release.

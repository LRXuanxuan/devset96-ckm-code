# Reproduction notes

Environment used for every recorded number: **Windows 11, Python 3.12.14**, packages pinned in `requirements.txt`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "sionna_small_validation"
```

`ROOT` in M0/M1 resolves to the repository root, so place the dataset at `output/devset96_20261027/release_05/` or pass `--index`.

## M0 — static estimator control

```powershell
& .\.venv\Scripts\python.exe `
    .\algorithm_development\m0_baselines\run_bench.py `
    --out .\output\algorithm_m0_my_run
```

Expected: 8 validation windows × 2 beam modes × 3 nominal SNR = 48 conditions, 144 scores, ~15 s. Recorded output: `results/m0_run_02/`.

Reference macro-average NMSE (dB, `rotating`):

| Estimator | 0 dB | 10 dB | 20 dB |
|---|---:|---:|---:|
| LS | 10.35 | 1.25 | −4.19 |
| LMMSE | 2.26 | −1.31 | −4.49 |
| OMP | −2.54 | −4.94 | −6.70 |

## M1 — controlled dynamic model

```powershell
& .\.venv\Scripts\python.exe .\algorithm_development\m1_dynamics\run_controls.py --out .\output\algorithm_m1_my_controls
& .\.venv\Scripts\python.exe .\algorithm_development\m1_dynamics\run_sbl_comparison.py --out .\output\algorithm_m1_my_sbl
& .\.venv\Scripts\python.exe .\algorithm_development\m1_dynamics\run_grid_checks.py --out .\output\algorithm_m1_my_grid
```

Recorded: `run_sbl_comparison` → 45 pairs / 90 rows / 328.24 s / `status=passed`; 34 of 45 pairs met the stopping criterion, 11 hit the iteration cap. Recorded output: `results/m1_run_03_sbl_20261030/`.

## Checks

- Every script asserts its own internal controls and writes `status` into `results.json`; treat `passed` as “the script's own assertions held”, not as an external validation of the method.
- M0 asserts the CP condition on real dynamic data; if a path delay exceeds the CP, the run stops instead of silently truncating (this happened in `run_01`).

# devset96-ckm-code

Sionna-RT **96-route development dataset generator** and the **M0 / M1 algorithm code** used for dynamic channel-knowledge-map (CKM) and MIMO channel-estimation research.

> **Status.** Research code accompanying a manuscript under preparation. It is not a polished library: scripts are versioned snapshots (`_v2 … _v5`) and several internal artifacts (dataset, plans, RT qualification files) are intentionally not published. See [docs/PIPELINE.md](docs/PIPELINE.md) for what is required to regenerate data.

---

## 1. Layout

| Path | Content |
|---|---|
| `sionna_small_validation/validation/` | Dataset construction package: RT scene loading (`cities.py`), route/trajectory expansion (`devset_routes.py`), streaming HDF5 writer (`devset_storage_v5.py`), phase/CSI references, waveform & observation synthesis (`mimo_observation.py`, `dataset_observations.py`), dataset views (`devset_view*.py`) |
| `sionna_small_validation/run_devset*.py` | Generation entry points (v1–v5). `run_devset_v5.py` is the generator used for release 05 |
| `algorithm_development/m0_baselines/` | **M0**: frozen-channel static estimators — LS, spatial LMMSE, MMV-OMP; plus the dataset bridge and benchmark runner |
| `algorithm_development/m1_dynamics/` | **M1**: angle–delay–Doppler parametric model, OMP / complex-Gaussian EM-SBL estimators, grid tools and controlled comparisons |
| `results/` | Recorded result JSON/CSV of the M0 and M1 runs quoted in the manuscript |
| `docs/` | Dataset schema, generation pipeline, reproduction steps |

---

## 2. Dataset in one page

The development release (`release_05`) contains **96 route families** (train 64 / val 16 / test 16) generated with Sionna RT 2.2 on three built-in scenes (Munich, Florence, Étoile).

| Item | Value |
|---|---|
| Observation tensor | `full_fft` complex64 `(57, 16, 256)` = 3 bursts × 19 symbols, 16 Rx antennas, 256 FFT bins |
| Symbol timing | symbol period 320 samples = 10.4167 µs; CP 64 samples = 2.083 µs; fs = 30.72 MHz; SCS = 120 kHz |
| Target label | `target_csi` complex64 `(16, 64, 16)` = `[rx, tx, freq]`, one slice at t₀ + 2 ms; 16 frequencies (±12 MHz, 1.6 MHz spacing) of which 6 sit on FFT bins and 10 do not |
| Beams | DFT codebook `W` `(57, 64)`, columns 0…56 (`rotating`, rank 57) or rank 1 (`fixed`) |
| History map | 1024 = 16 (UE 4×4 angular DFT) × 16 (cyclic delay, **625 ns** period) × 4 (known Tx beams); **power only, no phase** |
| Survey | 4 designs × 128 static positions per family, 25 dB measurement noise, base environment |
| Motion | 72 material families / 24 vehicle families; UE speed 0/10/15/30 m/s = 10/13/39/34 families |
| CFO | one draw per family from N(0, 500²), unknown, absolute-time phase; realized range −1489 … +1124 Hz |
| CP overrun | 10.47 % of online states have at least one path delay > CP (max 4.753 µs) |

Full verified field list: [docs/DATASET.md](docs/DATASET.md).

---

## 3. Install

Tested on **Windows 11 + Python 3.12.14**. Sionna RT requires a matching Mitsuba / Dr.Jit pair.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

M0/M1 import `validation.*`, so the `sionna_small_validation/` directory must be importable. Either run every script from the repository root with `PYTHONPATH` set, or execute from inside that directory:

```powershell
$env:PYTHONPATH = "sionna_small_validation"
```

`ROOT` inside M0/M1 is resolved as `Path(__file__).resolve().parents[2]`, i.e. the repository root. The dataset is therefore expected at:

```text
<repo>/output/devset96_20261027/release_05/
```

(`run_bench.py --index` can point anywhere; the M1 controls contain their own controlled synthetic cases.)

---

## 4. Running M0

```powershell
$env:PYTHONPATH = "sionna_small_validation"
& .\.venv\Scripts\python.exe `
    .\algorithm_development\m0_baselines\run_bench.py `
    --out .\output\algorithm_m0_my_run
```

M0 evaluates three static estimators on a **frozen direct-cutoff channel** (Doppler and CFO set to 0) over 8 validation windows × 2 beam modes × nominal SNR 0/10/20 dB. Recorded run: 48 conditions / 144 scores / 15.32 s.

## 5. Running M1

```powershell
$env:PYTHONPATH = "sionna_small_validation"
& .\.venv\Scripts\python.exe .\algorithm_development\m1_dynamics\run_controls.py --out .\output\algorithm_m1_my_controls
& .\.venv\Scripts\python.exe .\algorithm_development\m1_dynamics\run_sbl_comparison.py --out .\output\algorithm_m1_my_sbl
& .\.venv\Scripts\python.exe .\algorithm_development\m1_dynamics\run_grid_checks.py --out .\output\algorithm_m1_my_grid
```

Recorded SBL comparison: 45 pairs / 90 rows, 328.24 s, `status=passed`.

---

## 6. Scope and evidence boundaries (read before citing numbers)

1. **M0 is not a 6 ms predictor.** It scores the frozen channel at the cutoff time t₀ (`label_scope="frozen direct-cutoff channel on 256 FFT frequencies; not future CSI"`). Real dynamic data are used only for diagnostics.
2. **Nominal SNR ≠ realized SNR.** The nominal reference is the family's historical survey average power; per-window offset can be −20 dB in the tail. Always state which one a table uses.
3. **Residual ≠ correctness.** Under `rotating` beams, `W W† = I₅₇`, so the LS observation residual is numerically zero even when channel NMSE is large.
4. **M0 windows are `pass_index=1`** (6 wood façades, 2 with vehicles), not the base environment.
5. **M1 is a controlled single/double-path model**, not a full 96-route future-CSI evaluation.
6. **The history map has power only** and a 625 ns periodic delay ambiguity; it is a prior, not ground truth.

---

## 7. Not included

- HDF5 dataset (`release_05/`, ~GB scale) and RT artifacts.
- Route plans and V4 qualification evidence (`plan_096_qualified_*.json`, screening reports, source locks). `run_devset_v5.py` verifies SHA-256 bindings to those files by design, so full regeneration requires them.
- Trained checkpoints (none are used by M0/M1).

The dataset itself can be requested for academic reproduction; the generation chain is otherwise complete in this repository.

---

## 8. License and citation

Code is released under the MIT License (see [LICENSE](LICENSE)). If you use the dataset or algorithms, please cite the associated manuscript once published; until then cite this repository URL.

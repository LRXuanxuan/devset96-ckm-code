# Recorded results

These files are the exact outputs quoted in the manuscript draft. They are kept for provenance, not as a benchmark.

| File | Content |
|---|---|
| `m0_run_02/results.json` | M0 run over 8 validation windows × 2 beam modes × nominal SNR 0/10/20 dB; `label_scope="frozen direct-cutoff channel on 256 FFT frequencies; not future CSI"`; `test_split_accessed=false`; `new_RT_calls=0` |
| `m0_run_02/static_control_scores.csv` | Per-sample M0 scores |
| `m1_run_03_sbl_20261030/results.json` | M1 OMP vs complex-Gaussian EM-SBL comparison, 45 pairs / 90 rows, `status=passed`, 328.24 s |

Notes:

- All M0 windows are `pass_index=1` (6 wood façades, 2 with a car), i.e. **not** the base environment.
- Nominal SNR is referenced to the family survey average power; per-window offsets are reported inside the JSON.
- The M0 JSON contains no future-CSI prediction scores; do not relabel it as such.

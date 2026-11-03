# Dataset schema — devset96 (`release_05`)

All numbers below were verified against the generated HDF5 files and the generation reports on 2026-10-29. They describe the **96-route development release**, not a public benchmark.

## 1. Scale

| Item | Value |
|---|---|
| Route families | 96 = train 64 / validation 16 / test 16 |
| Admitted tasks | 194,715 (train 127,024 / val 38,316 / test 29,375) |
| `eval_selected` | 26,622 |

Each family is one HDF5 shard under `routes/{train,validation,test}/<family>.h5` and contains `paths`, `states`, `survey_features`, `tasks`, `golden_cfr`.

## 2. Observation tensor

- `input.observation.full_fft`: complex64 `(57, 16, 256)` = 3 bursts × 19 OFDM symbols × 16 Rx × 256 FFT bins.
- Burst start times: 0, 1.901, 3.802 ms. Each burst lasts 198 µs; the gap between bursts is 1.70 ms.
- Symbol period: 320 samples = 10.4167 µs; CP: 64 samples = 2.083 µs; `fs` = 30.72 MHz; SCS = 120 kHz.

## 3. Arrays and beams

- BS array: 8 × 8 = 64 elements; UE array: 4 × 4 = 16 elements; half-wavelength spacing 5.3534 mm at 28 GHz; isotropic single-V polarization; `synthetic_array=True`.
- Pilot beamforming matrix `W` `(57, 64)` = columns 0…56 of the DFT codebook.
  - `rotating`: rank 57 (columns 57…63 are missing, i.e. 7 consecutive directions).
  - `fixed`: rank 1.

## 4. Label

- `target_csi`: complex64 `(16, 64, 16)` with axis order `[rx, tx, freq]`.
- It is a **single time slice** at `t₀ + 2 ms`, not a trajectory.
- The 16 frequency points span ±12 MHz with 1.6 MHz spacing; **6 lie exactly on FFT bins and 10 do not**, so a nearest-bin substitution is not exact.

## 5. History map (channel knowledge map)

- Shape 1024 = 16 (UE 4×4 angular DFT) × 16 (cyclic delay, period **625 ns**) × 4 (known Tx DFT beams).
- **Power only — no instantaneous phase.**
- Query: nearest 16 survey points within a 5 m bandwidth; effective sample count `= 1/Σw²`.
- The map is always built from the **base** environment, even for `pass1` states.

## 6. Survey

- 4 designs × 128 static positions per family (planned 49,152, qualified 48,340).
- UE is static with a fixed orientation; base environment; 25 dB measurement noise.

## 7. Environment and motion

- 72 material families (pass1 replaces façades with wood) and 24 vehicle families (pass1 adds a car).
- Cities: Munich 40, Florence 40, Étoile 16.
- UE speed 0 / 10 / 15 / 30 m/s in 10 / 13 / 39 / 34 families respectively; the car moves at 8 m/s.

## 8. Impairments

- **CFO**: one draw per family from N(0, 500²) Hz, unknown to the receiver, applied as an absolute-time phase `e^{j2π ε t}`. Realized values (recomputed from the seeds) span −1489 … +1124 Hz; 12 of 96 families have |ε| > 800 Hz.
- **ISI**: 10.47 % of online states have at least one path delay exceeding the CP (12.95 % over all states); 20 of 96 shards contain such paths; maximum 4.753 µs (`munich_family_031`).

## 9. Known interpretation traps

1. The map has no phase and a 625 ns periodic delay ambiguity; it is a prior, never a label.
2. Nominal SNR refers to the survey average power, not the per-task received power.
3. `rotating` beams make the LS residual identically zero (`W W† = I₅₇`), so residual-based validation must use `fixed` beams.
4. `target_csi` is at a single future instant; do not read it as a continuous trajectory.

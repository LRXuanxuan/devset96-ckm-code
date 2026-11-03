# Generation pipeline

The dataset was produced by a staged pipeline. Only the **code** is published here; the dataset, route plans, and qualification evidence are not.

## Stage 0 — Route planning

Output: a plan JSON (`plan_096_qualified_*.json`) with one record per route family:

```json
{
  "route_family_id": "...",
  "city": "munich | florence | etoile",
  "split": "train | validation | test",
  "v4_qualification": {
    "report_path": "...", "report_sha256": "...",
    "plan_path": "...",   "plan_sha256": "...",
    "source_lock_path": "...", "source_lock_sha256": "..."
  }
}
```

`run_devset_v5.py::validate_qualification` checks that every binding is present and that the SHA-256 of each referenced file is unchanged before any ray tracing starts.

## Stage 1 — Physical screening / qualification

Sionna RT is used to screen candidate geometries (path existence, delay spread, CP coverage, LoS/NLoS intent) and to write a screening report plus a source lock that pins the exact runtime sources used for the screen. Those artifacts are referenced by the plan and are **not** in this repository.

## Stage 2 — Generation

`run_devset_v5.py` (adapter) drives `validation/devset_generate_v5.py::generate_route` through `validation/devset_runtime_v5.py`:

```powershell
& .\.venv\Scripts\python.exe .\sionna_small_validation\run_devset_v5.py `
    --plan   .\output\devset96_20261027\plans\plan_096_qualified_20261028.json `
    --output .\output\devset96_20261027\release_05 `
    --backend llvm_ad_mono_polarized `
    --workers 8 --samples-per-src 100000 --retry-samples 300000 `
    --stage-limit 96 --resume
```

Key arguments: `--batch-start/--batch-stop` split the 96 families, `--worker-slot` shards them across worker processes, `--resume` allows restarting on existing shards (each shard is audited before reuse), and `--audit-only` skips writing.

## Stage 3 — Storage and per-shard audit

`validation/devset_storage_v5.py` streams truth, observations, and map records into one HDF5 per family, writing `.report.json` alongside. `audit_route_file_v5()` re-reads the shard and checks the storage contract (schema, hashes, burst timing, CP accounting).

## Stage 4 — Views and observation synthesis

- `validation/devset_view_v5.py` — dataset view used by the algorithm side.
- `validation/devset_view.py` — noise scaling and map conventions (noise calibration around L248).
- `validation/dataset_observations.py` — `_bursts`: 3 bursts × 19 symbols.
- `validation/mimo_observation.py` — `make_schedule`: QPSK pilots + DFT beams; `simulate_schedule`: observation synthesis including CFO and CP-limited ISI.

## Stage 5 — Algorithm bridge

`algorithm_development/m0_baselines/devset_bridge.py` reads `task_index.h5` and the family shards, builds training covariance statistics, and exposes validation cases. M1 controls additionally use `validation.cities.reconstruct` and the same observation synthesis.

## What would be needed to regenerate from scratch

1. The qualified plan JSON and its V4 qualification report + source lock (path, hash, and budget binding).
2. A Sionna RT 2.2.0 environment (see `requirements.txt`).
3. Ray-tracing budget and disk space: the 96-family release is GB-scale.

Steps 1 is environment-specific; contact the author for academic reproduction requests.

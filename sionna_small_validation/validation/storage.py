"""Small, explicit HDF5 I/O validation; no ray tracing is performed here.

Call ``run_storage`` from a script protected by ``if __name__ == '__main__'``.
The multiprocessing checks deliberately use spawn, including on Linux.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import traceback
from typing import Any, Mapping

import h5py
import numpy as np


SCHEMA_NAME = "sionna_small_validation.ragged_paths"
SCHEMA_VERSION = "1.0"
VALUE_FIELDS = ("gain_real", "gain_imag", "delay_sec", "doppler_hz", "path_id")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(value: Any, dtype: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result)


def _synthetic_payload() -> dict[str, Any]:
    records = []
    # Empty and uneven records exercise offsets, padding-free storage and chunks.
    for index, count in enumerate((3, 0, 6, 1, 9, 2, 5, 8, 4, 7, 10)):
        path = np.arange(count, dtype=np.float32)
        gain = (100 * index + path + 0.25) + 1j * (-1000 - 10 * index - 3 * path - 0.5)
        records.append(
            {
                "trajectory_id": index // 4,
                "time_sec": index * 0.00013,
                "gain": gain.astype(np.complex64),
                "delay_sec": (index + 1 + np.arange(count)) * 1e-8,
                "doppler_hz": 37 * path - 91 * index,
                "path_id": np.arange(count, dtype=np.int64),
            }
        )
    return {
        "kind": "synthetic_io_fixture",
        "provenance": {"description": "Deterministic storage fixture; not RF or RT data"},
        "records": records,
    }


def _normalize_payload(payload: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    supplied = dict(_synthetic_payload() if payload is None else payload)
    raw_records = supplied.get("records")
    if not isinstance(raw_records, (list, tuple)) or not raw_records:
        raise ValueError("payload.records must be a nonempty sequence of record mappings")
    records = []
    for index, source in enumerate(raw_records):
        if not isinstance(source, Mapping):
            raise ValueError(f"record {index} must be a mapping")
        if "gain" in source:
            gain = _vector(source["gain"], np.complex64, "gain")
        elif "gain_real" in source and "gain_imag" in source:
            real = _vector(source["gain_real"], np.float32, "gain_real")
            imag = _vector(source["gain_imag"], np.float32, "gain_imag")
            if real.shape != imag.shape:
                raise ValueError(f"record {index}: real/imag shapes differ")
            gain = np.asarray(real + 1j * imag, dtype=np.complex64)
        else:
            raise ValueError(f"record {index}: missing gain or gain_real/gain_imag")
        delay = _vector(source["delay_sec"], np.float64, "delay_sec")
        if delay.shape != gain.shape or np.any(delay < 0):
            raise ValueError(f"record {index}: delays must match gain and be nonnegative")
        doppler = _vector(source.get("doppler_hz", np.zeros(gain.size)), np.float32, "doppler_hz")
        path_id = _vector(source.get("path_id", np.arange(gain.size)), np.int64, "path_id")
        if doppler.shape != gain.shape or path_id.shape != gain.shape:
            raise ValueError(f"record {index}: path field lengths differ")
        time_sec = float(source["time_sec"])
        if not np.isfinite(time_sec):
            raise ValueError(f"record {index}: time_sec must be finite")
        records.append(
            {
                "trajectory_id": int(source.get("trajectory_id", 0)),
                "time_sec": time_sec,
                "gain_real": np.ascontiguousarray(gain.real),
                "gain_imag": np.ascontiguousarray(gain.imag),
                "delay_sec": delay,
                "doppler_hz": doppler,
                "path_id": path_id,
                "doppler_provided": "doppler_hz" in source,
                "path_id_provided": "path_id" in source,
            }
        )
    metadata = {
        "kind": str(supplied.get("kind", "provided_path_records")),
        "provenance": supplied.get("provenance", {}),
        "storage_validation_does_not_validate_rf_physics": True,
        "numeric_casts": {"gain": "complex64 split as two float32", "delay_sec": "float64", "doppler_hz": "float32"},
    }
    return metadata, records


def _sentinel() -> np.ndarray:
    i, j, k = np.indices((3, 5, 7), dtype=np.float32)
    return np.asarray((1000 * i + 100 * j + k + 0.125) + 1j * (-3 * i - 11 * j - 7 * k - 0.375), dtype=np.complex64)


def _dataset(group: h5py.Group, name: str, values: np.ndarray, axes: list[str], units: str, chunk: int = 8) -> h5py.Dataset:
    array = np.asarray(values)
    options: dict[str, Any] = {}
    if array.ndim == 1 and array.size:
        options = {"chunks": (min(chunk, array.size),), "compression": "gzip", "compression_opts": 1, "fletcher32": True}
    node = group.create_dataset(name, data=array, **options)
    node.attrs["axes"] = json.dumps(axes)
    node.attrs["units"] = units
    return node


def _write_h5(path: Path, metadata: dict[str, Any], records: list[dict[str, Any]]) -> np.ndarray:
    lengths = np.array([item["gain_real"].size for item in records], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_name"] = SCHEMA_NAME
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["storage_order"] = "python_native_explicit_axes"
        handle.attrs["complex_encoding"] = "split_real_imag_float32"
        handle.attrs["data_origin"] = metadata["kind"]
        handle.attrs["provenance_json"] = json.dumps(metadata, default=_json_default, sort_keys=True)
        handle.attrs["complete"] = False
        record_group = handle.create_group("records")
        _dataset(record_group, "path_offsets", offsets, ["record_boundary"], "path_index")
        _dataset(record_group, "time_sec", np.array([r["time_sec"] for r in records], dtype=np.float64), ["record"], "s")
        _dataset(record_group, "trajectory_id", np.array([r["trajectory_id"] for r in records], dtype=np.int64), ["record"], "identifier")
        _dataset(record_group, "doppler_provided", np.array([r["doppler_provided"] for r in records], dtype=np.bool_), ["record"], "boolean")
        _dataset(record_group, "path_id_provided", np.array([r["path_id_provided"] for r in records], dtype=np.bool_), ["record"], "boolean")
        paths = handle.create_group("paths")
        units = {"gain_real": "dimensionless_input_coefficient", "gain_imag": "dimensionless_input_coefficient", "delay_sec": "s", "doppler_hz": "Hz", "path_id": "identifier_local_to_record_unless_provenance_states_otherwise"}
        for field in VALUE_FIELDS:
            node = _dataset(paths, field, np.concatenate([r[field] for r in records]), ["flat_path"], units[field])
            if field.startswith("gain_"):
                node.attrs["complex_pair"] = "gain_real + 1j*gain_imag"
        fixture = handle.create_group("io_fixture")
        fixture.attrs["origin"] = "synthetic_io_sentinel_not_rf_data"
        sentinel = _sentinel()
        for suffix, array in (("real", sentinel.real), ("imag", sentinel.imag)):
            node = fixture.create_dataset("sentinel_" + suffix, data=array, chunks=(1, 2, 3), compression="gzip", fletcher32=True)
            node.attrs["axes"] = json.dumps(["sample", "row", "column"])
            node.attrs["units"] = "synthetic_arbitrary_unit"
        handle.attrs["complete"] = True
        handle.flush()
    return offsets


class LazyPathReader:
    """A path-only pickleable reader; every process opens its own HDF5 handle."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).resolve())
        self._handle: h5py.File | None = None
        self._pid: int | None = None

    def _open(self) -> h5py.File:
        if self._pid != os.getpid():
            self.close()
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
            self._pid = os.getpid()
            if self._handle.attrs.get("schema_name") != SCHEMA_NAME or not bool(self._handle.attrs.get("complete", False)):
                self.close()
                raise ValueError("Not a completed native path-schema file")
        return self._handle

    def __len__(self) -> int:
        return int(self._open()["records/time_sec"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        handle = self._open()
        if index < 0 or index >= handle["records/time_sec"].shape[0]:
            raise IndexError(index)
        start, stop = (int(x) for x in handle["records/path_offsets"][index:index + 2])
        record = {field: np.asarray(handle[f"paths/{field}"][start:stop]) for field in VALUE_FIELDS}
        record["time_sec"] = float(handle["records/time_sec"][index])
        record["trajectory_id"] = int(handle["records/trajectory_id"][index])
        record["doppler_provided"] = bool(handle["records/doppler_provided"][index])
        record["path_id_provided"] = bool(handle["records/path_id_provided"][index])
        return record

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "_handle": None, "_pid": None}

    def __enter__(self) -> "LazyPathReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _record_digest(record: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for field in (*VALUE_FIELDS, "time_sec", "trajectory_id", "doppler_provided", "path_id_provided"):
        value = np.asarray(record[field])
        digest.update(json.dumps([field, list(value.shape), value.dtype.str]).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _worker_read(path: str, indices: list[int], results: Any) -> None:
    try:
        with LazyPathReader(path) as reader:
            digests = [_record_digest(reader[index]) for index in indices]
            results.put({"pid": os.getpid(), "opened_in_pid": reader._pid, "digests": digests, "ok": True})
    except Exception:
        results.put({"pid": os.getpid(), "ok": False, "error": traceback.format_exc()})


def _process_parity(path: Path, indices: list[int], expected: list[str], count: int) -> dict[str, Any]:
    if count == 0:
        with LazyPathReader(path) as reader:
            observed = [_record_digest(reader[index]) for index in indices]
        return {"processes": 0, "ok": observed == expected, "pid": os.getpid()}
    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [context.Process(target=_worker_read, args=(str(path), indices, results)) for _ in range(count)]
    received = []
    try:
        for process in workers:
            process.start()
        for _ in workers:
            try:
                received.append(results.get(timeout=45))
            except queue.Empty as exc:
                raise RuntimeError("Spawned storage reader did not return within 45 seconds") from exc
        for process in workers:
            process.join(timeout=10)
        ok = all(item.get("ok") and item["digests"] == expected and item["pid"] == item["opened_in_pid"] for item in received)
        ok = ok and all(process.exitcode == 0 for process in workers)
        return {"processes": count, "start_method": "spawn", "ok": bool(ok), "workers": received, "exitcodes": [p.exitcode for p in workers]}
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()


def _run_storage(out_dir: Path, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata, records = _normalize_payload(payload)
    final = out_dir / "storage_path_records.h5"
    partial = final.with_name(final.name + ".partial")
    offsets = _write_h5(partial, metadata, records)
    # Closing, then reopening, prevents an in-memory buffer check masquerading as disk I/O.
    with h5py.File(partial, "r") as handle:
        if not bool(handle.attrs["complete"]) or not np.array_equal(handle["records/path_offsets"][:], offsets):
            raise AssertionError("Partial-file precommit validation failed")
    os.replace(partial, final)

    incomplete = out_dir / "storage_deliberately_incomplete.h5.partial"
    with h5py.File(incomplete, "w") as handle:
        handle.attrs["complete"] = False
        handle.attrs["origin"] = "synthetic_crash_recovery_fixture"
    manifest_path = out_dir / "storage_manifest.json"
    manifest = {"schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "files": [{"path": final.name, "sha256": _file_hash(final), "bytes": final.stat().st_size}], "ignore_suffixes": [".partial"]}
    _atomic_json(manifest_path, manifest)
    reloaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_ok = all(not entry["path"].endswith(".partial") for entry in reloaded_manifest["files"]) and incomplete.name not in {entry["path"] for entry in reloaded_manifest["files"]}

    with LazyPathReader(final) as reader:
        init_is_lazy = reader._handle is None
        count = len(reader)
        roundtrip = count == len(records)
        for index, expected in enumerate(records):
            actual = reader[index]
            roundtrip = roundtrip and all(np.array_equal(actual[field], expected[field]) for field in VALUE_FIELDS)
            roundtrip = roundtrip and actual["time_sec"] == expected["time_sec"] and actual["trajectory_id"] == expected["trajectory_id"]
            roundtrip = roundtrip and actual["doppler_provided"] == expected["doppler_provided"] and actual["path_id_provided"] == expected["path_id_provided"]
        indices = sorted(set((0, min(1, count - 1), count // 2, count - 1)))
        expected_digests = [_record_digest(reader[index]) for index in indices]
        first_paths = int(reader[0]["gain_real"].size)
    with h5py.File(final, "r") as handle:
        actual_offsets = handle["records/path_offsets"][:]
        offset_ok = actual_offsets[0] == 0 and actual_offsets[-1] == handle["paths/gain_real"].size and np.all(np.diff(actual_offsets) >= 0)
        sentinel = np.asarray(handle["io_fixture/sentinel_real"][:]) + 1j * np.asarray(handle["io_fixture/sentinel_imag"][:])
        sentinel_ok = np.array_equal(sentinel, _sentinel())
        reject_conjugate = not np.array_equal(_sentinel(), _sentinel().conj())
        reject_axis_swap = _sentinel().shape != _sentinel().transpose(2, 1, 0).shape
        window = (slice(None), slice(1, 4), slice(1, 6))
        sentinel_window = handle["io_fixture/sentinel_real"][window] + 1j * handle["io_fixture/sentinel_imag"][window]
        boundary_ok = np.array_equal(sentinel_window, _sentinel()[window])
        flat = handle["paths/delay_sec"]
        flat_boundary = {"applicable": False, "ok": True}
        if flat.chunks and flat.size > flat.chunks[0]:
            boundary = flat.chunks[0]
            selection = slice(max(0, boundary - 2), min(flat.size, boundary + 3))
            reference = np.concatenate([item["delay_sec"] for item in records])
            flat_boundary = {"applicable": True, "chunk_length": boundary, "range": [selection.start, selection.stop], "ok": bool(np.array_equal(flat[selection], reference[selection]))}
        axes_ok = all("axes" in item.attrs and "units" in item.attrs for group in ("records", "paths", "io_fixture") for item in handle[group].values())
        schema_ok = handle.attrs["storage_order"] == "python_native_explicit_axes" and handle.attrs["complex_encoding"] == "split_real_imag_float32"

    parity = [_process_parity(final, indices, expected_digests, count) for count in (0, 2, 4)]
    checks = {"native_schema_attributes": bool(axes_ok and schema_ok), "ragged_offsets": bool(offset_ok), "lossless_after_declared_dtype_casts": bool(roundtrip), "complex_sentinel_roundtrip": bool(sentinel_ok), "negative_control_rejects_conjugation": reject_conjugate, "negative_control_rejects_axis_swap": reject_axis_swap, "cross_chunk_sentinel_window": bool(boundary_ok), "cross_chunk_flat_values": flat_boundary["ok"], "partial_excluded_from_manifest": manifest_ok, "lazy_reader_initialization": init_is_lazy, "reader_process_parity": all(item["ok"] for item in parity)}
    return {"status": "passed" if all(checks.values()) else "failed", "passed": all(checks.values()), "data_origin": metadata, "records": len(records), "total_paths": int(offsets[-1]), "checks": checks, "process_checks": parity, "fixed_indices": indices, "flat_chunk_boundary": flat_boundary, "lazy_interface": {"initialization_opens_file": not init_is_lazy, "read_unit": "one record using offsets and path slices", "first_record_paths": first_paths, "whole_file_loaded_by_reader": False, "rss_claim": "No RSS bound is inferred; interface behavior was checked"}, "artifacts": {"h5": str(final), "manifest": str(manifest_path), "deliberate_incomplete_fixture": str(incomplete)}, "versions": {"python": sys.version, "numpy": np.__version__, "h5py": h5py.__version__, "hdf5": h5py.version.hdf5_version}, "limits": ["I/O validation does not establish RT or RF correctness", "Lossless roundtrip is relative to the explicitly declared storage dtypes", "No MATLAB or PyTorch roundtrip was run", "Atomic rename assumes one writer and a local same-filesystem destination; crash durability is not claimed"]}


def run_storage(out_dir: str | Path, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate native HDF5 storage and spawn readers, returning a JSON-safe dict.

    A record requires ``time_sec``, ``gain`` (1-D complex) or split gain fields,
    and ``delay_sec``. Optional fields are ``trajectory_id``, ``doppler_hz`` and
    ``path_id``. Missing Doppler/ID fields are marked explicitly in the file.
    Output names are owned by this validation and replaced on repeated runs.
    """
    directory = Path(out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_storage(directory, payload)
    except Exception:
        result = {"status": "failed", "passed": False, "error": traceback.format_exc(), "limits": ["Storage validation did not complete"]}
    _atomic_json(directory / "storage_validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_storage(arguments.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

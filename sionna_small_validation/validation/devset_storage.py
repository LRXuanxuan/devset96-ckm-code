"""Streaming compact route shards for the algorithm-development dataset.

This is a new contract, not a way to bypass the v2 producer's native-target
protection. Every target references a directly traced state with a persisted
native reconstruction check. Dense native CFR is only a sparse audit cache.

One writer owns one shard. A flushed, atomic sidecar checkpoint permits trimming
an interrupted append if the HDF5 file remains readable; it is not a power-loss
or arbitrary HDF5 corruption guarantee. No estimator should open partial files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import warnings
from typing import Any, Mapping

import h5py
import numpy as np

from .cities import COMPACT_DTYPE, FC, FREQUENCIES

SCHEMA = "kap_devset.route.v1"
STATE_DTYPE = np.dtype([
    ("time_sec", "<f8"), ("rx_xyz_m", "<f8", (3,)),
    ("rx_velocity_mps", "<f8", (3,)), ("rx_pose_rad", "<f8", (3,)),
    ("tx_xyz_m", "<f8", (3,)), ("tx_pose_rad", "<f8", (3,)),
    ("car_xyz_m", "<f8", (3,)), ("car_velocity_mps", "<f8", (3,)),
    ("car_pose_rad", "<f8", (3,)), ("has_dynamic_object", "?"),
    ("quality_flags", "<u4"), ("native_relative_error", "<f8"),
    ("native_checked", "?"), ("has_los", "?"), ("role", "S32"),
    ("pass_index", "<i2"), ("survey_design", "<i2"),
    ("receiver_anchor", "?"), ("receiver_validated_until_sec", "<f8"),
])
TASK_DTYPE = np.dtype([
    ("task_index", "<i8"), ("task_id", "S64"), ("pass_index", "<i2"), ("survey_design", "<i2"),
    ("cutoff_sec", "<f8"), ("history_start_sec", "<f8"),
    ("history_end_sec", "<f8"), ("target_time_sec", "<f8"),
    ("target_record_index", "<i8"), ("noise_seed", "<u8"),
    ("state_noise_seed", "<u8"), ("eval_selected", "?"), ("admitted", "?"), ("flags", "<u4"),
])
INTERVAL_DTYPE = np.dtype([
    ("pass_index", "<i2"), ("left_index", "<i8"),
    ("midpoint_index", "<i8"), ("right_index", "<i8"),
    ("left_time_sec", "<f8"), ("right_time_sec", "<f8"),
    ("midpoint_nmse_db", "<f8"), ("right_nmse_db", "<f8"),
    ("level", "<i2"), ("passed", "?"), ("used_as_leaf", "?"),
])
SURVEY_DTYPE = np.dtype([
    ("record_index", "<i8"), ("design_index", "<i2"), ("noise_seed", "<u8"),
    ("reported_xyz_m", "<f8", (3,)), ("noise_variance", "<f8"),
    ("measured_cfr_power", "<f8"),
])


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=_json_default, allow_nan=False)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(_json(value), encoding="utf-8")
    os.replace(temporary, path)


def _diagnostic_value(value):
    """Losslessly tag nonfinite diagnostics; never relax strict config JSON."""
    if isinstance(value, np.ndarray):
        return _diagnostic_value(value.tolist())
    if isinstance(value, np.generic):
        return _diagnostic_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"__nonfinite_float__": "nan" if np.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"}
    if isinstance(value, complex):
        return {"__complex__": [_diagnostic_value(value.real), _diagnostic_value(value.imag)]}
    if isinstance(value, Path):
        return str(value)
    return value


def _row(dtype, values, *, defaults=None):
    values = dict(values)
    unknown = set(values).difference(dtype.names)
    if unknown:
        raise ValueError(f"Unknown {dtype.names} fields: {sorted(unknown)}")
    row = np.zeros((), dtype=dtype)
    for name, value in (defaults or {}).items():
        row[name] = value
    for name, value in values.items():
        if dtype[name].kind == "S":
            encoded = value.encode("utf-8") if isinstance(value, str) else value
            if len(encoded) > dtype[name].itemsize:
                raise ValueError(f"{name} exceeds its fixed-width UTF-8 field")
            row[name] = encoded
        else:
            row[name] = value
    return row


def _append(dataset, values):
    values = np.asarray(values, dtype=dataset.dtype)
    n = len(dataset)
    dataset.resize((n + len(values),) + dataset.shape[1:])
    dataset[n:] = values


class RouteWriter:
    """Batched append-only compact truth, with explicit interval/target checks.

    ``append_snapshot`` returns its final row index before the buffer flushes.
    Snapshots may be rejected and retained; only admitted states can be tasks.
    A finite native error must be explicitly paired with ``native_checked``.
    ``update_snapshot`` is restricted to quality/receiver-certificate fields.
    """

    _UPDATE_FIELDS = {"quality_flags", "receiver_anchor",
                      "receiver_validated_until_sec"}

    def __init__(self, path, config: Mapping[str, Any], *, resume=False,
                 batch_size=128, warning_bytes=512_000_000,
                 hard_budget_bytes=None):
        self.path = Path(path).resolve()
        self.partial_path = self.path.with_name(self.path.name + ".partial")
        self.checkpoint_path = self.path.with_name(self.path.name + ".checkpoint.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self.config_text = _json(self.config)
        self.config_hash = hashlib.sha256(self.config_text.encode()).hexdigest()
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.warning_bytes = None if warning_bytes is None else int(warning_bytes)
        self.hard_budget_bytes = None if hard_budget_bytes is None else int(hard_budget_bytes)
        self._warned = False
        self._states = []
        self._paths = []
        self._goldens = []
        self._updates = {}
        self._intervals = []
        self._tasks = []
        self._survey_index = []
        self._survey_features = []
        self._events = []
        self._closed = False
        self._sequence = 0
        self.checkpoint_metadata = {}
        self._committed_metadata = {}
        if self.path.exists():
            raise FileExistsError(f"Completed shard already exists: {self.path}")
        if self.partial_path.exists():
            if not resume:
                raise FileExistsError(f"Partial shard exists; explicit resume required: {self.partial_path}")
            self.handle = h5py.File(self.partial_path, "r+")
            try:
                self._resume()
            except BaseException:
                self.handle.close()
                raise
        else:
            self.handle = h5py.File(self.partial_path, "x")
            self._initialize()
            self.flush()

    def _dataset(self, name, shape, maxshape, dtype, axes, units, chunks=None):
        node = self.handle.create_dataset(name, shape=shape, maxshape=maxshape,
                                          dtype=dtype, chunks=chunks or True,
                                          compression="gzip", compression_opts=1,
                                          shuffle=True, fletcher32=True)
        node.attrs.update(axes=axes, units=units)
        return node

    def _initialize(self):
        f = self.handle
        f.attrs.update(schema=SCHEMA, generation_complete=False,
                       access="private_truth; estimator must use a public view",
                       compact_bytes=128, carrier_hz=FC,
                       config_sha256=self.config_hash,
                       phase_reference="native baseband rx0 tx0; no extra carrier phase",
                       path_identity="snapshot_local; no inferred cross-time physical ID",
                       target_contract="direct RT record + persisted native check; sparse dense cache",
                       quality_schema="snapshot_flags_and_checked_intervals.v1")
        f.create_dataset("config_json", data=self.config_text, dtype=h5py.string_dtype())
        f.create_dataset("frequencies_hz", data=FREQUENCIES).attrs.update(axes="frequency", units="Hz")
        self._dataset("states", (0,), (None,), STATE_DTYPE, "state", "field_specific_SI", (256,))
        self._dataset("paths", (0,), (None,), COMPACT_DTYPE, "flat_path",
                      "b00:linear_complex;delay:s;doppler:Hz;k:unit_GCS_vector", (4096,))
        self._dataset("offsets", (1,), (None,), "<i8", "state_boundary", "path_index", (1024,))
        f["offsets"][0] = 0
        self._dataset("tasks", (0,), (None,), TASK_DTYPE, "task", "field_specific_SI", (512,))
        self._dataset("intervals", (0,), (None,), INTERVAL_DTYPE, "interval", "s;dB;indices", (512,))
        self._dataset("survey_index", (0,), (None,), SURVEY_DTYPE, "survey", "field_specific_SI", (128,))
        self._dataset("survey_features", (0, 1024), (None, 1024), "<f4",
                      "survey,rx_angular_cyclic_delay_power", "linear_power", (32, 1024))
        self._dataset("golden_cfr/record_index", (0,), (None,), "<i8", "golden", "state_index", (32,))
        for name in ("real", "imag"):
            self._dataset("golden_cfr/" + name, (0, 16, 64, 16),
                          (None, 16, 64, 16), "<f4", "golden,rx,tx,frequency",
                          "linear_channel_coefficient", (1, 16, 64, 16))
        # Variable JSON retains actual failure/retry details without repeating
        # a large per-state settings object in the successful common case.
        f.create_dataset("events", shape=(0,), maxshape=(None,),
                         dtype=h5py.string_dtype(), chunks=(128,))
        f["events"].attrs.update(axes="event", units="JSON_ledger",
                                nonfinite_encoding="explicit __nonfinite_float__ tags; strict JSON")

    def _lengths(self):
        return {name: len(self.handle[name]) for name in (
            "states", "paths", "offsets", "tasks", "intervals", "survey_index",
            "survey_features", "golden_cfr/record_index", "golden_cfr/real",
            "golden_cfr/imag", "events")}

    def _resume(self):
        f = self.handle
        if f.attrs.get("schema") != SCHEMA or f.attrs.get("config_sha256") != self.config_hash:
            raise ValueError("Resume schema/config mismatch")
        if bool(f.attrs.get("generation_complete", False)):
            raise ValueError("Completed partial shard must be reviewed and finalized, not appended")
        if not self.checkpoint_path.exists():
            raise ValueError("Partial shard lacks an atomic checkpoint")
        cp = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if cp["config_sha256"] != self.config_hash:
            raise ValueError("Checkpoint configuration mismatch")
        for name, length in cp["lengths"].items():
            node = f[name]
            if len(node) < length:
                raise ValueError(f"Checkpoint refers to data not present in {name}")
            node.resize((length,) + node.shape[1:])
        if int(f["offsets"][-1]) != len(f["paths"]) or len(f["offsets"]) != len(f["states"]) + 1:
            raise ValueError("Checkpoint ragged offsets are inconsistent")
        # Updates to already-committed states are replayed from the checkpoint
        # so dirty certificate extensions cannot survive interrupted writes.
        for key, value in cp.get("certificate_updates", {}).items():
            index = int(key)
            row = f["states"][index]
            for field, item in value.items():
                row[field] = item
            f["states"][index] = row
        self._committed_certificates = cp.get("certificate_updates", {})
        self.checkpoint_metadata = dict(cp.get("metadata", {}))
        self._committed_metadata = dict(self.checkpoint_metadata)
        self._sequence = int(cp["sequence"])
        f.flush()

    @property
    def state_count(self):
        return len(self.handle["states"]) + len(self._states)

    @property
    def task_count(self):
        return len(self.handle["tasks"]) + len(self._tasks)

    def append_snapshot(self, compact, metadata=None, *, native_golden=None, **fields):
        if self._closed:
            raise RuntimeError("Writer is closed")
        values = dict(metadata or {})
        overlap = set(values).intersection(fields)
        if overlap:
            raise ValueError(f"Duplicate snapshot fields: {sorted(overlap)}")
        values.update(fields)
        values.pop("record_id", None)  # row index is the shard-local identifier
        array = np.asarray(compact)
        if array.dtype != COMPACT_DTYPE or array.ndim != 1:
            raise ValueError("compact must be a one-dimensional COMPACT_DTYPE array")
        if np.any(array["depth"] > 3):
            raise ValueError("compact128 supports at most three interaction slots")
        defaults = dict(time_sec=np.nan, receiver_validated_until_sec=np.nan,
                        native_relative_error=np.nan, survey_design=-1,
                        pass_index=-1, role="unspecified", quality_flags=0,
                        native_checked=False)
        if "time_sec" in values:
            defaults["receiver_validated_until_sec"] = values["time_sec"]
        row = _row(STATE_DTYPE, values, defaults=defaults)
        for name in ("time_sec", "rx_xyz_m", "rx_velocity_mps", "rx_pose_rad", "tx_xyz_m", "tx_pose_rad"):
            if not np.all(np.isfinite(row[name])):
                raise ValueError(f"Snapshot {name} is non-finite")
        if bool(row["native_checked"]) and (not np.isfinite(row["native_relative_error"]) or row["native_relative_error"] < 0):
            raise ValueError("native_checked requires a finite nonnegative stored relative error")
        if not len(array) and not int(row["quality_flags"]):
            raise ValueError("An empty unresolved state requires a nonzero quality flag")
        index = self.state_count
        if native_golden is not None:
            native = np.asarray(native_golden, dtype=np.complex64)
            if native.shape != (16, 64, 16) or not np.isfinite(native).all():
                raise ValueError("Sparse native golden must have finite [16,64,16] values")
        self._states.append(row)
        self._paths.append(array.copy())
        if native_golden is not None:
            self._goldens.append((index, native.copy()))
        if len(self._states) >= self.batch_size:
            self.flush()
        return index

    def append_snapshots(self, records):
        indices = []
        for record in records:
            record = dict(record)
            compact = record.pop("compact")
            native = record.pop("native_golden", None)
            indices.append(self.append_snapshot(compact, record, native_golden=native))
        return indices

    def update_snapshot(self, index, **fields):
        if set(fields).difference(self._UPDATE_FIELDS):
            raise ValueError("Only quality and receiver-certificate updates are permitted")
        index = int(index)
        if index < 0 or index >= self.state_count:
            raise IndexError(index)
        if index >= len(self.handle["states"]):
            row = self._states[index - len(self.handle["states"])]
            for name, value in fields.items():
                row[name] = value
        else:
            self._updates.setdefault(index, {}).update(fields)

    def append_interval(self, values=None, **fields):
        data = dict(values or {}); data.update(fields)
        row = _row(INTERVAL_DTYPE, data, defaults=dict(
            midpoint_nmse_db=np.nan, right_nmse_db=np.nan, midpoint_index=-1,
            left_index=-1, right_index=-1, used_as_leaf=True))
        if not (np.isfinite(row["left_time_sec"]) and np.isfinite(row["right_time_sec"])):
            raise ValueError("Interval times must be finite")
        if row["right_time_sec"] <= row["left_time_sec"]:
            raise ValueError("Interval must have positive duration")
        if row["passed"] and not (np.isfinite(row["midpoint_nmse_db"]) and np.isfinite(row["right_nmse_db"])):
            raise ValueError("Passed interval requires finite midpoint/right checks")
        self._intervals.append(row)

    def append_task(self, values=None, **fields):
        data = dict(values or {}); data.update(fields)
        data.setdefault("task_index", self.task_count)
        data.setdefault("task_id", hashlib.sha256((self.config_hash + "/" + str(data["task_index"])).encode()).hexdigest()[:24])
        cutoff = float(data.get("cutoff_sec", np.nan))
        data.setdefault("history_end_sec", cutoff)
        data.setdefault("history_start_sec", cutoff - .004)
        row = _row(TASK_DTYPE, data, defaults=dict(survey_design=0, target_record_index=-1, admitted=True))
        if not np.isfinite([row[name] for name in ("cutoff_sec", "history_start_sec", "history_end_sec", "target_time_sec")]).all():
            raise ValueError("Task times must be finite")
        if not row["history_start_sec"] < row["history_end_sec"] <= row["cutoff_sec"] < row["target_time_sec"]:
            raise ValueError("Task history/cutoff/target ordering is invalid")
        self._tasks.append(row)
        return int(row["task_index"])

    def append_survey_feature(self, record_index, features, *, design_index,
                              noise_seed, reported_xyz_m, noise_variance,
                              measured_cfr_power):
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (1024,) or not np.isfinite(features).all() or np.any(features < 0):
            raise ValueError("Survey feature must be 1024 finite nonnegative powers")
        row = _row(SURVEY_DTYPE, dict(record_index=record_index, design_index=design_index,
                   noise_seed=noise_seed, reported_xyz_m=reported_xyz_m,
                   noise_variance=noise_variance, measured_cfr_power=measured_cfr_power))
        if not np.isfinite(row["reported_xyz_m"]).all() or not np.isfinite([noise_variance, measured_cfr_power]).all() or min(noise_variance, measured_cfr_power) < 0:
            raise ValueError("Survey report/variance is invalid")
        self._survey_index.append(row)
        self._survey_features.append(features.copy())

    def append_event(self, kind, payload):
        self._events.append(_json(dict(kind=str(kind), payload=_diagnostic_value(payload))))

    def flush(self, metadata=None):
        if self._closed:
            raise RuntimeError("Writer is closed")
        f = self.handle
        old = len(f["states"])
        if metadata is not None:
            self.checkpoint_metadata.update(metadata)
        certificates = dict(getattr(self, "_committed_certificates", {}))
        for index in self._updates:
            row = f["states"][index]
            certificates.setdefault(str(index), {
                key: row[key].item() for key in self._UPDATE_FIELDS})
        # Persist rollback values before any in-place certificate mutation.
        # New append tails can be discarded by their committed lengths.
        intent = dict(schema=SCHEMA + ".checkpoint", sequence=self._sequence,
                      config_sha256=self.config_hash, lengths=self._lengths(),
                      certificate_updates=certificates,
                      metadata=self._committed_metadata)
        _atomic_json(self.checkpoint_path, intent)
        if self._states:
            counts = np.asarray([len(item) for item in self._paths], dtype=np.int64)
            base = len(f["paths"])
            flat = np.concatenate(self._paths)
            _append(f["paths"], flat)
            _append(f["offsets"], base + np.cumsum(counts))
            _append(f["states"], np.asarray(self._states, dtype=STATE_DTYPE))
        for index, fields in self._updates.items():
            row = f["states"][index]
            # The previous values are kept until the sidecar is committed. On
            # resume use the committed certificate snapshot, never dirty HDF5.
            certificates.setdefault(str(index), {
                key: row[key].item() for key in self._UPDATE_FIELDS})
            for name, value in fields.items():
                row[name] = value
            f["states"][index] = row
        # The checkpoint records all certificate values which have ever been
        # changed in place. This is small relative to path arrays (one/anchor).
        for index in self._updates:
            row = f["states"][index]
            certificates[str(index)] = {key: row[key].item() for key in self._UPDATE_FIELDS}
        for name, rows, dtype in (("tasks", self._tasks, TASK_DTYPE),
                                  ("intervals", self._intervals, INTERVAL_DTYPE),
                                  ("survey_index", self._survey_index, SURVEY_DTYPE)):
            if rows:
                _append(f[name], np.asarray(rows, dtype=dtype))
        if self._survey_features:
            _append(f["survey_features"], np.stack(self._survey_features))
        if self._goldens:
            _append(f["golden_cfr/record_index"], [item[0] for item in self._goldens])
            values = np.stack([item[1] for item in self._goldens])
            _append(f["golden_cfr/real"], values.real)
            _append(f["golden_cfr/imag"], values.imag)
        if self._events:
            _append(f["events"], self._events)
        f.flush()
        self._sequence += 1
        cp = dict(schema=SCHEMA + ".checkpoint", sequence=self._sequence,
                  config_sha256=self.config_hash, lengths=self._lengths(),
                  certificate_updates=certificates,
                  metadata=self.checkpoint_metadata)
        _atomic_json(self.checkpoint_path, cp)
        self._committed_certificates = certificates
        self._committed_metadata = dict(self.checkpoint_metadata)
        for buffer in (self._states, self._paths, self._goldens, self._intervals,
                       self._tasks, self._survey_index, self._survey_features,
                       self._events):
            buffer.clear()
        self._updates.clear()
        disk_bytes = self.partial_path.stat().st_size
        if self.warning_bytes is not None and disk_bytes > self.warning_bytes and not self._warned:
            warnings.warn(f"Route shard grew to {disk_bytes} B; review capacity ledger", RuntimeWarning)
            self._warned = True
        if self.hard_budget_bytes is not None and disk_bytes > self.hard_budget_bytes:
            raise RuntimeError(f"Shard exceeds explicit byte budget ({disk_bytes} B); data preserved")
        return dict(**cp, disk_bytes=disk_bytes, appended_states=len(f["states"]) - old)

    def finish(self, *, require_tasks=True):
        self.flush()
        result = audit_route_file(self.partial_path, require_complete=False,
                                  require_tasks=require_tasks)
        if not result["passed"]:
            raise ValueError("Shard contract failed: " + "; ".join(result["failures"]))
        self.handle.attrs["generation_complete"] = True
        self.handle.flush()
        self.handle.close()
        self._closed = True
        os.replace(self.partial_path, self.path)
        result["path"] = str(self.path)
        result["disk_bytes"] = self.path.stat().st_size
        # Retain the small checkpoint as provenance; completed readers only use
        # the HDF5 marker and never append to a published file.
        return result

    def close(self):
        if not self._closed:
            self.handle.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def read_snapshot(path, index, *, require_complete=True):
    """Private evaluator utility; every call opens its own process-local file."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != SCHEMA or (require_complete and not f.attrs.get("generation_complete", False)):
            raise ValueError("Not a completed compact development shard")
        index = int(index)
        if index < 0 or index >= len(f["states"]):
            raise IndexError(index)
        lo, hi = map(int, f["offsets"][index:index + 2])
        row = f["states"][index]
        result = {name: row[name] for name in STATE_DTYPE.names}
        result["compact"] = f["paths"][lo:hi]
        return result


def audit_route_file(path, *, require_complete=True, require_tasks=True):
    """Disk checks, independent of RT imports and full dense materialization."""
    failures = []
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != SCHEMA:
            failures.append("schema mismatch")
        if require_complete and not bool(f.attrs.get("generation_complete", False)):
            failures.append("missing completion marker")
        states = f["states"][:]
        offsets = f["offsets"][:]
        tasks = f["tasks"][:]
        intervals = f["intervals"][:]
        if f["paths"].dtype != COMPACT_DTYPE or f["paths"].dtype.itemsize != 128:
            failures.append("invalid compact path dtype")
        if len(offsets) != len(states) + 1 or offsets[0] != 0 or offsets[-1] != len(f["paths"]) or np.any(np.diff(offsets) < 0):
            failures.append("invalid ragged offsets")
        if require_tasks and not np.any(tasks["admitted"]):
            failures.append("no admitted tasks")
        if len(f["survey_index"]) != len(f["survey_features"]):
            failures.append("survey features/index length mismatch")
        config = json.loads(f["config_json"][()].decode())
        tolerance = float(config.get("native_relative_error_tolerance", 1e-4))
        if not np.isfinite(tolerance) or tolerance <= 0:
            failures.append("native tolerance must be positive and finite")
        interval_threshold = float(config.get("interval_nmse_threshold_db", -30.0))
        if not np.isfinite(interval_threshold):
            failures.append("interval NMSE threshold must be finite")
        # All selectors below operate on in-memory structured arrays. Each
        # dataset is read once; no per-task HDF5 row reads or dense CFR scans.
        good_native = (states["native_checked"] &
                       np.isfinite(states["native_relative_error"]) &
                       (states["native_relative_error"] >= 0) &
                       (states["native_relative_error"] < tolerance))
        interval_valid = np.ones(len(intervals), dtype=bool)
        if len(intervals):
            indices = np.column_stack([intervals[name] for name in (
                "left_index", "midpoint_index", "right_index")])
            reference_ok = np.all((indices >= 0) & (indices < len(states)), axis=1)
            if not np.all(reference_ok):
                failures.append("interval state index out of range")
            interval_valid &= reference_ok
            if np.any(reference_ok):
                chosen = np.flatnonzero(reference_ok)
                triple = states[indices[chosen]]
                rows = intervals[chosen]
                expected = np.column_stack([rows["left_time_sec"],
                    (rows["left_time_sec"] + rows["right_time_sec"]) / 2,
                    rows["right_time_sec"]])
                time_ok = (np.isfinite(expected).all(axis=1) &
                           (rows["right_time_sec"] > rows["left_time_sec"]) &
                           np.isfinite(triple["time_sec"]).all(axis=1) &
                           np.all(np.abs(triple["time_sec"] - expected) <= 1e-10, axis=1))
                pass_ok = np.all(triple["pass_index"] == rows["pass_index"][:, None], axis=1)
                if not np.all(time_ok):
                    failures.append("interval reference time/order mismatch")
                if not np.all(pass_ok):
                    failures.append("interval state pass mismatch")
                interval_valid[chosen] &= time_ok & pass_ok
                passed = rows["passed"]
                if np.any(passed):
                    active = chosen[passed]
                    active_triple = triple[passed]
                    active_rows = rows[passed]
                    quality_ok = np.all(active_triple["quality_flags"] == 0, axis=1)
                    native_ok = np.all(good_native[indices[active]], axis=1)
                    nmse = np.column_stack([active_rows["midpoint_nmse_db"], active_rows["right_nmse_db"]])
                    nmse_ok = np.isfinite(nmse).all(axis=1) & np.all(nmse <= interval_threshold, axis=1)
                    certificate = active_triple[:, 0]["receiver_validated_until_sec"]
                    certificate_ok = (active_triple[:, 0]["receiver_anchor"] &
                        np.isfinite(certificate) &
                        (certificate + 1e-12 >= active_rows["right_time_sec"]))
                    if not np.all(quality_ok):
                        failures.append("passed interval has unadmitted endpoint/midpoint")
                    if not np.all(native_ok):
                        failures.append("passed interval lacks native checks")
                    if not np.all(nmse_ok):
                        failures.append("passed interval violates declared midpoint/right NMSE threshold")
                    if not np.all(certificate_ok):
                        failures.append("passed interval lacks finite receiver certificate")
                    interval_valid[active] &= quality_ok & native_ok & nmse_ok & certificate_ok
        merged_intervals = {}
        for pass_index in np.unique(intervals["pass_index"]):
            selected = intervals[(intervals["pass_index"] == pass_index) & intervals["passed"] & intervals["used_as_leaf"] & interval_valid]
            spans = sorted((float(item["left_time_sec"]), float(item["right_time_sec"])) for item in selected)
            merged = []
            for left, right in spans:
                if merged and left <= merged[-1][1] + 1e-12:
                    merged[-1][1] = max(merged[-1][1], right)
                else:
                    merged.append([left, right])
            merged_intervals[int(pass_index)] = merged
        if np.any(~tasks["admitted"] & (tasks["flags"] == 0)):
            failures.append("rejected task lacks a rejection flag")
        selected_tasks = tasks[tasks["admitted"]]
        if len(selected_tasks):
            indices = selected_tasks["target_record_index"]
            reference_ok = (indices >= 0) & (indices < len(states))
            if not np.all(reference_ok):
                failures.append("target index out of range")
            valid_tasks = selected_tasks[reference_ok]
            indices = indices[reference_ok]
            target_states = states[indices]
            if np.any(valid_tasks["flags"] != 0) or np.any(target_states["quality_flags"] != 0) or np.any(offsets[indices + 1] == offsets[indices]):
                failures.append("unadmitted target")
            if not np.all(good_native[indices]):
                failures.append("target lacks a passing independent native check")
            if np.any(~np.isfinite(target_states["time_sec"])) or np.any(np.abs(target_states["time_sec"] - valid_tasks["target_time_sec"]) > 1e-10) or np.any(target_states["pass_index"] != valid_tasks["pass_index"]):
                failures.append("target reference time/pass mismatch")
            time_fields = np.column_stack([selected_tasks[name] for name in (
                "history_start_sec", "history_end_sec", "cutoff_sec", "target_time_sec")])
            if not np.isfinite(time_fields).all() or np.any(time_fields[:, 0] >= time_fields[:, 1]) or np.any(time_fields[:, 1] > time_fields[:, 2]) or np.any(time_fields[:, 2] >= time_fields[:, 3]):
                failures.append("causal task time ordering violated")
            for pass_index in np.unique(selected_tasks["pass_index"]):
                rows = selected_tasks[selected_tasks["pass_index"] == pass_index]
                spans = np.asarray(merged_intervals.get(int(pass_index), []), dtype=np.float64).reshape(-1, 2)
                if not len(spans):
                    failures.append("admitted task history crosses unchecked/unsafe intervals"); break
                # Non-overlapping sorted union: the final start <= task start
                # is the only interval which could cover this entire history.
                candidate = np.searchsorted(spans[:, 0], rows["history_start_sec"] + 1e-12, side="right") - 1
                safe = candidate >= 0
                safe[safe] &= spans[candidate[safe], 1] + 1e-12 >= rows["history_end_sec"][safe]
                if not np.all(safe):
                    failures.append("admitted task history crosses unchecked/unsafe intervals"); break
        goldens = f["golden_cfr/record_index"][:]
        if len(goldens) != len(f["golden_cfr/real"]) or len(goldens) != len(f["golden_cfr/imag"]):
            failures.append("sparse golden length mismatch")
        if np.any(goldens < 0) or np.any(goldens >= len(states)):
            failures.append("sparse golden index out of range")
        result = dict(passed=not failures, failures=failures, states=len(states),
                      path_records=len(f["paths"]), tasks=len(tasks), intervals=len(intervals),
                      admitted_tasks=int(np.count_nonzero(tasks["admitted"])),
                      rejected_tasks=int(np.count_nonzero(~tasks["admitted"])),
                      surveys=len(f["survey_index"]), dense_goldens=len(goldens),
                      rejected_states=int(np.count_nonzero(states["quality_flags"])),
                      unsafe_leaf_intervals=int(np.count_nonzero(intervals["used_as_leaf"] & ~intervals["passed"])) if len(intervals) else 0,
                      nominal_path_bytes=len(f["paths"]) * 128)
    return result

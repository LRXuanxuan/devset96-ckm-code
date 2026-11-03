"""Isolated 64x16 compact-adapter controls; never runs city geometry probes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

_DLL_HANDLES = []
CONTROLS = [
    {"name": "los_yaw_a", "kind": "los", "tx_yaw": 0.35, "rx_yaw": -0.45},
    {"name": "los_yaw_b", "kind": "los", "tx_yaw": -0.60, "rx_yaw": 0.80},
    {"name": "reflection_yaw_a", "kind": "reflection", "tx_yaw": 0.35, "rx_yaw": -0.45},
    {"name": "reflection_yaw_b", "kind": "reflection", "tx_yaw": -0.60, "rx_yaw": 0.80},
]


def _relative_error(estimate, reference):
    return float(np.linalg.norm((estimate-reference).ravel()) /
                 max(float(np.linalg.norm(reference.ravel())), 1e-30))


def _worker(index: int, out_dir: Path, backend: str = 'auto') -> dict:
    project = Path(__file__).resolve().parents[1]
    runtime = project / ".runtime" / "bin"
    if os.name == "nt":
        os.environ.setdefault('SIONNA_VALIDATION_CPU_THREADS','1')
        if os.environ.get('SIONNA_WINDOWS_JIT_COMPAT')!='1':
            os.environ.setdefault('DRJIT_LIBLLVM_PATH',str(runtime/'LLVM-C.dll'))
            _DLL_HANDLES.append(os.add_dll_directory(str(runtime)))
    from .common import setup_backend, save_json, portable_asset, environment, sha256

    out_dir.mkdir(parents=True, exist_ok=True)
    control = CONTROLS[index]
    path = out_dir / (control["name"] + ".json")
    report = {"control": control, "status": "RUNNING", "stage": "setup_backend",
              "runtime_dll": str(runtime / "LLVM-C.dll"),
              "scope": "Single small RT solve; fixed geometry/pattern/polarization/pose; 64 Tx x 16 Rx synthetic array.",
              "relative_error_tolerance": 1e-4}

    def stage(name):
        report["stage"] = name
        save_json(path, report)
        print(f"[{control['name']}] {name}", flush=True)

    try:
        stage("setup_backend")
        backend = setup_backend(backend)
        report['runtime_dll']=os.environ.get('DRJIT_LIBLLVM_PATH')
        import mitsuba as mi
        import sionna.rt as rt
        from .cities import (extract_compact, reconstruct, native_cfr,
                             stable_objects, xyz_rows, complex_pair,
                             FREQUENCIES, FC, COMPACT_DTYPE)

        report["environment"] = environment(backend)
        report["cities_adapter_sha256"] = sha256(project / "validation" / "cities.py")
        stage("load_small_scene")
        if control["kind"] == "los":
            scene = rt.load_scene()
            tx_xyz, rx_xyz = [0., 0., 1.], [10., 2., 1.5]
        else:
            scene = rt.load_scene(portable_asset(rt.scene.simple_reflector), merge_shapes=False)
            tx_xyz, rx_xyz = [-0.2, 0.1, 1.], [0.2, 0.1, 1.]
        scene.frequency = FC
        scene.tx_array = rt.PlanarArray(num_rows=8, num_cols=8, vertical_spacing=.5,
                                         horizontal_spacing=.5, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=4, num_cols=4, vertical_spacing=.5,
                                         horizontal_spacing=.5, pattern="iso", polarization="V")
        tx = rt.Transmitter("tx", position=tx_xyz, orientation=[control["tx_yaw"], 0., 0.])
        rx = rt.Receiver("rx", position=rx_xyz, orientation=[control["rx_yaw"], 0., 0.])
        rx.velocity = [30., 0., 0.]
        scene.add([tx, rx])
        mapping, objects = stable_objects(scene, control["name"])
        solver = rt.PathSolver(deterministic=True)
        solver.loop_mode = "evaluated"
        report.update(tx_xyz_m=tx_xyz, rx_xyz_m=rx_xyz, rx_velocity_mps=[30., 0., 0.],
                      solver_deterministic=True, solver_loop_mode="evaluated",
                      path_record_bytes=int(COMPACT_DTYPE.itemsize), stable_objects=objects)
        stage("native_rt_solve")
        started = time.perf_counter()
        paths = solver(scene=scene, max_depth=0 if control["kind"] == "los" else 1,
                       samples_per_src=8192, max_num_paths_per_src=4096,
                       synthetic_array=True, los=control["kind"] == "los",
                       specular_reflection=control["kind"] != "los",
                       diffuse_reflection=False, refraction=False, diffraction=False,
                       seed=20261025)
        report["rt_elapsed_sec"] = time.perf_counter()-started
        stage("extract_compact_raw_adapter")
        compact = extract_compact(paths, mapping)
        report["valid_paths"] = len(compact)
        report["path_depths"] = compact["depth"].astype(int).tolist()
        if not len(compact):
            raise RuntimeError("No valid physical path: no zero-energy reconstruction pass is permitted")
        ptx = xyz_rows(scene.tx_array.rotate(scene.wavelength, tx.orientation))
        prx = xyz_rows(scene.rx_array.rotate(scene.wavelength, rx.orientation))
        stage("native_cfr_t0_helper")
        native_zero = native_cfr(paths)
        reconstructed_zero = reconstruct(compact, ptx, prx)
        report["native_cfr_shape"] = list(native_zero.shape)
        report["t0_helper_relative_error"] = _relative_error(reconstructed_zero, native_zero)
        stage("native_cfr_five_time_points")
        native = complex_pair(paths.cfr(
            frequencies=mi.Float(FREQUENCIES), sampling_frequency=100_000.,
            num_time_steps=5, normalize_delays=False, normalize=False,
            out_type="numpy"))[0, :, 0, :, :, :]
        times = np.arange(5, dtype=float)*10e-6
        errors = []
        for t_index, time_sec in enumerate(times):
            reconstructed = reconstruct(compact, ptx, prx, delta_t=float(time_sec))
            errors.append(_relative_error(reconstructed, native[:, :, t_index, :]))
        report["times_sec"] = times.tolist()
        report["relative_errors_by_time"] = errors
        report["native_t0_helper_vs_time_array_relative_error"] = _relative_error(native_zero, native[:, :, 0, :])
        report["max_relative_error"] = max([report["t0_helper_relative_error"], *errors])
        report["checks"] = {
            "actual_mimo_shape_16_64_16": list(native_zero.shape) == [16, 64, 16],
            "record_size_128B": int(COMPACT_DTYPE.itemsize) == 128,
            "nonzero_reference_energy": bool(np.linalg.norm(native_zero)>0),
            "compact_t0_matches_native_helper": report["t0_helper_relative_error"] <= 1e-4,
            "compact_local_doppler_matches_five_native_times": max(errors) <= 1e-4,
            "native_helper_matches_time_array_t0": report["native_t0_helper_vs_time_array_relative_error"] <= 1e-6,
        }
        report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
        report["stage"] = "completed"
    except Exception as error:
        report.update(status="ERROR", error=str(error), traceback=traceback.format_exc())
    save_json(path, report)
    return report


def run_compact_controls(out_dir: str | Path, timeout_sec: int = 90, backend: str = 'auto') -> dict:
    """At most four serial child processes; preserve native failures verbatim."""
    from .common import save_json, sha256
    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[1]
    report = {"status": "RUNNING", "controls": [],
              "scope": "Fixed-pose single-polarization 64x16 synthetic-array adapter controls only; no city probes.",
              "script_sha256": sha256(Path(__file__)),
              "cities_adapter_sha256": sha256(project/"validation"/"cities.py"),
              "note": "Native nonzero exits remain process failures even if a child wrote numerical metrics."}
    for index, control in enumerate(CONTROLS):
        command = [sys.executable, "-u", "-m", "validation.compact_controls",
                   "--worker", str(index), "--out-dir", str(destination),'--backend',backend]
        row = {"name": control["name"]}
        try:
            completed = subprocess.run(command, cwd=project, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace",
                                       timeout=timeout_sec)
            row.update(returncode=completed.returncode,
                       returncode_unsigned_hex=hex(completed.returncode & 0xffffffff),
                       stdout=completed.stdout, stderr=completed.stderr)
        except subprocess.TimeoutExpired as error:
            row.update(returncode=None, timeout_sec=timeout_sec,
                       stdout=str(error.stdout), stderr=str(error.stderr))
        child_path = destination/(control["name"]+".json")
        child = json.loads(child_path.read_text(encoding="utf-8")) if child_path.exists() else None
        row["worker_report"] = child
        row["status"] = ("PROCESS_FAILURE" if row["returncode"] != 0 else
                         child.get("status", "ERROR") if child else "ERROR")
        report["controls"].append(row)
        save_json(destination/"compact_controls_report.json", report)
        print(json.dumps({"name": row["name"], "status": row["status"],
                          "returncode": row["returncode"],
                          "max_relative_error": (child or {}).get("max_relative_error"),
                          "last_stage": (child or {}).get("stage")}), flush=True)
    report["status"] = "PASS" if all(row["status"]=="PASS" for row in report["controls"]) else "FAIL"
    report["counts"] = {name:sum(row["status"]==name for row in report["controls"])
                        for name in ("PASS", "FAIL", "ERROR", "PROCESS_FAILURE")}
    report["report_path"] = str(destination/"compact_controls_report.json")
    save_json(destination/"compact_controls_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(4))
    parser.add_argument("--out-dir", type=Path, default=Path("output/compact_controls"))
    parser.add_argument('--backend',default='auto',choices=['auto','llvm_ad_mono_polarized','cuda_ad_mono_polarized'])
    parser.add_argument('--windows-jit-compat',action='store_true')
    args = parser.parse_args()
    if args.windows_jit_compat:os.environ['SIONNA_WINDOWS_JIT_COMPAT']='1'
    if args.worker is not None:
        report = _worker(args.worker, args.out_dir.resolve(),args.backend)
    else:
        report = run_compact_controls(args.out_dir,backend=args.backend)
    print(json.dumps({key:report.get(key) for key in ("status","counts","report_path")}),flush=True)
    return 0 if report.get('status')=='PASS' else 1


if __name__ == "__main__":
    raise SystemExit(main())

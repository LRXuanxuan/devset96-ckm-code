"""Small, actual Sionna RT analytic controls; no dataset generation.

The caller selects the Mitsuba variant BEFORE calling ``run_physics``.  This
module never changes it.  ``Paths.a`` is a passband path coefficient, whereas
``Paths.cir(normalize_delays=False)`` includes the carrier propagation phase.
Only the latter should be used as the baseband reference for CFR synthesis.
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np


C0 = 299_792_458.0
FC = 28_000_000_000.0
FS_TIME = 1_000_000.0
NUM_TIME = 257
FREQUENCIES = np.linspace(-12_000_000.0, 12_000_000.0, 33)


def _array(value: Any) -> np.ndarray:
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _complex(value: Any) -> np.ndarray:
    if isinstance(value, tuple):
        return _array(value[0]) + 1j * _array(value[1])
    return _array(value)


def _relative_error(observed: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected.ravel())), 1e-30)
    return float(np.linalg.norm((observed - expected).ravel()) / denominator)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _angle_direction(theta: float, phi: float) -> np.ndarray:
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def _geometry(kind: str, rx_velocity: np.ndarray, mirror_speed: float,
              time: float = 0.0) -> dict[str, Any]:
    if kind == "los":
        tx = np.array([0.0, 0.0, 1.0])
        rx = np.array([10.0, 0.0, 1.0]) + time * rx_velocity
        direction = _unit(rx - tx)
        length = float(np.linalg.norm(rx - tx))
        derivative = float(np.dot(direction, rx_velocity))
        return {
            "tx": tx, "rx": rx, "mirror_z": 0.0, "length": length,
            "fd_hz": -FC / C0 * derivative,
            "departure_direction": direction,
            "arrival_toward_source": -direction,
        }
    # The bundled simple_reflector is the square [-.5,.5]^2 in z=0.
    # Small coordinates keep the reflection point well inside its boundary.
    # Offset y from the square's triangulation diagonal: an exactly-on-edge
    # specular point can be returned twice after float32 object transforms.
    # We retain the strict one-path check rather than deduplicating silently.
    tx = np.array([-0.2, 0.1, 1.0])
    rx = np.array([0.2, 0.1, 1.0]) + time * rx_velocity
    mirror_z = mirror_speed * time
    image_tx = tx.copy()
    image_tx[2] = 2.0 * mirror_z - tx[2]
    image_ray = rx - image_tx
    length = float(np.linalg.norm(image_ray))
    image_velocity = np.array([0.0, 0.0, 2.0 * mirror_speed])
    derivative = float(np.dot(_unit(image_ray), rx_velocity - image_velocity))
    fraction = (mirror_z - image_tx[2]) / (rx[2] - image_tx[2])
    reflection = image_tx + fraction * image_ray
    return {
        "tx": tx, "rx": rx, "mirror_z": mirror_z, "length": length,
        "fd_hz": -FC / C0 * derivative,
        "departure_direction": _unit(reflection - tx),
        "arrival_toward_source": _unit(reflection - rx),
    }


def _controls() -> list[dict[str, Any]]:
    result = []
    for kind in ("los", "reflection"):
        for label, velocity in (
            ("static", [0.0, 0.0, 0.0]),
            ("rx_positive_x", [30.0, 0.0, 0.0]),
            ("rx_negative_x", [-30.0, 0.0, 0.0]),
            ("rx_transverse_y", [0.0, 30.0, 0.0]),
        ):
            result.append({"name": f"{kind}_{label}", "kind": kind,
                           "velocity": velocity, "mirror_speed": 0.0})
    for label, velocity, mirror in (
        ("static", [0.0, 0.0, 0.0], 0.0),
        ("mirror_only", [0.0, 0.0, 0.0], 10.0),
        ("rx_only", [30.0, 0.0, 0.0], 0.0),
        ("rx_and_mirror", [30.0, 0.0, 0.0], 10.0),
    ):
        result.append({"name": f"moving_reflector_{label}",
                       "kind": "reflection", "velocity": velocity,
                       "mirror_speed": mirror})
    return result


def _trace(rt: Any, solver: Any, control: dict[str, Any], time: float = 0.0,
           require_single: bool = True):
    velocity = np.asarray(control["velocity"], dtype=np.float64)
    geometry = _geometry(control["kind"], velocity,
                         float(control["mirror_speed"]), time)
    if control["kind"] == "los":
        scene = rt.load_scene()
    else:
        # Mitsuba's Windows file reader can reject existing Unicode paths.
        # The shared helper stages unchanged assets in the agreed ASCII cache.
        from .common import portable_asset
        xml_path = Path(portable_asset(rt.scene.simple_reflector))
        xml = ElementTree.fromstring(xml_path.read_text(encoding="utf-8"))
        shape = xml.find("shape")
        if shape is None:
            raise RuntimeError("Bundled reflector XML contains no shape")
        mesh = shape.find("string[@name='filename']")
        if mesh is None:
            raise RuntimeError("Bundled reflector XML contains no mesh filename")
        mesh.set("value", (xml_path.parent / mesh.attrib["value"]).as_posix())
        # Recreate each displaced geometry directly, avoiding stale acceleration
        # structures from in-place transforms during +/-time retracing.
        transform = ElementTree.SubElement(shape, "transform", name="to_world")
        ElementTree.SubElement(transform, "translate", z=str(geometry["mirror_z"]))
        scene = rt.load_scene_from_string(ElementTree.tostring(xml, encoding="unicode"),
                                          merge_shapes=False)
    scene.frequency = FC
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    tx = rt.Transmitter("tx", position=geometry["tx"].tolist())
    rx = rt.Receiver("rx", position=geometry["rx"].tolist())
    rx.velocity = velocity.tolist()
    scene.add([tx, rx])
    if control["kind"] != "los":
        mirror = scene.get("reflector")
        if mirror is None:
            raise RuntimeError("Bundled simple_reflector has no 'reflector' object")
        mirror.velocity = [0.0, 0.0, control["mirror_speed"]]
    paths = solver(
        scene=scene, max_depth=0 if control["kind"] == "los" else 1,
        max_num_paths_per_src=4096, samples_per_src=8192,
        synthetic_array=True, los=control["kind"] == "los",
        specular_reflection=control["kind"] != "los",
        diffuse_reflection=False, refraction=False, diffraction=False, seed=47,
    )
    # This helper returns Paths while its locally constructed scene would
    # otherwise leave scope. Keep the geometry/material owners alive through
    # the caller's later CIR/CFR and finite-difference exports. This validation
    # reference does not modify the vendor's Paths implementation or solver.
    paths._validation_scene_owner = scene
    # Materialize the primary fields before any NumPy export or local-owner
    # teardown. Do not request visualization-only lazy interaction assembly.
    import drjit as dr
    dr.eval(*paths.a, paths.tau, paths.valid, paths.doppler,
            paths.theta_t, paths.phi_t, paths.theta_r, paths.phi_r)
    dr.sync_thread()
    valid = _array(paths.valid).reshape(-1).astype(bool)
    count = int(valid.sum())
    if count == 0 or (require_single and count != 1):
        raise RuntimeError(f"Expected one valid path, observed {int(valid.sum())}")
    if count > 1:
        # Do not silently deduplicate the RT response. A representative delay
        # may be used ONLY for this analytic diagnostic when all returned
        # delays and Dopplers coincide. The caller records the multiplicity
        # and fails its separate one-path check.
        valid_tau = _array(paths.tau).reshape(-1)[valid]
        valid_fd = _array(paths.doppler).reshape(-1)[valid]
        if np.ptp(valid_tau) > 1e-13 or np.ptp(valid_fd) > 0.05:
            raise RuntimeError("Multiple distinct paths cannot be associated in this single-path control")
    return paths, geometry, int(np.flatnonzero(valid)[0])


def _evaluate_control(rt: Any, mi: Any, solver: Any,
                      control: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    paths, geometry, idx = _trace(rt, solver, control)
    tau = float(_array(paths.tau).reshape(-1)[idx])
    fd_rt = float(_array(paths.doppler).reshape(-1)[idx])
    theta_t = float(_array(paths.theta_t).reshape(-1)[idx])
    phi_t = float(_array(paths.phi_t).reshape(-1)[idx])
    theta_r = float(_array(paths.theta_r).reshape(-1)[idx])
    phi_r = float(_array(paths.phi_r).reshape(-1)[idx])
    departure = _angle_direction(theta_t, phi_t)
    arrival = _angle_direction(theta_r, phi_r)
    velocity = np.asarray(control["velocity"], dtype=np.float64)
    mirror_velocity = np.array([0.0, 0.0, control["mirror_speed"]])
    fd_projection = FC / C0 * float(np.dot(velocity, arrival))
    if control["kind"] != "los":
        fd_projection += FC / C0 * float(
            np.dot(mirror_velocity, -arrival - departure))

    cir, tau_cir = paths.cir(
        sampling_frequency=FS_TIME, num_time_steps=NUM_TIME,
        normalize_delays=False, out_type="numpy")
    cir = _complex(cir)
    # SISO permits a simple (path,time) view without ambiguous axis squeezing.
    coefficients = cir.reshape(-1, cir.shape[-2], cir.shape[-1])[0]
    delays = _array(tau_cir).reshape(-1)
    frequencies = FREQUENCIES.astype(np.float32)
    reference_cfr = np.einsum(
        "lt,lf->tf", coefficients.astype(np.complex128),
        np.exp(-2j * np.pi * delays.astype(np.float64)[:, None]
               * frequencies.astype(np.float64)[None, :]))
    native_cfr = _complex(paths.cfr(
        frequencies=mi.Float(frequencies), sampling_frequency=FS_TIME,
        num_time_steps=NUM_TIME, normalize_delays=False, normalize=False,
        out_type="numpy")).reshape(NUM_TIME, frequencies.size)
    cfr_error = _relative_error(native_cfr, reference_cfr)

    # This independent float64 evaluation intentionally exposes the native
    # float32 carrier-phase floor instead of manufacturing exact agreement.
    a_passband = _complex(paths.a).reshape(-1)[idx]
    a_baseband = coefficients[idx]
    expected_baseband_zero = a_passband * np.exp(-2j * np.pi * FC * tau)
    carrier_contract_error = float(abs(a_baseband[0] - expected_baseband_zero)
                                   / max(abs(expected_baseband_zero), 1e-30))
    times = np.arange(NUM_TIME, dtype=np.float64) / FS_TIME
    relative_phase = np.unwrap(np.angle(a_baseband * np.conj(a_baseband[0])))
    fd_phase_fit = float(np.polyfit(times, relative_phase, 1)[0] / (2 * np.pi))
    fd_fit_error = abs(fd_phase_fit - fd_rt)
    static_change = float(np.max(np.abs(a_baseband - a_baseband[0]))
                          / max(abs(a_baseband[0]), 1e-30))

    finite_differences = []
    for step_sec in (1e-4, 2e-4):
        plus, _, plus_idx = _trace(rt, solver, control, step_sec, require_single=False)
        minus, _, minus_idx = _trace(rt, solver, control, -step_sec, require_single=False)
        tau_plus = float(_array(plus.tau).reshape(-1)[plus_idx])
        tau_minus = float(_array(minus.tau).reshape(-1)[minus_idx])
        fd_difference = -FC * (tau_plus - tau_minus) / (2 * step_sec)
        finite_differences.append({
            "central_difference_half_step_sec": step_sec,
            "rt_fd_from_delay_derivative_hz": fd_difference,
            "absolute_error_vs_geometric_fd_hz": abs(fd_difference - geometry["fd_hz"]),
            "plus_valid_path_count": int(_array(plus.valid).sum()),
            "minus_valid_path_count": int(_array(minus.valid).sum()),
            "plus_returned_delays_sec": _array(plus.tau).reshape(-1)[_array(plus.valid).reshape(-1).astype(bool)].astype(float).tolist(),
            "minus_returned_delays_sec": _array(minus.tau).reshape(-1)[_array(minus.valid).reshape(-1).astype(bool)].astype(float).tolist(),
        })
    direction_error = max(
        float(np.linalg.norm(departure - geometry["departure_direction"])),
        float(np.linalg.norm(arrival - geometry["arrival_toward_source"])))
    geometry_fd_error = abs(fd_rt - geometry["fd_hz"])
    projection_error = abs(fd_rt - fd_projection)
    derivative_error = max(item["absolute_error_vs_geometric_fd_hz"]
                           for item in finite_differences)
    delay_error = abs(tau - geometry["length"] / C0)
    # Engineering tolerances, declared before execution. They are NOT claims
    # about achieved accuracy. Delay derivatives use larger displacement steps
    # because arbitrarily fine differencing amplifies native float32 rounding.
    tolerances = {
        "cir_to_cfr_relative_l2": 1e-4,
        "passband_to_baseband_relative_complex_error": 2e-2,
        "direction_unit_vector_l2": 1e-5,
        "geometric_delay_absolute_sec": 1e-13,
        "analytic_and_projection_fd_absolute_hz": 0.05 + 1e-5 * abs(geometry["fd_hz"]),
        "phase_fit_fd_absolute_hz": 0.02 + 1e-5 * abs(fd_rt),
        "delay_derivative_fd_absolute_hz": 2.0 + 0.003 * abs(geometry["fd_hz"]),
        "static_relative_coefficient_change": 1e-6,
    }
    checks = {
        "one_valid_path": True,
        "cir_to_cfr": cfr_error <= tolerances["cir_to_cfr_relative_l2"],
        "passband_baseband_contract": carrier_contract_error <= tolerances["passband_to_baseband_relative_complex_error"],
        "direction_convention": direction_error <= tolerances["direction_unit_vector_l2"],
        "absolute_delay": delay_error <= tolerances["geometric_delay_absolute_sec"],
        "analytic_fd": geometry_fd_error <= tolerances["analytic_and_projection_fd_absolute_hz"],
        "angle_velocity_projection_fd": projection_error <= tolerances["analytic_and_projection_fd_absolute_hz"],
        "local_doppler_phase_evolution": fd_fit_error <= tolerances["phase_fit_fd_absolute_hz"],
        "retraced_delay_derivative": derivative_error <= tolerances["delay_derivative_fd_absolute_hz"],
        "single_path_during_geometry_retracing": all(
            item["plus_valid_path_count"] == 1 and item["minus_valid_path_count"] == 1
            for item in finite_differences),
    }
    is_static = not np.any(velocity) and control["mirror_speed"] == 0
    if is_static:
        checks["static_limit"] = static_change <= tolerances["static_relative_coefficient_change"] and abs(fd_rt) <= 1e-6
    result = {
        "name": control["name"], "status": "PASS" if all(checks.values()) else "FAIL",
        "rx_velocity_mps": control["velocity"],
        "reflector_normal_velocity_mps": control["mirror_speed"],
        "carrier_hz": FC, "valid_paths": 1,
        "geometric_length_m": geometry["length"], "rt_absolute_delay_sec": tau,
        "delay_absolute_error_sec": delay_error,
        "analytic_fd_hz": geometry["fd_hz"], "rt_fd_hz": fd_rt,
        "angle_projection_fd_hz": fd_projection, "phase_fit_fd_hz": fd_phase_fit,
        "analytic_fd_absolute_error_hz": geometry_fd_error,
        "projection_fd_absolute_error_hz": projection_error,
        "direction_unit_vector_error": direction_error,
        "cir_to_cfr_relative_l2_error": cfr_error,
        "passband_baseband_relative_complex_error": carrier_contract_error,
        "static_relative_coefficient_change": static_change,
        "delay_derivative_steps": finite_differences,
        "tolerances": tolerances, "checks": checks,
    }
    if not checks["single_path_during_geometry_retracing"]:
        result["limitation"] = (
            "Displaced reflector RT returned repeated equal-delay/equal-Doppler paths. "
            "A representative delay was used only for the derivative diagnostic; "
            "the single-path multiplicity check remains FAIL. No RT coefficients "
            "were removed or combined to manufacture a pass."
        )
    plot_data = {
        "name": control["name"], "times_sec": times,
        "relative_phase": relative_phase, "analytic_fd_hz": geometry["fd_hz"],
        "cfr_frequency_relative_error": np.linalg.norm(native_cfr - reference_cfr, axis=0)
        / np.maximum(np.linalg.norm(reference_cfr, axis=0), 1e-30),
    }
    return result, plot_data


def _make_plot(out_dir: Path, results: list[dict[str, Any]],
               plot_data: list[dict[str, Any]]) -> list[str]:
    if not plot_data:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = [result for result in results if "rt_fd_hz" in result]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    theory = np.array([result["analytic_fd_hz"] for result in successful])
    measured = np.array([result["rt_fd_hz"] for result in successful])
    axes[0, 0].scatter(theory, measured)
    bounds = [min(theory.min(), measured.min()) - 50,
              max(theory.max(), measured.max()) + 50]
    axes[0, 0].plot(bounds, bounds, "k--", linewidth=1)
    axes[0, 0].set(xlabel="Analytic Doppler [Hz]", ylabel="Sionna Doppler [Hz]",
                   title="Actual LoS / reflection controls")
    for step_index in (0, 1):
        values = [result["delay_derivative_steps"][step_index]["absolute_error_vs_geometric_fd_hz"]
                  for result in successful]
        step_us = successful[0]["delay_derivative_steps"][step_index]["central_difference_half_step_sec"] * 1e6
        axes[0, 1].plot(np.arange(len(values)), values, "o-", label=f"half-step {step_us:g} us")
    axes[0, 1].set(xlabel="Control index", ylabel="Absolute Doppler error [Hz]",
                   title="Geometry retraced at +/- time")
    axes[0, 1].legend(fontsize=8)
    for item in plot_data:
        if item["name"] in ("los_rx_positive_x", "moving_reflector_mirror_only"):
            axes[1, 0].plot(item["times_sec"] * 1e6, item["relative_phase"], label=item["name"])
            axes[1, 0].plot(item["times_sec"] * 1e6,
                            2 * np.pi * item["analytic_fd_hz"] * item["times_sec"],
                            "k--", linewidth=0.8)
    axes[1, 0].set(xlabel="Local time [us]", ylabel="Relative baseband phase [rad]",
                   title="CIR phase and analytic slope")
    axes[1, 0].legend(fontsize=8)
    maximum = np.max([item["cfr_frequency_relative_error"] for item in plot_data], axis=0)
    axes[1, 1].semilogy(FREQUENCIES / 1e6, np.maximum(maximum, 1e-16), "o-")
    axes[1, 1].set(xlabel="Baseband frequency offset [MHz]", ylabel="Relative CFR error",
                   title="CIR Fourier synthesis vs native cfr()")
    for axis in axes.ravel():
        axis.grid(True, alpha=0.25)
    destination = out_dir / "physics_controls.png"
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return [str(destination.resolve())]


def run_physics(out_dir: str | Path,
                backend: str = "llvm_ad_mono_polarized") -> dict[str, Any]:
    """Run real RT controls and save a JSON report and measured-data plot.

    Failures and unavailable dependencies are reported explicitly. The caller
    owns backend selection; in particular this function never substitutes CUDA
    for a requested CPU backend or reports skipped work as passing.
    """
    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "module": "analytic_sionna_physics_controls",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "requested_backend": backend, "status": "ERROR", "controls": [],
        "llvm_library_path": os.environ.get("DRJIT_LIBLLVM_PATH"),
        "sources": [
            "https://github.com/NVlabs/sionna-rt/blob/v2.2.0/src/sionna/rt/path_solvers/paths.py",
            "https://github.com/NVlabs/sionna-rt/blob/v2.2.0/src/sionna/rt/scenes/simple_reflector/simple_reflector.xml",
        ],
        "conventions": {
            "Paths.a": "Passband coefficient; carrier propagation phase is not yet included.",
            "Paths.cir": "Baseband coefficient includes exp(-j*2*pi*fc*tau); no second carrier phase.",
            "delay_normalization": False, "CFR_normalization": False,
            "frequency_offsets_hz": FREQUENCIES.tolist(),
            "local_time_sampling_hz": FS_TIME, "local_time_steps": NUM_TIME,
            "warning": "Fixed-path Doppler time evolution is a local model. This module does not validate long routes, dense OFDM waveforms, city interpolation, or compressed 128B archives.",
        },
        "tolerance_note": "Declared engineering checks; achieved values come only from this run. Float64 labels do not repair native RT float32 precision.",
    }
    plots: list[dict[str, Any]] = []
    try:
        import mitsuba as mi
        if mi.variant() != backend:
            raise RuntimeError(f"Caller must select backend {backend!r}; active variant is {mi.variant()!r}")
        import sionna.rt as rt
        report["active_backend"] = mi.variant()
        report["versions"] = {name: metadata.version(name) for name in
                              ("sionna-rt", "mitsuba", "drjit", "numpy", "matplotlib")}
        if mi.variant() != backend:
            raise RuntimeError(f"Import changed backend to {mi.variant()!r}; requested {backend!r}")
        solver = rt.PathSolver(deterministic=True)
        # On this Windows/portable-LLVM CPU runtime, symbolic compilation has
        # exhibited unresolved AVX constant symbols. Evaluated mode is selected
        # explicitly and recorded; it does not waive any physics checks.
        solver.loop_mode = "evaluated"
        report["solver_loop_mode"] = solver.loop_mode
        for control in _controls():
            try:
                result, data = _evaluate_control(rt, mi, solver, control)
                report["controls"].append(result)
                plots.append(data)
            except Exception as error:
                report["controls"].append({"name": control["name"], "status": "ERROR",
                                           "error": str(error), "traceback": traceback.format_exc()})
        report["plots"] = _make_plot(destination, report["controls"], plots)
        statuses = [item["status"] for item in report["controls"]]
        report["counts"] = {status: statuses.count(status) for status in ("PASS", "FAIL", "ERROR")}
        core = [item for item in report["controls"]
                if not item["name"].startswith("moving_reflector_")]
        report["required_los_and_fixed_reflector_controls_status"] = (
            "PASS" if len(core) == 8 and all(item["status"] == "PASS" for item in core)
            else "FAIL")
        report["status"] = ("ERROR" if "ERROR" in statuses else
                             "FAIL" if "FAIL" in statuses else "PASS")
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["report_path"] = str(destination / "physics_report.json")
    (destination / "physics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    return report

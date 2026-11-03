"""NumPy OFDM consistency checks, deliberately separate from NR and RT truth.

Receive-time convention, after CP removal, for integer path delays d_l::

    r[n] = exp(j*2*pi*f_cfo*(CP+n)/Fs)
           sum_l a_l exp(j*2*pi*nu_l*(CP+n)/Fs) x[(n-d_l) mod N].

With unitary FFTs, C(epsilon)[k,m] = mean_n exp(j*2*pi*(m-k+epsilon)*n/N).
Consequently A = phase_cfo C(eps_cfo) sum_l a_l phase_l C(eps_l) D(d_l).
This module checks that expression against delayed time-domain samples.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np


SEED = 20261025
NFFT = 256
CP = 64
DEFAULT_FS_HZ = 30.72e6
RELATIVE_L2_TOLERANCE = 2e-12


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), np.finfo(float).tiny))


def _time_domain_receive(
    symbols: np.ndarray,
    delays: np.ndarray,
    gains: np.ndarray,
    doppler_hz: np.ndarray,
    fs_hz: float,
    cp: int,
    oscillator_cfo_hz: float = 0.0,
) -> np.ndarray:
    """Linear time-domain channel; zeros precede this single CP-extended symbol.

    No circular-convolution assumption is used to implement the channel.  For
    CP >= max(delay), the useful receive samples reduce to circular convolution.
    Coefficients are referenced to q=0, the start of the transmitted CP.
    """
    x = np.fft.ifft(symbols, norm="ortho")
    with_cp = np.concatenate((x[-cp:], x)) if cp else x.copy()
    received = np.zeros(with_cp.size, dtype=np.complex128)
    q = np.arange(with_cp.size)
    for delay, gain, fd in zip(delays, gains, doppler_hz, strict=True):
        if int(delay) != delay or delay < 0:
            raise ValueError("This fixture requires nonnegative integer sample delays")
        delay = int(delay)
        if delay >= with_cp.size:
            continue
        # Phase is indexed by receive time, rather than by the delayed input.
        phase = np.exp(2j * np.pi * float(fd) * q[delay:] / fs_hz)
        received[delay:] += gain * phase * with_cp[: with_cp.size - delay]
    received *= np.exp(2j * np.pi * oscillator_cfo_hz * q / fs_hz)
    return np.fft.fft(received[cp : cp + symbols.size], norm="ortho")


def _static_cfr(delays: np.ndarray, gains: np.ndarray, nfft: int) -> np.ndarray:
    frequency_index = np.arange(nfft)
    return np.sum(gains[:, None] * np.exp(-2j * np.pi * delays[:, None] * frequency_index / nfft), axis=0)


def _ici_matrix(nfft: int, normalized_frequency: float) -> np.ndarray:
    modulation = np.exp(2j * np.pi * normalized_frequency * np.arange(nfft) / nfft)
    kernel = np.fft.fft(modulation) / nfft
    row, column = np.indices((nfft, nfft))
    return kernel[(row - column) % nfft]


def _ici_operator(
    delays: np.ndarray,
    gains: np.ndarray,
    doppler_hz: np.ndarray,
    fs_hz: float,
    cp: int,
    nfft: int,
    oscillator_cfo_hz: float = 0.0,
) -> np.ndarray:
    scs_hz = fs_hz / nfft
    path_operator = np.zeros((nfft, nfft), dtype=np.complex128)
    for delay, gain, fd in zip(delays, gains, doppler_hz, strict=True):
        frequency_phase = np.exp(-2j * np.pi * np.arange(nfft) * delay / nfft)
        cp_phase = np.exp(2j * np.pi * fd * cp / fs_hz)
        path_operator += gain * cp_phase * _ici_matrix(nfft, fd / scs_hz) * frequency_phase[None, :]
    common_phase = np.exp(2j * np.pi * oscillator_cfo_hz * cp / fs_hz)
    return common_phase * (_ici_matrix(nfft, oscillator_cfo_hz / scs_hz) @ path_operator)


def _combined_frequency_operator(
    delays: np.ndarray,
    gains: np.ndarray,
    doppler_hz: np.ndarray,
    fs_hz: float,
    cp: int,
    nfft: int,
    oscillator_cfo_hz: float,
) -> np.ndarray:
    """Equivalent algebraic check; it does not identify CFO and Doppler."""
    result = np.zeros((nfft, nfft), dtype=np.complex128)
    for delay, gain, fd in zip(delays, gains, doppler_hz, strict=True):
        total_frequency = fd + oscillator_cfo_hz
        phase = np.exp(2j * np.pi * total_frequency * cp / fs_hz)
        delay_phase = np.exp(-2j * np.pi * np.arange(nfft) * delay / nfft)
        result += gain * phase * _ici_matrix(nfft, total_frequency / (fs_hz / nfft)) * delay_phase[None, :]
    return result


def _noise_check(rng: np.random.Generator) -> tuple[dict[str, Any], np.ndarray]:
    sample_count = 2**18
    specified_variance = 0.025  # E[|n|^2], rather than variance of one component.
    noise = np.sqrt(specified_variance / 2) * (rng.standard_normal(sample_count) + 1j * rng.standard_normal(sample_count))
    measured_variance = float(np.mean(np.abs(noise) ** 2))
    relative_variance_error = abs(measured_variance / specified_variance - 1)
    statistical_tolerance = 6 / np.sqrt(sample_count)
    real_variance = float(np.var(noise.real))
    imag_variance = float(np.var(noise.imag))
    mean_limit = 6 * np.sqrt(specified_variance / (2 * sample_count))
    pseudo_covariance_ratio = float(abs(np.mean(noise**2)) / specified_variance)
    noise_fft = np.fft.fft(noise.reshape(-1, NFFT), norm="ortho", axis=-1)
    fft_variance = float(np.mean(np.abs(noise_fft) ** 2))
    parseval_error = abs(fft_variance - measured_variance) / measured_variance
    checks = {
        "total_complex_variance": bool(relative_variance_error <= statistical_tolerance),
        "real_variance_half_total": bool(abs(2 * real_variance / specified_variance - 1) <= statistical_tolerance),
        "imag_variance_half_total": bool(abs(2 * imag_variance / specified_variance - 1) <= statistical_tolerance),
        "zero_mean_real_imag": bool(abs(noise.real.mean()) <= mean_limit and abs(noise.imag.mean()) <= mean_limit),
        "proper_complex_pseudocovariance": bool(pseudo_covariance_ratio <= 6 * np.sqrt(2 / sample_count)),
        "unitary_fft_preserves_noise_energy": bool(parseval_error < RELATIVE_L2_TOLERANCE),
    }
    report = {
        "sample_count": sample_count,
        "specified_E_abs_n_squared": specified_variance,
        "measured_E_abs_n_squared": measured_variance,
        "real_variance": real_variance,
        "imag_variance": imag_variance,
        "variance_relative_error": float(relative_variance_error),
        "variance_relative_tolerance": float(statistical_tolerance),
        "mean_component_absolute_tolerance": float(mean_limit),
        "pseudocovariance_to_variance_ratio": pseudo_covariance_ratio,
        "fft_energy_relative_error": float(parseval_error),
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": "One seeded finite-sample sanity check; statistical bounds are not a proof of a noise model",
    }
    return report, noise


def _payload_case(payload: Mapping[str, Any] | None, symbols: np.ndarray) -> dict[str, Any]:
    if payload is None:
        return {"status": "not_provided", "physical_truth_claim": False}
    records = payload.get("records", [])
    for index, record in enumerate(records):
        if "gain" in record:
            gains = np.asarray(record["gain"], dtype=np.complex128).reshape(-1)
        elif "gain_real" in record and "gain_imag" in record:
            gains = np.asarray(record["gain_real"]) + 1j * np.asarray(record["gain_imag"])
            gains = np.asarray(gains, dtype=np.complex128).reshape(-1)
        else:
            continue
        if gains.size:
            break
    else:
        return {"status": "skipped_no_nonempty_record", "physical_truth_claim": False}
    delay_sec = np.asarray(record["delay_sec"], dtype=np.float64).reshape(-1)
    if delay_sec.shape != gains.shape or not np.all(np.isfinite(delay_sec)) or np.any(delay_sec < 0):
        raise ValueError("Additional payload has invalid path delays")
    if not np.all(np.isfinite(gains)):
        raise ValueError("Additional payload has non-finite path gains")
    delay_reference_sec = float(np.min(delay_sec))
    relative_delay = delay_sec - delay_reference_sec
    quantized = np.rint(relative_delay * DEFAULT_FS_HZ).astype(np.int64)
    error_sec = quantized / DEFAULT_FS_HZ - relative_delay
    fd = np.asarray(record.get("doppler_hz", np.zeros(gains.size)), dtype=np.float64).reshape(-1)
    if fd.shape != gains.shape or not np.all(np.isfinite(fd)):
        raise ValueError("Additional payload has invalid Doppler values")
    report: dict[str, Any] = {
        "status": "quantization_report_only",
        "source_kind": str(payload.get("kind", "provided_path_records")),
        "source_record_index": index,
        "paths": int(gains.size),
        "doppler_provided": "doppler_hz" in record,
        "delay_reference_sec": delay_reference_sec,
        "quantization": "nearest integer sample after removing common delay for timing synchronization",
        "sample_period_sec": 1 / DEFAULT_FS_HZ,
        "max_absolute_delay_quantization_error_sec": float(np.max(np.abs(error_sec))),
        "half_sample_error_bound_sec": 0.5 / DEFAULT_FS_HZ,
        "maximum_quantized_delay_samples": int(quantized.max()),
        "physical_truth_claim": False,
        "limits": "Integer-delay baseband surrogate of one input record; no fractional-delay or RT propagation accuracy is validated. Input gains anchor the local CP-start epoch.",
    }
    if quantized.max() > CP:
        report["status"] = "skipped_waveform_comparison_cp_too_short"
        return report
    observed = _time_domain_receive(symbols, quantized, gains, fd, DEFAULT_FS_HZ, CP)
    expected = _ici_operator(quantized, gains, fd, DEFAULT_FS_HZ, CP, NFFT) @ symbols
    error = _relative_error(observed, expected)
    report.update(status="passed_quantized_surrogate", waveform_relative_l2_error=error, tolerance=RELATIVE_L2_TOLERANCE)
    report["passed"] = bool(error < RELATIVE_L2_TOLERANCE)
    return report


def _make_figure(path: Path, static_errors: dict[str, np.ndarray], dynamic: list[dict[str, Any]], leakage: list[dict[str, Any]], noise: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, values in static_errors.items():
        axes[0, 0].semilogy(np.maximum(values, 1e-18), label=name)
    axes[0, 0].set(xlabel="FFT-bin index", ylabel="Absolute output difference", title="Static integer-delay CP fixture")
    axes[0, 0].legend(fontsize=8)
    labels = [f"{item['subcarrier_spacing_hz'] / 1000:.0f}k / {item['case']}" for item in dynamic]
    axes[0, 1].bar(np.arange(len(dynamic)), [max(item["relative_l2_error"], 1e-18) for item in dynamic])
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(RELATIVE_L2_TOLERANCE, color="black", linestyle="--", label="Acceptance tolerance")
    axes[0, 1].set_ylim(1e-16, 1e-11)
    axes[0, 1].set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right", fontsize=7)
    axes[0, 1].set(ylabel="Relative L2 error", title="Time-domain vs per-path ICI operator")
    axes[0, 1].legend(fontsize=8)
    for scs in (30000.0, 120000.0):
        cases = [item for item in leakage if item["subcarrier_spacing_hz"] == scs]
        axes[1, 0].plot([r["path_doppler_hz"] for r in cases], [100 * r["off_diagonal_energy_fraction"] for r in cases], marker="o", label=f"SCS {scs / 1000:.0f} kHz")
    axes[1, 0].set(xlabel="Single-path Doppler (Hz)", ylabel="Energy outside transmitted bin (%)", title="One-active-bin ICI probe")
    axes[1, 0].legend(fontsize=8)
    sigma = np.sqrt(0.025 / 2)
    axes[1, 1].hist(noise.real / sigma, bins=70, range=(-4, 4), density=True, alpha=0.55, label="Real / sigma")
    axes[1, 1].hist(noise.imag / sigma, bins=70, range=(-4, 4), density=True, alpha=0.45, label="Imag / sigma")
    z = np.linspace(-4, 4, 200)
    axes[1, 1].plot(z, np.exp(-z**2 / 2) / np.sqrt(2 * np.pi), color="black", label="N(0,1)")
    axes[1, 1].set(xlabel="Normalized noise component", ylabel="Density", title="Complex AWGN fixture")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle("Generic OFDM numerical validation — synthetic fixtures, not NR or RT ground truth", fontsize=12)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _run_waveform(out_dir: Path, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    symbols = np.exp(1j * (np.pi / 4 + (np.pi / 2) * rng.integers(0, 4, NFFT)))
    delays = np.array([0, 3, 19, 47], dtype=np.int64)
    gains = np.array([1 + 0.2j, 0.35 - 0.17j, -0.18 + 0.31j, 0.12 + 0.09j])
    static = []
    static_errors = {}
    for name, taps, amplitudes in (
        ("single path", np.array([11]), np.array([0.7 - 0.4j])),
        ("four paths", delays, gains),
    ):
        expected = _static_cfr(taps, amplitudes, NFFT) * symbols
        actual = _time_domain_receive(symbols, taps, amplitudes, np.zeros(taps.size), DEFAULT_FS_HZ, CP)
        static_errors[name] = np.abs(actual - expected)
        static.append({"case": name, "integer_delays_samples": taps.tolist(), "relative_l2_error": _relative_error(actual, expected)})

    insufficient_cp = _time_domain_receive(symbols, delays, gains, np.zeros(delays.size), DEFAULT_FS_HZ, 8)
    insufficient_cp_error = _relative_error(insufficient_cp, _static_cfr(delays, gains, NFFT) * symbols)
    dynamic = []
    leakage = []
    counterexample_errors = []
    operator_algebra_errors = []
    for fs_hz in (DEFAULT_FS_HZ, 7.68e6):
        scs_hz = fs_hz / NFFT
        for name, fd, cfo in (
            ("common", np.full(4, 2802.0), 0.0),
            ("per-path", np.array([0.0, 2802.0, -5604.0, 5604.0]), 0.0),
            ("per-path+CFO", np.array([0.0, 2802.0, -5604.0, 5604.0]), 1234.0),
        ):
            operator = _ici_operator(delays, gains, fd, fs_hz, CP, NFFT, cfo)
            observed = _time_domain_receive(symbols, delays, gains, fd, fs_hz, CP, cfo)
            expected = operator @ symbols
            dynamic.append({"case": name, "sample_rate_hz": fs_hz, "subcarrier_spacing_hz": scs_hz, "path_doppler_hz": fd.tolist(), "oscillator_cfo_hz": cfo, "relative_l2_error": _relative_error(observed, expected)})
            combined = _combined_frequency_operator(delays, gains, fd, fs_hz, CP, NFFT, cfo)
            operator_algebra_errors.append(_relative_error(operator, combined))
            if name == "per-path":
                mean_fd = float(np.sum(np.abs(gains) ** 2 * fd) / np.sum(np.abs(gains) ** 2))
                common_operator = np.exp(2j * np.pi * mean_fd * CP / fs_hz) * _ici_matrix(NFFT, mean_fd / scs_hz) * _static_cfr(delays, gains, NFFT)[None, :]
                counterexample_errors.append({"subcarrier_spacing_hz": scs_hz, "common_approximation_relative_error": _relative_error(common_operator @ symbols, expected)})
        probe = np.zeros(NFFT, dtype=np.complex128)
        probe[23] = 1.0
        for fd_hz in (0.0, 2802.0, 5604.0):
            observed = _time_domain_receive(probe, np.array([0]), np.array([1 + 0j]), np.array([fd_hz]), fs_hz, CP)
            kernel = _ici_matrix(NFFT, fd_hz / scs_hz)[:, 23]
            off_diagonal = float((np.sum(np.abs(observed) ** 2) - abs(observed[23]) ** 2) / np.sum(np.abs(observed) ** 2))
            predicted = float((np.sum(np.abs(kernel) ** 2) - abs(kernel[23]) ** 2) / np.sum(np.abs(kernel) ** 2))
            leakage.append({"subcarrier_spacing_hz": scs_hz, "path_doppler_hz": fd_hz, "normalized_doppler": fd_hz / scs_hz, "oscillator_cfo_hz": 0.0, "off_diagonal_energy_fraction": max(0.0, off_diagonal), "operator_predicted_fraction": max(0.0, predicted), "absolute_fraction_error": abs(off_diagonal - predicted)})
    noise_report, noise = _noise_check(rng)
    additional = _payload_case(payload, symbols)
    figure_path = out_dir / "ofdm_validation.png"
    _make_figure(figure_path, static_errors, dynamic, leakage, noise)
    checks = {
        "static_single_and_multipath_fft_equal_HX": all(item["relative_l2_error"] < RELATIVE_L2_TOLERANCE for item in static),
        "time_varying_waveform_equals_per_path_ici_operator": all(item["relative_l2_error"] < RELATIVE_L2_TOLERANCE for item in dynamic),
        "cfo_path_operator_factorization": max(operator_algebra_errors) < RELATIVE_L2_TOLERANCE,
        "ici_energy_matches_operator": max(item["absolute_fraction_error"] for item in leakage) < RELATIVE_L2_TOLERANCE,
        "insufficient_cp_negative_control_detected": insufficient_cp_error > 1e-4,
        "single_common_frequency_counterexample_detected": all(item["common_approximation_relative_error"] > 1e-4 for item in counterexample_errors),
        "complex_noise_checks": noise_report["passed"],
        "additional_quantized_payload_case_if_executed": bool(additional.get("passed", True)),
        "figure_created": figure_path.is_file(),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "seed": SEED,
        "relative_l2_tolerance": RELATIVE_L2_TOLERANCE,
        "default_configuration": {"nfft": NFFT, "cp_samples": CP, "sample_rate_hz": DEFAULT_FS_HZ, "subcarrier_spacing_hz": DEFAULT_FS_HZ / NFFT, "fft_normalization": "ortho", "constellation": "full-grid QPSK fixture, no NR resource mapping"},
        "additional_configuration": {"sample_rate_hz": 7.68e6, "subcarrier_spacing_hz": 30000.0},
        "channel_convention": "Path coefficient at CP-start epoch; Doppler and oscillator CFO multiply receive-time sample q. No repeated carrier-delay phase is inserted.",
        "static_cases": static,
        "dynamic_cases": dynamic,
        "ici_probe_cases": leakage,
        "cfo_factorization_max_relative_l2_error": max(operator_algebra_errors),
        "negative_controls": {"cp_8_max_delay_47_relative_error": insufficient_cp_error, "single_common_frequency_approximation": counterexample_errors},
        "noise": noise_report,
        "additional_payload_case": additional,
        "artifacts": {"figure": str(figure_path), "report": str(out_dir / "waveform_validation.json")},
        "versions": {"python": sys.version, "numpy": np.__version__},
        "limits": ["Generic SISO OFDM numerical consistency, not a 3GPP NR implementation", "Integer-sample channel fixtures are synthetic and are not RT ground truth", "Per-path constant Doppler over one symbol; no path birth/death or fractional-delay implementation", "CFO and path Doppler are separate inputs; this does not prove they are separately identifiable", "No measured RF, MIMO receiver, BER/BLER, or link-budget validation is claimed"],
    }


def run_waveform(out_dir: str | Path, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run small OFDM checks and optionally quantify one RT record's tap rounding."""
    directory = Path(out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_waveform(directory, payload)
    except Exception:
        result = {"status": "failed", "passed": False, "error": traceback.format_exc()}
    target = directory / "waveform_validation.json"
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(partial, target)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_waveform(args.out_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

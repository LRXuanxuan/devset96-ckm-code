"""Algebraic controls for the grid adaptation; no RT or dataset inference."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from grid_tools import (bu_style_delay_grid, centered_dft,
                        angular_analysis, angular_synthesis)


def nmse(estimate, truth):
    return float(np.linalg.norm(estimate - truth) ** 2 / np.linalg.norm(truth) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    # Synthetic design constants only, not privileged statistics from a route.
    grid = bu_style_delay_grid(8, 4e-6, 0.9e-6, 0.75)
    probes = np.unique(np.r_[np.linspace(0, 4e-6, 1001), grid.edges_sec,
                            np.nextafter(grid.edges_sec[1:-1], 0),
                            np.nextafter(grid.edges_sec[1:-1], np.inf)])
    cells, fractions = grid.encode(probes)
    delay_error = float(np.max(np.abs(grid.decode(cells, fractions) - probes)))
    assert delay_error < 1e-20 and fractions.min() >= 0 and fractions.max() <= 1
    nearest_probe = 0.29e-6
    nearest = int(np.argmin(np.abs(grid.nodes_sec - nearest_probe)))
    nearest_offset = float(nearest_probe - grid.nodes_sec[nearest])
    assert nearest_offset < 0
    cell, fraction = grid.encode(nearest_probe)

    ur, ut = centered_dft(4), centered_dft(8)
    unitarity_error = max(float(np.linalg.norm(ur @ ur.conj().T - np.eye(4))),
                         float(np.linalg.norm(ut @ ut.conj().T - np.eye(8))))
    # An intentionally off-grid steering vector still has an exact complete
    # finite-array DFT expansion, although one coefficient is not sufficient.
    h = np.exp(1j * np.pi * np.arange(4) * 0.42)[:, None] * np.exp(
        -1j * np.pi * np.arange(8) * 0.4)[None, :]
    angular = angular_analysis(h, ur, ut)
    reconstruction_error = nmse(angular_synthesis(angular, ur, ut), h)
    single_coefficient_fraction = float(np.max(abs(angular) ** 2) / np.sum(abs(angular) ** 2))
    assert unitarity_error < 1e-12 and reconstruction_error < 1e-24
    assert single_coefficient_fraction < 0.999

    # Two orthogonal spatial signatures, one physical delay, two frequencies.
    # A single time-invariant angular matrix times one exponential has rank
    # at most one over time. Best rank-one error is therefore a LOWER BOUND
    # on every single-frequency-per-delay representation's error.
    c1, c2 = np.zeros((4, 8), complex), np.zeros((4, 8), complex)
    c1[0, 0] = 1
    c2[1, 1] = 1
    a1, a2 = (angular_synthesis(c, ur, ut).ravel() for c in (c1, c2))
    times = np.arange(64) * 1e-4
    h_time = a1[:, None] * np.exp(2j * np.pi * 250 * times)[None, :] + (
        a2[:, None] * np.exp(-2j * np.pi * 500 * times)[None, :])
    singular = np.linalg.svd(h_time, compute_uv=False)
    rank_one_lower_bound = float(np.sum(singular[1:] ** 2) / np.sum(singular ** 2))
    assert rank_one_lower_bound > 0.4

    # The survey's 1.6 MHz tone spacing creates a 0.625 us delay period.
    frequencies = -12e6 + np.arange(16) * 1.6e6
    period = 1 / 1.6e6
    tau1 = 0.31e-6
    h1 = np.exp(-2j * np.pi * frequencies * tau1)
    h2 = np.exp(-2j * np.pi * frequencies * (tau1 + period))
    p1 = abs(np.fft.ifft(h1, norm="ortho")) ** 2
    p2 = abs(np.fft.ifft(h2, norm="ortho")) ** 2
    periodic_power_error = nmse(p2, p1)
    assert periodic_power_error < 1e-24

    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("grid_tools.py").resolve(),
               root / "sionna_small_validation/validation/dataset_v2.py"]
    result = dict(
        status="passed", created_at=datetime.now(timezone.utc).isoformat(),
        delay=dict(edges_sec=grid.edges_sec.tolist(), number_of_probes=len(probes),
                   maximum_roundtrip_error_sec=delay_error,
                   nearest_assignment_negative_offset_sec=nearest_offset,
                   consistent_left_cell=int(cell), consistent_fraction=float(fraction)),
        angle=dict(unitarity_frobenius_error=unitarity_error,
                   off_grid_full_dft_reconstruction_nmse=reconstruction_error,
                   largest_coefficient_energy_fraction=single_coefficient_fraction),
        collision=dict(frequencies_hz=[250, -500], equal_delay=True,
                       spatial_time_singular_values=singular[:3].tolist(),
                       any_rank_one_time_model_nmse_lower_bound=rank_one_lower_bound,
                       interpretation="channel representation lower bound, not receiver-estimator benchmark"),
        periodic_map=dict(tone_spacing_hz=1.6e6, delay_period_sec=period,
                          cyclic_bin_spacing_sec=period / 16,
                          delays_sec=[tau1, tau1 + period], power_feature_nmse=periodic_power_error),
        source_binding=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        bu_reproduction=False, joint_continuous_solver_implemented=False,
        original_devset_prediction_scores=False, new_RT_calls=0, test_split_accessed=False)
    (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

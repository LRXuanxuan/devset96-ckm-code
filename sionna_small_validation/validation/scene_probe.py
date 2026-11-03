"""Small, geometric city-scene probe; no radio path solver is called.

The top-down test certifies sampled open-air corridors over the built-in
z=0 ground plane. It does not infer road semantics, traffic laws or RF quality.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np


def probe_scene(scene, name: str, outdir) -> dict:
    """Return bbox, raster, and up to three 30-m open-ground route proposals."""
    import mitsuba as mi
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bbox = scene.mi_scene.bbox()
    bmin = np.asarray([float(bbox.min.x), float(bbox.min.y), float(bbox.min.z)])
    bmax = np.asarray([float(bbox.max.x), float(bbox.max.y), float(bbox.max.z)])
    # All three audited city assets have a flat ground mesh at z=0.
    ground_z = 0.0
    center = np.clip(np.zeros(2), bmin[:2] + 105.0, bmax[:2] - 105.0)
    step = 1.0
    xs = np.arange(center[0] - 100.0, center[0] + 100.0 + step / 2, step)
    ys = np.arange(center[1] - 100.0, center[1] + 100.0 + step / 2, step)
    xx, yy = np.meshgrid(xs, ys)
    origin = mi.Point3f(mi.Float(xx.ravel().astype(np.float32)),
                        mi.Float(yy.ravel().astype(np.float32)),
                        float(bmax[2] + 10.0))
    ray = mi.Ray3f(origin, mi.Vector3f(0.0, 0.0, -1.0))
    hit = scene.mi_scene.ray_intersect(ray)
    valid = np.asarray(hit.is_valid(), dtype=bool).reshape(xx.shape)
    height = np.asarray(hit.p.z, dtype=np.float64).reshape(xx.shape)
    nz = np.asarray(hit.n.z, dtype=np.float64).reshape(xx.shape)
    ground = valid & (np.abs(height - ground_z) < 0.02) & (np.abs(nz) > 0.9)
    # Reject enclosed courtyards: keep ground connected to the ROI boundary.
    accessible = np.zeros_like(ground)
    boundary = np.zeros_like(ground)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    todo = deque(zip(*np.where(ground & boundary)))
    for j, i in todo:
        accessible[j, i] = True
    while todo:
        j, i = todo.popleft()
        for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
            if (0 <= jj < ground.shape[0] and 0 <= ii < ground.shape[1]
                    and ground[jj, ii] and not accessible[jj, ii]):
                accessible[jj, ii] = True
                todo.append((jj, ii))
    # 5x5 erosion protects a +/-2m footprint, including diagonal side offsets.
    safe = np.zeros_like(ground)
    safe[2:-2, 2:-2] = np.lib.stride_tricks.sliding_window_view(
        accessible, (5, 5)
    ).all(axis=(-1, -2))

    iy, ix = np.where(safe)
    # Limit search to every other raster point to keep proposal work small.
    keep = ((ix % 2) == 0) & ((iy % 2) == 0)
    iy, ix = iy[keep], ix[keep]
    samples = np.arange(-15.0, 15.0 + 0.5, step)
    proposals = []
    for heading in np.arange(12) * math.pi / 12.0:
        dx, dy = math.cos(heading), math.sin(heading)
        qi = ix[:, None] + np.rint(samples[None, :] * dx / step).astype(int)
        qj = iy[:, None] + np.rint(samples[None, :] * dy / step).astype(int)
        inside = ((qi >= 0) & (qi < len(xs)) & (qj >= 0) & (qj < len(ys))).all(axis=1)
        qi = np.clip(qi, 0, len(xs) - 1)
        qj = np.clip(qj, 0, len(ys) - 1)
        passes = inside & safe[qj, qi].all(axis=1)
        for cx, cy in zip(ix[passes], iy[passes]):
            x, y = float(xs[cx]), float(ys[cy])
            proposals.append((x * x + y * y, x, y, float(heading)))
    proposals.sort()

    routes = []
    for _, x, y, heading in proposals:
        if any(np.linalg.norm(np.asarray([x, y]) - np.asarray(r["center_xyz"][:2])) < 15.0
               for r in routes):
            continue
        direction = np.asarray([math.cos(heading), math.sin(heading)])
        start = np.asarray([x, y]) - 15.0 * direction
        end = np.asarray([x, y]) + 15.0 * direction
        # Prefer a non-coincident open-air TX near one end of the route.
        preferred = start - 10.0 * direction
        candidates = np.column_stack((xx[safe], yy[safe]))
        distance_from_center = np.linalg.norm(candidates - [x, y], axis=1)
        allow = distance_from_center >= 12.0
        candidates = candidates[allow]
        tx_xy = candidates[np.argmin(np.sum((candidates - preferred) ** 2, axis=1))]
        routes.append({
            "route_id": f"{name}_open_corridor_{len(routes) + 1}",
            "center_xyz": [x, y, 1.5],
            "heading_rad": heading,
            "heading_deg": math.degrees(heading),
            "start_xyz": [float(start[0]), float(start[1]), 1.5],
            "end_xyz": [float(end[0]), float(end[1]), 1.5],
            "length_m": 30.0,
            "sample_step_m": step,
            "lateral_clearance_sample_m": 2.0,
            "suggested_tx_xyz": [float(tx_xy[0]), float(tx_xy[1]), 12.0],
            "suggested_rx_xyz": [x, y, 1.5],
            "geometric_status": "sampled open ground; road semantics and radio paths not verified",
        })
        tx_xyz = np.asarray(routes[-1]["suggested_tx_xyz"], dtype=float)
        rx_xyz = np.asarray(routes[-1]["suggested_rx_xyz"], dtype=float)
        delta = rx_xyz - tx_xyz
        distance = float(np.linalg.norm(delta))
        direct_ray = mi.Ray3f(mi.Point3f(tx_xyz.tolist()),
                              mi.Vector3f((delta / distance).tolist()))
        direct_ray.maxt = distance - 1e-3
        blocked = bool(np.asarray(scene.mi_scene.ray_test(direct_ray)).reshape(-1)[0])
        routes[-1]["center_has_geometric_los"] = not blocked
        if len(routes) == 3:
            break

    # A separate geometry-only NLoS proposal: never replace or RF-rank routes.
    nlos_tx_candidate = None
    if routes:
        jj, ii = np.indices(safe.shape)
        selection = safe & ((ii % 5) == 0) & ((jj % 5) == 0)
        tx_candidates = np.column_stack((xx[selection], yy[selection],
                                         np.full(np.count_nonzero(selection), 12.0)))
        rx_center = np.asarray(routes[0]["suggested_rx_xyz"], dtype=float)
        deltas = rx_center[None, :] - tx_candidates
        distances = np.linalg.norm(deltas, axis=1)
        keep = (distances >= 25.0) & (distances <= 90.0)
        tx_candidates, deltas, distances = tx_candidates[keep], deltas[keep], distances[keep]
        if len(distances):
            directions = deltas / distances[:, None]
            direct_rays = mi.Ray3f(
                mi.Point3f(*(mi.Float(tx_candidates[:, i].astype(np.float32)) for i in range(3))),
                mi.Vector3f(*(mi.Float(directions[:, i].astype(np.float32)) for i in range(3))))
            direct_rays.maxt = mi.Float((distances - 1e-3).astype(np.float32))
            obstruction = np.asarray(scene.mi_scene.ray_test(direct_rays), dtype=bool)
            if np.any(obstruction):
                blocked_indices = np.where(obstruction)[0]
                chosen = blocked_indices[np.argmin(distances[blocked_indices])]
                nlos_tx_candidate = {
                    "tx_xyz": tx_candidates[chosen].tolist(),
                    "rx_xyz": rx_center.tolist(),
                    "tx_rx_distance_m": float(distances[chosen]),
                    "center_has_geometric_los": False,
                    "selection_rule": "nearest geometric-obstructed candidate among open-ground 5m grid points at 25-90m distance; not RF-ranked",
                    "num_geometric_candidates": int(len(distances)),
                    "num_directly_blocked_candidates": int(np.count_nonzero(obstruction)),
                    "rf_status": "not_traced; reflection/diffraction channel nonzero still requires PathSolver",
                }

    # Save modest arrays for independent inspection; no channel tensors stored.
    raster_path = outdir / f"{name}_ground_mask.npz"
    np.savez_compressed(raster_path, x_m=xs, y_m=ys, first_hit_z_m=height,
                        ground_mask=ground, boundary_connected_ground_mask=accessible,
                        route_clearance_mask=safe)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.imshow(ground.astype(float), origin="lower", cmap="gray", vmin=0, vmax=1,
              extent=[xs[0] - step / 2, xs[-1] + step / 2,
                      ys[0] - step / 2, ys[-1] + step / 2], interpolation="nearest")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for route, color in zip(routes, colors):
        a, b = route["start_xyz"], route["end_xyz"]
        tx = route["suggested_tx_xyz"]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=2.5,
                label=route["route_id"].rsplit("_", 1)[-1] + ": open corridor")
        ax.scatter([a[0]], [a[1]], marker="o", color=color, s=22)
        ax.scatter([tx[0]], [tx[1]], marker="^", color=color, s=55)
    if nlos_tx_candidate:
        tx_nlos = nlos_tx_candidate["tx_xyz"]
        rx_nlos = nlos_tx_candidate["rx_xyz"]
        ax.scatter([tx_nlos[0]], [tx_nlos[1]], marker="x", color="#CC79A7", s=65,
                   label="blocked direct ray TX candidate")
        ax.plot([tx_nlos[0], rx_nlos[0]], [tx_nlos[1], rx_nlos[1]],
                ls="--", color="#CC79A7", lw=1.2)
    ax.set_title(f"{name}: first-hit ground mask, central 200 m ROI\n"
                 "white = unobstructed z=0 ground; triangle = proposed TX")
    ax.set_xlabel("Global x [m]")
    ax.set_ylabel("Global y [m]")
    ax.set_aspect("equal")
    if routes:
        ax.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    plot_path = outdir / f"{name}_ground_mask.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    result = {
        "scene": name,
        "mitsuba_variant": mi.variant(),
        "bbox_min_xyz_m": bmin.tolist(),
        "bbox_max_xyz_m": bmax.tolist(),
        "roi_center_xy_m": center.tolist(),
        "roi_extent_m": [200.0, 200.0],
        "ground_plane_z_m": ground_z,
        "ray_count": int(xx.size),
        "open_ground_fraction_in_roi": float(np.mean(ground)),
        "boundary_connected_ground_fraction_in_roi": float(np.mean(accessible)),
        "raster_spacing_m": step,
        "routes": routes,
        "safe_route_found": bool(routes),
        "route_center_xyz": routes[0]["center_xyz"] if routes else None,
        "heading_rad": routes[0]["heading_rad"] if routes else None,
        "suggested_tx_xyz": routes[0]["suggested_tx_xyz"] if routes else None,
        "suggested_rx_xyz": routes[0]["suggested_rx_xyz"] if routes else None,
        "nlos_tx_candidate": nlos_tx_candidate,
        "topview_png": str(plot_path.resolve()),
        "ground_raster_npz": str(raster_path.resolve()),
        "limitations": [
            "Top-down clearance is sampled at 1m raster spacing with a 5x5 erosion; sub-meter thin obstacles can be missed.",
            "First-hit z=0 removes roofs and covered ground; semantic drivable roads are not identified.",
            "Candidates are in a central ROI and are not chosen by received power or LoS, so no RF selection claim is made.",
            "Dynamic traffic footprint and fine route samples require separate validation before generation.",
        ],
    }
    (outdir / f"{name}_scene_probe.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    try:
        from .common import portable_asset, setup_backend
    except ImportError:
        from common import portable_asset, setup_backend
    setup_backend("llvm_ad_mono_polarized")
    import sionna.rt as rt
    output_dir = Path(__file__).resolve().parents[1] / "output" / "scene_probe"
    summaries = []
    for city in ("munich", "florence", "etoile"):
        scene_ = rt.load_scene(portable_asset(getattr(rt.scene, city)))
        report = probe_scene(scene_, city, output_dir)
        summaries.append(report)
        print(json.dumps({"scene": city, "routes": report["routes"],
                          "topview_png": report["topview_png"]}, ensure_ascii=False), flush=True)
    (output_dir / "scene_probe_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

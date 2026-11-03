"""Bounded physical-admission diagnostics, retaining every native RT path.

These experiments distinguish search/duplicate failures from interpolation error.
They do not silently repair, filter or certify the old archives.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from .common import portable_asset, save_json, sha256
from .cities import (FC, extract_compact, native_cfr, reconstruct,
                     stable_objects, xyz_rows, defined_nmse)


def _solver(rt):
    solver = rt.PathSolver(deterministic=True)
    solver.loop_mode = 'evaluated'
    return solver


def _settings(samples=20000, depth=2, seed=20261025, diffraction=False):
    return dict(max_depth=depth, samples_per_src=samples,
                max_num_paths_per_src=4096, synthetic_array=True,
                los=True, specular_reflection=True, diffuse_reflection=False,
                refraction=False, diffraction=diffraction, seed=seed)


def metadata(paths):
    """Version-pinned source-buffer data in the native public path order."""
    valid = np.asarray(paths.valid).reshape(-1).astype(bool)
    tau = np.asarray(paths.tau).reshape(-1)
    selected = np.flatnonzero(valid & (tau >= 0))
    if not len(selected):
        return {'objects': [], 'primitives': [], 'interaction_types': [],
                'vertices_xyz_m': [], 'native_path_index': [],
                'exact_duplicate_geometry_pairs': []}
    permutation = np.asarray(paths._path_ind).astype(np.int64)
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation))
    buf = paths._paths_buffer
    def take(field):
        return np.array(field.numpy(), copy=True)[inverse][selected]
    objects, primitives, types = take(buf.shapes), take(buf.primitives), take(buf.interaction_types)
    vertices = np.stack([take(buf.vertices_x), take(buf.vertices_y), take(buf.vertices_z)], axis=-1)
    duplicates = []
    for i in range(len(selected)):
        for j in range(i):
            if (np.array_equal(objects[i], objects[j])
                    and np.array_equal(primitives[i], primitives[j])
                    and np.array_equal(types[i], types[j])
                    and np.array_equal(vertices[i], vertices[j])
                    and tau[selected[i]] == tau[selected[j]]):
                duplicates.append([j, i])
    return {'objects': objects, 'primitives': primitives, 'interaction_types': types,
            'vertices_xyz_m': vertices, 'native_path_index': selected,
            'exact_duplicate_geometry_pairs': duplicates}


def _snapshot(scene, solver, object_map, settings):
    tick = time.perf_counter()
    paths = solver(scene, **settings)
    # Keep all geometry owners alive until exports are materialized.
    paths._validation_scene_owner = scene
    compact = extract_compact(paths, object_map)
    H = native_cfr(paths)
    ptx = xyz_rows(scene.tx_array.rotate(scene.wavelength, scene.get('tx').orientation))
    prx = xyz_rows(scene.rx_array.rotate(scene.wavelength, scene.get('rx').orientation))
    reconstructed = reconstruct(compact, ptx, prx)
    record = {'path_count': len(compact), 'nonzero_cfr': bool(np.linalg.norm(H) > 0),
              'has_los': bool(np.any(compact['depth'] == 0)),
              'delay_sec': compact['delay_sec'], 'doppler_hz': compact['doppler_hz'],
              'coefficient_real': compact['b00'].real, 'coefficient_imag': compact['b00'].imag,
              'interaction_object_ids': compact['object_ids'],
              'depth': compact['depth'], 'raw_geometry': metadata(paths),
              'compact_native_cfr_nmse_db': defined_nmse(reconstructed, H),
              'solve_seconds': time.perf_counter() - tick}
    return record, compact, H


def _city_geometry(rt, city, source_root):
    """Reproduce the actual archived first anchor, rather than reselect a route."""
    directory = source_root / ('city_' + city) / 'cities'
    report = json.loads((directory / 'cities_report.json').read_text(encoding='utf-8'))
    with h5py.File(directory / 'compact128.h5', 'r') as f:
        fragments = f['fragment'].asstr()[:]
        states = f['state'].asstr()[:]
        frag = 1 if city == 'munich' else 3
        index = int(np.flatnonzero((fragments == f'{city}_{frag}') & (states == 'static'))[0])
        info = {name: f[name][index] for name in ('rx_xyz_m', 'tx_xyz_m',
                  'rx_orientation_rad', 'tx_orientation_rad', 'rx_relative_m', 'tx_relative_m')}
    scene = rt.load_scene(portable_asset(getattr(rt.scene, city)))
    scene.frequency = FC
    scene.tx_array = rt.PlanarArray(num_rows=8, num_cols=8, vertical_spacing=.5,
                                   horizontal_spacing=.5, pattern='iso', polarization='V')
    scene.rx_array = rt.PlanarArray(num_rows=4, num_cols=4, vertical_spacing=.5,
                                   horizontal_spacing=.5, pattern='iso', polarization='V')
    tx = rt.Transmitter('tx', position=info['tx_xyz_m'].tolist(), orientation=info['tx_orientation_rad'].tolist())
    rx = rt.Receiver('rx', position=info['rx_xyz_m'].tolist(), orientation=info['rx_orientation_rad'].tolist())
    direction = np.array([np.cos(info['rx_orientation_rad'][0]), np.sin(info['rx_orientation_rad'][0]), 0.])
    rx.velocity = (30 * direction).tolist()
    scene.add([tx, rx])
    object_map, object_records = stable_objects(scene, city)
    return scene, rx, direction, info, object_map, object_records, report


def etoile_visibility_probe(scene, info, direction, original_record):
    """Image-source geometry and shadow-ray check of a disappearing reflection.

    This is an independent triangle/occlusion diagnostic, not an assertion that
    RT can distinguish every neighboring physical edge at submillimeter scale.
    """
    import mitsuba as mi
    meta = original_record['raw_geometry']
    # The first one-bounce branch is the branch absent at +5 mm in the audited
    # archive. Select by geometry depth, without assuming a runtime object ID.
    p = next(i for i, depth in enumerate(original_record['depth']) if depth == 1)
    runtime_id = int(meta['objects'][p][0]); primitive = int(meta['primitives'][p][0])
    obj = next(o for o in scene.objects.values() if int(o.object_id) == runtime_id)
    params = mi.traverse(obj.mi_mesh)
    vertex = np.asarray(params['vertex_positions'], dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(params['faces']).reshape(-1, 3)
    triangle = vertex[faces[primitive]]
    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    normal /= np.linalg.norm(normal)
    tx = info['tx_xyz_m']; rx0 = info['rx_xyz_m']
    image = tx - 2*normal*np.dot(tx-triangle[0], normal)
    records = []
    for offset in np.linspace(0., .01, 21):
        rx = rx0 + offset*direction
        q = image + np.dot(triangle[0]-image, normal)/np.dot(rx-image, normal)*(rx-image)
        basis = np.column_stack([triangle[1]-triangle[0], triangle[2]-triangle[0]])
        uv = np.linalg.lstsq(basis, q-triangle[0], rcond=None)[0]
        bary = np.array([1-uv.sum(), uv[0], uv[1]])
        checks = []
        for epsilon in (1e-4, 1e-3):
            segment_checks = []
            for a, b in ((tx, q), (q, rx)):
                distance = float(np.linalg.norm(b-a)); d = (b-a)/distance
                ray = mi.Ray3f(mi.Point3f((a+epsilon*d).tolist()), mi.Vector3f(d.tolist()))
                ray.maxt = distance-2*epsilon
                hit = scene.mi_scene.ray_intersect(ray)
                blocked = bool(np.asarray(hit.is_valid()).reshape(-1)[0])
                segment_checks.append({'blocked': blocked,
                                       'occluder_primitive': int(np.asarray(hit.prim_index).reshape(-1)[0]) if blocked else None,
                                       'distance_to_hit_m': float(np.asarray(hit.t).reshape(-1)[0]) if blocked else None})
            checks.append({'endpoint_epsilon_m': epsilon, 'segments_tx_q_q_rx': segment_checks})
        records.append({'offset_m': float(offset), 'image_source_reflection_xyz_m': q,
                        'barycentric': bary, 'inside_original_triangle': bool(np.min(bary) >= -1e-8),
                        'shadow_checks': checks})
    edge_vertices = set(faces[primitive][[0, 2]].tolist())
    adjacent = []
    for face_index in np.flatnonzero(np.sum(np.isin(faces, list(edge_vertices)), axis=1) == 2):
        tri = vertex[faces[face_index]]
        n = np.cross(tri[1]-tri[0], tri[2]-tri[0]); n /= np.linalg.norm(n)
        adjacent.append({'primitive': int(face_index), 'triangle_xyz_m': tri,
                         'normal_dot_original': float(np.dot(n, normal))})
    return {'object_name': obj.name, 'runtime_object_id': runtime_id,
            'primitive': primitive, 'triangle_xyz_m': triangle, 'records': records,
            'triangles_adjacent_to_crossed_edge': adjacent,
            'scope': 'One disappearing one-bounce branch only; no path deduplication or zero filling.'}


def run_city_admission(out, city, source_root):
    import sionna.rt as rt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    scene, rx, direction, info, object_map, objects, source = _city_geometry(rt, city, Path(source_root))
    solver = _solver(rt)
    settings_grid = [
        ('original_repeat_0', _settings()),
        ('original_repeat_1', _settings()),
        ('seed_7', _settings(seed=7)),
        ('seed_19', _settings(seed=19)),
        ('rays_100k', _settings(samples=100000)),
        ('rays_300k', _settings(samples=300000)),
        ('depth_3_100k', _settings(samples=100000, depth=3)),
        ('depth_3_diffraction_100k', _settings(samples=100000, depth=3, diffraction=True)),
    ]
    records = []; cfr = {}; compacts = {}
    offsets = [0.] if city == 'munich' else [0., .005, .01]
    for label, settings in settings_grid:
        for offset in offsets:
            rx.position = (info['rx_xyz_m'] + offset * direction).tolist()
            rec, compact, H = _snapshot(scene, solver, object_map, settings)
            key = f'{label}_{offset:.6f}'
            rec.update(key=key, configuration=label, offset_m=offset, settings=settings,
                       rx_xyz_m=xyz_rows(rx.position)[0])
            records.append(rec); cfr[key] = H; compacts[key] = compact
            print(f'[admission] {city} {key}: {len(compact)} paths', flush=True)
            save_json(out / 'progress.json', {'city': city, 'records': records})
    midpoint = []
    if city == 'etoile':
        for label, settings in settings_grid:
            left = compacts[f'{label}_0.000000']
            mid_H = cfr[f'{label}_0.005000']
            estimate = reconstruct(left, info['tx_relative_m'], info['rx_relative_m'], delta_t=.005 / 30)
            midpoint.append({'configuration': label, 'extrapolation_5mm_nmse_db': defined_nmse(estimate, mid_H)})
        # Explicit local refinement: actual RT truth is stored at every 1 mm;
        # never replace the difficult truth by an interpolated channel.
        for offset in np.linspace(0., .01, 11):
            rx.position = (info['rx_xyz_m'] + offset * direction).tolist()
            settings = _settings(samples=300000)
            rec, compact, H = _snapshot(scene, solver, object_map, settings)
            key = f'refine_1mm_{offset:.6f}'
            rec.update(key=key, configuration='refine_1mm_300k', offset_m=float(offset), settings=settings)
            records.append(rec); cfr[key] = H; compacts[key] = compact
        # Measure every one-mm frozen-left interval against its independently
        # retraced right endpoint. This is diagnostic, not an admission by fiat.
        for i in range(10):
            lk = f'refine_1mm_{i*.001:.6f}'; rk = f'refine_1mm_{(i+1)*.001:.6f}'
            estimate = reconstruct(compacts[lk], info['tx_relative_m'], info['rx_relative_m'], delta_t=.001 / 30)
            midpoint.append({'configuration': 'refine_1mm_300k', 'left_offset_m': i*.001,
                             'extrapolation_1mm_nmse_db': defined_nmse(estimate, cfr[rk])})
        # Keep the edge-crossing zone at 100 micrometre spatial increments,
        # independently retraced.  This quantifies numerical edge sensitivity;
        # a tiny interval with discontinuous support remains explicitly unsafe.
        for offset in np.linspace(.0035, .006, 26):
            rx.position = (info['rx_xyz_m'] + offset * direction).tolist()
            settings = _settings(samples=300000)
            rec, compact, H = _snapshot(scene, solver, object_map, settings)
            key = f'refine_100um_{offset:.6f}'
            rec.update(key=key, configuration='refine_100um_300k', offset_m=float(offset), settings=settings)
            records.append(rec); cfr[key] = H; compacts[key] = compact
    np.savez_compressed(out / 'native_cfr.npz', **cfr)
    result = {'status': 'diagnostic_completed', 'city': city,
              'source_archive': str(Path(source_root) / ('city_' + city) / 'cities' / 'compact128.h5'),
              'source_fragment': 1 if city == 'munich' else 3,
              'source_archive_sha256': sha256(Path(source_root) / ('city_' + city) / 'cities' / 'compact128.h5'),
              'source_geometry': info, 'object_id_map': objects, 'records': records,
              'midpoint_diagnostics': midpoint,
              'scope': 'One original failed geometry per city, full native retained paths; no filtering or scientific admission implied.',
              'warning': 'Finite ray/depth checks cannot prove global search convergence. Equal object IDs do not establish temporal physical-path identity.'}
    if city == 'etoile':
        result['disappearing_branch_triangle_shadow_check'] = etoile_visibility_probe(scene, info, direction, records[0])
    save_json(out / 'report.json', result)
    return result


def run_mirror_admission(out):
    import sionna.rt as rt
    from .physics import _trace
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    records = []
    for mirror_speed in (0., 10.):
        for t in (-.0001, 0., .0001):
            control = {'kind': 'reflection', 'velocity': [0., 0., 0.], 'mirror_speed': mirror_speed}
            paths, geometry, idx = _trace(rt, _solver(rt), control, time=t, require_single=False)
            meta = metadata(paths)
            valid = np.asarray(paths.valid).reshape(-1)
            fd = np.asarray(paths.doppler).reshape(-1)[valid]
            records.append({'mirror_speed_mps': mirror_speed, 'time_sec': t,
                            'native_path_count': int(np.sum(valid)), 'geometry': geometry,
                            'doppler_hz': fd, 'doppler_analytic_hz': geometry['fd_hz'],
                            'doppler_max_error_hz': float(np.max(np.abs(fd-geometry['fd_hz']))),
                            'raw_geometry': meta})
    result = {'status': 'diagnostic_completed', 'records': records,
              'duplicate_policy': 'Retained every native path; duplicates quarantined rather than silently removed.',
              'scope': 'Independent moving-reflector control; city meshes unchanged.'}
    save_json(out / 'report.json', result)
    return result


def run_dynamic_admission(out):
    """A mesh vehicle traverses an actual LoS; stopped UE remains stopped."""
    import mitsuba as mi
    import sionna.rt as rt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    scene = rt.load_scene(); scene.frequency = FC
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern='iso', polarization='V')
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern='iso', polarization='V')
    tx = rt.Transmitter('tx', position=[-10., 0., 1.0])
    rx = rt.Receiver('rx', position=[10., 0., 1.0]); rx.velocity = [0., 0., 0.]
    scene.add([tx, rx]); solver = _solver(rt)
    material = rt.ITURadioMaterial('moving_metal', itu_type='metal', thickness=.01)
    car = rt.SceneObject(fname=portable_asset(rt.scene.low_poly_car), name='crossing_car', radio_material=material)
    # The bundled mesh dimensions are preserved; only a documented 1.5x height
    # factor matches the old van setup. This remains a controlled RT object.
    car.scaling = mi.Vector3f(1., 1., 1.5)
    bb = car.mi_mesh.bbox()
    height = float(bb.max.z - bb.min.z)
    car.velocity = [0., 8., 0.]
    scene.add(car)
    object_map, objects = stable_objects(scene, 'controlled_car_crossing')
    car_id = next(x['stable_id'] for x in objects if x['name'] == 'crossing_car')
    records = []; cfr = {}; settings = _settings(samples=100000, depth=2)
    for t in np.linspace(-1., 1., 9):
        car.position = [0., 8.*float(t), height/2]
        rec, compact, H = _snapshot(scene, solver, object_map, settings)
        car_path = np.any(compact['object_ids'] == car_id, axis=1)
        fd_projection = FC / 299792458.0 * (-(compact['k_tx'][car_path]+compact['k_rx'][car_path]) @ np.array([0., 8., 0.]))
        rec.update(time_sec=float(t), car_xyz_m=xyz_rows(car.position)[0], car_velocity_mps=[0., 8., 0.],
                   stopped_ue=True, car_interacting_path_count=int(car_path.sum()),
                   car_interacting_doppler_hz=compact['doppler_hz'][car_path],
                   car_doppler_projection_hz=fd_projection,
                   car_doppler_projection_max_error_hz=float(np.max(np.abs(fd_projection-compact['doppler_hz'][car_path]))) if np.any(car_path) else None,
                   car_doppler_projection_relative_error=float(np.linalg.norm(fd_projection-compact['doppler_hz'][car_path])/np.linalg.norm(fd_projection)) if np.any(car_path) else None)
        records.append(rec); cfr[f't_{t:+.3f}'] = H
        print(f'[admission] crossing t={t:+.3f} paths={len(compact)} car={int(car_path.sum())} los={rec["has_los"]}', flush=True)
    scene.remove(car)
    baseline_map, _ = stable_objects(scene, 'controlled_car_crossing')
    baseline, _, baseline_H = _snapshot(scene, solver, baseline_map, settings)
    np.savez_compressed(out / 'native_cfr.npz', baseline=baseline_H, **cfr)
    result = {'status': 'diagnostic_completed', 'scenario': 'isolated actual vehicle mesh crossing fixed TX/RX LoS',
              'duration_sec': 2., 'ue_velocity_mps': [0., 0., 0.],
              'vehicle_bbox_height_m': height, 'vehicle_scaling': [1., 1., 1.5],
              'settings': settings, 'object_id_map': objects, 'baseline_no_vehicle': baseline,
              'records': records, 'crossing_los_changed': len(set(x['has_los'] for x in records)) > 1,
              'vehicle_reflection_covered': any(x['car_interacting_path_count'] > 0 for x in records),
              'scope': 'Controlled geometry, not a real-city traffic dataset or calibrated car electromagnetic model.'}
    save_json(out / 'report.json', result)
    return result

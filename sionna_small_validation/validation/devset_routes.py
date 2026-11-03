"""Geometry-only candidates for the grouped 96-family development release.

No candidate is RF-certified here.  In particular a blocked direct ray does
not establish a nonempty NLoS channel, and a proposed traffic trajectory does
not establish a radio event.  The generation worker must retain this distinction.
"""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import save_json

MECHANISMS = ('stable_los', 'reflection_nlos', 'visibility_boundary', 'dynamic_event')
TARGET_COUNTS = {'munich': 40, 'florence': 40, 'etoile': 16}
SEED = 20261027


def _hash_int(text):
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'little')


def _ray_blocked(scene, start, end):
    import mitsuba as mi
    start, end = np.broadcast_arrays(np.asarray(start, float), np.asarray(end, float))
    start, end = start.reshape(-1, 3), end.reshape(-1, 3)
    delta = end-start
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 1e-3):
        raise ValueError('Coincident visibility-test endpoints')
    rays = mi.Ray3f(mi.Point3f(*(mi.Float(start[:, k].astype(np.float32)) for k in range(3))),
                   mi.Vector3f(*(mi.Float((delta[:, k]/distance).astype(np.float32)) for k in range(3))))
    rays.maxt = mi.Float((distance-1e-3).astype(np.float32))
    return np.asarray(scene.mi_scene.ray_test(rays), bool).reshape(-1)


def _ground_grid(scene, spacing=1.0):
    import mitsuba as mi
    from scipy.ndimage import binary_erosion, label
    bb = scene.mi_scene.bbox()
    lo = np.array([float(bb.min.x), float(bb.min.y), float(bb.min.z)])
    hi = np.array([float(bb.max.x), float(bb.max.y), float(bb.max.z)])
    xs = np.arange(math.ceil(lo[0]+12), math.floor(hi[0]-12)+.01, spacing)
    ys = np.arange(math.ceil(lo[1]+12), math.floor(hi[1]-12)+.01, spacing)
    xx, yy = np.meshgrid(xs, ys)
    origins = mi.Point3f(mi.Float(xx.ravel().astype(np.float32)),
                        mi.Float(yy.ravel().astype(np.float32)), float(hi[2]+5))
    hit = scene.mi_scene.ray_intersect(mi.Ray3f(origins, mi.Vector3f(0, 0, -1)))
    valid = np.asarray(hit.is_valid(), bool).reshape(xx.shape)
    z = np.asarray(hit.p.z, float).reshape(xx.shape)
    normal_z = np.asarray(hit.n.z, float).reshape(xx.shape)
    ground = valid & (np.abs(z) < .02) & (np.abs(normal_z) > .9)
    labels, _ = label(ground)
    boundary_ids = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    boundary_ids = boundary_ids[boundary_ids != 0]
    connected = np.isin(labels, boundary_ids)
    safe = binary_erosion(connected, np.ones((3, 3), bool), border_value=0)
    return dict(x=xs, y=ys, first_hit_z=z, ground=ground, connected=connected,
                safe=safe, bbox_min=lo, bbox_max=hi, spacing=float(spacing))


def _safe_xy(grid, xy):
    xy = np.asarray(xy, float)
    shape = xy.shape[:-1]
    ix = np.rint((xy[..., 0]-grid['x'][0])/grid['spacing']).astype(int)
    iy = np.rint((xy[..., 1]-grid['y'][0])/grid['spacing']).astype(int)
    inside = (ix >= 0) & (ix < len(grid['x'])) & (iy >= 0) & (iy < len(grid['y']))
    return (inside & grid['safe'][np.clip(iy, 0, len(grid['y'])-1),
                                  np.clip(ix, 0, len(grid['x'])-1)]).reshape(shape)


def _region(city, xy, block_m=200.0):
    # Region assignment is fixed from geometry coordinates before any RF trace.
    ij = np.floor(np.asarray(xy)/block_m).astype(int)
    rid = f'{city}_region_{ij[0]:+03d}_{ij[1]:+03d}'
    split = 'test' if city == 'etoile' else ('validation' if _hash_int(rid) % 5 == 0 else 'train')
    return rid, split


def _split_guard(city, positions, margin=12.0):
    # A route and a 12m neighbourhood must not cross a split boundary. Regions
    # on the same side of the split can meet without pretending independence.
    positions = np.asarray(positions, float)
    rid, split = _region(city, positions[len(positions)//2, :2])
    for p in positions[::5]:
        for off in ((0, 0), (margin, 0), (-margin, 0), (0, margin), (0, -margin)):
            if _region(city, p[:2]+off)[1] != split:
                return None
    return rid, split


def _tx_options(scene, grid, center, route_positions, rng):
    jj, ii = np.where(grid['safe'])
    coarse = ((ii % 4) == 0) & ((jj % 4) == 0)
    xy = np.column_stack((grid['x'][ii[coarse]], grid['y'][jj[coarse]]))
    distance = np.linalg.norm(xy-center[:2], axis=1)
    keep = (distance >= 25) & (distance <= 115)
    xy, distance = xy[keep], distance[keep]
    if not len(xy):
        return []
    # Angle-stratified deterministic subsampling, unrelated to received power.
    angles = np.arctan2(xy[:, 1]-center[1], xy[:, 0]-center[0])
    selected = []
    for k in range(24):
        q = np.flatnonzero((angles >= -np.pi+k*np.pi/12) &
                           (angles < -np.pi+(k+1)*np.pi/12))
        if len(q):
            order = q[np.argsort(distance[q])]
            for fraction in (.15, .5, .85):
                selected.append(order[min(int(len(order)*fraction), len(order)-1)])
    selected = np.unique(selected)
    tx = np.column_stack((xy[selected], np.full(len(selected), 8.0)))
    positions = route_positions[::2]
    blocked = _ray_blocked(scene, tx[:, None, :], positions[None, :, :]).reshape(len(tx), len(positions))
    options = []
    for point, visibility in zip(tx, ~blocked):
        fraction = float(np.mean(visibility))
        changes = int(np.count_nonzero(visibility[1:] != visibility[:-1]))
        options.append(dict(tx_xyz_m=point.tolist(), geometric_los=visibility.tolist(),
                            los_fraction=fraction, los_switch_count=changes,
                            distance_m=float(np.linalg.norm(point-center))))
    return options


def _dynamic_proposal(grid, center, tx, direction, duration):
    # A single uncalibrated vehicle crosses the direct ray near the UE. The
    # production worker must evaluate its actual mesh/visibility contribution.
    toward = np.asarray(tx)-center
    toward[2] = 0
    toward /= np.linalg.norm(toward)
    crossing = np.array([-toward[1], toward[0], 0.])
    contact = center+5*toward
    contact[2] = 0
    half_travel = min(7.5, max(3.0, 5.0*duration))
    points = contact + np.linspace(-half_travel, half_travel, 31)[:, None]*crossing
    if not np.all(_safe_xy(grid, points[:, :2])):
        return None
    start, end = points[0], points[-1]
    return dict(object_asset='low_poly_car', material='metal',
                start_ground_xyz_m=start.tolist(), end_ground_xyz_m=end.tolist(),
                velocity_mps=((end-start)/duration).tolist(),
                heading_rad=float(np.arctan2(crossing[1], crossing[0])),
                duration_sec=float(duration), crossing_time_sec=float(duration/2),
                status='geometric_proposal; actual moving-mesh radio event must be verified',
                ground_clearance_scope='1m raster with 1m erosion; mesh footprint still requires checking')


@lru_cache(maxsize=256)
def _curve_table(control_key):
    control = np.asarray(control_key, float).reshape(3, 3)
    u = np.linspace(0., 1., 4097)
    points = ((1-u[:, None])**2*control[0]+2*u[:, None]*(1-u[:, None])*control[1]
              +u[:, None]**2*control[2])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return u, arc


def _add_curved_candidates(scene, grid, candidates, city):
    count = 0
    quota = 8 if city != 'etoile' else 4
    for route in candidates:
        if count >= quota:
            break
        if route['intended_mechanism'] == 'dynamic_event' or route['ue_motion_mode'] != 'constant_velocity':
            continue
        endpoints = np.asarray(route['waypoints_xyz_m'], float)
        center = np.asarray(route['center_xyz_m'], float)
        direction = endpoints[1]-endpoints[0]
        direction /= np.linalg.norm(direction)
        lateral = np.array([-direction[1], direction[0], 0.])
        # A shallow, continuously differentiable 20m bend; classification and
        # geometric clearance are checked again on its actual curved samples.
        for side in (1., -1.):
            control = np.stack((endpoints[0], center+2*side*lateral, endpoints[1]))
            u = np.linspace(0., 1., 81)
            points = ((1-u[:, None])**2*control[0]+2*u[:, None]*(1-u[:, None])*control[1]
                      +u[:, None]**2*control[2])
            if not _safe_xy(grid, points[:, :2]).all() or _split_guard(city, points) is None:
                continue
            if _ray_blocked(scene, points[:-1], points[1:]).any():
                continue
            visibility = ~_ray_blocked(scene, np.asarray(route['tx_xyz_m']), points[::4])
            fraction = float(np.mean(visibility))
            switches = int(np.count_nonzero(visibility[1:] != visibility[:-1]))
            mechanism = route['intended_mechanism']
            matches = (fraction == 1 if mechanism == 'stable_los' else fraction == 0
                       if mechanism == 'reflection_nlos' else .15 <= fraction <= .85 and switches > 0)
            if not matches:
                continue
            curve_u, arc = _curve_table(tuple(control.ravel()))
            route['curve_control_points_xyz_m'] = control.tolist()
            route['trajectory_type'] = 'quadratic_bezier_constant_arclength_speed'
            route['length_m'] = route['actual_ue_travel_m'] = float(arc[-1])
            route['duration_sec'] = float(arc[-1]/route['speed_mps'])
            route['yaw_changes_with_velocity'] = True
            route['geometry_los_profile'] = visibility.tolist()
            route['geometry_profile_arclength_m'] = np.interp(u[::4], curve_u, arc).tolist()
            route['geometry_los_fraction'] = fraction
            route['geometry_los_switch_count'] = switches
            count += 1
            break
    return count


def discover_city_routes(scene, city, outdir, *, minimum_center_distance_m=40.0,
                         candidate_count=None, seed=SEED):
    """Discover 60/60/24 geometry candidates; RF acceptance is a later stage.

    Returns a JSON-serializable manifest. Callers can increase candidate_count
    without changing family/group definitions. Selected centres never overlap
    within ``minimum_center_distance_m``; the manifest reports that this alone
    does not prove statistical independence or different street semantics.
    """
    if city not in TARGET_COUNTS:
        raise ValueError(city)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    candidate_count = candidate_count or (24 if city == 'etoile' else 60)
    if candidate_count % 4:
        raise ValueError('candidate_count must be divisible by four')
    grid = _ground_grid(scene)
    np.savez_compressed(outdir/f'{city}_expanded_ground.npz', x_m=grid['x'], y_m=grid['y'],
                        first_hit_z_m=grid['first_hit_z'], ground_mask=grid['ground'],
                        boundary_connected_ground_mask=grid['connected'],
                        route_clearance_mask=grid['safe'])
    rng = np.random.default_rng(seed+_hash_int(city) % 100000)
    jj, ii = np.where(grid['safe'])
    coarse = ((ii % 6) == 0) & ((jj % 6) == 0)
    centers = np.column_stack((grid['x'][ii[coarse]], grid['y'][jj[coarse]],
                               np.full(np.count_nonzero(coarse), 1.5)))
    rng.shuffle(centers)
    required = {}
    for mechanism in MECHANISMS:
        per_class = candidate_count//4
        if city == 'etoile':
            required[('test', mechanism)] = per_class
        else:
            val = per_class//5
            required[('train', mechanism)] = per_class-val
            required[('validation', mechanism)] = val
    counts = {key: 0 for key in required}
    candidates = []
    examined = 0
    headings = np.arange(12)*np.pi/12
    for center in centers:
        if all(counts[k] >= n for k, n in required.items()):
            break
        if any(np.linalg.norm(center[:2]-np.asarray(r['center_xyz_m'])[:2]) < minimum_center_distance_m
               for r in candidates):
            continue
        rid, split = _region(city, center[:2])
        if all(counts[(split, m)] >= required[(split, m)] for m in MECHANISMS):
            continue
        possible = []
        heading_order = rng.permutation(headings)
        for heading in heading_order:
            direction = np.array([np.cos(heading), np.sin(heading), 0.])
            points = center+np.linspace(-10, 10, 41)[:, None]*direction
            if not np.all(_safe_xy(grid, points[:, :2])) or _split_guard(city, points) is None:
                continue
            # Fine direct route-clearance test complements top-down sampling.
            if _ray_blocked(scene, points[0], points[-1])[0]:
                continue
            possible.append((heading, direction, points))
            if len(possible) == 2:
                break
        if not possible:
            continue
        examined += 1
        # Assign difficult geometric classes first, still without RF ranking.
        outstanding = sorted(MECHANISMS, key=lambda m: (0 if m == 'visibility_boundary' else
                             1 if m == 'reflection_nlos' else 2 if m == 'dynamic_event' else 3,
                             counts[(split, m)]/max(required[(split, m)], 1)))
        made = False
        for heading, direction, points in possible:
            options = _tx_options(scene, grid, center, points, rng)
            for mechanism in outstanding:
                if counts[(split, mechanism)] >= required[(split, mechanism)]:
                    continue
                eligible = [o for o in options if
                            (o['los_fraction'] == 1 if mechanism in ('stable_los', 'dynamic_event') else
                             o['los_fraction'] == 0 if mechanism == 'reflection_nlos' else
                             .15 <= o['los_fraction'] <= .85 and o['los_switch_count'] >= 1)]
                eligible.sort(key=lambda o: (o['distance_m'], o['tx_xyz_m']))
                for option in eligible:
                    local_index = len(candidates)
                    speed = 15.0 if local_index % 2 == 0 else 30.0
                    duration = 20.0/speed
                    dynamic = _dynamic_proposal(grid, center, option['tx_xyz_m'], direction, duration) if mechanism == 'dynamic_event' else None
                    if mechanism == 'dynamic_event' and dynamic is None:
                        continue
                    family = f'{city}_family_{local_index:03d}'
                    stationary = mechanism == 'dynamic_event' and counts[(split, mechanism)] % 4 == 0
                    candidates.append(dict(candidate_id=family, route_family_id=family, city=city,
                        region_id=rid, split=split, intended_mechanism=mechanism,
                        verification_status='geometry_only; RF_pending',
                        center_xyz_m=center.tolist(), waypoints_xyz_m=[points[0].tolist(), points[-1].tolist()],
                        length_m=20.0, nominal_length_m=20.0,
                        actual_ue_travel_m=0.0 if stationary else 20.0,
                        ue_motion_mode='stationary_event' if stationary else 'constant_velocity',
                        heading_rad=float(heading), speed_mps=0.0 if stationary else speed,
                        time_budget_reference_speed_mps=speed, duration_sec=duration,
                        tx_xyz_m=option['tx_xyz_m'], tx_height_m=8.0,
                        geometry_los_profile=option['geometric_los'],
                        geometry_profile_arclength_m=np.linspace(0, 20, len(option['geometric_los'])).tolist(),
                        geometry_los_fraction=option['los_fraction'],
                        geometry_los_switch_count=option['los_switch_count'],
                        geometry_visibility_scope='sampled direct rays, not native path-support labels',
                        dynamic_object=dynamic,
                        paired_environment='moving_vehicle' if mechanism == 'dynamic_event' else 'vertical_facades_wood',
                        survey_designs=4, survey_positions_per_design=128,
                        survey_protocol='independent historical base-scene positions and noise; target-region-map allowed',
                        motion_scope='geometric trajectory proposal, not calibrated natural traffic',
                        selection_order=local_index))
                    counts[(split, mechanism)] += 1
                    made = True
                    break
                if made:
                    break
            if made:
                break
    curved_count = _add_curved_candidates(scene, grid, candidates, city)
    result = dict(schema='kap_devset.routes.geometry.v1', city=city, seed=seed,
        target_accepted_families=TARGET_COUNTS[city], requested_candidates=candidate_count,
        proposed_candidates=len(candidates), candidates=candidates,
        counts={f'{s}/{m}': n for (s, m), n in counts.items()},
        deficits={f'{s}/{m}': required[(s, m)]-n for (s, m), n in counts.items() if n < required[(s, m)]},
        source_bbox_min_xyz_m=grid['bbox_min'].tolist(), source_bbox_max_xyz_m=grid['bbox_max'].tolist(),
        ground_raster_spacing_m=grid['spacing'], route_center_min_distance_m=minimum_center_distance_m,
        route_length_m=20.0, examined_route_centers=examined,
        curved_candidate_count=curved_count,
        region_rule='200m global-XY blocks; deterministic region hash mod5 validation; all Etoile test',
        split_guard_m=12.0,
        limitations=['Geometric accessibility is not semantic drivable-road certification.',
                     'Minimum centre separation and region grouping do not prove statistical independence.',
                     'RF acceptance, nonempty reflected channels and dynamic events require actual native RT.',
                     'No received power, target label or learned-model performance influenced these proposals.',
                     'LoS/NLoS and switch profiles are finite geometric checks; thin/sub-grid objects may remain.'])
    save_json(outdir/f'{city}_route_candidates.json', result)
    return result


def trajectory_at(route, time_sec):
    """Return UE position, velocity, yaw and proposed vehicle ground position."""
    t = float(time_sec)
    duration = float(route['duration_sec'])
    if t < -1e-9 or t > duration+1e-9:
        raise ValueError('Time outside route support')
    points = np.asarray(route['waypoints_xyz_m'], float)
    if route.get('ue_motion_mode') == 'stationary_event':
        position = np.asarray(route['center_xyz_m'], float)
        velocity = np.zeros(3)
    elif route.get('curve_control_points_xyz_m'):
        control = np.asarray(route['curve_control_points_xyz_m'], float)
        u_table, arc = _curve_table(tuple(control.ravel()))
        u = float(np.interp(np.clip(t/duration, 0, 1)*arc[-1], arc, u_table))
        position = (1-u)**2*control[0]+2*u*(1-u)*control[1]+u**2*control[2]
        tangent = 2*(1-u)*(control[1]-control[0])+2*u*(control[2]-control[1])
        velocity = tangent/np.linalg.norm(tangent)*(arc[-1]/duration)
    else:
        position = points[0]+(points[-1]-points[0])*np.clip(t/duration, 0, 1)
        velocity = (points[-1]-points[0])/duration
    dynamic = route.get('dynamic_object')
    vehicle = (np.asarray(dynamic['start_ground_xyz_m'])+t*np.asarray(dynamic['velocity_mps'])
               if dynamic else None)
    yaw = float(np.arctan2(velocity[1], velocity[0])) if np.linalg.norm(velocity) else route['heading_rad']
    return position, velocity, np.array([yaw, 0., 0.]), vehicle

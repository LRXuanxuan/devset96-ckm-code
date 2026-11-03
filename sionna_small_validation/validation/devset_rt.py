"""Native RT adapter shared by screening and streamed development generation.

No paths are removed or merged. A failed solve gets one bounded retry of the
same propagation model and retains the original arrays for quarantine.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from .cities import (FC, FREQUENCIES, extract_compact, native_cfr, reconstruct,
                     stable_objects, xyz_rows)
from .common import portable_asset
from .devset_routes import trajectory_at, SEED
from .material_policy import prepare_facade_scene, apply_facade_intervention
from .path_quality import assess_snapshot


def exact_metadata(paths):
    """Version-pinned raw metadata, with hash-assisted exact duplicate checks.

    Hashes only select equality candidates. Array equality confirms duplicates,
    including the old +0/-0 semantics; nonfinite rows never compare equal.
    """
    valid = np.asarray(paths.valid).reshape(-1).astype(bool)
    tau = np.asarray(paths.tau).reshape(-1)
    selected = np.flatnonzero(valid & (tau >= 0))
    if not len(selected):
        return dict(objects=np.empty((0, 3), np.uint32), primitives=np.empty((0, 3), np.uint32),
                    interaction_types=np.empty((0, 3), np.uint32), vertices_xyz_m=np.empty((0, 3, 3)),
                    native_path_index=selected, exact_duplicate_geometry_pairs=[])
    permutation = np.asarray(paths._path_ind).astype(np.int64)
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation))
    buf = paths._paths_buffer
    def take(field):
        return np.array(field.numpy(), copy=True)[inverse][selected]
    objects, primitives, types = take(buf.shapes), take(buf.primitives), take(buf.interaction_types)
    vertices = np.stack([take(buf.vertices_x), take(buf.vertices_y), take(buf.vertices_z)], axis=-1)
    duplicates, seen = [], {}
    for i in range(len(selected)):
        row_v = vertices[i].copy()
        row_tau = np.asarray([tau[selected[i]]]).copy()
        if not np.isfinite(row_v).all() or not np.isfinite(row_tau).all():
            continue
        row_v[row_v == 0] = 0
        row_tau[row_tau == 0] = 0
        digest = hashlib.sha256()
        for arr in (objects[i], primitives[i], types[i], row_v, row_tau):
            digest.update(np.ascontiguousarray(arr).tobytes())
        key = digest.digest()
        prior = seen.setdefault(key, [])
        for j in prior:
            if (np.array_equal(objects[i], objects[j]) and np.array_equal(primitives[i], primitives[j])
                    and np.array_equal(types[i], types[j]) and np.array_equal(vertices[i], vertices[j])
                    and tau[selected[i]] == tau[selected[j]]):
                duplicates.append([j, i])
        prior.append(i)
    return dict(objects=objects, primitives=primitives, interaction_types=types,
                vertices_xyz_m=vertices, native_path_index=selected,
                exact_duplicate_geometry_pairs=duplicates)


class DevsetRT:
    """One isolated city scene. Call setup_backend before construction."""
    def __init__(self, city, samples=30000):
        import mitsuba as mi
        import sionna.rt as rt
        self.rt, self.mi, self.city, self.samples = rt, mi, city, int(samples)
        if samples < 1:
            raise ValueError('Positive search effort required')
        self.asset = portable_asset(getattr(rt.scene, city))
        prepared, self.material_manifest = prepare_facade_scene(self.asset)
        self.scene = rt.load_scene(prepared)
        self.scene.frequency = FC
        self.scene.tx_array = rt.PlanarArray(num_rows=8, num_cols=8, vertical_spacing=.5,
                                            horizontal_spacing=.5, pattern='iso', polarization='V')
        self.scene.rx_array = rt.PlanarArray(num_rows=4, num_cols=4, vertical_spacing=.5,
                                            horizontal_spacing=.5, pattern='iso', polarization='V')
        self.tx = rt.Transmitter('devset_tx', position=[0, 0, 8])
        self.rx = rt.Receiver('devset_rx', position=[1, 0, 1.5])
        self.scene.add([self.tx, self.rx])
        self._base_materials = {name: obj.radio_material for name, obj in self.scene.objects.items()}
        self.material_changes, self.material_guard = apply_facade_intervention(self.scene, rt, self.material_manifest)
        self._wood_materials = {row['object']: self.scene.objects[row['object']].radio_material
                                for row in self.material_changes}
        for name, material in self._base_materials.items():
            self.scene.objects[name].radio_material = material
        self.settings = dict(max_depth=3, samples_per_src=self.samples, max_num_paths_per_src=4096,
                             synthetic_array=True, los=True, specular_reflection=True,
                             diffuse_reflection=False, refraction=False, diffraction=False, seed=SEED)
        self.solver = rt.PathSolver(deterministic=True)
        self.solver.loop_mode = 'evaluated'
        self.route = None
        self.environment = 0
        self.car = None
        self.car_added = False
        self.car_stable_id = None
        self.car_height = 0.
        self.car_motion = None
        self.dynamic_geometry_guard = None
        self.object_map, self.object_records = stable_objects(self.scene, city)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._remove_car()

    def _remove_car(self):
        if self.car_added:
            self.scene.remove(self.car.name)
            self.car_added = False

    def configure(self, route):
        self._remove_car()
        self.route = dict(route)
        self.tx.position = route['tx_xyz_m']
        self.tx.velocity = [0, 0, 0]
        self.tx.look_at(self.mi.Point3f(route['center_xyz_m']))
        self.tx_pose = xyz_rows(self.tx.orientation)[0]
        self.car_motion = None
        self.dynamic_geometry_guard = None
        if route.get('dynamic_object'):
            proposed = route['dynamic_object']
            direction = np.asarray(proposed['velocity_mps'], float)
            direction /= np.linalg.norm(direction)
            center = (np.asarray(proposed['start_ground_xyz_m'])+np.asarray(proposed['end_ground_xyz_m']))/2
            if 'dynamic_placement_distance_m' in route:
                ue_center = np.asarray(route['center_xyz_m'], float)
                toward_tx = np.asarray(route['tx_xyz_m'], float)-ue_center
                toward_tx[2] = 0.
                toward_tx /= np.linalg.norm(toward_tx)
                center = ue_center+float(route['dynamic_placement_distance_m'])*toward_tx
                center[2] = 0.
            duration = route['duration_sec']
            # A declared 8m/s event follows a subset of the geometrically checked
            # crossing segment; its midpoint and total event duration are fixed.
            self.car_motion = dict(start_ground_xyz_m=(center-direction*8*duration/2).tolist(),
                                   velocity_mps=(direction*8).tolist(),
                                   heading_rad=float(np.arctan2(direction[1], direction[0])),
                                   duration_sec=float(duration), speed_mps=8.0)
        self.set_environment(0)
        records = {r['stable_id']: r for r in self.object_records}
        if route.get('dynamic_object'):
            self.set_environment(1)
            records.update({r['stable_id']: r for r in self.object_records})
            self.set_environment(0)
        stable_records = [{k: v for k, v in records[key].items() if k != 'runtime_id_at_capture'}
                           for key in sorted(records)]
        zero_pose = self.mi.Point3f(0., 0., 0.)
        return dict(tx_array_local_m=xyz_rows(self.scene.tx_array.rotate(self.scene.wavelength, zero_pose)),
                    rx_array_local_m=xyz_rows(self.scene.rx_array.rotate(self.scene.wavelength, zero_pose)),
                    tx_pose_rad=self.tx_pose.copy(), carrier_hz=FC, frequencies_hz=FREQUENCIES.copy(),
                    stable_object_ids=stable_records, solver_settings=self.settings.copy(),
                    material_guard=self.material_guard, material_policy=self.material_manifest['policy_version'],
                    dynamic_motion=self.car_motion, dynamic_geometry_guard=self.dynamic_geometry_guard,
                    dynamic_vehicle_scope='synthetic metallic car on real geometry; not measured traffic')

    def set_environment(self, pass_index):
        if self.route is None:
            raise RuntimeError('Configure a route before setting environment')
        if pass_index not in (0, 1):
            raise ValueError('Only paired passes 0/1 are declared')
        self._remove_car()
        for name, material in self._base_materials.items():
            self.scene.objects[name].radio_material = material
        if pass_index == 1 and not self.route.get('dynamic_object'):
            for name, material in self._wood_materials.items():
                self.scene.objects[name].radio_material = material
        if pass_index == 1 and self.route.get('dynamic_object'):
            if self.car is None:
                material = self.rt.ITURadioMaterial(name='devset_vehicle_metal', itu_type='metal', thickness=.01)
                self.car = self.rt.SceneObject(fname=portable_asset(self.rt.scene.low_poly_car),
                                               name='devset_vehicle', radio_material=material)
                self.car.scaling = self.mi.Vector3f(1., 1., 1.5)
                bounds = self.car.mi_mesh.bbox()
                self.car_height = float(bounds.max.z-bounds.min.z)
                self.car_floor_z_at_zero = float(bounds.min.z)
                self.car_local_min = np.array([float(bounds.min.x), float(bounds.min.y), float(bounds.min.z)])
                self.car_local_max = np.array([float(bounds.max.x), float(bounds.max.y), float(bounds.max.z)])
            if self.dynamic_geometry_guard is None:
                self.dynamic_geometry_guard = self._check_dynamic_geometry()
                if not self.dynamic_geometry_guard['passed']:
                    raise ValueError('Dynamic mesh trajectory failed geometry guard: '+str(self.dynamic_geometry_guard))
            self.car.position = (np.asarray(self.car_motion['start_ground_xyz_m'])
                                 +[0, 0, -self.car_floor_z_at_zero]).tolist()
            self.car.orientation = [self.car_motion['heading_rad'], 0, 0]
            self.car.velocity = self.car_motion['velocity_mps']
            self.scene.add(self.car)
            self.car_added = True
        self.environment = int(pass_index)
        self.object_map, self.object_records = stable_objects(self.scene, self.city)
        self.car_stable_id = next((row['stable_id'] for row in self.object_records
                                   if row['name'] == 'devset_vehicle'), None)

    def _check_dynamic_geometry(self):
        times = np.linspace(0., self.route['duration_sec'], 41)
        centers = (np.asarray(self.car_motion['start_ground_xyz_m'])[None, :]
                   +times[:, None]*np.asarray(self.car_motion['velocity_mps'])[None, :])
        yaw = self.car_motion['heading_rad']
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = np.array([[x, y] for x in (self.car_local_min[0], 0., self.car_local_max[0])
                            for y in (self.car_local_min[1], 0., self.car_local_max[1])])
        footprint = centers[:, None, :2]+(corners@rot.T)[None, :, :]
        xy = footprint.reshape(-1, 2)
        bb = self.scene.mi_scene.bbox()
        origins = self.mi.Point3f(self.mi.Float(xy[:, 0].astype(np.float32)),
                                 self.mi.Float(xy[:, 1].astype(np.float32)), float(bb.max.z+5.))
        hit = self.scene.mi_scene.ray_intersect(self.mi.Ray3f(origins, self.mi.Vector3f(0, 0, -1)))
        valid = np.asarray(hit.is_valid(), bool).reshape(-1)
        z = np.asarray(hit.p.z, float).reshape(-1)
        ground = valid & (abs(z) < .02)
        ue = np.array([trajectory_at(self.route, t)[0] for t in times])
        ue_in_local = (ue[:, :2]-centers[:, :2])@rot
        collision = ((ue_in_local >= self.car_local_min[:2]-.1).all(axis=1)
                     & (ue_in_local <= self.car_local_max[:2]+.1).all(axis=1)
                     & (ue[:, 2] <= self.car_height+.1))
        return dict(passed=bool(ground.all() and not collision.any()), samples=41,
                    footprint_ground_checks=int(len(xy)), invalid_ground_samples=int((~ground).sum()),
                    ue_bbox_collision_samples=int(collision.sum()), vehicle_height_m=self.car_height,
                    actual_speed_mps=8.0,
                    scope='sampled static-ground footprint and UE conservative bbox clearance; not natural traffic certification')

    def trace(self, time_sec, position=None, velocity=None, pose=None):
        t = float(time_sec)
        p, v, orientation, _ = trajectory_at(self.route, t) if position is None else (
            np.asarray(position, float), np.zeros(3) if velocity is None else np.asarray(velocity, float),
            np.array([self.route['heading_rad'], 0., 0.]) if pose is None else np.asarray(pose, float), None)
        if velocity is not None:
            v = np.asarray(velocity, float)
        if pose is not None:
            orientation = np.asarray(pose, float)
        self.rx.position, self.rx.velocity, self.rx.orientation = p.tolist(), v.tolist(), orientation.tolist()
        car_xyz, car_velocity, car_pose = np.zeros(3), np.zeros(3), np.zeros(3)
        if self.car_added:
            ground = np.asarray(self.car_motion['start_ground_xyz_m'])+t*np.asarray(self.car_motion['velocity_mps'])
            car_xyz = ground+np.array([0., 0., -self.car_floor_z_at_zero])
            car_velocity = np.asarray(self.car_motion['velocity_mps'], float)
            car_pose = np.array([self.car_motion['heading_rad'], 0., 0.])
            self.car.position, self.car.velocity = car_xyz.tolist(), car_velocity.tolist()
        txrel = xyz_rows(self.scene.tx_array.rotate(self.scene.wavelength, self.tx.orientation))
        rxrel = xyz_rows(self.scene.rx_array.rotate(self.scene.wavelength, self.rx.orientation))
        attempts = []
        for number in range(2):
            settings = self.settings.copy()
            if number:
                settings.update(samples_per_src=max(self.samples, 100000), seed=SEED+1)
            tick = time.perf_counter()
            paths = self.solver(self.scene, **settings)
            paths._validation_scene_owner = self.scene
            compact = extract_compact(paths, self.object_map)
            native = native_cfr(paths)
            rebuilt = reconstruct(compact, txrel, rxrel)
            denom = float(np.linalg.norm(native))
            error = float(np.linalg.norm(rebuilt-native)/denom) if np.isfinite(denom) and denom > 0 else None
            if error is not None and not np.isfinite(error):
                error = None
            raw = exact_metadata(paths)
            quality = assess_snapshot(compact, native, raw, error, settings['max_num_paths_per_src'])
            elapsed = time.perf_counter()-tick
            attempt = dict(attempt=number, settings=settings, solve_seconds=elapsed,
                           path_count=len(compact), quality=quality, native_relative_error=error,
                           native_power=float(np.mean(abs(native)**2)) if np.isfinite(native).all() else None,
                           exact_duplicate_geometry_pairs=raw['exact_duplicate_geometry_pairs'])
            if not quality['admitted']:
                attempt.update(compact=compact.copy(), native=native.copy(), raw=raw)
            attempts.append(attempt)
            if quality['admitted']:
                break
        return dict(compact=compact, native=native, quality=quality, native_relative_error=error,
                    raw=raw, attempts=attempts, solve_seconds=sum(x['solve_seconds'] for x in attempts),
                    tx_relative_m=txrel, rx_relative_m=rxrel,
                    state_fields=dict(time_sec=t, rx_xyz_m=p, rx_velocity_mps=v, rx_pose_rad=orientation,
                        tx_xyz_m=xyz_rows(self.tx.position)[0], tx_pose_rad=self.tx_pose.copy(),
                        car_xyz_m=car_xyz, car_velocity_mps=car_velocity, car_pose_rad=car_pose,
                        has_dynamic_object=self.car_added, quality_flags=quality['flags'],
                        native_relative_error=np.nan if error is None else error,
                        native_checked=bool(error is not None and np.isfinite(error)),
                        has_los=bool(np.any(compact['depth'] == 0)), pass_index=self.environment,
                        receiver_anchor=False, receiver_validated_until_sec=t))

"""V2 runtime: canonical object IDs, fixed survey pose and declared search effort.

The frozen pilot adapter remains untouched. The propagation model is unchanged;
100k-ray finite search is the default and bad snapshots get one 300k retry.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .cities import FC,extract_compact,native_cfr,reconstruct,xyz_rows
from .common import portable_asset,sha256
from .devset_object_ids import CanonicalObjectCatalog,VERSION as OBJECT_ID_VERSION
from .devset_routes import trajectory_at
from .devset_rt import DevsetRT,exact_metadata
from .path_quality import assess_snapshot

VERSION='kap_devset.runtime.v2'


class DevsetRTV2(DevsetRT):
    def __init__(self,city,samples=100000,retry_samples=300000):
        if int(samples)<100000 or int(retry_samples)<int(samples):
            raise ValueError('V2 finite-search policy requires primary>=100k and retry>=primary')
        super().__init__(city,samples=int(samples))
        self.retry_samples=int(retry_samples)
        self.settings['seed']=20261028
        self.object_catalog=CanonicalObjectCatalog(self.scene,city)
        self.object_map=self.object_catalog.runtime_mapping(self.scene)
        self.object_records=self.object_catalog.records_for(self.scene)
        self.survey_reference_pose_rad=None

    def set_environment(self,pass_index):
        super().set_environment(pass_index)
        if not hasattr(self,'object_catalog'):return
        if self.car is not None:
            self.object_catalog.register_rigid_object(self.car,role='traffic_vehicle',
                source_mesh_path=portable_asset(self.rt.scene.low_poly_car),instance_key='vehicle_0')
        self.object_map=self.object_catalog.runtime_mapping(self.scene)
        self.object_records=self.object_catalog.records_for(self.scene)
        self.car_stable_id=self.object_catalog.id_for(self.car) if self.car_added else None

    def _canonical_material_guard(self):
        baseline={str(self.object_catalog.id_for(self.scene.objects[name])):material.name
                  for name,material in self._base_materials.items()}
        intervened=dict(baseline);changes=[]
        for change in self.material_changes:
            ident=self.object_catalog.id_for(self.scene.objects[change['object']])
            item={k:v for k,v in change.items() if k!='object'}
            item.update(object_id=ident,object='static_mesh_'+next(
                r['identity_sha256'][:20] for r in self.object_catalog.records_for(self.scene) if r['stable_id']==ident))
            intervened[str(ident)]=change['new_material'];changes.append(item)
        changes.sort(key=lambda r:r['object_id'])
        return dict(passed=self.material_guard['passed'],changed_objects=changes,
            unchanged_object_count=self.material_guard['unchanged_object_count'],
            only_explicit_facade_groups_changed=True,geometry_hashes_unchanged=True,
            baseline_materials=baseline,intervened_materials=intervened,
            object_identity_schema=OBJECT_ID_VERSION,
            alias_scope='renderer auto names excluded; runtime aliases are diagnostic only')

    def configure(self,route):
        extras=super().configure(route)
        self.survey_reference_pose_rad=np.array([float(route['heading_rad']),0.,0.])
        extras['material_guard']=self._canonical_material_guard()
        extras['stable_object_ids']=self.object_catalog.records_for(self.scene,
            extra_objects=[self.car] if route.get('dynamic_object') and self.car is not None else [])
        extras.update(runtime_adapter_version=VERSION,object_identity_schema=OBJECT_ID_VERSION,
            survey_reference_pose_rad=self.survey_reference_pose_rad.copy(),
            survey_reference_pose_scope='fixed GCS yaw for every independent historical capture; online uses instantaneous trajectory pose',
            coordinate_scope='original built-in city world coordinates; no centering/translation was introduced',
            finite_search_policy=dict(primary_rays=self.samples,primary_seed=20261028,
                quality_failure_retry_rays=self.retry_samples,retry_seed=20261029,
                retries=1,model_changed_on_retry=False,global_convergence_certified=False,
                seed_scope='pure deterministic specular Fibonacci emission is not an independent random-seed experiment'),
            runtime_adapter_source_sha256={p.name:sha256(p) for p in
                (Path(__file__),Path(__file__).with_name('devset_object_ids.py'))},
            run_provenance=dict(getattr(self,'run_provenance',{})))
        return extras

    def trace(self,time_sec,position=None,velocity=None,pose=None):
        t=float(time_sec)
        if position is None:
            p,v,orientation,_=trajectory_at(self.route,t)
        else:
            p=np.asarray(position,float)
            v=np.zeros(3) if velocity is None else np.asarray(velocity,float)
            orientation=np.array([self.route['heading_rad'],0.,0.]) if pose is None else np.asarray(pose,float)
        if velocity is not None:v=np.asarray(velocity,float)
        if pose is not None:orientation=np.asarray(pose,float)
        if t<0 and position is not None:
            orientation=self.survey_reference_pose_rad.copy()
        self.rx.position,self.rx.velocity,self.rx.orientation=p.tolist(),v.tolist(),orientation.tolist()
        car_xyz,car_velocity,car_pose=np.zeros(3),np.zeros(3),np.zeros(3)
        if self.car_added:
            ground=np.asarray(self.car_motion['start_ground_xyz_m'])+t*np.asarray(self.car_motion['velocity_mps'])
            car_xyz=ground+np.array([0.,0.,-self.car_floor_z_at_zero])
            car_velocity=np.asarray(self.car_motion['velocity_mps'],float)
            car_pose=np.array([self.car_motion['heading_rad'],0.,0.])
            self.car.position,self.car.velocity=car_xyz.tolist(),car_velocity.tolist()
        txrel=xyz_rows(self.scene.tx_array.rotate(self.scene.wavelength,self.tx.orientation))
        rxrel=xyz_rows(self.scene.rx_array.rotate(self.scene.wavelength,self.rx.orientation))
        attempts=[]
        for number in range(2):
            settings=self.settings.copy()
            if number:settings.update(samples_per_src=self.retry_samples,seed=20261029)
            tick=time.perf_counter()
            paths=self.solver(self.scene,**settings);paths._validation_scene_owner=self.scene
            compact=extract_compact(paths,self.object_map);native=native_cfr(paths)
            rebuilt=reconstruct(compact,txrel,rxrel);denom=float(np.linalg.norm(native))
            error=float(np.linalg.norm(rebuilt-native)/denom) if np.isfinite(denom) and denom>0 else None
            if error is not None and not np.isfinite(error):error=None
            raw=exact_metadata(paths)
            quality=assess_snapshot(compact,native,raw,error,settings['max_num_paths_per_src'])
            elapsed=time.perf_counter()-tick
            item=dict(attempt=number,settings=settings,solve_seconds=elapsed,path_count=len(compact),quality=quality,
                native_relative_error=error,native_power=float(np.mean(abs(native)**2)) if np.isfinite(native).all() else None,
                exact_duplicate_geometry_pairs=raw['exact_duplicate_geometry_pairs'])
            if not quality['admitted']:item.update(compact=compact.copy(),native=native.copy(),raw=raw)
            attempts.append(item)
            if quality['admitted']:break
        return dict(compact=compact,native=native,quality=quality,native_relative_error=error,raw=raw,attempts=attempts,
            solve_seconds=sum(x['solve_seconds'] for x in attempts),tx_relative_m=txrel,rx_relative_m=rxrel,
            state_fields=dict(time_sec=t,rx_xyz_m=p,rx_velocity_mps=v,rx_pose_rad=orientation,
                tx_xyz_m=xyz_rows(self.tx.position)[0],tx_pose_rad=self.tx_pose.copy(),
                car_xyz_m=car_xyz,car_velocity_mps=car_velocity,car_pose_rad=car_pose,
                has_dynamic_object=self.car_added,quality_flags=quality['flags'],
                native_relative_error=np.nan if error is None else error,
                native_checked=bool(error is not None and np.isfinite(error)),
                has_los=bool(np.any(compact['depth']==0)),pass_index=self.environment,
                receiver_anchor=False,receiver_validated_until_sec=t))

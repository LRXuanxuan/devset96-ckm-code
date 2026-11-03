"""V3 runtime: certified unique specular rays and correctly grounded rigid cars.

Raw native candidates/full CFR remain a separate diagnostic archive. Estimation
truth uses only image-certified representatives and an independently masked
native CFR reference. Normalization is not an additional ray-tracing solve.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .cities import reconstruct,xyz_rows
from .common import sha256
from .devset_routes import trajectory_at
from .devset_runtime_v2 import DevsetRTV2
from .devset_specular_audit import (SpecularSceneGeometry,audit_specular_paths,
                                   check_private_contract,VERSION as AUDIT_VERSION)
from .path_quality import assess_snapshot

VERSION='kap_devset.runtime.v3'
UNRESOLVED_SPECULAR_CERTIFICATE=64


class _CandidateCounter:
    def __init__(self,original):self.original=original;self.record=None
    def __getattr__(self,name):return getattr(self.original,name)
    def __call__(self,**kwargs):
        buffer=self.original(**kwargs)
        self.record=dict(candidate_count_before_refinement=int(buffer.paths_counter[0]),
                         candidate_buffer_capacity=int(buffer.buffer_size))
        return buffer


class DevsetRTV3(DevsetRTV2):
    def __init__(self,city,samples=100000,retry_samples=300000):
        super().__init__(city,samples=samples,retry_samples=retry_samples)
        check_private_contract()
        self.geometry_cache=SpecularSceneGeometry()
        self.solver_call_count=0;self.route_solver_calls=0
        self._car_template=None;self._car_param_scene_id=None;self._car_scene_params=None
        self._last_vehicle_guard=None

    def _capture_car_template(self):
        if self._car_template is not None:return
        params=self.mi.traverse(self.car.mi_mesh)
        vertices=np.asarray(params['vertex_positions'],float).reshape(-1,3)
        center=(vertices.min(axis=0)+vertices.max(axis=0))/2
        # The first call occurs immediately after construction/scaling, before
        # the parent's first orientation setter. This immutable local shape
        # prevents cumulative in-place float32 translation/rotation drift.
        self._car_template=vertices-center
        self._car_template_faces=np.asarray(params['faces']).copy()

    def _place_vehicle(self,t):
        self._capture_car_template()
        yaw=float(self.car_motion['heading_rad']);pose=self.mi.Point3f(yaw,0.,0.)
        if not np.array_equal(np.asarray(self.car.orientation),np.asarray(pose)):
            self.car.orientation=pose
        rot=np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
        rotated=self._car_template@rot.T
        # SceneObject.position denotes the world AABB centre, not the source
        # PLY origin. Recenter the rotated shape before setting that centre.
        rotated-=.5*(rotated.min(axis=0)+rotated.max(axis=0))
        ground=np.asarray(self.car_motion['start_ground_xyz_m'],float)+float(t)*np.asarray(self.car_motion['velocity_mps'],float)
        center=ground+np.array([0.,0.,self.car_height/2])
        world=(rotated+center).astype(np.float32)
        if self.car_added:
            scene_id=id(self.scene.mi_scene)
            if self._car_param_scene_id!=scene_id:
                self._car_scene_params=self.mi.traverse(self.scene.mi_scene);self._car_param_scene_id=scene_id
            key=self.car.mi_mesh.id()+'.vertex_positions'
            if key not in self._car_scene_params:raise RuntimeError('Fresh scene traversal cannot resolve the rigid vehicle mesh')
            self._car_scene_params[key]=self.mi.Float(world.ravel());self._car_scene_params.update()
            self.scene.scene_geometry_updated()
        else:
            params=self.mi.traverse(self.car.mi_mesh)
            params['vertex_positions']=self.mi.Float(world.ravel());params.update()
        self.car.velocity=self.car_motion['velocity_mps']
        bb=self.car.mi_mesh.bbox();actual=xyz_rows(self.car.position)[0]
        floor_error=float(bb.min.z-ground[2]);top_error=float(bb.max.z-(ground[2]+self.car_height))
        if max(abs(floor_error),abs(top_error))>1e-5:raise ValueError('Actual world vehicle bbox is not correctly grounded')
        self._last_vehicle_guard=dict(actual_bbox_min_z_m=float(bb.min.z),actual_bbox_max_z_m=float(bb.max.z),
            requested_ground_z_m=float(ground[2]),floor_error_m=floor_error,top_error_m=top_error,
            actual_aabb_center_xyz_m=actual,placement_reference='AABB centre = ground + height/2',
            deterministic_rigid_source_reconstruction=True)
        return actual

    def _check_dynamic_geometry(self):
        self._capture_car_template()
        times=np.linspace(0.,self.route['duration_sec'],41);footprints=[];collisions=[];floor_errors=[]
        yaw=float(self.car_motion['heading_rad']);rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        for t in times:
            center=self._place_vehicle(t)
            vertices=np.asarray(self.mi.traverse(self.car.mi_mesh)['vertex_positions'],float).reshape(-1,3)
            local=(vertices[:,:2]-center[:2])@rot
            lo,hi=local.min(axis=0),local.max(axis=0)
            points=np.array([[x,y] for x in (lo[0],(lo[0]+hi[0])/2,hi[0])
                              for y in (lo[1],(lo[1]+hi[1])/2,hi[1])])
            footprints.append(center[:2]+points@rot.T)
            ue=trajectory_at(self.route,t)[0];ue_local=(ue[:2]-center[:2])@rot
            bb=self.car.mi_mesh.bbox()
            collisions.append(bool((ue_local>=lo-.1).all() and (ue_local<=hi+.1).all()
                                   and float(bb.min.z)-.1<=ue[2]<=float(bb.max.z)+.1))
            floor_errors.append(abs(self._last_vehicle_guard['floor_error_m']))
        xy=np.asarray(footprints).reshape(-1,2);bb=self.scene.mi_scene.bbox()
        origins=self.mi.Point3f(self.mi.Float(xy[:,0].astype(np.float32)),
            self.mi.Float(xy[:,1].astype(np.float32)),float(bb.max.z+5.))
        hits=self.scene.mi_scene.ray_intersect(self.mi.Ray3f(origins,self.mi.Vector3f(0.,0.,-1.)))
        valid=np.asarray(hits.is_valid(),bool).reshape(-1);z=np.asarray(hits.p.z,float).reshape(-1)
        ground=valid&(abs(z)<.02)
        return dict(passed=bool(ground.all() and not any(collisions) and max(floor_errors)<=1e-5),samples=41,
            footprint_ground_checks=len(xy),invalid_ground_samples=int((~ground).sum()),
            ue_bbox_collision_samples=int(sum(collisions)),vehicle_height_m=self.car_height,actual_speed_mps=8.,
            max_actual_bbox_ground_error_m=float(max(floor_errors)),bbox_ground_tolerance_m=1e-5,
            placement_reference='actual world AABB centre at ground+height/2',
            geometry_source='immutable scaled local vertices; deterministic rigid reconstruction at each timestamp',
            scope='finite footprint/UE bbox/actual-world-ground checks; not natural traffic calibration')

    def set_environment(self,pass_index):
        super().set_environment(pass_index)
        self._car_param_scene_id=None;self._car_scene_params=None
        if self.car_added:self._place_vehicle(0.)

    def configure(self,route):
        extras=super().configure(route);self.route_solver_calls=0
        self.survey_reference_pose_rad=xyz_rows(self.mi.Point3f(self.survey_reference_pose_rad.tolist()))[0]
        extras['survey_reference_pose_rad']=self.survey_reference_pose_rad.copy()
        extras.update(runtime_adapter_version=VERSION,
            native_reference_contract='analytic-certified one representative per full ordered planar specular triangle sequence; independently masked native Paths.cfr; original full native CFR separately archived',
            specular_normalization=dict(version=AUDIT_VERSION,
                grouping='exact active object/primitive/specular sequence; inactive slots ignored; different complete topologies never merged',
                geometry='actual world triangles, Float64 images, strict triangle interior, reflection law, full-scene Float64 BVH visibility',
                representative='closest joint image-point and analytic-delay residual; actual-plane and projected-optical-length consistency, no universal Euclidean point cutoff',
                unresolved_blocks_admission=True,raw_candidate_archive=True,
                exact_electromagnetic_field_error_bound=False),
            candidate_capacity_measurement='actual candidate-generator counter before shrink/image/refinement; full compact count is not the capacity test',
            quality_flag_extension={str(UNRESOLVED_SPECULAR_CERTIFICATE):'UNRESOLVED_SPECULAR_CERTIFICATE'},
            truth_coordinate_scope='actual native Float32 endpoint/pose/velocity values retained as Float64; requested values are diagnostic',
            dynamic_placement_reference='AABB centre=ground+height/2; immutable local mesh rigid reconstruction; actual bboxmin verified',
            runtime_adapter_source_sha256={p.name:sha256(p) for p in (Path(__file__),
                Path(__file__).with_name('devset_specular_audit.py'),Path(__file__).with_name('devset_runtime_v2.py'),
                Path(__file__).with_name('devset_object_ids.py'))},
            private_native_source_contract=check_private_contract())
        return extras

    def trace(self,time_sec,position=None,velocity=None,pose=None):
        t=float(time_sec)
        if position is None:p,v,orientation,_=trajectory_at(self.route,t)
        else:
            p=np.asarray(position,float);v=np.zeros(3) if velocity is None else np.asarray(velocity,float)
            orientation=np.array([self.route['heading_rad'],0.,0.]) if pose is None else np.asarray(pose,float)
        if velocity is not None:v=np.asarray(velocity,float)
        if pose is not None:orientation=np.asarray(pose,float)
        if t<0 and position is not None:orientation=self.survey_reference_pose_rad.copy()
        self.rx.position,self.rx.velocity,self.rx.orientation=p.tolist(),v.tolist(),orientation.tolist()
        car_xyz,car_velocity,car_pose=np.zeros(3),np.zeros(3),np.zeros(3)
        if self.car_added:
            car_xyz=self._place_vehicle(t);car_velocity=xyz_rows(self.car.velocity)[0];car_pose=xyz_rows(self.car.orientation)[0]
        attempts=[];raw_archives=[];normalization_seconds=0.;solver_seconds=0.
        for number in range(2):
            settings=self.settings.copy()
            if number:settings.update(samples_per_src=self.retry_samples,seed=20261029)
            original=self.solver._candidate_generator;counter=_CandidateCounter(original)
            self.solver._candidate_generator=counter;tick=time.perf_counter()
            self.solver_call_count+=1;self.route_solver_calls+=1
            try:paths=self.solver(self.scene,**settings)
            finally:self.solver._candidate_generator=original
            elapsed_solver=time.perf_counter()-tick;solver_seconds+=elapsed_solver
            paths._validation_scene_owner=self.scene
            started=time.perf_counter()
            result=audit_specular_paths(paths,self.scene,self.object_map,geometry_cache=self.geometry_cache,
                dynamic_object_ids=[self.car_stable_id] if self.car_added else [])
            elapsed_normalization=time.perf_counter()-started;normalization_seconds+=elapsed_normalization
            compact,native,raw,report=result['compact'],result['native'],result['raw'],result['report']
            keep_indices=np.asarray(report['keep_compact_indices'],int)
            selected_raw={name:np.asarray(raw[name])[keep_indices] for name in
                ('objects','primitives','interaction_types','vertices_xyz_m','native_path_index')}
            selected_raw['exact_duplicate_geometry_pairs']=[]
            txrel,rxrel=result['tx_relative_m'],result['rx_relative_m']
            raw_norm=float(np.linalg.norm(result['raw_native']))
            raw_error=float(np.linalg.norm(reconstruct(result['raw_compact'],txrel,rxrel)-result['raw_native'])/raw_norm) if np.isfinite(raw_norm) and raw_norm>0 else None
            raw_quality=assess_snapshot(result['raw_compact'],result['raw_native'],raw,raw_error,settings['max_num_paths_per_src'])
            capacity=counter.record or dict(candidate_count_before_refinement=None,candidate_buffer_capacity=None)
            capacity_hit=bool(capacity['candidate_count_before_refinement'] is None or
                capacity['candidate_count_before_refinement']>=capacity['candidate_buffer_capacity'])
            capacity=dict(capacity,candidate_count=capacity['candidate_count_before_refinement'],
                capacity=capacity['candidate_buffer_capacity'],hit_capacity=capacity_hit)
            norm=float(np.linalg.norm(native));error=report['selected_compact_vs_masked_native_relative_error']
            selected_meta=dict(exact_duplicate_geometry_pairs=[])
            quality=assess_snapshot(compact,native,selected_meta,error,settings['max_num_paths_per_src'])
            if not report['admitted']:
                quality['flags']|=UNRESOLVED_SPECULAR_CERTIFICATE;quality['reasons'].append('UNRESOLVED_SPECULAR_CERTIFICATE')
            if capacity_hit:
                quality['flags']|=32;raw_quality['flags']|=32
                if 'SEARCH_CAPACITY_REACHED' not in quality['reasons']:quality['reasons'].append('SEARCH_CAPACITY_REACHED')
                if 'SEARCH_CAPACITY_REACHED' not in raw_quality['reasons']:raw_quality['reasons'].append('SEARCH_CAPACITY_REACHED')
            quality['admitted']=not bool(quality['flags']);raw_quality['admitted']=not bool(raw_quality['flags'])
            report.update(candidate_capacity=capacity,candidate_capacity_reached=capacity_hit,
                native_reference_contract='selected analytic-certified native contribution sum; original full native sum is raw archive',
                vehicle_world_bbox_guard=self._last_vehicle_guard if self.car_added else None)
            if report['duplicate_topology_groups'] or report['unresolved_group_count'] or not report['admitted'] or not raw_quality['admitted']:
                raw_archives.append(dict(compact=result['raw_compact'],native=result['raw_native'],raw=raw,
                    normalization_report=report,native_relative_error=raw_error,quality=raw_quality,
                    solver_attempt=number,settings=settings.copy()))
            item=dict(attempt=number,settings=settings.copy(),quality=quality,path_count=len(compact),
                raw_path_count=len(result['raw_compact']),native_relative_error=error,
                native_power=float(np.mean(abs(native)**2)) if np.isfinite(native).all() else None,
                solve_seconds=elapsed_solver,normalization_seconds=elapsed_normalization,
                candidate_capacity=capacity,normalization_is_additional_solver_call=False)
            if not quality['admitted']:item.update(compact=compact.copy(),native=native.copy(),raw=selected_raw)
            attempts.append(item)
            if quality['admitted']:break
        return dict(compact=compact,native=native,quality=quality,native_relative_error=error,raw=selected_raw,
            normalization_report=report,raw_archives=raw_archives,attempts=attempts,
            candidate_capacity=dict(candidate_count=capacity['candidate_count_before_refinement'],
                capacity=capacity['candidate_buffer_capacity'],hit_capacity=capacity_hit),
            actual_solver_calls=len(attempts),solve_seconds=solver_seconds,normalization_seconds=normalization_seconds,
            tx_relative_m=txrel,rx_relative_m=rxrel,
            state_fields=dict(time_sec=t,rx_xyz_m=xyz_rows(self.rx.position)[0],rx_velocity_mps=xyz_rows(self.rx.velocity)[0],
                rx_pose_rad=xyz_rows(self.rx.orientation)[0],tx_xyz_m=xyz_rows(self.tx.position)[0],tx_pose_rad=self.tx_pose.copy(),
                car_xyz_m=car_xyz,car_velocity_mps=car_velocity,car_pose_rad=car_pose,
                has_dynamic_object=self.car_added,quality_flags=quality['flags'],
                native_relative_error=np.nan if error is None else error,native_checked=bool(error is not None and np.isfinite(error)),
                has_los=bool(np.any(compact['depth']==0)),pass_index=self.environment,
                receiver_anchor=False,receiver_validated_until_sec=t))

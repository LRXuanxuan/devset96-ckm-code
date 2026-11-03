"""Native transport with Float64 physical-image propagation phase.

Frozen V4 geometry, candidate search, fields and raw evidence remain available.
The estimation reference changes explicitly to the independently accumulated
full-array native pre-carrier transport and double-precision image delay.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np

from .cities import reconstruct,xyz_rows
from .common import sha256
from .devset_routes import trajectory_at
from .devset_runtime_v4 import DevsetRTV4
from .devset_runtime_v3 import _CandidateCounter,UNRESOLVED_SPECULAR_CERTIFICATE
from .devset_phase_reference_v5 import audit_phase_paths,VERSION as PHASE_VERSION
from .path_quality import assess_snapshot

VERSION='kap_devset.runtime.v5'


class DevsetRTV5(DevsetRTV4):
    def configure(self,route):
        extras=super().configure(route)
        extras.update(runtime_adapter_version=VERSION,
            phase_reference_version=PHASE_VERSION,
            native_carrier_frequency_hz=float(np.asarray(self.scene.frequency).reshape(-1)[0]),
            native_reference_contract='selected masked native Paths.a full-array transport with Float64 exp[-j2pi(fc+f)physical_image_delay], independent of compact reconstruction; original native.cir/cfr and original delays remain raw',
            phase_reference_contract=dict(version=PHASE_VERSION,
                native_transport_unchanged=True,original_native_cir_archived=True,
                float64_image_delay=True,actual_quantized_endpoints=True,exact_em_error_bound=False,
                geometry='V4 exact physical-patch Float64 image delay at actual native Float32 endpoints; no continuous-manifest endpoint substitution',
                independent_reference='full selected masked native Paths.a[Rx,Tx,path] summed in complex128 after Float64 carrier-plus-baseband phase rotation',
                compact='b00=native pre-carrier a00*exp(-j2pi fc image_delay); delay=image_delay; native k and Doppler unchanged',
                every_actual_rt_attempt_keeps_original_raw_archive=True,
                original_v4_normalization_report_semantics_preserved=True,
                original_native_cfr_is_the_corrected_reference=False,
                exact_electromagnetic_field_error_bound=False),
            runtime_adapter_source_sha256={p.name:sha256(p) for p in (Path(__file__),
                Path(__file__).with_name('devset_phase_reference_v5.py'),Path(__file__).with_name('devset_runtime_v4.py'),
                Path(__file__).with_name('devset_specular_audit_v4.py'),Path(__file__).with_name('devset_surface_patches.py'),
                Path(__file__).with_name('devset_runtime_v3.py'),Path(__file__).with_name('devset_specular_audit.py'),
                Path(__file__).with_name('devset_runtime_v2.py'),Path(__file__).with_name('devset_object_ids.py'))})
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
            paths._validation_scene_owner=self.scene;started=time.perf_counter()
            phase=audit_phase_paths(paths,self.scene,self.object_map,geometry_cache=self.geometry_cache,
                patch_catalog=self.patch_catalog,dynamic_object_ids=[self.car_stable_id] if self.car_added else [])
            elapsed_normalization=time.perf_counter()-started;normalization_seconds+=elapsed_normalization
            base=phase['base'];compact,native=phase['compact'],phase['native'];raw,report=base['raw'],base['report']
            keep=np.asarray(report['keep_compact_indices'],int)
            selected_raw={name:np.asarray(raw[name])[keep] for name in
                ('objects','primitives','interaction_types','vertices_xyz_m','native_path_index')}
            selected_raw['exact_duplicate_geometry_pairs']=[]
            txrel,rxrel=base['tx_relative_m'],base['rx_relative_m']
            raw_norm=float(np.linalg.norm(base['raw_native']))
            raw_error=float(np.linalg.norm(reconstruct(base['raw_compact'],txrel,rxrel)-base['raw_native'])/raw_norm) if np.isfinite(raw_norm) and raw_norm>0 else None
            raw_quality=assess_snapshot(base['raw_compact'],base['raw_native'],raw,raw_error,settings['max_num_paths_per_src'])
            capacity=counter.record or dict(candidate_count_before_refinement=None,candidate_buffer_capacity=None)
            capacity_hit=bool(capacity['candidate_count_before_refinement'] is None or
                capacity['candidate_count_before_refinement']>=capacity['candidate_buffer_capacity'])
            capacity=dict(capacity,candidate_count=capacity['candidate_count_before_refinement'],
                capacity=capacity['candidate_buffer_capacity'],hit_capacity=capacity_hit)
            error=phase['native_relative_error']
            quality=assess_snapshot(compact,native,dict(exact_duplicate_geometry_pairs=[]),error,settings['max_num_paths_per_src'])
            if not report['admitted']:
                quality['flags']|=UNRESOLVED_SPECULAR_CERTIFICATE;quality['reasons'].append('UNRESOLVED_SPECULAR_CERTIFICATE')
            if capacity_hit:
                quality['flags']|=32;raw_quality['flags']|=32
                if 'SEARCH_CAPACITY_REACHED' not in quality['reasons']:quality['reasons'].append('SEARCH_CAPACITY_REACHED')
                if 'SEARCH_CAPACITY_REACHED' not in raw_quality['reasons']:raw_quality['reasons'].append('SEARCH_CAPACITY_REACHED')
            quality['admitted']=not bool(quality['flags']);raw_quality['admitted']=not bool(raw_quality['flags'])
            # Preserve the V4 report's original native.cfr interpretation.
            report.update(candidate_capacity=capacity,candidate_capacity_reached=capacity_hit,
                vehicle_world_bbox_guard=self._last_vehicle_guard if self.car_added else None)
            raw_archives.append(dict(compact=base['raw_compact'],native=base['raw_native'],raw=raw,
                normalization_report=report,native_relative_error=raw_error,quality=raw_quality,
                phase_correction=phase['phase_correction'],phase_reference_report=phase['phase_reference_report'],
                original_selected_native_cfr=phase['original_selected_native_cfr'],
                solver_attempt=number,settings=settings.copy()))
            item=dict(attempt=number,settings=settings.copy(),quality=quality,path_count=len(compact),
                raw_path_count=len(base['raw_compact']),native_relative_error=error,
                native_power=float(np.mean(abs(native)**2)) if np.isfinite(native).all() else None,
                solve_seconds=elapsed_solver,normalization_seconds=elapsed_normalization,
                candidate_capacity=capacity,normalization_is_additional_solver_call=False)
            if not quality['admitted']:item.update(compact=compact.copy(),native=native.copy(),raw=selected_raw)
            attempts.append(item)
            if quality['admitted']:break
        return dict(compact=compact,native=native,quality=quality,native_relative_error=error,raw=selected_raw,
            normalization_report=report,phase_correction=phase['phase_correction'],phase_reference_report=phase['phase_reference_report'],
            original_selected_native_cfr=phase['original_selected_native_cfr'],
            raw_archives=raw_archives,attempts=attempts,
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

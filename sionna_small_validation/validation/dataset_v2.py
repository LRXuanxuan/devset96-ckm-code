"""Small independent survey / online RT product, with a strict estimator view.

This module deliberately keeps evaluation truth in a different HDF5 file.
The map contains measured historical power statistics, never online phases.
No RT import occurs until ``generate_city`` is called by an isolated worker.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import math

import h5py
import numpy as np

from .cities import (COMPACT_DTYPE, FC, FREQUENCIES, extract_compact,
                     native_cfr, reconstruct, stable_objects, xyz_rows)
from .common import save_json, sha256, environment
from .material_policy import prepare_facade_scene, apply_facade_intervention
from .path_quality import assess_snapshot

SEED = 20261027
SPLITS = {'munich': 'train', 'florence': 'validation', 'etoile': 'test'}
SURVEY_SIZES = (4, 8, 16, 32)


def write_complex(group, name, value, axes):
    value = np.asarray(value, np.complex64)
    child = group.create_group(name)
    child.attrs['axes'] = axes
    child.create_dataset('real', data=value.real, compression='gzip', compression_opts=1)
    child.create_dataset('imag', data=value.imag, compression='gzip', compression_opts=1)
    return child


def read_complex(group):
    return group['real'][:] + 1j * group['imag'][:]


def power_features(measured_h):
    """Rx 2D beam x delay power after four known Tx DFT beams.

    The frequency grid has 1.6 MHz spacing: these delay bins are cyclic
    1/(16*1.6MHz), not absolute path-delay estimates. All operations consume
    only noisy historical sounding, with no native path labels.
    """
    n = np.arange(64)
    beams = np.exp(2j*np.pi*n[:, None]*np.array([0, 5, 19, 37])[None, :]/64) / 8
    y = np.einsum('rtf,tb->rfb', measured_h, beams)
    y = y.reshape(4, 4, 16, 4)
    transformed = np.fft.ifft(np.fft.fft2(y, axes=(0, 1), norm='ortho'), axis=2, norm='ortho')
    return (np.abs(transformed)**2).astype(np.float32).ravel()


def _archive_truth(path, records, goldens):
    temp = path.with_suffix('.h5.partial')
    offsets = np.r_[0, np.cumsum([len(s['compact']) for s in records])].astype('<i8')
    with h5py.File(temp, 'x') as f:
        f.attrs.update(schema='kap_dataset_v2.truth.v2', carrier_hz=FC,
                       access='evaluation_only; not an estimator input',
                       phase_reference='native baseband rx0 tx0; no extra carrier phase',
                       synthetic_array=True, compact_bytes=128,
                       quality_schema='snapshot_flags_and_checked_intervals.v1')
        f.create_dataset('paths', data=np.concatenate([s['compact'] for s in records]), chunks=True)
        f.create_dataset('offsets', data=offsets)
        for name in ('quality_flags','receiver_anchor','receiver_validated_until_sec'):
            f.create_dataset(name,data=[s[name] for s in records])
        for name in ('time_sec', 'rx_xyz_m', 'rx_velocity_mps', 'tx_xyz_m',
                     'tx_relative_m', 'rx_relative_m', 'rx_pose_rad', 'tx_pose_rad'):
            f.create_dataset(name, data=[s[name] for s in records])
        for name in ('role', 'pass_id', 'scene_version', 'route_family_id', 'split', 'record_id'):
            f.create_dataset(name, data=np.array([s[name] for s in records], dtype=h5py.string_dtype()))
        f.create_dataset('frequencies_hz', data=FREQUENCIES)
        group = f.create_group('golden_cfr')
        group.create_dataset('record_index', data=[g[0] for g in goldens])
        write_complex(group, 'values', np.stack([g[1] for g in goldens]), 'golden,rx,tx,frequency')
        f['paths'].attrs['path_index_scope'] = 'snapshot_local; no temporal physical identity'
    with h5py.File(temp, 'r') as f:
        if f['paths'].dtype != COMPACT_DTYPE or len(f['offsets']) != len(records)+1:
            raise AssertionError('Invalid ragged compact archive')
        if not np.array_equal(f['offsets'][:], offsets):
            raise AssertionError('Offsets changed during HDF5 save')
    temp.replace(path)


def generate_city(city, out, backend, samples=30000, duration=.010,
                  anchor_step=.01, survey_count=32):
    import mitsuba as mi
    import sionna.rt as rt
    from .common import portable_asset
    from .scene_probe import probe_scene

    if (samples<1 or survey_count<32 or not np.isfinite(duration) or duration<.008
            or not np.isfinite(anchor_step) or anchor_step<=0):
        raise ValueError('Positive rays/step, >=32 surveys and duration>=8ms required')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED + list(SPLITS).index(city)*1000)
    asset = portable_asset(getattr(rt.scene, city))
    prepared_asset,material_manifest=prepare_facade_scene(asset)
    save_json(out/'material_geometry_manifest.json',material_manifest)
    scene = rt.load_scene(prepared_asset)
    scene.frequency = FC
    probe = probe_scene(scene, city, out/'geometry')
    proposal = probe['routes'][0]
    center = np.array(proposal['center_xyz'], float)
    heading = proposal['heading_rad']
    direction = np.array([math.cos(heading), math.sin(heading), 0.])
    lateral = np.array([-direction[1], direction[0], 0.])
    family = city + '_corridor1_family'
    scene.tx_array = rt.PlanarArray(num_rows=8, num_cols=8, vertical_spacing=.5,
                                    horizontal_spacing=.5, pattern='iso', polarization='V')
    scene.rx_array = rt.PlanarArray(num_rows=4, num_cols=4, vertical_spacing=.5,
                                    horizontal_spacing=.5, pattern='iso', polarization='V')
    tx = rt.Transmitter('tx', position=proposal['suggested_tx_xyz'])
    rx = rt.Receiver('rx', position=center.tolist())
    scene.add([tx, rx]); tx.look_at(mi.Point3f(center.tolist()))
    rx.orientation = [heading, 0, 0]
    txrel = xyz_rows(scene.tx_array.rotate(scene.wavelength, tx.orientation))
    rxrel = xyz_rows(scene.rx_array.rotate(scene.wavelength, rx.orientation))
    settings = dict(max_depth=3, samples_per_src=samples, max_num_paths_per_src=4096,
                    synthetic_array=True, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False, diffraction=False, seed=SEED)
    solver = rt.PathSolver(deterministic=True); solver.loop_mode = 'evaluated'
    mapping, objects = stable_objects(scene, city)
    records, goldens, native_errors = [], [], []
    intervals, survey_h, survey_indices = [], [], []
    changes = [];material_guard=None;quality_records=[];solve_attempts=[]

    def trace(t, position, velocity, role, pass_id, version, keep_golden=False):
        rx.position = np.asarray(position).tolist(); rx.velocity = np.asarray(velocity).tolist()
        idx = len(records)
        from .admission import metadata
        # Bounded retries change candidate-search effort only. They do not
        # switch the declared propagation model or delete duplicate paths.
        for attempt in range(2):
            effective=settings.copy()
            if attempt:effective.update(samples_per_src=max(samples,100000),seed=SEED+1)
            paths=solver(scene,**effective);paths._validation_scene_owner=scene
            compact=extract_compact(paths,mapping);native=native_cfr(paths)
            rebuilt=reconstruct(compact,txrel,rxrel);denom=float(np.linalg.norm(native))
            err=float(np.linalg.norm(rebuilt-native)/denom) if np.isfinite(denom) and denom>0 else None
            if err is not None and not np.isfinite(err):err=None
            raw=metadata(paths);quality=assess_snapshot(compact,native,raw,err,effective['max_num_paths_per_src'])
            solve_attempts.append(dict(record_index=idx,attempt=attempt,settings=effective,
                path_count=len(compact),quality=quality,relative_error=err))
            if quality['admitted']:break
            np.savez_compressed(out/f'quarantine_attempt_{idx:05d}_{attempt}.npz',compact=compact,native_cfr=native)
            save_json(out/f'quarantine_attempt_{idx:05d}_{attempt}.json',dict(quality=quality,raw_geometry=raw,
                      time_sec=float(t),position_m=np.asarray(position),settings=effective))
        records.append(dict(record_id=f'{city}_{idx:06d}', compact=compact, time_sec=float(t),
                            rx_xyz_m=np.array(position), rx_velocity_mps=np.array(velocity),
                            tx_xyz_m=xyz_rows(tx.position)[0], tx_relative_m=txrel, rx_relative_m=rxrel,
                            rx_pose_rad=np.array([heading, 0, 0]), tx_pose_rad=xyz_rows(tx.orientation)[0],
                            role=role, pass_id=pass_id, scene_version=version,
                            route_family_id=family, split=SPLITS[city],quality_flags=quality['flags'],
                            receiver_anchor=role=='online_anchor',receiver_validated_until_sec=float(t)))
        quality_records.append(dict(record_index=idx,**quality))
        native_errors.append(dict(record_index=idx, count=len(compact), relative_error=err,
                                  finite=bool(np.isfinite(native).all()), nonempty=denom>0))
        if keep_golden:
            goldens.append((idx, native))
        return idx, native

    # Separate capture positions, times, noise streams and labels. Survey is
    # allowed in a held-out target city under the target-region-map protocol.
    u = (np.arange(survey_count) + rng.uniform(.15, .85, survey_count))/survey_count
    rng.shuffle(u)
    for j in range(survey_count):
        position = center + (u[j]-.5)*30*direction + rng.uniform(-.2, .2)*lateral
        idx, h = trace(-86400+j*.01, position, np.zeros(3), 'historical_survey',
                       f'{city}_survey', 'base', j in (0, survey_count-1))
        survey_indices.append(idx); survey_h.append(h)
        if records[idx]['quality_flags']:
            save_json(out/'quality_failure.json',dict(reason='Unadmitted historical survey cannot build a map',
                      quality_records=quality_records,solve_attempts=solve_attempts))
            raise RuntimeError('Unadmitted historical survey; raw attempts retained, no map published')
    survey_h = np.stack(survey_h)
    survey_variance = np.mean(np.abs(survey_h)**2, axis=(1, 2, 3))*10**(-25/10)
    noise = (rng.standard_normal(survey_h.shape)+1j*rng.standard_normal(survey_h.shape))
    measured = survey_h + noise*np.sqrt(survey_variance[:, None, None, None]/2)
    features = np.stack([power_features(h) for h in measured])
    # Survey position reporting noise is independent of online state reports.
    survey_reported = np.array([records[i]['rx_xyz_m'] for i in survey_indices]) + rng.normal(0, .25, (survey_count, 3))

    count = int(math.ceil(duration*30/anchor_step))+1
    base_times = np.unique(np.r_[np.linspace(0,duration,count),.008])
    online = {}
    for version in ('base', 'vertical_facades_wood'):
        if version != 'base':
            changes,material_guard=apply_facade_intervention(scene,rt,material_manifest)
            save_json(out/'material_intervention_guard.json',material_guard)
        pass_id = city + '_' + version + '_pass0'
        traced = {}
        def online_trace(t, role='online_anchor'):
            key = round(float(t), 14)
            if key not in traced:
                pos = center + (t-duration/2)*30*direction
                traced[key] = trace(t, pos, 30*direction, role, pass_id, version,
                                     abs(t)<1e-12 or abs(t-.008)<1e-12 or abs(t-duration)<1e-12)
            return traced[key]
        for j, t in enumerate(base_times):
            online_trace(float(t))
        # Refine failed local propagation intervals, never interpolate complex
        # CFR across them. Both midpoint truth and the failure remain archived.
        def check_interval(left, right, level):
            a, hleft = online_trace(left)
            c, hright=online_trace(right)
            for ix in (a,c):records[ix]['receiver_anchor']=True
            mid = (left+right)/2
            b, hmid = online_trace(mid, 'online_midpoint')
            estimate = reconstruct(records[a]['compact'], txrel, rxrel, delta_t=mid-left)
            denom = np.linalg.norm(hmid)
            error = float(20*np.log10(max(np.linalg.norm(estimate-hmid)/denom, 1e-150))) if np.isfinite(denom) and denom>0 else None
            if error is not None and not np.isfinite(error):error=None
            right_estimate=reconstruct(records[a]['compact'],txrel,rxrel,delta_t=right-left)
            rdenom=np.linalg.norm(hright)
            right_error=float(20*np.log10(max(np.linalg.norm(right_estimate-hright)/rdenom,1e-150))) if np.isfinite(rdenom) and rdenom>0 else None
            if right_error is not None and not np.isfinite(right_error):right_error=None
            good_states=all(records[i]['quality_flags']==0 for i in (a,b,c))
            passed = good_states and error is not None and error <= -30 and right_error is not None and right_error<=-30
            intervals.append(dict(pass_id=pass_id, left_index=a, midpoint_index=b,
                                  left_time_sec=left, right_time_sec=right,
                                  right_index=c,level=level, midpoint_nmse_db=error,right_nmse_db=right_error,
                                  snapshot_quality_passed=good_states,passed=passed,
                                  used_as_leaf=passed or level>=3))
            if passed:records[a]['receiver_validated_until_sec']=max(records[a]['receiver_validated_until_sec'],right)
            if not passed and level<3:
                check_interval(left, mid, level+1); check_interval(mid, right, level+1)
        for left, right in zip(base_times[:-1], base_times[1:]):
            check_interval(float(left), float(right), 0)
        online[pass_id] = sorted([v[0] for v in traced.values() if records[v[0]]['receiver_anchor']], key=lambda i: records[i]['time_sec'])
        print(f'[v2] {city} {version}: {len(online[pass_id])} RT records', flush=True)

    _archive_truth(out/'truth.h5', records, goldens)
    # Native dense sounding is an explicitly ideal offline measurement model;
    # online waveform validation is a separate product, not implied by this.
    temp = out/'survey.h5.partial'
    with h5py.File(temp, 'x') as f:
        f.attrs.update(schema='kap_dataset_v2.survey.v1', survey_id=city+'_survey',
                       historical_scene_version='base', historical_time_reference_sec=-86400.,
                       measurement_model='offline ideal dense CFR sounding plus iid circular complex Gaussian noise',
                       capture_time_semantics='independent static captures with logical historical timestamps; not a continuous acquisition route',
                       sounding_snr_db=25., route_family_id=family, split=SPLITS[city])
        f.create_dataset('reported_position_m', data=survey_reported)
        f.create_dataset('position_noise_covariance_m2', data=np.eye(3)*.25**2)
        f.create_dataset('survey_time_sec', data=[records[i]['time_sec'] for i in survey_indices])
        f.create_dataset('survey_sample_id', data=np.arange(survey_count))
        f.create_dataset('noise_variance', data=survey_variance)
        f.create_dataset('power_features', data=features)
        f.create_dataset('frequencies_hz', data=FREQUENCIES)
        write_complex(f, 'measured_dense_cfr', measured, 'survey,rx,tx,frequency')
    temp.replace(out/'survey.h5')

    public_tasks, task_queries, evaluation_tasks = [], [], {}
    temp = out/'reported_states.h5.partial'
    with h5py.File(temp, 'x') as f:
        f.attrs['schema'] = 'kap_dataset_v2.reported_states.v1'
        f.attrs['noise_model'] = 'independent Gaussian position/velocity/yaw errors; all timestamps receiver-known'
        for pass_id, indices in online.items():
            # Opaque public IDs hide matched/mismatch evaluation labels.
            public_pass = hashlib.sha256((str(SEED)+city+pass_id).encode()).hexdigest()[:20]
            g = f.create_group(public_pass)
            # Reports are on a fixed device clock, independent of private RT
            # midpoint errors/adaptive refinement. Pair variants share errors.
            times = base_times.copy()
            report_rng = np.random.default_rng(SEED+list(SPLITS).index(city)*1000+701)
            positions = center + (times[:,None]-duration/2)*30*direction + report_rng.normal(0,.25,(len(times),3))
            velocities = np.tile(30*direction,(len(times),1)) + report_rng.normal(0,.5,(len(times),3))
            poses = np.tile([heading,0,0],(len(times),1)) + report_rng.normal(0,.01,(len(times),3))
            for name, value in (('report_time_sec', times), ('reported_position_m', positions),
                                ('reported_velocity_mps', velocities), ('reported_pose_rad', poses)):
                g.create_dataset(name, data=value)
            for name, sigma in (('position_covariance_m2', .25), ('velocity_covariance_m2s2', .5), ('pose_covariance_rad2', .01)):
                g.create_dataset(name, data=np.eye(3)*sigma**2)
            # Six milliseconds is the input cutoff; target delay is 2ms.
            # Task target may be changed without duplicating propagation truth.
            cutoff = .006; history_start = .002; target = .008
            if duration < target:
                raise ValueError('duration must cover 4ms history and 2ms target delay')
            q = np.flatnonzero(times<=cutoff+1e-12)[-1]
            task_id = hashlib.sha256((public_pass+'task0').encode()).hexdigest()[:20]
            public_tasks.append(dict(task_id=task_id, city=city, split=SPLITS[city],
                route_family_id=family, pass_id=public_pass, input_cutoff_sec=cutoff,
                history_start_sec=history_start, target_time_sec=target,
                target_region_map_available=True, map_versions=list(SURVEY_SIZES),
                reported_state_file='reported_states.h5', map_file='maps.h5',
                observation_contract='full FFT across all Rx; actual sequential pilot symbol times; no future truth'))
            evaluation_tasks[task_id] = dict(truth_pass_id=pass_id,public_pass_id=public_pass,
                pairing_group=family+'_material_pair',pairing_role='matched' if '_base_' in pass_id else 'material_mismatch')
            task_queries.append(positions[q])
    temp.replace(out/'reported_states.h5')

    temp = out/'maps.h5.partial'
    with h5py.File(temp, 'x') as f:
        f.attrs.update(schema='kap_dataset_v2.maps.v1', historical_scene_version='base',
                       map_representation='Rx4x4 angular x cyclic-delay16 power in four known Tx DFT beams',
                       map_source='noisy independent historical survey only; no online truth or path labels',
                       uncertainty_scope='weighted finite-survey empirical variance; not a calibrated posterior',
                       target_region_map_available=True)
        f.create_dataset('query_task_id', data=np.array([t['task_id'] for t in public_tasks], dtype=h5py.string_dtype()))
        f.create_dataset('reported_query_position_m', data=task_queries)
        for size in SURVEY_SIZES:
            if size>survey_count:
                continue
            g = f.create_group('n'+str(size))
            g.create_dataset('survey_sample_ids', data=np.arange(size))
            means, variances, neffs, distances = [], [], [], []
            for query in task_queries:
                d = np.linalg.norm(survey_reported[:size]-query, axis=1)
                w = np.exp(-.5*(d/5.)**2); w /= w.sum()
                mean = w @ features[:size]
                correction = 1-float(w @ w)
                variance = (w @ ((features[:size]-mean)**2))/correction
                means.append(mean); variances.append(variance); neffs.append(1/float(w@w)); distances.append(d)
            g.create_dataset('power_mean', data=np.asarray(means, np.float32))
            g.create_dataset('power_empirical_variance', data=np.asarray(variances, np.float32))
            g.create_dataset('effective_sample_count', data=neffs)
            g.create_dataset('survey_query_distance_m', data=distances)
    temp.replace(out/'maps.h5')
    save_json(out/'tasks.json', dict(schema='kap_dataset_v2.tasks.v1', tasks=public_tasks))
    # Evaluator mapping is isolated from the task input contract.
    save_json(out/'evaluation_index.json', dict(access='evaluation_only', online_record_indices=online,
              survey_record_indices=survey_indices, intervals=intervals, material_changes=changes,
              scene_object_ids=objects, route_proposal=proposal, task_truth_mapping=evaluation_tasks))
    leaves = [i for i in intervals if i['used_as_leaf']]
    quality_report=dict(schema='snapshot_flags_and_checked_intervals.v1',
        status='passed' if all(q['admitted'] for q in quality_records) and all(i['passed'] for i in leaves) else 'needs_attention',
        physical_scope='Finite fixed-model numeric/local checks; not global propagation certification',
        snapshot_quality=quality_records,solve_attempts=solve_attempts,intervals=intervals,
        empty_is_physical_zero=False,duplicates_removed=False,
        unsafe_intervals_block_receiver=True,
        checked_interval_rule='frozen-left CFR agrees at midpoint AND right endpoint; finite screening only')
    save_json(out/'quality_report.json',quality_report)
    checks = dict(native_compact=all(x['finite'] and x['nonempty'] and x['relative_error']<1e-4 for x in native_errors),
                  local_midpoints=all(x['passed'] for x in leaves),
                  independent_historical_times=max(records[i]['time_sec'] for i in survey_indices)<0,
                  sufficient_window=duration>=.008,
                  snapshot_quality=all(q['admitted'] for q in quality_records),
                  material_intervention_guard=bool(material_guard and material_guard['passed']))
    report = dict(status='passed' if all(checks.values()) else 'needs_attention', checks=checks,
                  scope='local small dataset implementation; not full production scientific admission',
                  city=city, split=SPLITS[city], survey_count=survey_count,
                  task_count=len(public_tasks), online_passes={k:len(v) for k,v in online.items()},
                  total_rt_states=len(records), path_records=sum(len(s['compact']) for s in records),
                  path_count_max=max(len(s['compact']) for s in records),
                  duration_sec=duration, online_length_m=duration*30, base_anchor_step_m=anchor_step,
                  midpoint_checks=len(intervals), unresolved_leaf_count=sum(not i['passed'] for i in leaves),
                  native_errors=native_errors, solver_settings=settings, environment=environment(backend),
                  quality_schema=quality_report['schema'],total_rt_solve_attempts=len(solve_attempts),
                  receiver_anchor_count=sum(s['receiver_anchor'] for s in records),
                  material_policy=material_manifest['policy_version'],
                  source_sha256={p.name:sha256(p) for p in
                    [Path(__file__).parents[1]/'run_dataset_v2.py',*[Path(__file__).parent/name for name in
                    ('dataset_v2.py','material_policy.py','path_quality.py','cities.py','admission.py','common.py','windows_llvm.py')]]},
                  source_asset_sha256=sha256(asset), seed=SEED,
                  limitations=['One short straight route family per city, selected by geometric clearance, not RF performance.',
                     'A held-out city has its own independent historical map; this is target-region-map generalization.',
                     'Survey finite-sample empirical variance is not a calibrated predictive posterior.',
                     'Material changes are synthetic stress conditions on real geometry, not measured aging.',
                     'Midpoint checks are finite engineering checks, not a uniform error certificate.',
                     'No cross-anchor path identity is inferred from object IDs.',
                     'Depth3 specular-only RT, no diffraction/diffuse/refraction: production convergence still separate.'])
    report['files'] = {p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in out.glob('*') if p.is_file()}
    save_json(out/'generation_report.json', report)
    return report


class EstimatorView:
    """Allow-list loader: never opens a truth or evaluation-index file."""
    def __init__(self, city_directory):
        self.directory = Path(city_directory)

    def load(self, task_id, survey_size=16, observation_mode='rotating'):
        if survey_size not in SURVEY_SIZES or observation_mode not in ('fixed','rotating'):
            raise ValueError('Undeclared map version or observation mode')
        tasks = json.loads((self.directory/'tasks.json').read_text(encoding='utf-8'))['tasks']
        task = next(t for t in tasks if t['task_id']==task_id)
        private_task_keys={'pairing_role','pairing_group','scene_version','truth_pass_id','target_csi','paths'}
        if private_task_keys&set(task):raise ValueError('Private evaluation labels in public task')
        if not task['history_start_sec']<=task['input_cutoff_sec']<=task['target_time_sec']:
            raise ValueError('Invalid task time order')
        with h5py.File(self.directory/'reported_states.h5','r') as f:
            g=f[task['pass_id']]
            times=g['report_time_sec'][:]
            keep=(times>=task['history_start_sec']-1e-12)&(times<=task['input_cutoff_sec']+1e-12)
            states={name:g[name][:][keep] for name in ('report_time_sec','reported_position_m',
                    'reported_velocity_mps','reported_pose_rad')}
            states.update({name:g[name][:] for name in
                ('position_covariance_m2','velocity_covariance_m2s2','pose_covariance_rad2')})
        with h5py.File(self.directory/'maps.h5','r') as f:
            q=list(f['query_task_id'].asstr()[:]).index(task_id)
            group=f['n'+str(survey_size)]
            prior={name:group[name][q] for name in ('power_mean','power_empirical_variance','effective_sample_count')}
        result=dict(task=task,reported_states=states,map=prior)
        observation_path=self.directory/'observations.h5'
        if observation_path.exists():
            with h5py.File(observation_path,'r') as f:
                if str(f.attrs.get('schema','')).endswith('.v2') and not bool(f.attrs.get('generation_complete',False)):
                    raise ValueError('Receiver product has no successful completion marker')
                if task_id in f:
                    g=f[task_id][observation_mode]
                    allowed={'full_fft','pilot_y','symbols','tx_beams','start_sample','receive_time_sec',
                             'pilot_mask','burst_id','noise_variance','cfo_is_known','cfo_prior_std_hz',
                             'fs_hz','nfft','cp_samples'}
                    if set(g)-allowed:raise ValueError('Undeclared fields in public receiver view')
                    result['observation']={name:read_complex(g[name]) if isinstance(g[name],h5py.Group) and 'real' in g[name]
                         else g[name][:] for name in g}
                    obs=result['observation'];rtimes=obs['receive_time_sec']
                    if (not np.isfinite(rtimes).all() or rtimes.min()<task['history_start_sec']-1e-12
                            or rtimes.max()>task['input_cutoff_sec']+1e-12):
                        raise ValueError('Receiver data outside the legal task history')
                    if obs['full_fft'].shape[::2]!=rtimes.shape or not np.isfinite(obs['full_fft']).all():
                        raise ValueError('Invalid full-FFT dimensions or nonfinite samples')
        return result


def audit_dataset(root):
    """Check real disk arrays, causal views, nested surveys and route splits."""
    root=Path(root); reports=[]; families={}; all_tasks=[]
    for city in SPLITS:
        directory=root/city
        if not (directory/'generation_report.json').exists():
            reports.append(dict(city=city,status='missing'));continue
        generated=json.loads((directory/'generation_report.json').read_text(encoding='utf-8'))
        tasks=json.loads((directory/'tasks.json').read_text(encoding='utf-8'))['tasks']; all_tasks.extend(tasks)
        checks={}
        with h5py.File(directory/'truth.h5','r') as f:
            offsets=f['offsets'][:]; p=f['paths'][:]
            checks['ragged_offsets_valid']=bool(offsets[0]==0 and offsets[-1]==len(p) and np.all(np.diff(offsets)>=0))
            checks['compound128']=p.dtype==COMPACT_DTYPE
            if f.attrs.get('quality_schema'):
                checks['snapshot_admission_flags_clear']=bool(np.all(f['quality_flags'][:]==0))
                anchors=f['receiver_anchor'][:].astype(bool);until=f['receiver_validated_until_sec'][:]
                checks['private_receiver_quality_fields']=bool(len(anchors)==len(f['time_sec']) and np.isfinite(until).all())
            role=f['role'].asstr()[:]; times=f['time_sec'][:]
            hist=np.flatnonzero(role=='historical_survey'); current=np.flatnonzero(role!='historical_survey')
            checks['survey_online_time_disjoint']=bool(times[hist].max()<times[current].min())
            distances=np.linalg.norm(f['rx_xyz_m'][:][hist,None,:]-f['rx_xyz_m'][:][None,current,:],axis=-1)
            checks['no_identical_survey_online_positions']=bool(distances.min()>1e-6)
            for family, split in zip(f['route_family_id'].asstr()[:],f['split'].asstr()[:]):
                families.setdefault(family,set()).add(split)
        with h5py.File(directory/'maps.h5','r') as f:
            ids=[f['n'+str(n)+'/survey_sample_ids'][:] for n in SURVEY_SIZES]
            checks['nested_survey_sets']=all(np.array_equal(ids[j],ids[j+1][:len(ids[j])]) for j in range(3))
            checks['map_only_statistics']=not any(x in f for x in ('b00','paths','golden_cfr','target_csi'))
            checks['finite_nonnegative_map_variance']=all(np.isfinite(f['n'+str(n)+'/power_empirical_variance'][:]).all()
                      and np.all(f['n'+str(n)+'/power_empirical_variance'][:]>=0) for n in SURVEY_SIZES)
        for task in tasks:
            view=EstimatorView(directory).load(task['task_id'])
            times=view['reported_states']['report_time_sec']
            checks[task['task_id']+'_causal']=bool(times.max()<=task['input_cutoff_sec']+1e-12 and times.min()>=task['history_start_sec']-1e-12)
            checks[task['task_id']+'_4ms_2ms']=bool(abs(task['input_cutoff_sec']-task['history_start_sec']-.004)<1e-12
                         and abs(task['target_time_sec']-task['input_cutoff_sec']-.002)<1e-12)
            checks[task['task_id']+'_no_evaluation_labels']=not any(k in task for k in ('pairing_role','pairing_group','scene_version'))
            checks[task['task_id']+'_opaque_id']=not any(x in str(task['task_id'])+str(task['pass_id']) for x in ('base','wood','matched','mismatch'))
        with h5py.File(directory/'reported_states.h5','r') as f:
            g0,g1=[f[t['pass_id']] for t in tasks]
            checks['paired_fixed_report_schedule_and_errors']=all(np.array_equal(g0[n][:],g1[n][:]) for n in
                ('report_time_sec','reported_position_m','reported_velocity_mps','reported_pose_rad'))
        reports.append(dict(city=city,status='passed' if all(checks.values()) else 'failed', checks=checks,
                            rt_states=generated['total_rt_states'],path_records=generated['path_records']))
    checks=dict(no_route_family_crosses_split=all(len(v)==1 for v in families.values()),
                all_three_cities_present=len(reports)==3 and all(r['status']!='missing' for r in reports),
                paired_material_tasks_same_split=all(len({t['split'] for t in all_tasks if t['city']==city})==1 for city in SPLITS))
    passed=all(checks.values()) and all(r['status']=='passed' for r in reports)
    report=dict(status='passed' if passed else 'failed', checks=checks,cities=reports,
                scope='format and causal access checks; not a model performance claim',
                no_target_map_protocol='not generated; target-city historical map is available in this pilot')
    save_json(root/'dataset_audit.json',report)
    return report

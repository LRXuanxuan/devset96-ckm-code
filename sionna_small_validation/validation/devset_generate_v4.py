"""V4 streaming physical-patch truth with complete raw candidate audit records."""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .cities import reconstruct, xyz_rows
from .common import save_json, sha256
from .dataset_v2 import power_features
from .devset_routes import trajectory_at
from .devset_storage_v4 import RouteWriterV4 as RouteWriter
from .devset_view import rotate_local_array


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], 'little')


def source_manifest():
    base = Path(__file__).resolve().parents[1]
    names = ['run_devset_v4.py', 'validation/devset_generate_v4.py', 'validation/devset_runtime_v4.py', 'validation/devset_specular_audit_v4.py', 'validation/devset_surface_patches.py', 'validation/devset_storage_v4.py', 'validation/devset_view_v4.py', 'validation/devset_view_v3.py', 'validation/devset_index.py', 'run_devset_v3.py', 'validation/devset_generate_v3.py', 'validation/devset_runtime_v3.py', 'validation/devset_specular_audit.py', 'validation/devset_storage_v3.py', 'validation/devset_runtime_v2.py', 'validation/devset_object_ids.py', 'validation/devset_view_v2.py', 'run_devset.py', 'validation/devset_generate.py', 'validation/devset_rt.py',
             'validation/devset_storage.py', 'validation/devset_view.py',
             'validation/devset_routes.py', 'validation/devset_motion.py',
             'validation/cities.py', 'validation/path_quality.py',
             'validation/material_policy.py', 'validation/mimo_observation.py',
             'validation/dataset_observations.py', 'validation/dataset_v2.py',
             'validation/common.py', 'validation/windows_llvm.py']
    return {n: sha256(base/n) for n in names}


def _key(t):
    return round(float(t), 12)


def _db_error(estimate, reference):
    norm = float(np.linalg.norm(reference))
    if not np.isfinite(norm) or norm <= 0:
        return float('nan')
    value = float(np.linalg.norm(estimate-reference)/norm)
    return 20*math.log10(max(value, 1e-150)) if math.isfinite(value) else float('nan')


def _coverage(intervals):
    merged = []
    for a, b in sorted((float(r['left_time_sec']), float(r['right_time_sec']))
                       for r in intervals if r['passed'] and r['used_as_leaf']):
        if merged and a <= merged[-1][1]+1e-12:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.asarray(merged, float).reshape(-1, 2)


def generate_route(context, route, path, *, resume=False, progress_path=None,
                   anchor_step_m=.01, max_refinement=3, samples=30000):
    """Generate both environments, all survey designs and all cutoff indices.

    Completion is local/numeric admission, not global propagation convergence.
    Failed states and all rejected windows remain in the private shard ledger.
    """
    path = Path(path)
    started = time.perf_counter()
    extras = context.configure(route)
    extras = dict(extras or {})
    # Native handles are process-local, whereas geometry identity is stable.
    if 'stable_object_ids' in extras:
        extras['stable_object_ids'] = [{k: v for k, v in item.items() if 'runtime' not in k}
                                       for item in extras['stable_object_ids']]
    config = dict(route=route, **extras, seed=20261027, anchor_step_m=anchor_step_m,
                  max_refinement=max_refinement, local_nmse_limit_db=-30., interval_nmse_threshold_db=-30.,
                  native_relative_error_tolerance=1e-4, history_sec=.004,
                  prediction_sec=.002, cutoff_step_sec=.001, report_interval_sec=.001,
                  max_local_age_sec=.01, map_bandwidth_m=5.,
                  position_report_std_m=.25, velocity_report_std_mps=.5,
                  pose_report_std_rad=.01, survey_snr_db=25.,
                  cfo_prior_std_hz=500., source_sha256=source_manifest(),
                  target_contract='Every task target is a direct RT state and native-checked; sparse full native cache',
                  trace_accounting_required=True,
                  storage_contract='compact128 unique physical patch branches; original native candidates plus compressed geometry ledger; on-demand complete FFT')
    config['asset_provenance'] = {k:context.material_manifest[k] for k in
        ('content_digest','source_asset_sha256','policy_version')}
    if 'tx_array_local_m' not in config or 'rx_array_local_m' not in config:
        raise ValueError('RT context did not declare local array geometry')
    writer = RouteWriter(path, config, resume=resume, batch_size=128, warning_bytes=512_000_000)
    cache = {}
    native_cache = OrderedDict()
    completed_intervals = {}
    intervals_by_pass = {0: [], 1: []}
    survey_done = set()
    existing_tasks = set()
    durations = []
    retry_count = 0
    final_flags = Counter()
    failure_flags = Counter()
    golden_failure_count = 0
    last_progress = 0.

    def progress(stage, force=False):
        nonlocal last_progress
        now = time.perf_counter()
        if not force and now-last_progress < 5:
            return
        last_progress = now
        if progress_path:
            save_json(progress_path, dict(route_family_id=route['route_family_id'], city=route['city'],
                      split=route['split'], mechanism=route['intended_mechanism'], stage=stage,
                      states=writer.state_count, tasks=writer.task_count,
                      elapsed_sec=now-started, final_quality_flags=dict(final_flags),
                      retries=retry_count, shard_bytes=writer.partial_path.stat().st_size,
                      state_time_p50_sec=float(np.median(durations)) if durations else None,
                      state_time_p95_sec=float(np.quantile(durations, .95)) if durations else None))

    def cached_record(index, row, compact):
        return dict(index=int(index), compact=compact, fields={n: row[n] for n in row.dtype.names})

    # Resume from actual committed rows, not a cursor that could lag an auto-flush.
    f = writer.handle
    if len(f['states']):
        states = f['states'][:]; offsets = f['offsets'][:]
        for i, row in enumerate(states):
            role = row['role'].decode().rstrip('\x00')
            if role not in ('quarantine_attempt','raw_native_candidate'):
                cache[(int(row['pass_index']), _key(row['time_sec']))] = cached_record(i, row, f['paths'][offsets[i]:offsets[i+1]])
                final_flags[str(int(row['quality_flags']))] += 1
        for r in f['intervals'][:]:
            item = {name: r[name] for name in r.dtype.names}
            completed_intervals[(int(r['pass_index']), _key(r['left_time_sec']), _key(r['right_time_sec']), int(r['level']))] = item
            intervals_by_pass[int(r['pass_index'])].append(item)
        survey_done = set(map(int, f['survey_index']['record_index']))
        existing_tasks = {(int(r['pass_index']), _key(r['cutoff_sec'])) for r in f['tasks'][:]}

    def remember_native(index, h):
        native_cache[int(index)] = h
        native_cache.move_to_end(int(index))
        while len(native_cache) > 24:
            native_cache.popitem(last=False)

    def account_native_call(index, result, revisit=False):
        capacity = result.get('candidate_capacity')
        if not isinstance(capacity, dict) or 'candidate_count' not in capacity or 'capacity' not in capacity:
            raise ValueError('V4 requires the actual pre-pruning candidate counter, not final compact length')
        writer.append_trace_accounting(index, int(result['actual_solver_calls']), source_revisit=revisit,
            final_candidate_count=int(capacity['candidate_count']), candidate_capacity=int(capacity['capacity']))

    def archive_evidence(result, fields, t, pass_index):
        """Archive normalization changes and failed attempts, including revisits."""
        nonlocal retry_count, golden_failure_count
        raw_record_indices = []
        raw_source_attempts = []
        for archive in result.get('raw_archives', []):
            raw_fields = dict(fields)
            raw_error = archive.get('native_relative_error')
            raw_fields.update(role='raw_native_candidate', receiver_anchor=False,
                native_checked=bool(raw_error is not None and np.isfinite(raw_error)),
                native_relative_error=float(raw_error) if raw_error is not None else np.nan,
                quality_flags=int(archive.get('quality', {}).get('flags', 0)),
                has_los=bool(np.any(archive['compact']['depth']==0)))
            raw_golden = archive.get('native')
            preserve = bool(golden_failure_count<8 and raw_golden is not None and np.isfinite(raw_golden).all())
            raw_index = writer.append_raw_candidate(archive['compact'], raw_fields, archive['raw'],
                archive['normalization_report'], native_golden=raw_golden if preserve else None)
            golden_failure_count += int(preserve)
            raw_record_indices.append(raw_index)
            raw_source_attempts.append(dict(raw_record_index=raw_index,
                solver_attempt=int(archive['solver_attempt']),
                raw_path_count=len(archive['compact']),
                selected_path_count=int(archive['normalization_report']['selected_path_count'])))
        for attempt in result.get('attempts', [])[:-1]:
            retry_count += 1
            bad = dict(fields)
            ae = attempt.get('native_relative_error')
            bad.update(role='quarantine_attempt', quality_flags=int(attempt['quality']['flags']),
                       native_relative_error=float(ae) if ae is not None else np.nan,
                       native_checked=bool(ae is not None and np.isfinite(ae)))
            failure_flags[str(bad['quality_flags'])] += 1
            compact = attempt.get('compact')
            if compact is not None:
                bad['has_los'] = bool(np.any(compact['depth'] == 0))
                gh = attempt.get('native')
                keep = (golden_failure_count < 8 and gh is not None and np.isfinite(gh).all())
                bad_index = writer.append_snapshot(compact, bad, native_golden=gh if keep else None)
                golden_failure_count += int(keep)
            else:
                bad_index = -1
            writer.append_event('failed_attempt', dict(record_index=bad_index, at_time_sec=float(t),
                pass_index=pass_index, attempt={k:v for k,v in attempt.items() if k not in ('compact','native','raw','raw_archives','normalization_report')},
                native_sha256=hashlib.sha256(np.asarray(attempt['native']).tobytes()).hexdigest() if 'native' in attempt else None))
        return raw_record_indices, raw_source_attempts

    def trace(t, pass_index, role, *, position=None, velocity=None, pose=None,
              survey_design=-1, keep_golden=False):
        nonlocal retry_count, golden_failure_count
        stop_file = getattr(context, 'stop_file', None)
        if stop_file is not None and Path(stop_file).exists():
            raise RuntimeError('Controlled stop requested at a complete trace boundary; checkpoint retained')
        key = (int(pass_index), _key(t))
        if key in cache:
            rec = cache[key]
            if rec['index'] not in native_cache:
                # Re-trace an evicted/resumed reference instead of inventing a
                # dense native target from the compact label being checked.
                result = context.trace(float(t), position=position, velocity=velocity, pose=pose)
                durations.append(float(result.get('solve_seconds', 0.))+float(result.get('normalization_seconds',0.)))
                revisit_fields = dict(result['state_fields'], time_sec=float(t), pass_index=int(pass_index),
                    role='reference_revisit', survey_design=int(survey_design), receiver_anchor=False,
                    receiver_validated_until_sec=float(t))
                raw_ids, raw_attempts = archive_evidence(result, revisit_fields, t, pass_index)
                h = result['native']
                old = reconstruct(rec['compact'], rotate_local_array(config['tx_array_local_m'], rec['fields']['tx_pose_rad']),
                                  rotate_local_array(config['rx_array_local_m'], rec['fields']['rx_pose_rad']))
                norm = float(np.linalg.norm(h))
                relative = float(np.linalg.norm(old-h)/norm) if np.isfinite(norm) and norm>0 else float('nan')
                fresh_error = result.get('native_relative_error')
                admitted = bool(result['quality'].get('admitted',False) and not result['quality']['flags']
                    and np.isfinite(h).all() and fresh_error is not None and np.isfinite(fresh_error)
                    and 0<=fresh_error<config['native_relative_error_tolerance'])
                agrees = bool(admitted and np.isfinite(relative) and relative<=2e-4)
                writer.append_event('native_reference_revisit', dict(primary_record_index=rec['index'],
                    at_time_sec=float(t), pass_index=int(pass_index), role='revisit_comparison_only',
                    raw_candidate_record_indices=raw_ids, archive_solver_attempts=raw_attempts,
                    final_solver_attempt=len(result['attempts'])-1, actual_solver_calls=int(result['actual_solver_calls']),
                    final_quality=result['quality'], native_relative_error=fresh_error,
                    compact_vs_fresh_reference_relative_error=relative, comparison_passed=agrees,
                    lineage_scope='Comparison with the original stored physical state; these new raw candidates are not its generation lineage'))
                account_native_call(rec['index'], result, revisit=True)
                if not agrees:
                    raise ValueError('Fresh reference revisit is unadmitted or disagrees with the stored native-checked record; evidence retained')
                remember_native(rec['index'], h)
            return rec, native_cache[rec['index']]
        result = context.trace(float(t), position=position, velocity=velocity, pose=pose)
        durations.append(float(result.get('solve_seconds', 0.))+float(result.get('normalization_seconds',0.)))
        fields = dict(result['state_fields'])
        # Caller owns the causal/archive role, never the native solver.
        fields.update(time_sec=float(t), pass_index=int(pass_index), role=role,
                      survey_design=int(survey_design), receiver_anchor=False,
                      receiver_validated_until_sec=float(t))
        quality = result['quality']; error = result.get('native_relative_error')
        fields.update(quality_flags=int(quality['flags']), native_relative_error=float(error) if error is not None else np.nan,
                      native_checked=bool(error is not None and np.isfinite(error)))
        raw_record_indices, raw_source_attempts = archive_evidence(result, fields, t, pass_index)
        h = result['native']
        golden = h if keep_golden and np.isfinite(h).all() else None
        index = writer.append_snapshot(result['compact'], fields, native_golden=golden)
        if raw_record_indices:
            final_attempt = len(result['attempts'])-1
            final_raw = next((r['raw_record_index'] for r in reversed(raw_source_attempts)
                              if r['solver_attempt']==final_attempt),None)
            writer.append_event('raw_to_selected_native_reference', dict(
                primary_record_index=index, raw_candidate_record_indices=raw_record_indices,
                archive_solver_attempts=raw_source_attempts,final_solver_attempt=final_attempt,
                selected_native_source_raw_record_index=final_raw,
                final_raw_equals_selected_without_subsetting=(final_raw is None),
                actual_solver_calls=int(result.get('actual_solver_calls',len(result.get('attempts',[])) or 1)),
                reference_kind='native.cfr of independently masked Paths copy after full-topology image certificate'))
        account_native_call(index, result)
        rec = dict(index=index, compact=result['compact'], fields=fields)
        cache[key] = rec; remember_native(index, h)
        final_flags[str(fields['quality_flags'])] += 1
        if fields['quality_flags']:
            writer.append_event('unadmitted_snapshot', dict(record_index=index, time_sec=float(t), pass_index=pass_index,
                quality=quality, raw=result.get('raw'), native_relative_error=error))
        progress('tracing')
        return rec, h

    try:
        context.set_environment(0)
        # Four independently sampled historical designs. Captures are static;
        # timestamps are logical prior-day acquisition times, not a drive.
        for design in range(4):
            for j in range(128):
                rng = np.random.default_rng(seed_for(route['route_family_id'], 'survey-position', design, j))
                fraction = (j+rng.uniform(.1, .9))/128
                survey_route = dict(route)
                if route.get('ue_motion_mode') == 'stationary_event':
                    survey_route['ue_motion_mode'] = 'constant_velocity'
                p, _, ppose, _ = trajectory_at(survey_route, fraction*route['duration_sec'])
                lateral = np.array([-np.sin(ppose[0]), np.cos(ppose[0]), 0.])
                p = p + rng.uniform(-.2, .2)*lateral
                t = -86400.-design*1000+j*.01
                prior_capture = cache.get((-1-design, _key(t)))
                if prior_capture is not None and prior_capture['index'] in survey_done:
                    continue
                rec, native = trace(t, -1-design, 'historical_survey', position=p, velocity=np.zeros(3),
                                    pose=ppose, survey_design=design, keep_golden=j == 0)
                if rec['index'] in survey_done:
                    continue
                if rec['fields']['quality_flags']:
                    # Preserve all 128 designed captures and record their
                    # failures. An unadmitted capture never becomes a map
                    # feature; it is not silently replaced by an easy point.
                    continue
                nseed = seed_for(route['route_family_id'], 'survey-noise', design, j)
                nrng = np.random.default_rng(nseed)
                variance = float(np.mean(abs(native)**2))*10**(-25/10)
                measured = native+np.sqrt(variance/2)*(nrng.standard_normal(native.shape)+1j*nrng.standard_normal(native.shape))
                reported = p+nrng.normal(0, .25, 3)
                writer.append_survey_feature(rec['index'], power_features(measured), design_index=design,
                    noise_seed=nseed, reported_xyz_m=reported, noise_variance=variance,
                    measured_cfr_power=float(np.mean(abs(measured)**2)))
                survey_done.add(rec['index'])
                if (j+1) % 32 == 0:
                    writer.flush(metadata=dict(stage='survey', design=design, survey_done=j+1))
            progress('survey', force=True)
        writer.flush(metadata=dict(stage='survey-complete'))
        survey_counts = np.bincount(writer.handle['survey_index']['design_index'], minlength=4)
        if np.any(survey_counts < 96):
            raise ValueError('A historical design has fewer than 96 admitted of 128 planned captures; retain diagnostic shard')

        duration = float(route['duration_sec'])
        cutoffs = np.arange(4, int(np.floor((duration-.002)*1000+1e-8))+1, dtype=np.int64)*.001
        targets = cutoffs+.002
        actual_dynamic = config.get('dynamic_motion') or {}
        max_speed = max(float(route.get('speed_mps', 0.)),
                        float(np.linalg.norm(actual_dynamic.get('velocity_mps', [0,0,0]))), 1.)
        nominal_n = int(math.ceil(duration*max_speed/anchor_step_m))
        golden_times = np.linspace(0, duration, 5)
        base_times = np.unique(np.round(np.r_[np.linspace(0, duration, nominal_n+1), cutoffs, targets, golden_times], 12))
        # A target can share a directly traced base record; its native check is
        # still independent of any history-to-target extrapolation.
        golden_keys = {_key(t) for t in golden_times}
        for pass_index in (0, 1):
            context.set_environment(pass_index)

            def online(t, role='online_anchor'):
                return trace(t, pass_index, role, keep_golden=_key(t) in golden_keys)

            def check_interval(left, right, level):
                ident = (pass_index, _key(left), _key(right), level)
                old = completed_intervals.get(ident)
                if old is not None:
                    if not old['passed'] and not old['used_as_leaf']:
                        mid = (left+right)/2
                        check_interval(left, mid, level+1); check_interval(mid, right, level+1)
                    return
                a, ha = online(left); c, hc = online(right)
                mid = (left+right)/2
                b, hb = online(mid, 'online_midpoint')
                ptx = rotate_local_array(config['tx_array_local_m'], a['fields']['tx_pose_rad'])
                prx = rotate_local_array(config['rx_array_local_m'], a['fields']['rx_pose_rad'])
                me = _db_error(reconstruct(a['compact'], ptx, prx, delta_t=mid-left), hb)
                re = _db_error(reconstruct(a['compact'], ptx, prx, delta_t=right-left), hc)
                good = all(not r['fields']['quality_flags'] for r in (a, b, c))
                passed = bool(good and np.isfinite(me) and np.isfinite(re) and max(me, re) <= -30.)
                row = dict(pass_index=pass_index, left_index=a['index'], midpoint_index=b['index'], right_index=c['index'],
                           left_time_sec=float(left), right_time_sec=float(right), midpoint_nmse_db=me,
                           right_nmse_db=re, level=level, passed=passed, used_as_leaf=passed or level >= max_refinement)
                writer.append_interval(row); intervals_by_pass[pass_index].append(row); completed_intervals[ident] = row
                # Only actually checked intervals license receiver propagation.
                for rec in (a, c):
                    rec['fields']['receiver_anchor'] = True
                    writer.update_snapshot(rec['index'], receiver_anchor=True)
                if passed:
                    bound = max(float(a['fields']['receiver_validated_until_sec']), float(right))
                    a['fields']['receiver_validated_until_sec'] = bound
                    writer.update_snapshot(a['index'], receiver_validated_until_sec=bound)
                elif level < max_refinement:
                    check_interval(left, mid, level+1); check_interval(mid, right, level+1)

            for j, (left, right) in enumerate(zip(base_times[:-1], base_times[1:])):
                check_interval(float(left), float(right), 0)
                if (j+1) % 64 == 0:
                    writer.flush(metadata=dict(stage='online', pass_index=pass_index, interval_done=j+1))
                    progress('online', force=True)
            writer.flush(metadata=dict(stage='task-enumeration', pass_index=pass_index))
            coverage = _coverage(intervals_by_pass[pass_index])
            fixed_eval = set(map(int, np.unique(np.rint(np.linspace(0, len(cutoffs)-1, min(160, len(cutoffs)))).astype(int))))
            for j, (cutoff, target) in enumerate(zip(cutoffs, targets)):
                if (pass_index, _key(cutoff)) in existing_tasks:
                    continue
                rec = cache[(pass_index, _key(target))]
                flags = 0
                if rec['fields']['quality_flags'] or not rec['fields']['native_checked']:
                    flags |= 1
                if not len(coverage):
                    flags |= 2
                else:
                    at = np.searchsorted(coverage[:, 0], cutoff-.004+1e-12, side='right')-1
                    if at < 0 or coverage[at, 1] < cutoff-1e-12:
                        flags |= 2
                writer.append_task(pass_index=pass_index, cutoff_sec=float(cutoff), history_start_sec=float(cutoff-.004),
                    history_end_sec=float(cutoff), target_time_sec=float(target), target_record_index=rec['index'],
                    noise_seed=seed_for(route['route_family_id'], 'receiver-noise', j),
                    state_noise_seed=seed_for(route['route_family_id'], 'reported-state'),
                    eval_selected=j in fixed_eval, admitted=flags == 0, flags=flags)
            writer.flush(metadata=dict(stage='pass-complete', pass_index=pass_index))
            progress('pass-complete', force=True)
        result = writer.finish()
        import h5py
        with h5py.File(path, 'r') as finished:
            final_states = finished['states'][:]
            quarantine = final_states['role'] == b'quarantine_attempt'
            raw_candidates = final_states['role'] == b'raw_native_candidate'
            primary = ~(quarantine | raw_candidates)
            result['raw_candidate_records'] = int(np.count_nonzero(raw_candidates))
            result['physical_snapshot_records'] = int(np.count_nonzero(primary))
            result['rejected_physical_states'] = int(np.count_nonzero(final_states['quality_flags'][primary]))
            result['actual_solver_calls_this_invocation'] = int(getattr(context,'route_solver_calls',0))
            accounting = finished['trace_accounting'][:]
            result['committed_solver_calls'] = int(accounting['actual_solver_calls'].sum())
            result['pre_prune_candidate_count_max'] = int(accounting['final_candidate_count'].max(initial=0))
            result['pre_prune_capacity_reached_calls'] = int(np.count_nonzero(accounting['final_candidate_count']>=accounting['candidate_capacity']))
            result['native_reference_kind'] = 'analytically certified unique specular branches; masked native.cfr; raw candidates archived'
            retry_count = int(np.count_nonzero(quarantine))
            final_flags = Counter(map(str, map(int, final_states['quality_flags'][primary])))
            failure_flags = Counter(map(str, map(int, final_states['quality_flags'][quarantine])))
        result.update(route_family_id=route['route_family_id'], city=route['city'], split=route['split'],
                      intended_mechanism=route['intended_mechanism'], elapsed_sec=time.perf_counter()-started,
                      retries=retry_count, final_quality_flags=dict(final_flags),
                      failed_attempt_flags=dict(failure_flags),
                      timing_scope='this worker invocation; totals and retries reconstructed from full committed shard',
                      state_time_p50_sec=float(np.median(durations)) if durations else None,
                      state_time_p95_sec=float(np.quantile(durations, .95)) if durations else None)
        save_json(path.with_suffix('.report.json'), result)
        if progress_path: save_json(progress_path, dict(stage='complete', **result))
        return result
    except BaseException:
        if not writer._closed:
            # A half-recorded trace or interrupted joint flush must roll back
            # to its prior committed checkpoint. Do not mask the original
            # failure with a second, correctly rejected flush.
            unresolved = (getattr(writer,'_requires_resume',False)
                          or bool(getattr(writer,'_pending_accounting_indices',()))
                          or bool(getattr(writer,'_awaiting_raw_accounting',False)))
            if not unresolved:
                writer.flush(metadata=dict(stage='exception-retained'))
                progress('exception-retained', force=True)
            writer.close()
        raise

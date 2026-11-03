"""Causal, on-demand algorithm inputs for compact route shards.

The synthesizer has private simulator access. Its returned public dictionaries
are allowlisted and contain neither paths, target labels, true CFO nor noise RNG
seeds. Evaluation labels use a separate method. Dense H/Y is not cached on disk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .cities import FC, reconstruct
from .dataset_observations import _bursts
from .mimo_observation import simulate_schedule
from .path_quality import require_target_admitted


MAP_SIZES = (4, 8, 16, 32, 64)


def _seed(*parts):
    data = json.dumps(parts, ensure_ascii=False, separators=(',', ':'), default=str)
    return int.from_bytes(hashlib.sha256(data.encode('utf-8')).digest()[:8], 'little')


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def rotate_local_array(local, pose):
    """Sionna orientation convention: Rz(alpha) Ry(beta) Rx(gamma)."""
    a, b, g = np.asarray(pose, float)
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rx = np.array([[1, 0, 0], [0, cg, -sg], [0, sg, cg]])
    return np.asarray(local, float) @ (rz @ ry @ rx).T


class DevsetView:
    """Read one completed route shard without importing Sionna or Mitsuba.

    Each call reads only the selected paths and historical features, then
    recreates the declared full-FFT observation. Public inputs are repeatable
    across workers and Python processes. ``draw`` changes receive noise only.
    """

    def __init__(self, route_file):
        self.path = Path(route_file)
        with h5py.File(self.path, 'r') as f:
            if _text(f.attrs.get('schema', '')) != 'kap_devset.route.v1':
                raise ValueError('Undeclared route schema')
            if not bool(f.attrs.get('generation_complete', False)):
                raise ValueError('Route shard has no successful completion marker')
            raw = _text(f['config_json'][()])
            if hashlib.sha256(raw.encode('utf-8')).hexdigest() != _text(f.attrs['config_sha256']):
                raise ValueError('Route config hash does not match its manifest')
            self.config = json.loads(raw)
            self.route = self.config.get('route', self.config)
            self.frequencies_hz = f['frequencies_hz'][:]
        self.family = str(self.route.get('family_id', self.route.get('route_family_id', self.path.stem)))
        self.tx_local = self._array('tx')
        self.rx_local = self._array('rx')

    def _array(self, side):
        value = self.config.get(side + '_array_local_m', self.config.get(side + '_relative_local_m'))
        array = np.asarray(value, float)
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError('Missing finite local array geometry: ' + side)
        return array

    def _task(self, f, task_index):
        if isinstance(task_index, str):
            matches = np.flatnonzero(f['tasks']['task_id'].astype('U') == task_index)
            if len(matches) != 1:
                raise KeyError(task_index)
            i = int(matches[0])
        else:
            i = int(task_index)
            if i < 0 or i >= len(f['tasks']):
                raise IndexError(i)
        task = f['tasks'][i]
        if ('admitted' in task.dtype.names and not bool(task['admitted'])) or int(task['flags']):
            raise ValueError('Task is retained for diagnosis and is not admitted')
        if not task['history_start_sec'] < task['history_end_sec'] <= task['cutoff_sec'] < task['target_time_sec']:
            raise ValueError('Invalid task time ordering')
        return i, task

    def task_indices(self, *, eval_only=False):
        with h5py.File(self.path, 'r') as f:
            tasks = f['tasks'][:]
        keep = tasks['flags'] == 0
        if 'admitted' in tasks.dtype.names:
            keep &= tasks['admitted']
        if eval_only:
            keep &= tasks['eval_selected']
        return np.flatnonzero(keep)

    def _snapshot(self, f, state_index, row=None):
        i = int(state_index)
        row = f['states'][i] if row is None else row
        error = float(row['native_relative_error'])
        tolerance = float(self.config.get('native_relative_error_tolerance', 1e-4))
        if not row['native_checked'] or not np.isfinite(error) or not 0 <= error < tolerance:
            raise ValueError('Requested RT state lacks a passing independent native check')
        lo, hi = f['offsets'][i:i + 2]
        return dict(record_index=i, time_sec=float(row['time_sec']),
                    compact=f['paths'][int(lo):int(hi)],
                    tx_relative_m=rotate_local_array(self.tx_local, row['tx_pose_rad']),
                    rx_relative_m=rotate_local_array(self.rx_local, row['rx_pose_rad']),
                    quality_flags=int(row['quality_flags']),
                    validated_until_sec=float(row['receiver_validated_until_sec']))

    def _public_task(self, index, task):
        task_id = _text(task['task_id']).rstrip('\x00') if 'task_id' in task.dtype.names else ''
        if not task_id:
            task_id = hashlib.sha256(f'{self.family}/{index}'.encode()).hexdigest()[:24]
        public_pass = hashlib.sha256(f'{self.family}/{int(task["pass_index"])}'.encode()).hexdigest()[:20]
        return dict(task_id=task_id, pass_id=public_pass, route_family_id=self.family,
                    split=self.route.get('split', self.config.get('split', 'unspecified')),
                    city=self.route.get('city', self.config.get('city', 'unspecified')),
                    history_start_sec=float(task['history_start_sec']),
                    input_cutoff_sec=float(task['cutoff_sec']),
                    target_time_sec=float(task['target_time_sec']),
                    target_region_map_available=True, map_versions=list(MAP_SIZES))

    def _reports(self, task):
        from .devset_motion import motion_at
        step = float(self.config.get('report_interval_sec', .001))
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Invalid fixed device-report clock')
        first = int(np.ceil((float(task['history_start_sec']) - 1e-12) / step))
        last = int(np.floor((float(task['cutoff_sec']) + 1e-12) / step))
        clocks = np.arange(first, last + 1, dtype=np.int64)
        if not len(clocks):
            raise ValueError('No reported state in this task history')
        times = clocks * step
        truth = [motion_at(self.route, float(t)) for t in times]
        positions, velocities, poses = [np.asarray([x[j] for x in truth], float) for j in range(3)]
        stds = [float(self.config.get(key, default)) for key, default in
                (('position_report_std_m', .25), ('velocity_report_std_mps', .5), ('pose_report_std_rad', .01))]
        if not np.isfinite(stds).all() or min(stds) < 0:
            raise ValueError('Invalid reported-state uncertainty')
        # The random stream is attached to the fixed physical report clock;
        # overlapping windows and the environment pair see the same reports.
        base = int(task['state_noise_seed'])
        noise = np.stack([np.random.default_rng(_seed(self.family, base, int(q), 'reported-state')).standard_normal((3, 3))
                          for q in clocks])
        return dict(report_time_sec=times,
                    reported_position_m=positions + stds[0] * noise[:, 0],
                    reported_velocity_mps=velocities + stds[1] * noise[:, 1],
                    reported_pose_rad=poses + stds[2] * noise[:, 2],
                    position_covariance_m2=np.eye(3) * stds[0] ** 2,
                    velocity_covariance_m2s2=np.eye(3) * stds[1] ** 2,
                    pose_covariance_rad2=np.eye(3) * stds[2] ** 2)

    def _map(self, f, query, mapn, design):
        rows = f['survey_index'][:]
        eligible = np.flatnonzero(rows['design_index'] == design)
        if len(eligible) < mapn:
            raise ValueError('Historical survey design has fewer samples than requested')
        historical = f['states'][:]
        refs = rows['record_index'][eligible]
        if (np.any(refs < 0) or np.any(refs >= len(historical))
                or np.any(historical['time_sec'][refs] >= 0)
                or np.any(historical['quality_flags'][refs] != 0)
                or np.any(historical['pass_index'][refs] >= 0)
                or np.any(~historical['native_checked'][refs])
                or np.any(~np.isfinite(historical['native_relative_error'][refs]))
                or np.any(historical['native_relative_error'][refs] < 0)
                or np.any(historical['native_relative_error'][refs] >= float(self.config.get('native_relative_error_tolerance', 1e-4)))):
            raise ValueError('Map source contains an unadmitted or online state')
        distances = np.linalg.norm(rows['reported_xyz_m'][eligible] - query, axis=1)
        order = np.argsort(distances, kind='stable')[:mapn]
        ids = eligible[order]
        # h5py fancy indices must be increasing; preserve kNN order explicitly.
        sorted_ids = np.sort(ids)
        features = f['survey_features'][sorted_ids][np.searchsorted(sorted_ids, ids)].astype(float)
        if not np.isfinite(features).all() or np.any(features < 0):
            raise ValueError('Map contains nonfinite or negative measured powers')
        d = distances[order]
        bandwidth = float(self.config.get('map_bandwidth_m', 5.0))
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError('Invalid historical map bandwidth')
        logs = -.5 * (d / bandwidth) ** 2
        weights = np.exp(logs - logs.max()); weights /= weights.sum()
        mean = weights @ features
        correction = 1 - float(weights @ weights)
        # A degenerately local empirical distribution has no estimable sample
        # variance. Retain that condition explicitly instead of NaN or infinity.
        estimable = correction > 1e-12
        variance = (weights @ ((features - mean) ** 2)) / correction if estimable else np.zeros_like(mean)
        power = rows['measured_cfr_power'][eligible]
        if not np.isfinite(power).all() or np.any(power < 0) or not np.mean(power) > 0:
            raise ValueError('Invalid independent historical receiver-power calibration')
        prior = dict(power_mean=mean.astype(np.float32), power_empirical_variance=variance.astype(np.float32),
                     effective_sample_count=1 / float(weights @ weights), sample_count=int(mapn),
                     variance_estimable=bool(estimable), survey_sample_ids=ids,
                     survey_query_distance_m=d, nearest_survey_distance_m=float(d.min()),
                     farthest_survey_distance_m=float(d.max()), reported_query_position_m=np.asarray(query),
                     survey_design_index=int(design),
                     uncertainty_scope='weighted finite-survey empirical variance; not a calibrated posterior')
        return prior, float(np.mean(power))

    def _condition(self, task_index, mapn, design):
        if mapn not in MAP_SIZES or int(design) != design or design < 0:
            raise ValueError('Undeclared map size or survey design')
        with h5py.File(self.path, 'r') as f:
            index, task = self._task(f, task_index)
            public = self._public_task(index, task)
            reports = self._reports(task)
            prior, calibration = self._map(f, reports['reported_position_m'][-1], mapn, design)
        return dict(task=public, reported_states=reports, map=prior), calibration

    def load_condition(self, task_index, mapn=16, design=0):
        """Prior-model inputs without paying for receiver waveform synthesis."""
        condition, _ = self._condition(task_index, mapn, design)
        return condition

    def load(self, task_index, mapn=16, design=0, beam='rotating', snr=10.0, draw=0):
        if beam not in ('fixed', 'rotating'):
            raise ValueError('Undeclared beam mode')
        if not np.isfinite(snr) or int(draw) != draw or draw < 0:
            raise ValueError('Invalid SNR or noise draw')
        condition, calibration = self._condition(task_index, mapn, design)
        public = condition['task']
        with h5py.File(self.path, 'r') as f:
            index, task = self._task(f, task_index)
            states = f['states'][:]
            candidates = np.flatnonzero((states['pass_index'] == task['pass_index']) & states['receiver_anchor'])
            candidates = candidates[np.argsort(states['time_sec'][candidates], kind='stable')]
            times = states['time_sec'][candidates]
            before = np.searchsorted(times, public['history_start_sec'], side='right') - 1
            after = np.searchsorted(times, public['input_cutoff_sec'], side='right')
            if before < 0:
                raise ValueError('No causal receiver anchor before task history')
            selected = candidates[before:after]
            snapshots = [self._snapshot(f, i, states[i]) for i in selected]
        schedule = _bursts(len(self.tx_local), public, beam)
        cfo_std = float(self.config.get('cfo_prior_std_hz', 500.0))
        if not np.isfinite(cfo_std) or cfo_std < 0:
            raise ValueError('Invalid common CFO prior')
        cfo = float(np.random.default_rng(_seed(self.family, self.config.get('cfo_seed', 20261027), 'cfo')).normal(0, cfo_std))
        noise_variance = calibration * 10 ** (-float(snr) / 10)
        simulation = simulate_schedule(snapshots, schedule, cfo_hz=cfo, noise_variance=noise_variance,
                                       noise_seed=_seed(int(task['noise_seed']), int(draw), 'receiver-noise'),
                                       max_local_age_sec=float(self.config.get('max_local_age_sec', .01)))
        received = simulation['receive_time_sec']
        if received.min() < public['history_start_sec'] - 1e-12 or received.max() > public['input_cutoff_sec'] + 1e-12:
            raise ValueError('Synthesized receiver extends outside causal task history')
        ledger = simulation['ledger'].copy()
        for key in ('noise_seed', 'cfo_hz', 'local_anchor_indices_used', 'max_local_age_sec',
                    'minimum_absolute_delay_sec', 'maximum_absolute_delay_sec', 'cp_covers_maximum_absolute_delay'):
            ledger.pop(key, None)
        observation = {key: simulation[key] for key in ('full_fft', 'pilot_y', 'pilot_mask', 'receive_time_sec')}
        observation.update({key: schedule[key] for key in ('symbols', 'tx_beams', 'start_sample', 'burst_id')})
        for key in ('full_fft', 'pilot_y', 'symbols', 'tx_beams'):
            observation[key] = np.asarray(observation[key], np.complex64)
        observation.update(noise_variance=np.array([noise_variance]), cfo_is_known=np.array([False]),
                           cfo_prior_std_hz=np.array([cfo_std]), fs_hz=np.array([schedule['fs_hz']]),
                           nfft=np.array([schedule['nfft']]), cp_samples=np.array([schedule['cp_samples']]),
                           resource_ledger=ledger)
        if observation['full_fft'].shape != (57, len(self.rx_local), 256) or not np.isfinite(observation['full_fft']).all():
            raise ValueError('Full receiver FFT is nonfinite or has unexpected axes')
        return dict(**condition, observation=observation)

    def load_evaluation(self, task_index, frequencies_hz=None):
        """Private label view: target is a directly traced, native-checked state."""
        with h5py.File(self.path, 'r') as f:
            index, task = self._task(f, task_index)
            target_index = int(task['target_record_index'])
            if target_index < 0 or target_index >= len(f['states']):
                raise ValueError('Target does not reference an RT state')
            row = f['states'][target_index]
            snapshot = self._snapshot(f, target_index, row)
            require_target_admitted(snapshot)
            tolerance = float(self.config.get('native_relative_error_tolerance', 1e-4))
            if (not row['native_checked'] or not np.isfinite(row['native_relative_error'])
                    or not 0 <= row['native_relative_error'] < tolerance
                    or int(row['pass_index']) != int(task['pass_index'])
                    or abs(float(row['time_sec']) - float(task['target_time_sec'])) > 1e-10):
                raise ValueError('Target lacks an independent native RT check at the declared time')
            frequencies = self.frequencies_hz if frequencies_hz is None else np.asarray(frequencies_hz, float)
            if frequencies.ndim != 1 or not np.isfinite(frequencies).all():
                raise ValueError('Invalid evaluation frequency grid')
            target = reconstruct(snapshot['compact'], snapshot['tx_relative_m'], snapshot['rx_relative_m'], frequencies)
            result = dict(task=self._public_task(index, task), target_csi=target,
                          target_time_sec=float(task['target_time_sec']), target_record_index=target_index,
                          target_frequencies_hz=frequencies, native_relative_error=float(row['native_relative_error']),
                          validation_source='independent RT state and persisted native reconstruction check')
            cached = np.flatnonzero(f['golden_cfr/record_index'][:] == target_index)
            if len(cached):
                g = int(cached[0])
                result['stored_native_golden_csi'] = f['golden_cfr/real'][g] + 1j * f['golden_cfr/imag'][g]
                result['stored_native_golden_frequencies_hz'] = self.frequencies_hz.copy()
            return result

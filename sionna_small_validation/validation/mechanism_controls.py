"""Finite-hypothesis, full-receiver identifiability controls and input audits.

These are mechanism diagnostics, not a trained estimator or novelty claim. The
Gaussian likelihood uses every available receive antenna/FFT component, including
ICI; only its exact sufficient statistic is used for scalar Bayes quadrature.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import h5py
import numpy as np

from .cities import COMPACT_DTYPE, C, FC, reconstruct
from .common import save_json, sha256


def audit_input_contract(tasks, survey_records, split_records):
    """Audit normalized sidecar records without reading any private truth.

    Callers translate their storage schema to these small dictionaries. A target
    time is task metadata, whereas future CSI/path/state is private evaluation
    data. Historical target-region surveys are legal in the declared map protocol.
    """
    errors = []
    surveys = {str(s['survey_id']): s for s in survey_records}
    if len(surveys) != len(survey_records):
        errors.append('duplicate survey_id')
    for key in ('route_family_id', 'environment_instance_id', 'physical_region_id'):
        owners = {}
        for row in split_records:
            value = row.get(key)
            if value is None:
                continue
            owners.setdefault(str(value), set()).add(str(row['split']))
        for value, splits in owners.items():
            if len(splits) > 1:
                errors.append(f'{key}={value} crosses splits: {sorted(splits)}')
    task_ids = set()
    for task in tasks:
        tid = str(task['task_id'])
        if tid in task_ids:
            errors.append(f'{tid}: duplicate task_id')
        task_ids.add(tid)
        cutoff = float(task['input_cutoff_sec'])
        start = float(task['history_start_sec'])
        target = float(task['target_time_sec'])
        if not start <= cutoff <= target:
            errors.append(f'{tid}: invalid history/cutoff/target order')
        for ob in task.get('observations', []):
            time = float(ob['time_sec'])
            if not start - 1e-12 <= time <= cutoff + 1e-12:
                errors.append(f'{tid}: observation outside legal history')
            if ob.get('source_kind') != 'online_pilot':
                errors.append(f'{tid}: non-pilot observation source')
        for sid in task.get('map_survey_ids', []):
            survey = surveys.get(str(sid))
            if survey is None:
                errors.append(f'{tid}: unknown survey_id {sid}')
                continue
            if survey.get('source_kind') != 'historical_survey':
                errors.append(f'{tid}: map depends on online/future truth')
            if float(survey['survey_time_sec']) >= start:
                errors.append(f'{tid}: survey is not earlier than online window')
    return {'passed': not errors, 'errors': errors, 'tasks': len(tasks),
            'surveys': len(survey_records), 'split_records': len(split_records),
            'scope': 'normalized metadata contract; not a security sandbox'}


def strict_estimator_view(view, allowed_keys):
    """Reject undeclared top-level fields and explicit private-label names."""
    errors = []
    extra = set(view) - set(allowed_keys)
    if extra:
        errors.append('undeclared estimator fields: ' + ', '.join(sorted(extra)))
    private = {'truth', 'oracle', 'paths', 'compact', 'target_csi', 'future_csi',
               'true_position_m', 'true_velocity_mps', 'native_golden',
               'visibility_labels', 'path_association', 'scene_change_labels',
               'noiseless_full_fft', 'true_cfo_hz', 'target_record_index'}

    def visit(value, prefix=''):
        if isinstance(value, dict):
            for key, item in value.items():
                path = f'{prefix}.{key}' if prefix else key
                if key.lower() in private:
                    errors.append('private evaluation field in estimator view: ' + path)
                visit(item, path)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(item, f'{prefix}[{i}]')
    visit(view)
    return {'passed': not errors, 'errors': errors}


def linear_target_identifiability(G, B, relative_tolerance=1e-10):
    """Numerically test ker(G) subset ker(B), for the stated finite dictionary."""
    G = np.asarray(G, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    if G.ndim != 2 or B.ndim != 2 or G.shape[1] != B.shape[1]:
        raise ValueError('G and B need matching parameter dimension')
    _, singular, vh = np.linalg.svd(G, full_matrices=True)
    threshold = relative_tolerance * (singular[0] if len(singular) else 1.)
    rank = int(np.count_nonzero(singular > threshold))
    null = vh[rank:].conj().T
    residual = float(np.linalg.norm(B @ null)) if null.shape[1] else 0.
    scale = max(float(np.linalg.norm(B)), np.finfo(float).tiny)
    return {'observation_rank': rank, 'parameter_dimension': G.shape[1],
            'nullity': int(null.shape[1]), 'target_null_residual': residual,
            'target_null_relative': residual / scale,
            'target_unique_in_noise_free_dictionary': bool(residual <= relative_tolerance * scale),
            'singular_values': singular.tolist(),
            'scope': 'finite, known short-window dictionary; basic linear algebra, not new theory'}


def binary_bayes_diagnostics(mu0, mu1, target0, target1, noise_variance, points=96):
    """Equal-prior CN(mu,sigma2 I) Bayes risk using Gauss-Hermite quadrature.

    Conditional log-likelihood ratio is N(+/-d_obs,2*d_obs). This reduction
    exactly retains all information for these two known means/covariances.
    Within-hypothesis uncertainty is zero here; model/map uncertainty is not
    magically covered by the reported risk.
    """
    if noise_variance <= 0:
        raise ValueError('positive complex-noise variance required')
    delta = np.asarray(mu1) - np.asarray(mu0)
    d_obs = float(np.vdot(delta.ravel(), delta.ravel()).real / noise_variance)
    dh = np.asarray(target1) - np.asarray(target0)
    d_target = float(np.vdot(dh.ravel(), dh.ravel()).real)
    target_scale = .5 * (float(np.vdot(np.ravel(target0), np.ravel(target0)).real)
                         + float(np.vdot(np.ravel(target1), np.ravel(target1)).real))
    if d_obs <= 20:
        z, w = np.polynomial.hermite.hermgauss(points)
        llr = d_obs + 2 * np.sqrt(d_obs) * z
        q = np.exp(-np.abs(llr))
        factor = float(np.dot(w, q / (1+q)**2) / np.sqrt(np.pi))
        log_factor = math.log(factor)
    else:
        # Direct conditional quadrature misses exponentially rare ambiguous
        # observations at large d. Integrate around LLR=0 under the mixture:
        # E[p0*p1] = exp(-d/4)/(2*sqrt(4*pi*d)) * integral_0^inf
        # exp(-l*l/(4*d))/cosh(l/2) dl. Stable also below exp underflow.
        from scipy.integrate import quad
        integral = quad(lambda llr: math.exp(-llr*llr/(4*d_obs)) *
                        (2*math.exp(-llr/2)/(1+math.exp(-llr))), 0, np.inf,
                        epsabs=1e-12, epsrel=1e-12)[0]
        log_factor = -d_obs/4 - math.log(2*math.sqrt(4*np.pi*d_obs)) + math.log(integral)
        factor = math.exp(log_factor) if log_factor > -745 else 0.
    risk = factor * d_target
    log_risk = log_factor + math.log(d_target) if d_target > 0 else None
    return {'d_obs_whitened': d_obs, 'target_squared_distance': d_target,
            'target_relative_squared_distance': d_target / target_scale if target_scale else None,
            'equal_prior_optimal_binary_error': .5 * math.erfc(np.sqrt(d_obs) / 2),
            'prior_target_risk': .25 * d_target,
            'expected_posterior_target_risk': risk,
            'log_expected_posterior_target_risk': log_risk,
            'normalized_expected_target_risk': risk / target_scale if target_scale else None,
            'noise_complex_variance': float(noise_variance),
            'likelihood': 'proper complex Gaussian; identical sigma2*I for both hypotheses',
            'hypothesis_prior': [.5, .5], 'quadrature_points': points}


def synthetic_snapshot(two_paths=True, second_gain=1., first_doppler=0., second_doppler=0.):
    paths = np.zeros(2 if two_paths else 1, dtype=COMPACT_DTYPE)
    paths['b00'] = [1., second_gain] if two_paths else [1.]
    paths['delay_sec'] = [23.7e-9, 71.3e-9] if two_paths else [23.7e-9]
    paths['doppler_hz'] = [first_doppler, second_doppler] if two_paths else [first_doppler]
    paths['k_tx'][0] = [1., 0., 0.]
    paths['k_rx'][0] = [1., 0., 0.]
    if two_paths:
        paths['k_tx'][1] = [0., 1., 0.]
        paths['k_rx'][1] = [0., 1., 0.]
    paths['flags'] = 1
    paths['object_ids'].fill(np.iinfo(np.uint64).max)
    half = C / FC / 2
    return {'time_sec': 0., 'compact': paths,
            'tx_relative_m': np.array([[0., 0., 0.], [0., half, 0.]]),
            'rx_relative_m': np.array([[0., 0., 0.], [0., half, 0.]])}


def target(snapshot, time_sec, frequencies=None):
    if frequencies is None:
        frequencies = np.fft.fftfreq(256, 1 / 30.72e6)
    return reconstruct(snapshot['compact'], snapshot['tx_relative_m'],
                       snapshot['rx_relative_m'], frequencies=frequencies,
                       delta_t=time_sec - snapshot['time_sec'])


def _receiver(snapshot, schedule, cfo_hz=0.):
    from .mimo_observation import simulate_schedule
    result = simulate_schedule([snapshot], schedule, carrier_hz=FC,
                               cfo_hz=cfo_hz, noise_variance=0., max_local_age_sec=.01)
    return np.asarray(result['full_fft']), result


def run_contract_controls():
    survey = [{'survey_id': 's0', 'survey_time_sec': -10.,
               'source_kind': 'historical_survey', 'scene_version': 'historic'}]
    task = {'task_id': 't0', 'input_cutoff_sec': .004, 'target_time_sec': .006,
            'history_start_sec': 0., 'map_survey_ids': ['s0'],
            'observations': [{'time_sec': .003, 'source_kind': 'online_pilot'}]}
    splits = [{'route_family_id': 'r0', 'environment_instance_id': 'e0',
               'physical_region_id': 'p0', 'split': 'train'}]
    legal = audit_input_contract([task], survey, splits)
    future = copy.deepcopy(task); future['observations'][0]['time_sec'] = .005
    wrong_source = copy.deepcopy(survey); wrong_source[0]['source_kind'] = 'online_truth'
    overlap = splits + [dict(splits[0], split='test')]
    negatives = {
        'future_observation': audit_input_contract([future], survey, splits),
        'map_from_online_truth': audit_input_contract([task], wrong_source, splits),
        'same_region_route_across_split': audit_input_contract([task], survey, overlap),
        'private_truth_in_public_view': strict_estimator_view(
            {'target_time_sec': .006, 'features': {'paths': np.array([1.])}},
            ['target_time_sec', 'features'])}
    return {'passed': legal['passed'] and all(not v['passed'] for v in negatives.values()),
            'legal_fixture': legal, 'negative_controls': negatives}


def verify_bayes_risk_with_full_noise(mu0, mu1, target0, target1, noise_variance,
                                     samples=12000, seed=20261027):
    """Independent full-vector likelihood/noise check of the scalar risk rule."""
    rng = np.random.default_rng(seed)
    a, b = np.ravel(mu0), np.ravel(mu1)
    truth = rng.integers(0, 2, samples)
    posterior_variances, squared_errors = [], []
    target_distance = float(np.linalg.norm(np.asarray(target1)-target0)**2)
    for start in range(0, samples, 128):
        index = truth[start:start+128]
        noise = (rng.standard_normal((len(index), len(a))) +
                 1j*rng.standard_normal((len(index), len(a)))) * np.sqrt(noise_variance/2)
        y = np.where(index[:, None], b, a) + noise
        llr = (np.sum(abs(y-a)**2, axis=1)-np.sum(abs(y-b)**2, axis=1))/noise_variance
        p = 1/(1+np.exp(-np.clip(llr, -700, 700)))
        posterior_variances.extend(p*(1-p)*target_distance)
        squared_errors.extend(np.where(index, (1-p)**2, p**2)*target_distance)
    variance = np.asarray(posterior_variances); error = np.asarray(squared_errors)
    analytic = binary_bayes_diagnostics(a, b, target0, target1, noise_variance)['expected_posterior_target_risk']
    se_var = float(variance.std(ddof=1)/np.sqrt(samples)); se_error = float(error.std(ddof=1)/np.sqrt(samples))
    # Six standard errors is a conservative reproducible numerical check,
    # not an empirical-coverage guarantee for an unknown channel prior.
    passed = abs(float(variance.mean())-analytic) < 6*se_var and abs(float(error.mean())-analytic) < 6*se_error
    return {'passed': bool(passed), 'samples': samples, 'seed': seed,
            'analytic_bayes_risk': analytic, 'full_vector_mean_posterior_variance': float(variance.mean()),
            'full_vector_mean_squared_error': float(error.mean()),
            'posterior_variance_standard_error': se_var, 'squared_error_standard_error': se_error,
            'scope': 'known equal-prior finite-hypothesis Gaussian calibration only'}


def run_mechanism_controls(out_dir, archives=()):
    from .mimo_observation import make_schedule, schedule_ledger
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    schedule = make_schedule(2, n_symbols=2, nfft=256, fs_hz=30.72e6,
                             cp_samples=64, start_time_sec=0., beam_mode='fixed', seed=20261027)
    fixed = copy.deepcopy(schedule)
    fixed['tx_beams'][:] = np.array([1., 1.]) / np.sqrt(2)
    rotated = copy.deepcopy(fixed)
    rotated['tx_beams'][1] = np.array([1., -1.]) / np.sqrt(2)
    a, b = synthetic_snapshot(second_gain=1.), synthetic_snapshot(second_gain=-1.)
    target_time = .002
    h0, h1 = target(a, target_time), target(b, target_time)
    mu0, _ = _receiver(a, fixed); mu1, _ = _receiver(b, fixed)
    rot0, _ = _receiver(a, rotated); rot1, _ = _receiver(b, rotated)
    noise_var = .2
    null_fixed = binary_bayes_diagnostics(mu0, mu1, h0, h1, noise_var)
    null_rotated = binary_bayes_diagnostics(rot0, rot1, h0, h1, noise_var)
    ledger_fixed, ledger_rotated = schedule_ledger(fixed), schedule_ledger(rotated)
    # Dictionary columns use the same physical channel/waveform receiver and
    # target, so the numerical kernel test matches the stated measurement.
    dictionaries = {}
    for name, action in [('fixed', fixed), ('rotated', rotated)]:
        columns_y, columns_h = [], []
        for k in range(2):
            unit = copy.deepcopy(a); unit['compact']['b00'][:] = 0
            unit['compact']['b00'][k] = 1
            y, _ = _receiver(unit, action)
            columns_y.append(y.ravel()); columns_h.append(target(unit, target_time).ravel())
        dictionaries[name] = linear_target_identifiability(np.column_stack(columns_y), np.column_stack(columns_h))
    # Instantaneous sample alias is deliberately a negative control: the actual
    # complete OFDM receiver retains intra-symbol time evolution and can break it.
    alias_schedule = copy.deepcopy(fixed)
    alias_schedule['start_sample'][1] = int(round(.001*alias_schedule['fs_hz']))
    starts = np.asarray(alias_schedule['start_sample'], dtype=np.int64)
    spacing = float((starts[1] - starts[0]) / alias_schedule['fs_hz'])
    alias_a = synthetic_snapshot(False, first_doppler=0.)
    alias_b = synthetic_snapshot(False, first_doppler=1 / spacing)
    alias0, _ = _receiver(alias_a, alias_schedule); alias1, _ = _receiver(alias_b, alias_schedule)
    alias_target_time = .00225
    alias_diag = binary_bayes_diagnostics(alias0, alias1, target(alias_a, alias_target_time),
                                           target(alias_b, alias_target_time), noise_var)
    instant_error = float(np.max(np.abs(np.exp(2j*np.pi/spacing * starts/alias_schedule['fs_hz']) - 1)))
    near_a = synthetic_snapshot(False, first_doppler=-50.)
    near_b = synthetic_snapshot(False, first_doppler=50.)
    near0, _ = _receiver(near_a, fixed); near1, _ = _receiver(near_b, fixed)
    near_diag = binary_bayes_diagnostics(near0, near1, target(near_a, target_time),
                                          target(near_b, target_time), noise_var)
    # Same two symbols, same waveform, same energy, but move the second symbol
    # later. Common target/cutoff accounts for the additional waiting latency.
    spread = copy.deepcopy(fixed)
    spread['start_sample'][1] = int(round(.0008 * spread['fs_hz']))
    spread0, _ = _receiver(near_a, spread); spread1, _ = _receiver(near_b, spread)
    spread_diag = binary_bayes_diagnostics(spread0, spread1, target(near_a, target_time),
                                            target(near_b, target_time), noise_var)
    bayes_check = verify_bayes_risk_with_full_noise(near0, near1, target(near_a, target_time),
                                                   target(near_b, target_time), noise_var)
    real = screen_real_archives(archives, out_dir)
    contracts = run_contract_controls()
    checks = {
        'fixed_beam_observation_null_preserved_full_receiver': null_fixed['d_obs_whitened'] < 1e-20,
        'different_target_in_observation_null': null_fixed['target_relative_squared_distance'] > .5,
        'rotated_beam_reduces_expected_target_risk': null_rotated['expected_posterior_target_risk'] < null_fixed['expected_posterior_target_risk'],
        'beam_comparison_identical_tx_energy_and_duration': abs(ledger_fixed['total_tx_energy']-ledger_rotated['total_tx_energy']) < 1e-10 and ledger_fixed['elapsed_sec']==ledger_rotated['elapsed_sec'],
        'kernel_condition_fails_for_fixed': not dictionaries['fixed']['target_unique_in_noise_free_dictionary'],
        'kernel_condition_holds_for_rotated': dictionaries['rotated']['target_unique_in_noise_free_dictionary'],
        'instantaneous_alias_fixture_correct': instant_error < 1e-10,
        'full_receiver_does_not_share_instantaneous_alias': alias_diag['d_obs_whitened'] > 1.,
        'later_same_energy_symbol_reduces_known_doppler_pair_risk': spread_diag['expected_posterior_target_risk'] < near_diag['expected_posterior_target_risk'],
        'bayes_risk_independent_full_noise_check': bayes_check['passed'],
        'contract_positive_and_negative_controls': contracts['passed']}
    result = {'status': 'passed' if all(checks.values()) else 'needs_attention',
              'scope': 'diagnostic fixtures and unfiltered RT archive screen, not algorithm advantage or novelty',
              'checks': checks, 'full_receiver_axes': 'symbol,receive_antenna,all_fft_bins',
              'noise_normalization': 'fixed proper-complex sigma2=.2 per full FFT component in controls',
              'target_time_sec': target_time,
              'spatial_null_control': {'fixed': null_fixed, 'rotated_same_resource': null_rotated,
                                      'ledgers': {'fixed': ledger_fixed, 'rotated': ledger_rotated},
                                      'dictionary_conditions': dictionaries},
              'instantaneous_doppler_alias_negative_control': {
                  'spacing_sec': spacing, 'doppler_difference_hz': 1/spacing,
                  'instantaneous_phase_error': instant_error, 'full_receiver': alias_diag,
                  'target_time_sec': alias_target_time, 'ledger': schedule_ledger(alias_schedule),
                  'interpretation': 'do not apply instantaneous-sampling alias formula to full OFDM receiver'},
              'near_doppler_finite_hypothesis_control': {
                  'doppler_hypotheses_hz': [-50., 50.], 'consecutive': near_diag,
                  'spread_same_energy': spread_diag, 'spread_ledger': schedule_ledger(spread),
                  'common_input_cutoff_sec': .0009,
                  'selection_scope': 'oracle-known two-hypothesis diagnostic; not an implementable learned map claim',
                  'caution': 'binary hypothesis entropy and target variance rankings need not demonstrate distinct target-aware design novelty'},
              'input_contract_controls': contracts, 'real_geometry_screen': real,
              'bayes_full_vector_noise_verification': bayes_check,
              'limitations': ['No trained flow or SBL comparison.',
                             'Known finite hypotheses omit within-branch/map-parameter uncertainty.',
                             'Synthetic 2 ms target uses explicitly declared locally frozen path parameters.',
                             'Real screen pairs are diagnostic and not a prevalence estimate for new independent routes.']}
    np.savez_compressed(out_dir/'full_receiver_control_arrays.npz', fixed0=mu0, fixed1=mu1,
                        rotated0=rot0, rotated1=rot1, target0=h0, target1=h1,
                        alias0=alias0, alias1=alias1, near0=near0, near1=near1,
                        spread0=spread0, spread1=spread1)
    save_json(out_dir/'mechanism_report.json', result)
    return result


def screen_real_archives(archives, out_dir):
    """Keep every matched static/dynamic pair; flag screened cases separately.

    One short observed symbol is propagated from its actual RT anchor. The
    target uses the corresponding record itself (current CSI), avoiding an
    unsupported multi-ms extrapolated RT target. This tests the representation
    and observation-likelihood diagnostic, not future-CSI generalization.
    """
    from .mimo_observation import make_schedule
    rows, archive_info = [], []
    for archive in archives:
        archive = Path(archive)
        with h5py.File(archive, 'r') as f:
            records = f['paths'][:]; offsets = f['offsets'][:]
            if all(k in f for k in ('state', 'fragment', 'time_sec')):
                states = f['state'].asstr()[:]; fragments = f['fragment'].asstr()[:]
                pair_semantics = 'legacy static versus van geometry'
            elif all(k in f for k in ('scene_version', 'route_family_id', 'role', 'time_sec')):
                versions = f['scene_version'].asstr()[:]
                roles = f['role'].asstr()[:]
                states = np.where(roles=='historical_survey', 'survey', np.where(versions=='base', 'static', 'dynamic'))
                fragments = f['route_family_id'].asstr()[:]
                pair_semantics = 'same route/time base material versus recorded alternative material policy'
            else:
                archive_info.append({'path': str(archive), 'sha256': sha256(archive), 'skipped': 'needs state/fragment or version/route/role pairing'})
                continue
            times = f['time_sec'][:]
            lookup = {(str(fr), float(t), str(st)): i for i, (fr, t, st) in enumerate(zip(fragments, times, states))}
            count = 0
            for (fr, tm, st), i in sorted(lookup.items()):
                if st != 'static' or (fr, tm, 'dynamic') not in lookup:
                    continue
                j = lookup[(fr, tm, 'dynamic')]
                if f.attrs.get('quality_schema'):
                    qualified=all(int(f['quality_flags'][ix])==0 and bool(f['receiver_anchor'][ix])
                        and float(f['receiver_validated_until_sec'][ix])>=float(times[ix])+320/30.72e6
                        for ix in (i,j))
                    if not qualified:
                        rows.append({'archive':str(archive),'fragment':fr,'time_sec':tm,
                            'eligible':False,'reason':'RT probe or unadmitted interval retained; not used as receiver truth'})
                        continue
                snapshots = []
                for ix in (i, j):
                    snapshots.append({'time_sec': float(times[ix]),
                        'compact': records[offsets[ix]:offsets[ix+1]],
                        'tx_relative_m': f['tx_relative_m'][ix], 'rx_relative_m': f['rx_relative_m'][ix]})
                if any(not len(s['compact']) for s in snapshots):
                    rows.append({'archive': str(archive), 'fragment': fr, 'time_sec': tm,
                                 'eligible': False, 'reason': 'empty RT candidate retained, normalized risk undefined'})
                    continue
                schedule = make_schedule(len(snapshots[0]['tx_relative_m']), n_symbols=1, nfft=256,
                    fs_hz=30.72e6, cp_samples=64, start_time_sec=tm, beam_mode='fixed', seed=20261027)
                y0, _ = _receiver(snapshots[0], schedule); y1, _ = _receiver(snapshots[1], schedule)
                h0 = target(snapshots[0], tm); h1 = target(snapshots[1], tm)
                # Fix aggregate reference received SNR at 10 dB. This variance
                # is shared by both members, never fitted separately to each.
                noise = max(float(np.vdot(y0.ravel(), y0.ravel()).real) / 10, np.finfo(float).tiny)
                diag = binary_bayes_diagnostics(y0, y1, h0, h1, noise)
                rows.append({'archive': str(archive), 'fragment': fr, 'time_sec': tm,
                    'eligible': True, 'snapshot_indices': [i, j], 'diagnostics': diag,
                    'near_ambiguous_target_pair': diag['d_obs_whitened'] <= 1 and diag['target_relative_squared_distance'] >= .5})
                count += 1
            archive_info.append({'path': str(archive), 'sha256': sha256(archive), 'eligible_pairs': count,
                                 'pair_semantics': pair_semantics})
    save_json(Path(out_dir)/'real_geometry_unfiltered_pairs.json', rows)
    eligible = [r for r in rows if r['eligible']]
    screened = [r for r in eligible if r['near_ambiguous_target_pair']]
    save_json(Path(out_dir)/'real_geometry_screened_pairs.json', screened)
    return {'archives': archive_info, 'unfiltered_pairs': len(rows), 'eligible_pairs': len(eligible),
            'empty_pairs_retained':sum('empty RT' in r.get('reason','') for r in rows),
            'ineligible_pairs_retained':len(rows)-len(eligible),'screened_near_ambiguous_pairs': len(screened),
            'thresholds_predeclared': {'d_obs_max': 1., 'target_relative_squared_distance_min': .5,
                                     'aggregate_reference_received_snr_linear': 10.},
            'target': 'same-anchor current full MIMO CFR; not an extrapolated future target',
            'all_pairs_file': str((Path(out_dir)/'real_geometry_unfiltered_pairs.json').resolve()),
            'interpretation': 'paired fixtures only; no claim of real-world harmful-ambiguity frequency'}


def audit_v2_products(dataset_root, out_dir):
    """Independently recompute map statistics and inspect actual legal views."""
    import json
    from .dataset_v2 import EstimatorView, power_features, read_complex, SURVEY_SIZES
    rows = []
    family_split = {}
    for directory in sorted(Path(dataset_root).iterdir()):
        if not directory.is_dir() or not (directory/'maps.h5').exists():
            continue
        errors, measurements = [], {}
        tasks = json.loads((directory/'tasks.json').read_text(encoding='utf-8'))['tasks']
        with h5py.File(directory/'survey.h5', 'r') as f:
            measured = read_complex(f['measured_dense_cfr'])
            saved = f['power_features'][:]
            positions = f['reported_position_m'][:]
            survey_times = f['survey_time_sec'][:]
            recomputed = np.stack([power_features(h) for h in measured])
            feature_rel = float(np.linalg.norm(recomputed-saved)/max(np.linalg.norm(saved), 1e-100))
            measurements['feature_recompute_relative_error'] = feature_rel
            if feature_rel > 2e-6:
                errors.append('historical power features do not match actual noisy measurement')
            if not max(survey_times) < min(t['history_start_sec'] for t in tasks):
                errors.append('survey not strictly historical')
        with h5py.File(directory/'maps.h5', 'r') as f:
            queries = f['reported_query_position_m'][:]
            maxmean, maxvar = 0., 0.
            previous = np.array([], dtype=int)
            for size in SURVEY_SIZES:
                group = f['n'+str(size)]
                ids = group['survey_sample_ids'][:]
                if not np.array_equal(ids[:len(previous)], previous):
                    errors.append('finite survey maps not nested')
                previous = ids
                for q, query in enumerate(queries):
                    distance = np.linalg.norm(positions[ids]-query, axis=1)
                    weight = np.exp(-.5*(distance/5)**2); weight /= weight.sum()
                    mean = weight @ recomputed[ids]
                    var = (weight @ ((recomputed[ids]-mean)**2))/(1-float(weight@weight))
                    a = float(np.linalg.norm(mean-group['power_mean'][q])/max(np.linalg.norm(mean), 1e-100))
                    b = float(np.linalg.norm(var-group['power_empirical_variance'][q])/max(np.linalg.norm(var), 1e-100))
                    maxmean, maxvar = max(a, maxmean), max(b, maxvar)
                    if not np.isfinite(var).all() or np.min(var) < 0:
                        errors.append('nonfinite/negative finite-map variance')
            measurements.update(map_mean_recompute_max_relative_error=maxmean,
                                map_variance_recompute_max_relative_error=maxvar)
            if maxmean > 2e-6 or maxvar > 2e-6:
                errors.append('map mean/variance differs from survey-only recomputation')
        report_times = []
        for task in tasks:
            family_split.setdefault(task['route_family_id'], set()).add(task['split'])
            view = EstimatorView(directory).load(task['task_id'])
            public_task = view['task']
            # Bookkeeping in tasks.json may be evaluation metadata, but it must
            # not be exposed by the estimator loader as a scene-change oracle.
            if any(k in public_task for k in ('pairing_role', 'pairing_group', 'scene_version')):
                errors.append(task['task_id']+': scene-change/pairing label in estimator task')
            strings = [v for v in public_task.values() if isinstance(v, str)]
            if any('walls_wood' in text or 'material_mismatch' in text for text in strings):
                errors.append(task['task_id']+': condition encoded in estimator string identifier')
            flat = strict_estimator_view(view, ['task', 'reported_states', 'map', 'observation'])
            errors.extend(flat['errors'])
            times = view['reported_states']['report_time_sec']
            report_times.append(times)
            if np.any(times > task['input_cutoff_sec']+1e-12) or np.any(times < task['history_start_sec']-1e-12):
                errors.append(task['task_id']+': reported states outside legal history')
        if len(report_times)==2 and not np.array_equal(report_times[0], report_times[1]):
            errors.append('matched/mismatch report schedule depends on private propagation truth')
        measurements['reported_counts'] = [len(t) for t in report_times]
        rows.append({'city': directory.name, 'passed': not errors, 'errors': errors,
                     'independent_recomputation': measurements})
    crosses = {k: sorted(v) for k, v in family_split.items() if len(v)>1}
    result = {'passed': bool(rows) and all(r['passed'] for r in rows) and not crosses,
              'cities': rows, 'cross_split_route_families': crosses,
              'scope': 'independent survey-statistic recomputation and actual estimator-view audit; no model training'}
    save_json(Path(out_dir)/'independent_v2_contract_audit.json', result)
    return result


def audit_main_receiver_pairs(dataset_root, out_dir):
    """Use the generated main receiver's complete Y, covariance and CSI target.

    The two-mean Bayes quantities condition on the captured common CFO. A
    separate minimum-distance CFO profile reports nuisance sensitivity and is
    not presented as a marginalized receiver error or a proven global minimum.
    """
    import json
    from scipy.optimize import minimize_scalar
    from .dataset_v2 import read_complex
    rows, errors = [], []
    for directory in sorted(Path(dataset_root).iterdir()):
        if not directory.is_dir() or not (directory/'evaluation_observations.h5').exists():
            continue
        mapping = json.loads((directory/'evaluation_index.json').read_text(encoding='utf8'))['task_truth_mapping']
        tasks = {t['task_id']: t for t in json.loads((directory/'tasks.json').read_text(encoding='utf8'))['tasks']}
        matched = [k for k, v in mapping.items() if v['pairing_role']=='matched']
        changed = [k for k, v in mapping.items() if v['pairing_role']=='material_mismatch']
        if len(matched)!=1 or len(changed)!=1:
            errors.append(directory.name+': expected one unfiltered matched/mismatch pair')
            continue
        a, b = matched[0], changed[0]
        with h5py.File(directory/'observations.h5', 'r') as public, h5py.File(directory/'evaluation_observations.h5', 'r') as private:
            target0, target1 = read_complex(private[a]['target_csi']), read_complex(private[b]['target_csi'])
            cfo0, cfo1 = float(private[a]['true_cfo_hz'][()]), float(private[b]['true_cfo_hz'][()])
            for mode in ('fixed', 'rotating'):
                p0, p1 = public[a][mode], public[b][mode]
                mu0 = np.asarray(read_complex(private[a][mode]['noiseless_full_fft']), np.complex128)
                mu1 = np.asarray(read_complex(private[b][mode]['noiseless_full_fft']), np.complex128)
                noise0, noise1 = float(p0['noise_variance'][0]), float(p1['noise_variance'][0])
                if noise0 != noise1:
                    errors.append(directory.name+'/'+mode+': candidate noise covariance differs')
                    continue
                if not np.array_equal(p0['receive_time_sec'][:], p1['receive_time_sec'][:]):
                    errors.append(directory.name+'/'+mode+': unequal candidate receive schedules')
                diag = binary_bayes_diagnostics(mu0, mu1, target0, target1, noise0)
                time = p0['receive_time_sec'][:]
                y0 = np.fft.ifft(mu0, axis=-1, norm='ortho')
                y1 = np.fft.ifft(mu1, axis=-1, norm='ortho')
                cross = np.sum(np.conj(y0)*y1, axis=1)
                energy = float(np.vdot(y0.ravel(),y0.ravel()).real+np.vdot(y1.ravel(),y1.ravel()).real)
                def distance(delta):
                    value = (energy-2*np.real(np.sum(cross*np.exp(2j*np.pi*delta*time))))/noise0
                    return max(float(value), 0.)
                grid = np.linspace(-2000., 2000., 201)
                values = np.array([distance(d) for d in grid])
                i = int(values.argmin())
                candidates = [(grid[i], values[i])]
                if 0<i<len(grid)-1:
                    fit = minimize_scalar(distance, bounds=(grid[i-1],grid[i+1]), method='bounded',
                                          options={'xatol': 1e-7})
                    candidates.append((float(fit.x), float(fit.fun)))
                delta, profiled = min(candidates, key=lambda pair: pair[1])
                ledger = json.loads(p0.attrs['ledger_json'])
                near = diag['d_obs_whitened'] <= 1 and diag['target_relative_squared_distance'] >= .5
                rows.append({'city': directory.name, 'pair_task_ids': [a,b], 'beam_mode': mode,
                    'full_receiver_shape': list(mu0.shape), 'common_cfo_hz': cfo0 if cfo0==cfo1 else None,
                    'input_cutoff_sec': tasks[a]['input_cutoff_sec'], 'target_time_sec': tasks[a]['target_time_sec'],
                    'diagnostics_conditional_on_captured_cfo': diag,
                    'cfo_nuisance_profile': {'min_d_obs_on_grid_then_local_fit': profiled,
                        'delta_cfo_hz': delta, 'search_range_hz': [-2000.,2000.], 'grid_step_hz': 20.,
                        'interpretation': 'profiled pair distance, not prior-marginal Bayes risk or a certified global minimum'},
                    'resource_ledger': {k:ledger[k] for k in ('n_symbols','occupied_resource_elements','total_tx_energy',
                                            'start_time_sec','end_time_sec','elapsed_sec')},
                    'near_ambiguous_target_pair_at_predeclared_threshold': bool(near)})
    result = {'passed': bool(rows) and not errors, 'errors': errors, 'unfiltered_city_pairs': len(rows)//2,
        'receiver_protocol_rows': len(rows), 'rows': rows,
        'near_ambiguous_rows': sum(r['near_ambiguous_target_pair_at_predeclared_threshold'] for r in rows),
        'thresholds_predeclared': {'d_obs_max':1., 'target_relative_squared_distance_min':.5},
        'scope': 'all generated main-protocol material pairs retained; three short correlated-route cases, not prevalence/generalization',
        'cfo_caution': 'Bayes error is conditional on known candidate CFO; actual receiver treats CFO as unknown; separate nuisance profile is diagnostic only'}
    save_json(Path(out_dir)/'main_receiver_unfiltered_pairs.json', result)
    return result

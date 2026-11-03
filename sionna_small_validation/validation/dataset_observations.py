"""Materialize causal online receiver views from separately archived RT truth.

This generator has evaluator access; the resulting observations.h5 deliberately
contains neither path records, target CSI, true CFO nor random noise seeds.
The independently stored evaluation_observations.h5 owns those labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .cities import FC, reconstruct
from .common import save_json, sha256
from .mimo_observation import make_schedule, simulate_schedule, schedule_ledger
from .path_quality import require_target_admitted


def _read_complex(group):
    return group['real'][:]+1j*group['imag'][:]


def _write_complex(group,name,data,axes):
    data=np.asarray(data,np.complex64)
    child=group.create_group(name);child.attrs['axes']=axes
    child.create_dataset('real',data=data.real,compression='gzip',compression_opts=1)
    child.create_dataset('imag',data=data.imag,compression='gzip',compression_opts=1)


def _snapshots(path,pass_id):
    with h5py.File(path,'r') as f:
        keep=np.flatnonzero(f['pass_id'].asstr()[:]==pass_id)
        if 'receiver_anchor' in f:
            keep=keep[f['receiver_anchor'][:][keep].astype(bool)]
        keep=keep[np.argsort(f['time_sec'][:][keep])]
        offsets=f['offsets'][:]
        records=[dict(record_index=int(i),time_sec=float(f['time_sec'][i]),
                  compact=f['paths'][offsets[i]:offsets[i+1]],
                  tx_relative_m=f['tx_relative_m'][i],rx_relative_m=f['rx_relative_m'][i]) for i in keep]
        if f.attrs.get('quality_schema'):
            if not all(name in f for name in ('quality_flags','receiver_validated_until_sec')):
                raise ValueError('Quality-tagged archive is missing required admission fields')
            for rec in records:
                i=rec['record_index'];rec['quality_flags']=int(f['quality_flags'][i])
                rec['validated_until_sec']=float(f['receiver_validated_until_sec'][i])
    return records


def _bursts(n_tx,task,mode):
    ss=make_schedule(n_tx,n_symbols=57,nfft=256,cp_samples=64,
                     start_time_sec=task['history_start_sec'],beam_mode=mode,seed=20261027)
    # 4ms history contains three 19-beam bursts.  Positions depend solely on
    # declared history/cutoff, never on online truth or target difficulty.
    fs=ss['fs_hz'];burst_length=19*320/fs
    first=task['history_start_sec'];last=task['input_cutoff_sec']-burst_length
    burst_starts=np.rint(np.linspace(first,last,3)*fs).astype(np.int64)
    ss['start_sample']=np.concatenate([q+np.arange(19)*320 for q in burst_starts])
    ss['burst_id']=np.repeat(np.arange(3),19)
    return ss


def _frozen_burst_model(snaps,ss):
    """Diagnostic frozen H at each burst start, ignoring within-burst ICI/CFO."""
    freqs=np.fft.fftfreq(ss['nfft'],1/ss['fs_hz'])
    output=np.zeros((len(ss['symbols']),len(snaps[0]['rx_relative_m']),ss['nfft']),complex)
    times=np.array([s['time_sec'] for s in snaps])
    for burst in range(3):
        rows=np.flatnonzero(ss['burst_id']==burst)
        t=ss['start_sample'][rows[0]]/ss['fs_hz']
        index=np.searchsorted(times,t,side='right')-1;s=snaps[index]
        h=reconstruct(s['compact'],s['tx_relative_m'],s['rx_relative_m'],freqs,delta_t=t-s['time_sec'])
        output[rows]=np.einsum('rtf,st,sf->srf',h,ss['tx_beams'][rows],ss['symbols'][rows])
    return output


def generate_observations(dataset_root, nominal_snr_db=10.0):
    root=Path(dataset_root);cities=[]
    for ci,city in enumerate(('munich','florence','etoile')):
        directory=root/city
        tasks_path=directory/'tasks.json'
        if not tasks_path.exists():
            cities.append(dict(city=city,status='missing_dataset'));continue
        tasks=json.loads(tasks_path.read_text(encoding='utf-8'))['tasks']
        with h5py.File(directory/'truth.h5','r') as truth:
            if truth.attrs.get('quality_schema')!='snapshot_flags_and_checked_intervals.v1':
                raise ValueError('Receiver generation requires admitted v2 truth; legacy archives remain readable for audit')
        evaluation_index=json.loads((directory/'evaluation_index.json').read_text(encoding='utf-8'))
        task_truth=evaluation_index.get('task_truth_mapping',{})
        generation=json.loads((directory/'generation_report.json').read_text(encoding='utf-8'))
        with h5py.File(directory/'survey.h5','r') as f:
            survey=_read_complex(f['measured_dense_cfr'])
        calibration_power=float(np.mean(abs(survey)**2))
        noise_variance=calibration_power*10**(-nominal_snr_db/10)
        # Same CFO and noise realization across the material pair and beam
        # modes: controls change propagation or beams rather than nuisance data.
        cfo_hz=float(np.random.default_rng(20261027+ci).normal(0,500))
        noise_seed=30261027+ci
        observation_path=directory/'observations.h5'
        evaluation_path=directory/'evaluation_observations.h5'
        if observation_path.exists() or evaluation_path.exists():
            raise FileExistsError('Preserve existing observations; choose a new dataset directory')
        records=[]
        with h5py.File(observation_path.with_suffix('.h5.partial'),'x') as obs, h5py.File(evaluation_path.with_suffix('.h5.partial'),'x') as evaluation:
            obs.attrs.update(schema='kap_dataset_v2.receiver_observations.v2',generation_complete=False,
                access='estimator_input; all antenna FFT bins are available including ICI',
                cfo_status='unknown oscillator CFO; prior std500Hz, true value and noise seeds are evaluation-only',
                noise_calibration='mean measured historical dense-CFR power divided by nominal SNR; shared across material and beam pairs',
                nominal_snr_db=float(nominal_snr_db),
                limitations='generic CP-OFDM ideal DAC; not an NR synchronizer or decoder')
            evaluation.attrs.update(schema='kap_dataset_v2.observation_evaluation.v2',access='evaluation_only',generation_complete=False)
            for task in tasks:
                truth_pass_id=task_truth.get(task['task_id'],{}).get('truth_pass_id',task['pass_id'])
                snaps=_snapshots(directory/'truth.h5',truth_pass_id)
                n_tx=len(snaps[0]['tx_relative_m'])
                og=obs.create_group(task['task_id']);eg=evaluation.create_group(task['task_id'])
                eg.create_dataset('true_cfo_hz',data=cfo_hz);eg.create_dataset('noise_seed',data=noise_seed)
                target_time=task['target_time_sec'];times=np.array([s['time_sec'] for s in snaps])
                target=snaps[int(np.argmin(abs(times-target_time)))]
                require_target_admitted(target)
                if abs(target['time_sec']-target_time)>1e-12:
                    raise ValueError('Target must have independently retraced RT truth at its declared time')
                freqs=np.linspace(-12e6,12e6,16,dtype=np.float32)
                target_h=reconstruct(target['compact'],target['tx_relative_m'],target['rx_relative_m'],freqs)
                _write_complex(eg,'target_csi',target_h,'rx,tx,frequency')
                eg.create_dataset('target_time_sec',data=target_time)
                eg.create_dataset('target_record_index',data=target['record_index'])
                eg.create_dataset('target_frequencies_hz',data=freqs)
                native_error=None
                native_validation_source=None
                with h5py.File(directory/'truth.h5','r') as f:
                    ids=f['golden_cfr/record_index'][:]
                    at=np.flatnonzero(ids==target['record_index'])
                    if len(at):
                        values=f['golden_cfr/values']
                        native=values['real'][int(at[0])]+1j*values['imag'][int(at[0])]
                        native_error=float(np.linalg.norm(target_h-native)/np.linalg.norm(native))
                        native_validation_source='stored independent native golden CFR'
                        _write_complex(eg,'target_native_csi',native,'rx,tx,frequency')
                if native_error is None:
                    raise ValueError('Admitted target requires stored native golden, not a report-only substitute')
                modes={}
                for mode in ('fixed','rotating'):
                    ss=_bursts(n_tx,task,mode)
                    result=simulate_schedule(snaps,ss,cfo_hz=cfo_hz,noise_variance=noise_variance,
                                             noise_seed=noise_seed,max_local_age_sec=.000333334)
                    ledger=result['ledger'].copy()
                    # Neither hidden oscillator state nor invertible noise RNG
                    # seed belongs in the estimator's resource description.
                    ledger.pop('noise_seed');ledger.pop('cfo_hz');ledger.pop('local_anchor_indices_used')
                    ledger.pop('max_local_age_sec')
                    ledger['model_local_age_bound_sec']=.000333334
                    ledger.pop('minimum_absolute_delay_sec');ledger.pop('maximum_absolute_delay_sec')
                    ledger.pop('cp_covers_maximum_absolute_delay')
                    g=og.create_group(mode)
                    g.attrs['ledger_json']=json.dumps(ledger)
                    g.attrs['input_contract']='all observed RX FFT bins and known pilots/beams/times; unknown CFO'
                    for name,arr,axes in (('full_fft',result['full_fft'],'symbol,rx,fft_bin'),
                                          ('pilot_y',result['pilot_y'],'symbol,rx,fft_bin'),
                                          ('symbols',ss['symbols'],'symbol,fft_bin'),
                                          ('tx_beams',ss['tx_beams'],'symbol,tx')):
                        _write_complex(g,name,arr,axes)
                    for name,value in (('start_sample',ss['start_sample']),('receive_time_sec',result['receive_time_sec']),
                                       ('pilot_mask',ss['pilot_mask']),('burst_id',ss['burst_id']),
                                       ('noise_variance',[noise_variance]),('cfo_is_known',[False]),
                                       ('cfo_prior_std_hz',[500.]),('fs_hz',[ss['fs_hz']]),
                                       ('nfft',[ss['nfft']]),('cp_samples',[ss['cp_samples']])):
                        g.create_dataset(name,data=value)
                    frozen=_frozen_burst_model(snaps,ss)
                    clean=simulate_schedule(snaps,ss,cfo_hz=0.,noise_variance=0.,max_local_age_sec=.000333334)
                    difference=float(np.linalg.norm(frozen-clean['full_fft'])/np.linalg.norm(clean['full_fft']))
                    ev=eg.create_group(mode)
                    _write_complex(ev,'noiseless_full_fft',result['noiseless_full_fft'],'symbol,rx,fft_bin')
                    ev.create_dataset('frozen_per_burst_relative_error',data=difference)
                    modes[mode]=dict(max_receive_time_sec=float(result['receive_time_sec'].max()),
                         min_receive_time_sec=float(result['receive_time_sec'].min()),
                         ledger=ledger,frozen_per_burst_relative_error=difference,
                         causal=bool(result['receive_time_sec'].max()<=task['input_cutoff_sec']+1e-12),
                         history_only=bool(result['receive_time_sec'].min()>=task['history_start_sec']-1e-12),
                         full_fft_shape=list(result['full_fft'].shape))
                same_resource=all(abs(modes['fixed']['ledger'][name]-modes['rotating']['ledger'][name])<1e-10
                        for name in ('total_tx_energy','elapsed_sec','occupied_resource_elements'))
                records.append(dict(task_id=task['task_id'],modes=modes,paired_resource_equal=same_resource,
                    target_time_sec=target_time,target_native_cfr_relative_error=native_error,
                    target_native_validation_source=native_validation_source,
                    target_native_validated=bool(native_error is not None and native_error<1e-4)))
            if not records or not all(r['paired_resource_equal'] and r['target_native_validated']
                   and all(r['modes'][m]['causal'] and r['modes'][m]['history_only'] for m in ('fixed','rotating')) for r in records):
                raise ValueError('Incomplete or invalid observation product; partial files retained without publication')
            obs.attrs['generation_complete']=True;evaluation.attrs['generation_complete']=True
        observation_path.with_suffix('.h5.partial').replace(observation_path)
        evaluation_path.with_suffix('.h5.partial').replace(evaluation_path)
        checks=dict(all_observation_times_causal=all(r['modes'][m]['causal'] and r['modes'][m]['history_only'] for r in records for m in ('fixed','rotating')),
                    fixed_rotating_same_resources=all(r['paired_resource_equal'] for r in records),
                    all_targets_native_validated=all(r['target_native_validated'] for r in records),
                    shared_pair_noise_variance=True,unknown_cfo_not_exposed=True)
        report=dict(city=city,status='passed' if all(checks.values()) else 'needs_attention',checks=checks,
                    nominal_snr_db=float(nominal_snr_db),survey_calibration_power=calibration_power,
                    common_noise_variance=noise_variance,tasks=records,
                    files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in (observation_path,evaluation_path)})
        save_json(directory/'observation_report.json',report);cities.append(report)
        print('[observations]',city,report['status'],flush=True)
    report=dict(schema='kap_dataset_v2.observation_integration.v1',cities=cities,
                status='passed' if len(cities)==3 and all(c['status']=='passed' for c in cities) else 'needs_attention',
                scope='actual time-varying fractional-delay MIMO receiver inputs, not an estimator performance benchmark')
    save_json(root/'observation_integration_report.json',report)
    return report


def audit_observations(dataset_root):
    """Audit disk products and every public loader variant, not report flags."""
    from .dataset_v2 import EstimatorView
    root=Path(dataset_root);cases=[]
    forbidden={'paths','b00','path_index','target_csi','true_cfo_hz','noise_seed','max_local_age_sec','local_anchor_indices_used',
               'pairing_role','pairing_group','scene_version','truth_pass_id'}
    opened=[];original_file=h5py.File
    def tracked_file(path,*args,**kwargs):
        opened.append(Path(path).name)
        return original_file(path,*args,**kwargs)
    def all_keys(value):
        result=set()
        if isinstance(value,dict):
            result.update(value)
            for child in value.values():result.update(all_keys(child))
        return result
    for city in ('munich','florence','etoile'):
        d=root/city;tasks=json.loads((d/'tasks.json').read_text(encoding='utf-8'))['tasks']
        checks={};variances=[];ledger_pairs=[]
        with original_file(d/'observations.h5','r') as f:
            names=[];f.visit(names.append)
            checks['no_hidden_fields_on_disk']=not any(set(name.split('/')) & forbidden for name in names)
            for task in tasks:
                modes=[]
                for mode in ('fixed','rotating'):
                    g=f[task['task_id']][mode];ledger=json.loads(g.attrs['ledger_json'])
                    checks[task['task_id']+'_'+mode+'_clean_ledger']=not bool(set(ledger)&forbidden) and 'cfo_hz' not in ledger
                    times=g['receive_time_sec'][:]
                    checks[task['task_id']+'_'+mode+'_actual_time_cutoff']=bool(times.max()<=task['input_cutoff_sec']+1e-12 and times.min()>=task['history_start_sec']-1e-12)
                    checks[task['task_id']+'_'+mode+'_full_fft_shape']=g['full_fft/real'].shape==(57,16,256)
                    checks[task['task_id']+'_'+mode+'_unknown_cfo']=not bool(g['cfo_is_known'][0])
                    checks[task['task_id']+'_'+mode+'_three_bursts']=np.array_equal(np.bincount(g['burst_id'][:]),[19,19,19])
                    variances.append(float(g['noise_variance'][0]));modes.append(ledger)
                ledger_pairs.append(modes)
        checks['all_pairs_share_noise_variance']=len(set(variances))==1
        checks['fixed_rotating_energy_resources_match']=all(all(abs(pair[0][name]-pair[1][name])<1e-10
            for name in ('elapsed_sec','total_tx_energy','occupied_resource_elements')) for pair in ledger_pairs)
        h5py.File=tracked_file
        try:
            for task in tasks:
                for size in (4,8,16,32):
                    for mode in ('fixed','rotating'):
                        view=EstimatorView(d).load(task['task_id'],survey_size=size,observation_mode=mode)
                        checks[f'public_view_{task["task_id"]}_n{size}_{mode}']=not bool(all_keys(view)&forbidden)
        finally:
            h5py.File=original_file
        cases.append(dict(city=city,status='passed' if all(checks.values()) else 'failed',checks=checks))
    allowed={'maps.h5','reported_states.h5','observations.h5'}
    checks=dict(public_loader_never_opens_truth=bool(opened and set(opened)<=allowed),
                all_city_disk_and_view_checks=all(c['status']=='passed' for c in cases))
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,cities=cases,
                public_loader_calls=48,opened_hdf5_names=sorted(set(opened)),
                limitation='Allow-list and causal data-contract audit, not a hostile process sandbox')
    save_json(root/'observation_contract_audit.json',result)
    return result

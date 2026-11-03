"""Fixed-bandwidth numerology controls on newly generated real-city RT paths.

All cases share Fs, active duration, resource-element count, emitted energy,
fixed spatial beam, oscillator CFO and survey-calibrated noise variance. This
is a physical receiver diagnostic, not an estimator or numerology ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .cities import FC, C, reconstruct
from .common import save_json, sha256
from .dataset_observations import _snapshots
from .mimo_observation import make_schedule, simulate_schedule, schedule_ledger


def _direct_sample_oracle(snaps,schedule,cfo_hz,selected):
    """Explicit continuous Fourier sums at selected RX0 receive samples.

    Uses no IFFT delay, channel reconstruction helper, or waveform engine.
    Sampling across early/middle/late symbols also covers multiple RT anchors.
    """
    fs=schedule['fs_hz'];n=schedule['nfft'];cp=schedule['cp_samples']
    starts=schedule['start_sample'];freqs=np.fft.fftfreq(n,1/fs)
    anchor_times=np.array([s['time_sec'] for s in snaps])
    values=[];anchor_indices=[]
    for symbol,index in selected:
        t=(starts[symbol]+cp+index)/fs
        ai=int(np.searchsorted(anchor_times,t,side='right')-1)
        s=snaps[ai];anchor_indices.append(ai)
        tx=np.asarray(s['tx_relative_m']);rx=np.asarray(s['rx_relative_m'])
        value=0j
        for p in s['compact']:
            delayed=t-float(p['delay_sec'])
            source=int(np.searchsorted(starts/fs,delayed,side='right')-1)
            if source<0 or delayed>=(starts[source]+cp+n)/fs:
                continue
            spatial_tx=np.exp(2j*np.pi*FC/C*((tx-tx[0])@p['k_tx']))
            # RX0 is the stored native coefficient reference, its phase is 1.
            beam_gain=spatial_tx@schedule['tx_beams'][source]
            input_value=np.sum(schedule['symbols'][source]*np.exp(
                2j*np.pi*freqs*(delayed-(starts[source]+cp)/fs)))/np.sqrt(n)
            value+=p['b00']*beam_gain*input_value*np.exp(2j*np.pi*p['doppler_hz']*(t-s['time_sec']))
        values.append(value*np.exp(2j*np.pi*cfo_hz*t))
    return np.asarray(values),sorted(set(anchor_indices))


def _relative(value,reference):
    return float(np.linalg.norm(value-reference)/max(np.linalg.norm(reference),np.finfo(float).tiny))


def _freeze_prediction(snaps,schedule):
    fs=schedule['fs_hz'];n=schedule['nfft']
    t=float(schedule['start_sample'][0]/fs)
    ai=int(np.searchsorted([s['time_sec'] for s in snaps],t,side='right')-1)
    s=snaps[ai]
    h=reconstruct(s['compact'],s['tx_relative_m'],s['rx_relative_m'],
                  np.fft.fftfreq(n,1/fs),delta_t=t-s['time_sec'])
    return np.einsum('rtf,st,sf->srf',h,schedule['tx_beams'],schedule['symbols'])


def _leakage(full_fft,tone_bin):
    total=float(np.sum(abs(full_fft)**2))
    desired=float(np.sum(abs(full_fft[:,:,tone_bin])**2))
    return float(max(0.,total-desired)/total) if total>0 else None


def run_controls(dataset_root,output):
    root=Path(dataset_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'scs_rt_report.json').exists():
        raise FileExistsError('Preserve existing control results; select a new output directory')
    cases=[];evaluation=[]
    for ci,city in enumerate(('munich','florence','etoile')):
        d=root/city
        tasks=json.loads((d/'tasks.json').read_text(encoding='utf-8'))['tasks']
        private=json.loads((d/'evaluation_index.json').read_text(encoding='utf-8'))
        mapping=private['task_truth_mapping']
        task=next(t for t in tasks if mapping[t['task_id']]['pairing_role']=='matched')
        snaps=_snapshots(d/'truth.h5',mapping[task['task_id']]['truth_pass_id'])
        with h5py.File(d/'observations.h5','r') as f:
            g=f[task['task_id']]['rotating']
            noise_variance=float(g['noise_variance'][0]);prior=float(g['cfo_prior_std_hz'][0])
        with h5py.File(d/'evaluation_observations.h5','r') as f:
            cfo_hz=float(f[task['task_id']]['true_cfo_hz'][()])
        # Equal variance across SCS, not per-case measured-SNR normalization.
        noise_seed=40261027+ci
        for n,n_symbols in ((1024,19),(512,38),(256,76)):
            scs=int(30.72e6/n);cp=n//4
            ss=make_schedule(len(snaps[0]['tx_relative_m']),n_symbols=n_symbols,nfft=n,
                             cp_samples=cp,start_time_sec=.0045,beam_mode='fixed',seed=20261027)
            # Fixed unit DC spatial beam prevents a numerology-dependent beam
            # schedule from confounding the SCS comparison.
            scale=np.sqrt(24320/schedule_ledger(ss)['total_tx_energy'])
            ss['symbols']*=scale
            result=simulate_schedule(snaps,ss,cfo_hz=cfo_hz,noise_variance=noise_variance,
                    noise_seed=noise_seed,max_local_age_sec=.000333334)
            clean=simulate_schedule(snaps,ss,cfo_hz=0.,noise_variance=0.,max_local_age_sec=.000333334)
            frozen=_freeze_prediction(snaps,ss)
            frozen_error=_relative(frozen,clean['full_fft'])
            # A common 1.2 MHz transmitted tone measures energy escaping its
            # original FFT bin, without conflating multipath selectivity with
            # unknown multi-tone QPSK symbols. No noise is added to this metric.
            tone_bin=int(round(1.2e6/(30.72e6/n)))
            tone=dict(ss);tone['symbols']=np.zeros_like(ss['symbols'])
            tone['symbols'][:,tone_bin]=np.sqrt(n)
            tone['pilot_mask']=np.abs(tone['symbols'])>0
            tone_no_cfo=simulate_schedule(snaps,tone,cfo_hz=0.,noise_variance=0.,max_local_age_sec=.000333334)
            tone_cfo=simulate_schedule(snaps,tone,cfo_hz=cfo_hz,noise_variance=0.,max_local_age_sec=.000333334)
            selected=[(s,k) for s in (0,n_symbols//2,n_symbols-1) for k in (0,1,n//4,n//2,n-2,n-1)]
            oracle,anchor_indices=_direct_sample_oracle(snaps,ss,cfo_hz,selected)
            time_samples=np.fft.ifft(result['noiseless_full_fft'],axis=-1,norm='ortho')
            observed=np.array([time_samples[s,0,k] for s,k in selected])
            oracle_error=_relative(observed,oracle)
            path=output/f'{city}_scs{scs//1000}k_receiver.npz'
            with path.with_suffix('.npz.partial').open('xb') as stream:
                np.savez_compressed(stream,full_fft=result['full_fft'].astype(np.complex64),
                     symbols=ss['symbols'].astype(np.complex64),tx_beams=ss['tx_beams'].astype(np.complex64),
                     receive_time_sec=result['receive_time_sec'],start_sample=ss['start_sample'],
                     pilot_mask=ss['pilot_mask'],fs_hz=[ss['fs_hz']],nfft=[n],cp_samples=[cp],
                     noise_variance=[noise_variance],cfo_is_known=[False],cfo_prior_std_hz=[prior])
            path.with_suffix('.npz.partial').replace(path)
            ledger=result['ledger'].copy()
            # Public physical resource record has no latent oscillator state,
            # RNG seed or adaptive truth-grid side information.
            for name in ('noise_seed','cfo_hz','local_anchor_indices_used','max_local_age_sec',
                         'minimum_absolute_delay_sec','maximum_absolute_delay_sec','cp_covers_maximum_absolute_delay'):
                ledger.pop(name,None)
            ledger['model_local_age_bound_sec']=.000333334
            flags=dict(causal=float(result['receive_time_sec'].max())<=task['input_cutoff_sec'],
                       within_history=float(result['receive_time_sec'].min())>=task['history_start_sec'],
                       oracle=oracle_error<2e-10,multiple_RT_anchors=len(anchor_indices)>=2,
                       receiver_full_fft_shape=result['full_fft'].shape==(n_symbols,16,n),
                       correct_occupied_RE=ledger['occupied_resource_elements']==19456,
                       correct_CP_inclusive_energy=abs(ledger['total_tx_energy']-24320)<1e-8,
                       correct_active_duration=abs(ledger['elapsed_sec']-19*1280/30.72e6)<1e-14)
            cases.append(dict(city=city,task_id=task['task_id'],scs_hz=scs,ledger=ledger,
                     independent_RX0_sample_oracle_relative_error=oracle_error,
                     oracle_sample_count=len(selected),oracle_uses_multiple_RT_anchors=len(anchor_indices)>=2,
                     frozen_H_without_CFO_relative_error=frozen_error,
                     common_tone_hz=1.2e6,
                     tone_outside_original_bin_fraction_no_CFO=_leakage(tone_no_cfo['full_fft'],tone_bin),
                     tone_outside_original_bin_fraction_with_unknown_CFO=_leakage(tone_cfo['full_fft'],tone_bin),
                     checks=flags,passed=all(flags.values()),
                     file=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha256(path))))
            evaluation.append(dict(city=city,scs_hz=scs,true_cfo_hz=cfo_hz,noise_seed=noise_seed,
                         matched_truth_pass_id=mapping[task['task_id']]['truth_pass_id'],
                         oracle_private_anchor_indices=anchor_indices))
            print(f'[SCS RT] {city} {scs//1000}kHz oracle={oracle_error:.3g} passed={all(flags.values())}',flush=True)
    checks=dict(nine_city_numerology_cases=len(cases)==9,all_case_checks=all(c['passed'] for c in cases),
                fixed_Fs=all(c['ledger']['fs_hz']==30.72e6 for c in cases),
                equal_duration_RE_and_emitted_energy=all(c['checks']['correct_active_duration'] and
                    c['checks']['correct_occupied_RE'] and c['checks']['correct_CP_inclusive_energy'] for c in cases),
                same_CFO_within_city=all(len({e['true_cfo_hz'] for e in evaluation if e['city']==city})==1 for city in ('munich','florence','etoile')),
                same_noise_variance_within_city=all(len({c['ledger']['noise_E_abs_n_squared'] for c in cases if c['city']==city})==1 for city in ('munich','florence','etoile')))
    report=dict(schema='kap_dataset_v2.real_rt_scs_controls.v1',status='passed' if all(checks.values()) else 'needs_attention',
                checks=checks,cases=cases,total_receiver_bytes=sum(c['file']['bytes'] for c in cases),
                scope='real-city RT local multi-anchor fractional-delay MIMO fixed-bandwidth physical controls',
                limits=['No additional ray tracing; reuses the newly generated independent dataset truth without counting new states.',
                        'Fixed DC transmit beam and full known QPSK pilot band; each SCS has different OFDM symbols and CP lengths.',
                        'Equal emitted energy, bandwidth, active duration and RE count do not imply identical information operators.',
                        'CFO is unknown to the receiver; actual CFO and noise RNG seeds are only in evaluation_only.json.',
                        'Tone leakage includes Doppler and any admitted anchor transition; CFO-inclusive metric also includes the common oscillator.',
                        'Frozen H and tone leakage are physical diagnostics, not an estimator ranking or a universal best-SCS claim.',
                        'Ideal DAC, synchronized timing/sampling clock, zero CFO initial phase; no NR synchronizer, decoder or BER claim.'])
    save_json(output/'evaluation_only.json',dict(access='evaluation_only; not estimator_input',cases=evaluation))
    disk_audit=_audit_npz(cases)
    save_json(output/'receiver_npz_audit.json',disk_audit)
    report['receiver_disk_audit_passed']=disk_audit['passed']
    if not disk_audit['passed']:
        report['status']='needs_attention'
    save_json(output/'scs_rt_report.json',report)
    return report


def _audit_npz(cases):
    """Read actual disk arrays and hashes; latent fields are not input data."""
    forbidden={'paths','b00','true_cfo_hz','noise_seed','max_local_age_sec',
               'local_anchor_indices_used','target_csi'}
    records=[]
    for case in cases:
        path=Path(case['file']['path'])
        with np.load(path,allow_pickle=False) as z:
            checks=dict(no_latent_fields=not bool(set(z.files)&forbidden),
                complex64=z['full_fft'].dtype==np.complex64,
                all_full_FFT_components=z['full_fft'].shape[0]*z['full_fft'].shape[2]==19456
                    and z['full_fft'].shape[1]==16,
                CFO_unknown=not bool(z['cfo_is_known'][0]),
                causal=float(z['receive_time_sec'].max())<=.006,
                hash_matches=sha256(path)==case['file']['sha256'])
            records.append(dict(city=case['city'],scs_hz=case['scs_hz'],checks=checks,
                                fields=z.files,passed=all(checks.values())))
    return dict(passed=all(x['passed'] for x in records),cases=records,
                scope='actual NPZ readback, forbidden latent fields, FFT shape/dtype, time cutoff and SHA256 checks')

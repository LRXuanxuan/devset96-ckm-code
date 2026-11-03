"""Single/two-path development controls, not release_05 performance scores."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from path_model import (Atom, Estimate, direction_y, cartesian_grid, ula, layout,
                        observation_basis, future_basis, grid_omp, predict,
                        reference_path_cfo, steering)

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'sionna_small_validation'))
from validation.cities import COMPACT_DTYPE, reconstruct
from validation.mimo_observation import make_schedule, simulate_schedule


def nmse(estimate,reference):
    den=float(np.vdot(reference,reference).real)
    if den<=0:
        raise ValueError('Zero-energy reference')
    return float(np.vdot(estimate-reference,estimate-reference).real/den)


def cn(rng,shape):
    return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def three_bursts(n_tx=4, nfft=64, fs=7.68e6, cutoff=.004):
    # Same 3x19 timing structure, SCS and CP duration as the full protocol.
    cp=nfft//4
    schedule=make_schedule(n_tx,n_symbols=57,nfft=nfft,fs_hz=fs,cp_samples=cp)
    start=np.rint(np.linspace(0,cutoff-19*(nfft+cp)/fs,3)*fs).astype(int)
    schedule['start_sample']=np.concatenate([s+np.arange(19)*(nfft+cp) for s in start])
    return schedule


def canonical_waveform(atoms,gains,schedule,tx,rx,cfo,reference=0.):
    """Independent existing receiver; truth access only inside this control."""
    p=np.zeros(len(atoms),COMPACT_DTYPE)
    for j,a in enumerate(atoms):
        p[j]['b00']=gains[j];p[j]['delay_sec']=a.delay_sec;p[j]['doppler_hz']=a.frequency_hz
        p[j]['k_tx']=a.tx_direction;p[j]['k_rx']=a.rx_direction;p[j]['flags']=1
    snapshot=dict(time_sec=reference,compact=p,tx_relative_m=tx,rx_relative_m=rx)
    result=simulate_schedule([snapshot],schedule,cfo_hz=cfo,max_local_age_sec=.01)
    return result['noiseless_full_fft'],p


def direct_fourier(atoms,gains,schedule,tx,rx,cfo,reference=0.):
    """Slow independent Fourier-polynomial evaluation; no fractional-shift IFFT."""
    fs,n,cp,starts,x,w,numbers,times=layout(schedule)
    t=times.ravel();freq=np.fft.fftfreq(n,1/fs)
    y=np.zeros((len(t),len(rx)),complex)
    for a,g in zip(atoms,gains):
        at=steering(tx,a.tx_direction,28e9);ar=steering(rx,a.rx_direction,28e9)
        scalar=np.zeros(len(t),complex)
        for s,start in enumerate(starts):
            shifted=t-a.delay_sec
            valid=(shifted>=start/fs)&(shifted<(start+n+cp)/fs)
            relative=shifted[valid]-(start+cp)/fs
            scalar[valid]+=(np.exp(2j*np.pi*relative[:,None]*freq[None,:])@x[s])/np.sqrt(n)*(w[s]@at)
        scalar*=g*np.exp(2j*np.pi*(a.frequency_hz*(t-reference)+cfo*t))
        y+=scalar[:,None]*ar[None,:]
    samples=y.reshape(len(starts),n,len(rx)).transpose(0,2,1)
    return np.fft.fft(samples,axis=-1,norm='ortho')


def mathematical_controls():
    tx=ula(4);rx=ula(2);reference=-.000271
    atoms=(Atom(3.37/30.72e6,direction_y(.37),direction_y(-.21),733.),
           Atom(12.12/30.72e6,direction_y(-.62),direction_y(.48),-981.))
    gains=np.array([1+.25j,.5-.25j]);cfo=317.
    errors={}
    # CP=16 covers both delays; CP=2 produces actual ISI. Both must match.
    for cp in (16,2):
        schedule=make_schedule(4,n_symbols=5,nfft=32,cp_samples=cp)
        phi=observation_basis(atoms,schedule,tx,rx,cfo_hz=cfo,reference_time_sec=reference)
        y=(phi@gains).reshape(5,2,32)
        native,p=canonical_waveform(atoms,gains,schedule,tx,rx,cfo,reference)
        brute=direct_fourier(atoms,gains,schedule,tx,rx,cfo,reference)
        errors[f'cp_{cp}_native_nmse']=nmse(y,native)
        errors[f'cp_{cp}_independent_fourier_nmse']=nmse(y,brute)
        assert errors[f'cp_{cp}_native_nmse']<1e-22
        assert errors[f'cp_{cp}_independent_fourier_nmse']<1e-20
    rng=np.random.default_rng(20261030)
    v=cn(rng,(phi.shape[0],));g=cn(rng,(2,))
    adjoint=abs(np.vdot(phi@g,v)-np.vdot(g,phi.conj().T@v))/max(abs(np.vdot(phi@g,v)),1)
    assert adjoint<1e-12
    freq=np.linspace(-12e6,12e6,16);future=.006
    b=future_basis(atoms,tx,rx,freq,future,reference_time_sec=reference)
    truth=reconstruct(p,tx,rx,freq,delta_t=future-reference).ravel()
    errors['future_matches_native_reconstruction_nmse']=nmse(b@gains,truth)
    assert errors['future_matches_native_reconstruction_nmse']<1e-24
    fit=np.linalg.lstsq(phi,phi@gains,rcond=1e-10)[0]
    errors['oracle_support_ls_future_nmse']=nmse(b@fit,truth)
    assert errors['oracle_support_ls_future_nmse']<1e-24
    # Nonzero reference time is intentional: omitting this gain transform fails.
    delta=125.
    shifted=tuple(replace(a,frequency_hz=a.frequency_hz-delta) for a in atoms)
    shifted_gain=gains*np.exp(-2j*np.pi*delta*reference)
    phi_shifted=observation_basis(shifted,schedule,tx,rx,cfo_hz=cfo+delta,reference_time_sec=reference)
    b_shifted=future_basis(shifted,tx,rx,freq,future,reference_time_sec=reference)
    errors['unknown_cfo_gauge_observation_nmse']=nmse(phi_shifted@shifted_gain,phi@gains)
    errors['unknown_cfo_gauge_propagation_future_nmse']=nmse(b_shifted@shifted_gain,b@gains)
    effective_a=(b@gains)*np.exp(2j*np.pi*cfo*future)
    effective_b=(b_shifted@shifted_gain)*np.exp(2j*np.pi*(cfo+delta)*future)
    errors['unknown_cfo_gauge_effective_future_nmse']=nmse(effective_b,effective_a)
    theoretical=4*np.sin(np.pi*delta*future)**2
    assert errors['unknown_cfo_gauge_observation_nmse']<1e-24
    assert errors['unknown_cfo_gauge_effective_future_nmse']<1e-24
    assert abs(errors['unknown_cfo_gauge_propagation_future_nmse']-theoretical)<1e-12
    # Do NOT confuse a symbol-rate alias example with the full received samples.
    coarse_t=np.arange(4)*.001;nu=125.;alias=1000.
    first=np.exp(2j*np.pi*nu*coarse_t);second=np.exp(2j*np.pi*(nu+alias)*coarse_t)
    errors['coarse_1ms_alias_history_nmse']=nmse(second,first)
    for label,tstar in [('on_grid',.006),('off_grid',.0045)]:
        errors[f'coarse_1ms_alias_future_{label}_nmse']=float(abs(np.exp(2j*np.pi*alias*tstar)-1)**2)
    assert errors['coarse_1ms_alias_history_nmse']<1e-25
    assert errors['coarse_1ms_alias_future_on_grid_nmse']<1e-25
    assert abs(errors['coarse_1ms_alias_future_off_grid_nmse']-4)<1e-12
    # Full protocol-scale forward compatibility, without a real-data truth load.
    tx_full=np.array([[0.,iy,iz] for iy in range(8) for iz in range(8)])*299792458./(2*28e9)
    rx_full=np.array([[0.,iy,iz] for iy in range(4) for iz in range(4)])*299792458./(2*28e9)
    large=three_bursts(64,256,30.72e6)
    operator=observation_basis(atoms,large,tx_full,rx_full,cfo_hz=cfo)
    native,_=canonical_waveform(atoms,gains,large,tx_full,rx_full,cfo)
    errors['full_57_16_256_native_nmse']=nmse((operator@gains).reshape(native.shape),native)
    assert errors['full_57_16_256_native_nmse']<1e-22
    return dict(status='passed',errors=errors,adjoint_relative_error=float(adjoint),
                gauge_delta_hz=delta,gauge_theoretical_future_nmse=float(theoretical),
                scope='Numerical verification of declared fixtures, not a general recovery guarantee')


def run(out):
    started=time.perf_counter();checks=mathematical_controls()
    print('Finite-CP waveform, future map, phase gauge and sampling controls passed',flush=True)
    tx=ula(4);rx=ula(2);schedule=three_bursts()
    freq=np.linspace(-3e6,3e6,16);cutoff=.004;future=.006;cfo=250.
    grid=cartesian_grid([-.5,0.,.5],[-.5,0.,.5],
                        [.15e-6,.35e-6,.55e-6,.75e-6,.95e-6],np.arange(-1000,1001,250))
    phi=observation_basis(grid,schedule,tx,rx)
    a=Atom(.35e-6,direction_y(.5),direction_y(-.5),750.)
    b=Atom(.75e-6,direction_y(-.5),direction_y(.5),-500.)
    fixtures=[('single_on_grid',(a,),np.array([1+.25j])),
              ('double_on_grid',(a,b),np.array([1+.25j,.5-.25j])),
              ('single_off_grid',(Atom(.423e-6,direction_y(.42),direction_y(-.4),635.),),np.array([1+.25j]))]
    rows=[];details=[];unknown=[]
    for case_id,(name,atoms,gains) in enumerate(fixtures):
        physical=observation_basis(atoms,schedule,tx,rx,cfo_hz=cfo)
        clean=(physical@gains).reshape(57,2,64)
        target=(future_basis(atoms,tx,rx,freq,future)@gains).reshape(2,4,16)
        exact=grid_omp(clean,schedule,grid,phi,known_cfo_hz=cfo,max_paths=3)
        prediction=predict(exact,tx,rx,freq,future)
        held=predict(exact,tx,rx,freq,cutoff)
        info=dict(fixture=name,selected=[asdict(x) for x in exact.atoms],**exact.information,
                  noiseless_future_nmse=nmse(prediction,target),hold_last_same_estimate_nmse=nmse(held,target))
        if 'off_grid' not in name:
            assert nmse(prediction,target)<1e-20
            assert set(exact.atoms)==set(atoms)
        details.append(info)
        # These synthetic SNRs use the fixture's received power, deliberately
        # unlike the historical-calibration SNR used by release_05.
        signal_power=float(np.mean(abs(clean)**2))
        for snr in (0,10,20):
            variance=signal_power*10**(-snr/10)
            for draw in range(5):
                rng=np.random.default_rng(20261029+case_id*100+draw)
                y=clean+np.sqrt(variance)*cn(rng,clean.shape)
                est=grid_omp(y,schedule,grid,phi,noise_variance=variance,known_cfo_hz=cfo,max_paths=3)
                h=predict(est,tx,rx,freq,future)
                rows.append(dict(fixture=name,snr_db=snr,draw=draw,future_nmse=nmse(h,target),
                                 paths_selected=len(est.atoms),observation_fit_nmse=est.information['residual_energy']/est.information['observation_energy']))
        total=grid_omp(clean,schedule,grid,phi,max_paths=3)
        guarded=False
        try:
            predict(total,tx,rx,freq,future)
        except ValueError:
            guarded=True
        assert guarded
        effective=predict(total,tx,rx,freq,future,target='effective')
        item=dict(fixture=name,frame=total.frame,physical_target_guard=guarded,
                  effective_future_nmse=nmse(effective,target*np.exp(2j*np.pi*cfo*future)),
                  selected_total_frequencies_hz=[x.frequency_hz for x in total.atoms])
        if 'off_grid' not in name:
            assert item['effective_future_nmse']<1e-20
            # Oracle association and Doppler supplied ONLY in this named control.
            associated=next(i for i,x in enumerate(total.atoms)
                            if x.tx_direction==a.tx_direction and x.rx_direction==a.rx_direction and x.delay_sec==a.delay_sec)
            calibrated=reference_path_cfo(total,associated,a.frequency_hz)
            recovered=predict(total,tx,rx,freq,future,cfo_anchor_hz=calibrated)
            biased=reference_path_cfo(total,associated,a.frequency_hz+100.)
            wrong=predict(total,tx,rx,freq,future,cfo_anchor_hz=biased)
            item.update(oracle_reference_path_cfo_hz=calibrated,
                        oracle_reference_future_nmse=nmse(recovered,target),
                        biased_reference_plus_100hz_future_nmse=nmse(wrong,target))
            assert abs(calibrated-cfo)<1e-12 and nmse(recovered,target)<1e-20
        unknown.append(item)
        print(name+': known-CFO prediction and unknown-CFO boundary checked',flush=True)
    # A 1kHz shift is not an exact alias of these full-sample waveform inputs.
    original=observation_basis((a,),schedule,tx,rx)[:,0]
    different=observation_basis((replace(a,frequency_hz=a.frequency_hz+1000),),schedule,tx,rx)[:,0]
    full_alias_nmse=nmse(different,original)
    assert full_alias_nmse>1e-3
    aggregate=[]
    for name,_,_ in fixtures:
        for snr in (0,10,20):
            values=[x['future_nmse'] for x in rows if x['fixture']==name and x['snr_db']==snr]
            aggregate.append(dict(fixture=name,snr_db=snr,draws=len(values),mean_linear_nmse=float(np.mean(values)),
                                  db_of_mean_nmse=float(10*np.log10(max(np.mean(values),1e-30)))))
    result=dict(status='passed',controls=checks,noiseless_fixtures=details,known_cfo_noise_trials=rows,
                known_cfo_aggregate=aggregate,unknown_cfo_controls=unknown,
                full_waveform_1khz_shift_nmse=full_alias_nmse,
                config=dict(carrier_hz=28e9,tx=4,rx=2,nfft=64,fs_hz=7.68e6,cp_samples=16,
                            history_sec=.004,future_sec=.002,target_frequencies_hz=freq.tolist(),
                            cutoff_sec=cutoff,target_time_sec=future,cfo_hz=cfo,grid_atoms=len(grid),max_paths=3,
                            noise_draws_per_snr=5,snr_definition='synthetic received-power SNR; not release_05 calibration',
                            path_assumptions='constant gains at reference, delay, directions and Doppler over history and future'),
                source_binding={str(p.relative_to(ROOT)):sha(p) for p in [
                    Path(__file__),Path(__file__).with_name('path_model.py'),
                    ROOT/'sionna_small_validation/validation/mimo_observation.py',
                    ROOT/'sionna_small_validation/validation/cities.py',
                    ROOT/'output/algorithm_m1_20261029/references/library_snapshot.json']},
                elapsed_sec=time.perf_counter()-started,test_split_accessed=False,new_RT_calls=0,
                original_devset_prediction_scores=False,
                limits=['Controlled synthetic fixtures, not measured channels or original devset future scores.',
                        'On-grid exact recovery is a positive control, not off-grid or multi-scene robustness.',
                        'Oracle CFO and reference-path Doppler are explicitly extra diagnostic information.',
                        'The independent delays/Dopplers model has no hard geometric constraint or path births.',
                        'Absolute propagation CSI is not identified by unrestricted unknown CFO and per-path frequencies.',
                        'No learned prior, neural model, full STEM-KL reproduction, or Bu reproduction was run.'])
    with (out/'results.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status='passed',seconds=result['elapsed_sec'],aggregate=aggregate,
                         noiseless=details,unknown=unknown),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    try:
        run(args.out.resolve())
    except Exception:
        (args.out/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    main()

"""Run mathematical controls and small static diagnostics from release_05.

Outputs are deliberately separate from the immutable data release. Test split
is never opened. Results are not scores on the original 2 ms prediction task.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from linear_models import (channel_matrix, channel_tensor, static_forward,
    remove_known_pilots, least_squares, lmmse, mmv_omp, planar_dictionary,
    sensing_diagnostics, relative_mse, db)
from devset_bridge import (ROOT, sha, fit_training_covariance, validation_cases,
                           load_case, make_control, dynamic_diagnostic)
from validation.cities import COMPACT_DTYPE, reconstruct, C, FC
from validation.mimo_observation import make_schedule, simulate_schedule


def complex_normal(rng,shape):
    return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)


def unitary_dft(n):
    return np.exp(2j*np.pi*np.arange(n)[:,None]*np.arange(n)[None,:]/n)/np.sqrt(n)


def math_controls():
    rng=np.random.default_rng(20261029)
    n,m=8,5
    h=complex_normal(rng,(3,n,17));w=unitary_dft(n)
    x=np.exp(.5j*np.pi*rng.integers(0,4,(n,17)))
    y=static_forward(h,w,x);z=remove_known_pilots(y,x)
    fit=least_squares(w,z)
    exact=relative_mse(fit,channel_matrix(h));assert exact<1e-24
    short=w[:m];truth=channel_matrix(h);projector=np.linalg.pinv(short)@short
    fit_short=least_squares(short,short@truth)
    projection_error=relative_mse(fit_short,projector@truth);assert projection_error<1e-24
    null=np.eye(n)-projector
    distinct=truth+null@complex_normal(rng,truth.shape)
    indistinguishable=relative_mse(short@distinct,short@truth)
    assert indistinguishable<1e-24 and np.linalg.norm(distinct-truth)>1
    trials=30000
    isotropic=complex_normal(rng,(n,trials))
    floor=relative_mse(least_squares(short,short@isotropic),isotropic)
    assert abs(floor-(n-m)/n)<.01
    r=np.diag(np.geomspace(1.,.03,n));sigma2=.2
    gaussian=np.sqrt(np.diag(r))[:,None]*complex_normal(rng,(n,trials))
    observed=short@gaussian+np.sqrt(sigma2)*complex_normal(rng,(m,trials))
    bayes=lmmse(short,observed,r,sigma2)
    empirical=float(np.mean(np.sum(abs(bayes-gaussian)**2,axis=0)))
    expected=float(np.trace(r-r@short.conj().T@np.linalg.solve(short@r@short.conj().T+sigma2*np.eye(m),short@r)).real)
    assert abs(empirical-expected)/expected<.03
    ls_risk=float(np.mean(np.sum(abs(least_squares(short,observed)-gaussian)**2,axis=0)))
    assert empirical<ls_risk
    # A planted shared support under an underdetermined sensing matrix.
    dictionary=np.eye(32,dtype=complex)
    q,_=np.linalg.qr(complex_normal(rng,(32,20)))
    sensing=q.conj().T
    sparse=np.zeros((32,80),complex);support=[2,11,25]
    sparse[support]=complex_normal(rng,(3,80))
    recovered,info=mmv_omp(sensing,sensing@sparse,dictionary,max_atoms=3)
    omp_error=relative_mse(recovered,sparse)
    assert set(info["selected_atoms"])==set(support) and omp_error<1e-22
    nonunit_rejected=False
    try:remove_known_pilots(y,2*x)
    except ValueError:nonunit_rejected=True
    assert nonunit_rejected
    # Independent waveform control: diagonal OFDM is recovered only when its
    # stated timing/CP assumptions hold. A too-short CP is a negative control.
    p=np.zeros(2,dtype=COMPACT_DTYPE)
    p["b00"]=[1+.2j,-.3+.1j];p["delay_sec"]=[3.37/30.72e6,9.12/30.72e6]
    p["k_tx"]=[[1,0,0],[0,1,0]];p["k_rx"]=[[1,0,0],[0,1,0]];p["flags"]=1
    array=np.array([[0,0,0],[0,C/FC/2,0]])
    snapshot=dict(time_sec=0.,compact=p,tx_relative_m=array,rx_relative_m=array)
    schedule=make_schedule(2,n_symbols=3,nfft=32,cp_samples=16)
    freq=np.fft.fftfreq(32,1/schedule["fs_hz"])
    reference=static_forward(reconstruct(p,array,array,freq),schedule["tx_beams"],schedule["symbols"])
    wave=simulate_schedule([snapshot],schedule)["noiseless_full_fft"]
    cp_match=relative_mse(reference,wave);assert cp_match<1e-24
    short_cp=make_schedule(2,n_symbols=3,nfft=32,cp_samples=2)
    short_wave=simulate_schedule([snapshot],short_cp)["noiseless_full_fft"]
    short_reference=static_forward(reconstruct(p,array,array,freq),short_cp["tx_beams"],short_cp["symbols"])
    cp_counterexample=relative_mse(short_reference,short_wave);assert cp_counterexample>1e-3
    return dict(status="passed",seed=20261029,full_rank_ls_nmse=exact,
        rank_deficient_ls_equals_projection_nmse=projection_error,
        different_channels_same_observation_nmse=indistinguishable,
        isotropic_ls_floor_measured=floor,isotropic_ls_floor_theory=(n-m)/n,
        gaussian_lmmse_risk_measured=empirical,gaussian_lmmse_risk_theory=expected,
        gaussian_ls_risk_measured=ls_risk,mmv_omp_planted_support=info["selected_atoms"],
        mmv_omp_planted_nmse=omp_error,nonunit_pilot_guard=nonunit_rejected,
        cp_adequate_waveform_formula_nmse=cp_match,short_cp_counterexample_nmse=cp_counterexample,
        scope="Deterministic numerical controls of stated mathematical cases; not proofs or real-data performance claims")


def run(index,out):
    started=time.perf_counter()
    controls=math_controls()
    print("Mathematical controls passed",flush=True)
    manifest=index.parent/"FINAL_DELIVERY_MANIFEST.json"
    before_manifest=sha(manifest);before_index=sha(index)
    release=json.loads(manifest.read_text("utf-8"))
    assert release["status"]=="passed" and before_index==release["index"]["sha256"]
    covariance,cov_info=fit_training_covariance(index)
    np.save(out/"training_spatial_covariance.npy",covariance,allow_pickle=False)
    rows=[];diagnostics=[];sources={}
    for ci,route in enumerate(validation_cases(index)):
        case=load_case(index,route)
        sources[case["metadata"]["family"]]=case["metadata"]["shard_sha256"]
        assert sources[case["metadata"]["family"]]==release["shards"][case["metadata"]["family"]]["sha256"]
        dictionary,_=planar_dictionary(case["tx_local_m"],case["carrier_hz"],16)
        for beam in ("fixed","rotating"):
            schedule,truth,noiseless,frozen_waveform=make_control(case,beam)
            w,x=schedule["tx_beams"],schedule["symbols"]
            formula=static_forward(truth,w,x)
            static_agreement=relative_mse(formula,frozen_waveform)
            delays=case["frozen"]["compact"]["delay_sec"]
            cp_covers=float(delays.max()) <= schedule["cp_samples"]/schedule["fs_hz"]
            if cp_covers: assert static_agreement<1e-20
            sensing=sensing_diagnostics(w,dictionary)
            if beam=="fixed":assert sensing["beam_rank"]==1
            else:assert sensing["beam_rank"]==57
            target_matrix=channel_matrix(truth)
            frozen_ls=least_squares(w,remove_known_pilots(noiseless,x))
            ls_floor=relative_mse(frozen_ls,target_matrix)
            dyn=dynamic_diagnostic(case,schedule)
            diag=dict(**case["metadata"],beam=beam,**sensing,
                static_waveform_formula_nmse=static_agreement,
                cp_covers_maximum_absolute_delay=cp_covers,
                minimum_delay_sec=float(delays.min()),maximum_delay_sec=float(delays.max()),
                cp_duration_sec=schedule["cp_samples"]/schedule["fs_hz"],
                noiseless_static_min_norm_ls_floor=ls_floor,
                label_scope="frozen direct-cutoff channel on 256 FFT frequencies; not future CSI",
                original_landmark_frequencies_exactly_on_fft=int(sum(np.min(abs(np.fft.fftfreq(256,1/schedule["fs_hz"])-f))<1e-4 for f in case["original_target_frequencies_hz"])))
            for key,obs in dyn.items():
                demodulated=remove_known_pilots(obs,x)
                estimated=least_squares(w,demodulated)
                diag[key+"_cutoff_frozen_forward_mismatch_nmse"]=relative_mse(formula,obs)
                diag[key+"_ls_observation_fit_nmse"]=relative_mse(w@estimated,demodulated)
                diag[key+"_static_ls_cutoff_channel_nmse"]=relative_mse(estimated,target_matrix)
            diag["dynamic_no_cfo_vs_frozen_waveform_nmse"]=relative_mse(frozen_waveform,dyn["dynamic_cfo_zero"])
            diag["common_cfo_change_only_nmse"]=relative_mse(dyn["dynamic_cfo_zero"],dyn["dynamic_cfo_original"])
            diagnostics.append(diag)
            # One common standardized noise draw for all algorithms and SNRs.
            rng=np.random.default_rng(20261029+ci)
            standard_noise=complex_normal(rng,noiseless.shape)
            for snr in (0.,10.,20.):
                variance=case["calibration_power"]*10**(-snr/10)
                observation=noiseless+np.sqrt(variance)*standard_noise
                z=remove_known_pilots(observation,x)
                estimates={"minimum_norm_ls":least_squares(w,z),
                           "training_spatial_lmmse":lmmse(w,z,covariance,variance)}
                estimates["angular_mmv_omp"],omp_info=mmv_omp(w,z,dictionary,variance,max_atoms=8)
                actual_snr=db(float(np.mean(abs(noiseless)**2))/variance)
                for name,estimate in estimates.items():
                    rows.append(dict(family=route["route_family_id"],city=route["city"],
                        mechanism=route["mechanism"],beam=beam,nominal_snr_db=snr,
                        actual_frozen_receive_snr_db=actual_snr,algorithm=name,
                        nmse_linear=relative_mse(estimate,target_matrix),
                        nmse_db=db(relative_mse(estimate,target_matrix)),
                        observation_fit_nmse=relative_mse(w@estimate,z),
                        selected_atoms=len(omp_info["selected_atoms"]) if name=="angular_mmv_omp" else -1))
        print("Validated frozen controls and dynamic diagnostics: "+route["route_family_id"],flush=True)
    aggregate=[]
    for beam in ("fixed","rotating"):
        for snr in (0.,10.,20.):
            for name in ("minimum_norm_ls","training_spatial_lmmse","angular_mmv_omp"):
                selected=[r for r in rows if r["beam"]==beam and r["nominal_snr_db"]==snr and r["algorithm"]==name]
                mean=float(np.mean([r["nmse_linear"] for r in selected]))
                aggregate.append(dict(beam=beam,nominal_snr_db=snr,algorithm=name,
                    n_validation_families=len(selected),mean_linear_nmse=mean,db_of_mean_linear_nmse=db(mean)))
    assert sha(manifest)==before_manifest and sha(index)==before_index
    result=dict(status="passed",model="M0 ideal static diagonal frequency-domain control; zero CFO; actual finite-CP waveform compared separately",controls=controls,
        covariance=cov_info,aggregate=aggregate,dynamic_diagnostics=diagnostics,
        source_binding=dict(manifest_sha256=before_manifest,index_sha256=before_index,validation_shards=sources,
                            local_source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")}),
        config=dict(training_families=8,training_labels_per_family=4,validation_families=8,
                    validation_tasks_per_family=1,snr_db=[0,10,20],beam_modes=["fixed","rotating"],
                    omp_grid_size=16,omp_max_atoms=8,seed=20261029),
        counts=dict(static_channel_noise_cases=48,algorithm_scores=len(rows),math_trials=30000),
        test_split_accessed=False,new_RT_calls=0,formal_future_prediction_scores=False,
        elapsed_sec=time.perf_counter()-started,
        limits=["M0 scores use ideal diagonal frequency-domain observations of a frozen channel, not production waveforms.",
                "Finite-CP frozen waveforms can differ from M0 even at zero CFO/Doppler; both positive and negative controls are retained.",
                "No time/frequency prior or map is used in these first baseline implementations.",
                "The covariance is fitted using training labels only; Gaussian optimality holds only in its assumed model.",
                "Grid OMP has off-grid bias and does not recover unobserved directions universally.",
                "Eight selected validation windows do not establish method ranking or statistical significance.",
                "Original targets have16 landmark frequencies,10 of which are off the FFT grid; no nearest-tone substitution is made."])
    with (out/"results.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    with (out/"static_control_scores.csv").open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status="passed",elapsed_sec=result["elapsed_sec"],aggregate=aggregate),ensure_ascii=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",type=Path,default=ROOT/"output/devset96_20261027/release_05/task_index.h5")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        run(args.index.resolve(),args.out.resolve())
    except Exception:
        (args.out/"FAILED.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()

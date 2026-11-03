"""Paired OMP/SBL comparison on the existing M1 controlled fixtures."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from path_model import (Atom,direction_y,cartesian_grid,ula,observation_basis,
                        future_basis,grid_omp,predict)
from sbl import prepare_dictionary,posterior_from_statistics,fit_sbl
from run_controls import ROOT,three_bursts,cn,nmse,sha


def validate_sbl():
    rng=np.random.default_rng(20261030)
    a=cn(rng,(24,9));y=cn(rng,(24,));variance=.3
    gamma=np.geomspace(.05,2,9)
    mean,cov,cost=posterior_from_statistics(a.conj().T@a,a.conj().T@y,
                    float(np.vdot(y,y).real),len(y),gamma,variance)
    prior=np.diag(gamma);cy=a@prior@a.conj().T+variance*np.eye(len(y))
    mean_direct=prior@a.conj().T@np.linalg.solve(cy,y)
    cov_direct=prior-prior@a.conj().T@np.linalg.solve(cy,a@prior)
    cost_direct=np.linalg.slogdet(cy)[1]+np.vdot(y,np.linalg.solve(cy,y)).real
    errors=dict(posterior_mean_relative_mse=nmse(mean,mean_direct),
                posterior_covariance_relative_mse=nmse(cov,cov_direct),
                evidence_absolute_error=float(abs(cost-cost_direct)))
    assert errors['posterior_mean_relative_mse']<1e-22
    assert errors['posterior_covariance_relative_mse']<1e-22
    assert errors['evidence_absolute_error']<1e-10
    # Independent dense-matrix check of an EM sequence, not a recovery claim.
    costs=[]
    for _ in range(30):
        mu,sigma,value=posterior_from_statistics(a.conj().T@a,a.conj().T@y,
                           float(np.vdot(y,y).real),len(y),gamma,variance)
        costs.append(value);gamma=abs(mu)**2+np.diag(sigma).real
    assert np.max(np.diff(costs))<1e-10
    return dict(status='passed',errors=errors,em_maximum_evidence_increase=float(max(0,np.max(np.diff(costs)))))


def run(out):
    start=time.perf_counter();controls=validate_sbl()
    original=ROOT/'output/algorithm_m1_20261029/run_01/results.json'
    old=json.loads(original.read_text('utf-8'))
    assert old['status']=='passed'
    for path,digest in old['source_binding'].items():
        assert sha(ROOT/path)==digest,path
    tx=ula(4);rx=ula(2);schedule=three_bursts();freq=np.linspace(-3e6,3e6,16)
    grid=cartesian_grid([-.5,0.,.5],[-.5,0.,.5],
              [.15e-6,.35e-6,.55e-6,.75e-6,.95e-6],np.arange(-1000,1001,250))
    phi=observation_basis(grid,schedule,tx,rx);prepared=prepare_dictionary(phi)
    a=Atom(.35e-6,direction_y(.5),direction_y(-.5),750.)
    b=Atom(.75e-6,direction_y(-.5),direction_y(.5),-500.)
    fixtures=[('single_on_grid',(a,),np.array([1+.25j])),
              ('double_on_grid',(a,b),np.array([1+.25j,.5-.25j])),
              ('single_off_grid',(Atom(.423e-6,direction_y(.42),direction_y(-.4),635.),),np.array([1+.25j]))]
    rows=[];unknown_checks=[]
    print('Independent complex posterior and EM evidence controls passed',flush=True)
    for case_id,(name,atoms,gains) in enumerate(fixtures):
        clean=(observation_basis(atoms,schedule,tx,rx,cfo_hz=250.)@gains).reshape(57,2,64)
        truth=(future_basis(atoms,tx,rx,freq,.006)@gains).reshape(2,4,16)
        for snr in (0,10,20):
            variance=float(np.mean(abs(clean)**2))*10**(-snr/10)
            for draw in range(5):
                rng=np.random.default_rng(20261029+case_id*100+draw)
                y=clean+np.sqrt(variance)*cn(rng,clean.shape)
                t=time.perf_counter()
                omp=grid_omp(y,schedule,grid,phi,noise_variance=variance,known_cfo_hz=250.,max_paths=3)
                omp_runtime=time.perf_counter()-t
                sbl=fit_sbl(y,schedule,grid,prepared,noise_variance=variance,known_cfo_hz=250.)
                for method,est in [('omp',omp),('sbl',sbl)]:
                    prediction=predict(est,tx,rx,freq,.006)
                    error=nmse(prediction,truth)
                    if method=='omp':
                        prior=next(x for x in old['known_cfo_noise_trials'] if x['fixture']==name and x['snr_db']==snr and x['draw']==draw)
                        assert np.isclose(error,prior['future_nmse'],rtol=1e-12,atol=1e-15)
                    else:
                        assert est.information['maximum_relative_evidence_increase']<1e-7
                    row=dict(fixture=name,snr_db=snr,draw=draw,algorithm=method,
                            future_nmse=error,observation_fit_nmse=est.information['residual_energy']/est.information['observation_energy'],
                            elapsed_sec=omp_runtime if method=='omp' else est.information['elapsed_sec'],
                            information=est.information)
                    rows.append(row)
                    # Preserve completed work even if an interactive turn is
                    # interrupted. Only results.json certifies a completed run.
                    with (out/'completed_trials.jsonl').open('a',encoding='utf-8') as checkpoint:
                        checkpoint.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            print(f'{name} SNR {snr}: 5 paired OMP/SBL draws complete',flush=True)
        # Unknown-CFO SBL can estimate total frequencies only. Check that the
        # common prediction API refuses unanchored physical CSI for SBL too.
        total=fit_sbl(y,schedule,grid,prepared,noise_variance=variance)
        guarded=False
        try:
            predict(total,tx,rx,freq,.006)
        except ValueError:
            guarded=True
        assert guarded
        effective=predict(total,tx,rx,freq,.006,target='effective')
        unknown_checks.append(dict(fixture=name,snr_db=20,draw=4,physical_prediction_guard=guarded,
                  effective_future_nmse=nmse(effective,truth*np.exp(2j*np.pi*250*.006)),
                  iterations=total.information['iterations'],converged=total.information['converged']))
    aggregate=[]
    for name,_,_ in fixtures:
        for snr in (0,10,20):
            for method in ('omp','sbl'):
                group=[r for r in rows if r['fixture']==name and r['snr_db']==snr and r['algorithm']==method]
                average=float(np.mean([r['future_nmse'] for r in group]))
                aggregate.append(dict(fixture=name,snr_db=snr,algorithm=method,n=5,
                    mean_linear_nmse=average,db_of_mean_nmse=float(10*np.log10(average)),
                    mean_runtime_sec=float(np.mean([r['elapsed_sec'] for r in group])),
                    sbl_converged=sum(r['information'].get('converged',False) for r in group) if method=='sbl' else None,
                    mean_iterations=float(np.mean([r['information']['iterations'] for r in group])) if method=='sbl' else None))
    result=dict(status='passed',created_at=datetime.now().astimezone().isoformat(),controls=controls,
                paired_results=rows,aggregate=aggregate,unknown_cfo_checks=unknown_checks,
                config=dict(grid_atoms=405,sbl_max_iterations=300,sbl_relative_tolerance=1e-5,
                            sbl_pruning=False,sbl_map_prior=False,sbl_off_grid=False,
                            sbl_noise='same known variance as OMP',sbl_cfo='same oracle CFO as OMP in main comparison',
                            future_horizon_sec=.002,noise_draws=5,original_control_result_sha256=sha(original)),
                source_binding={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
                elapsed_sec=time.perf_counter()-start,test_split_accessed=False,new_RT_calls=0,
                original_devset_prediction_scores=False,bu_reproduction=False)
    with (out/'results.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status='passed',seconds=result['elapsed_sec'],aggregate=aggregate),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    try:
        run(args.out.resolve())
    except Exception:
        (args.out/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':
    main()

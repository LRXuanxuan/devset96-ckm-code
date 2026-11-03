"""Actual HDF5 multi-process reads and future-input invariance checks."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil

import h5py
import numpy as np

from .dataset_v2 import EstimatorView
from .common import save_json


def fingerprint(value):
    digest=hashlib.sha256()
    def visit(v):
        if isinstance(v,dict):
            for key in sorted(v):digest.update(key.encode());visit(v[key])
        elif isinstance(v,np.ndarray):
            digest.update(str(v.dtype).encode());digest.update(str(v.shape).encode());digest.update(v.tobytes())
        elif isinstance(v,np.generic):visit(v.item())
        else:digest.update(json.dumps(v,sort_keys=True,ensure_ascii=False).encode())
    visit(value)
    return digest.hexdigest()


def _worker(args):
    directory,task_id,size,mode=args
    view=EstimatorView(directory).load(task_id,size,mode)
    if 'observation' not in view:
        raise AssertionError('Real generated observations are required for this IO audit')
    observation=view['observation']
    if observation['receive_time_sec'].max()>view['task']['input_cutoff_sec']+1e-12:
        raise AssertionError('Future receiver samples reached estimator view')
    if observation['full_fft'].shape!=(57,16,256):
        raise AssertionError('Unexpected real MIMO data axes')
    if not np.isfinite(observation['full_fft']).all():
        raise AssertionError('Nonfinite receiver array')
    return fingerprint(view)


def run_io_audit(dataset_root,output):
    root=Path(dataset_root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for city in ('munich','florence','etoile'):
        tasks=json.loads((root/city/'tasks.json').read_text(encoding='utf-8'))['tasks']
        jobs.extend((str(root/city),t['task_id'],n,mode) for t in tasks for n in (4,8,16,32) for mode in ('fixed','rotating'))
    reference=[_worker(job) for job in jobs]
    checks={'real_single_process_reads':len(reference)==48}
    readers={}
    for workers in (2,4):
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            actual=list(pool.map(_worker,jobs))
        checks[str(workers)+'_spawn_readers_exact']=actual==reference
        readers[str(workers)]=dict(task_views=len(actual),all_fingerprints_match=actual==reference)
    # A public-only fixture demonstrates that the ordinary estimator loader
    # does not require evaluation files, and future reports cannot change it.
    fixture=out/'public_only_fixture';fixture.mkdir()
    source=root/'munich'
    for name in ('tasks.json','maps.h5','reported_states.h5','observations.h5'):
        shutil.copy2(source/name,fixture/name)
    task=json.loads((fixture/'tasks.json').read_text(encoding='utf-8'))['tasks'][0]
    before=EstimatorView(fixture).load(task['task_id'])
    checks['works_without_evaluation_files']=not (fixture/'truth.h5').exists() and not (fixture/'evaluation_index.json').exists()
    with h5py.File(fixture/'reported_states.h5','r+') as f:
        g=f[task['pass_id']];future=g['report_time_sec'][:]>task['input_cutoff_sec']+1e-12
        for name in ('reported_position_m','reported_velocity_mps','reported_pose_rad'):
            value=g[name][:];value[future]=1e9;g[name][:]=value
    after=EstimatorView(fixture).load(task['task_id'])
    checks['future_report_mutation_does_not_change_input']=fingerprint(before)==fingerprint(after)
    checks['no_scenario_labels_in_public_task']=not any(k in before['task'] for k in
        ('pairing_role','scene_version','pairing_group','target_csi','truth_pass_id'))
    checks['no_hidden_cfo_or_noise_seed_in_observation']=not any(k in before['observation'] for k in
        ('true_cfo_hz','cfo_hz','noise_seed','noiseless_full_fft','paths','target_csi'))
    report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
                real_task_views=len(jobs),readers=readers,
                scope='real native HDF5 public-only estimator views; spawn CPU readers, not a PyTorch DataLoader test',
                dataset_root=str(root.resolve()))
    save_json(out/'io_audit.json',report)
    return report

"""Grouped, resumable 12 -> 32 -> 96 route generation with compact shards."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', default='llvm_ad_mono_polarized',
                   choices=['llvm_ad_mono_polarized', 'cuda_ad_mono_polarized'])
    p.add_argument('--windows-jit-compat', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--samples-per-src', type=int, default=30000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker-city', choices=['munich', 'florence', 'etoile'])
    p.add_argument('--worker-slot', type=int)
    p.add_argument('--batch-start', type=int, default=0)
    p.add_argument('--batch-stop', type=int, default=96)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--stage-limit', type=int, default=96, choices=[12,32,96])
    return p.parse_args()


def read_plan(path):
    from validation.common import sha256
    plan = json.loads(path.read_text('utf-8'))
    routes = plan['routes']
    names = [r['route_family_id'] for r in routes]
    if len(set(names)) != len(names):
        raise ValueError('A family cannot be duplicated by deployment, speed or environment')
    target=int(plan.get('target_families',96))
    if not 1<=len(routes)<=target or (plan.get('selection_complete') and len(routes)!=target):
        raise ValueError('Plan count/selection-complete declaration is inconsistent')
    for r in routes:
        if r.get('verification_status') == 'geometry_only; RF_pending':
            raise ValueError('Unscreened geometry cannot be a formal route')
        if r['city'] == 'etoile' and r['split'] != 'test':
            raise ValueError('Etoile development protocol is sealed test')
    regions = {}
    for r in routes:
        old = regions.setdefault(r['region_id'], r['split'])
        if old != r['split']:
            raise ValueError('A spatial region appears across splits')
    return plan, sha256(path)


def worker(args):
    from validation.common import save_json
    if args.windows_jit_compat:
        os.environ['SIONNA_WINDOWS_JIT_COMPAT'] = '1'
    os.environ['SIONNA_VALIDATION_CPU_THREADS'] = str(args.threads)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from validation.common import setup_backend
    backend = setup_backend(args.backend)
    from validation.devset_rt import DevsetRT
    from validation.devset_generate import generate_route
    plan, plan_hash = read_plan(args.plan.resolve())
    out = args.output.resolve()
    selected = plan['routes'][args.batch_start:args.batch_stop]
    if args.worker_city:
        selected = [r for r in selected if r['city'] == args.worker_city]
        worker_name = args.worker_city
    else:
        selected = [r for j,r in enumerate(selected) if j % args.workers == args.worker_slot]
        worker_name = f'slot_{args.worker_slot:02d}'
    results = []
    contexts = {}
    with ExitStack() as stack:
        for route in selected:
            if route['city'] not in contexts:
                ctx = stack.enter_context(DevsetRT(route['city'], samples=args.samples_per_src))
                if plan.get('physics', {}).get('diffraction', False):
                    ctx.settings['diffraction'] = True
                contexts[route['city']] = ctx
            context = contexts[route['city']]
            family = route['route_family_id']
            shard = out/'routes'/route['split']/(family+'.h5')
            shard.parent.mkdir(parents=True, exist_ok=True)
            report_file = shard.with_suffix('.report.json')
            if shard.exists():
                if not args.resume:
                    raise FileExistsError('Published shard exists; choose --resume to validate and reuse it')
                from validation.devset_storage import audit_route_file
                disk = audit_route_file(shard)
                if not disk['passed']:
                    raise ValueError('Existing shard fails disk contract: '+family)
                row = json.loads(report_file.read_text('utf-8')) if report_file.exists() else disk
                row.update(route_family_id=family, reused=True)
                results.append(row)
                continue
            if shutil.disk_usage(out).free < 20_000_000_000:
                raise OSError('Less than20GB free; preserve shards and stop before starting another route')
            try:
                print('[route start] '+family+' '+route['intended_mechanism'], flush=True)
                row = generate_route(context, route, shard, resume=args.resume,
                    progress_path=out/'progress'/(worker_name+'.json'), samples=args.samples_per_src)
                row['backend'] = backend
                row['plan_sha256'] = plan_hash
                save_json(report_file, row)
                results.append(row)
                print('[route complete] '+family+' tasks='+str(row.get('admitted_tasks')), flush=True)
            except Exception:
                row = dict(route_family_id=family, passed=False, error=traceback.format_exc(),
                           shard=str(shard), diagnostic_partial=str(shard)+'.partial')
                save_json(out/'errors'/(family+'.json'), row)
                results.append(row)
                print('[route retained failure] '+family, flush=True)
            save_json(out/'workers'/(worker_name+'.json'), dict(worker=worker_name,
                results=results, batch_start=args.batch_start, batch_stop=args.batch_stop))
    save_json(out/'workers'/(worker_name+'.json'), dict(worker=worker_name, results=results,
        complete=True, batch_start=args.batch_start, batch_stop=args.batch_stop))
    return 0 if all(r.get('passed', False) for r in results) else 1


def audit_release(out, plan, plan_hash):
    import h5py
    import numpy as np
    from validation.common import save_json, sha256
    from validation.devset_storage import audit_route_file
    rows = []
    counters = Counter()
    for route in plan['routes']:
        p = out/'routes'/route['split']/(route['route_family_id']+'.h5')
        if not p.exists():
            rows.append(dict(route_family_id=route['route_family_id'], city=route['city'],
                             split=route['split'], passed=False, reason='not_published'))
            continue
        disk = audit_route_file(p)
        with h5py.File(p, 'r') as f:
            states = f['states'][:]; tasks=f['tasks'][:]; intervals=f['intervals'][:]
            config=json.loads(f['config_json'][()].decode())
            survey=f['survey_index'][:]
            design_counts=np.bincount(survey['design_index'],minlength=4)
            primary=states['role'] != b'quarantine_attempt'
            online=primary & (states['pass_index'] >= 0)
            counts=Counter(map(int,states['quality_flags'][primary]))
            row=dict(disk, route_family_id=route['route_family_id'], city=route['city'],
                region_id=route['region_id'], split=route['split'],
                intended_mechanism=route['intended_mechanism'],
                verified_mechanism=route.get('verified_mechanism'),
                bytes=p.stat().st_size, sha256=sha256(p),
                planned_survey_captures=512, admitted_surveys_per_design=design_counts.tolist(),
                snapshot_quality_flags={str(k):v for k,v in counts.items()},
                online_has_los_fraction=float(np.mean(states['has_los'][online])),
                maximum_native_relative_error=float(np.nanmax(states['native_relative_error'][primary])),
                base_anchors=int(np.count_nonzero(states['role']==b'online_anchor')),
                midpoint_states=int(np.count_nonzero(states['role']==b'online_midpoint')),
                quarantine_attempts=int(np.count_nonzero(~primary)),
                nominal_anchor_step_m=config['anchor_step_m'],
                source_sha256=config['source_sha256'])
            for key in ('states','path_records','tasks','admitted_tasks','rejected_tasks','surveys','dense_goldens','rejected_states','unsafe_leaf_intervals'):
                counters[key] += int(disk.get(key,0))
        rows.append(row)
    completed=sum(r['passed'] for r in rows)
    files=[p for p in out.rglob('*') if p.is_file()]
    total_bytes=sum(p.stat().st_size for p in files)
    result=dict(schema='kap_devset.release_manifest.v1', plan_sha256=plan_hash,
        target_families=int(plan.get('target_families',96)), selected_families=len(plan['routes']), completed_families=completed,
        status='completed' if completed==int(plan.get('target_families',96)) else 'incomplete',
        physical_admission='finite_snapshot_and_interval_checks; global propagation convergence not certified',
        counts=dict(counters), split_counts=dict(Counter(r['split'] for r in rows if r['passed'])),
        mechanism_counts=dict(Counter(r['intended_mechanism'] for r in rows if r['passed'])),
        total_disk_bytes=total_bytes, routes=rows,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='algorithm development; no model training result; engineering test inspection is disclosed')
    save_json(out/'release_manifest.json',result)
    return result


def main():
    args=parse_args()
    if args.threads<1 or args.samples_per_src<1 or not 1<=args.workers<=12:
        raise ValueError('Positive thread and sample counts required')
    if args.worker_city or args.worker_slot is not None:
        return worker(args)
    from validation.common import save_json
    plan,plan_hash=read_plan(args.plan.resolve())
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ('logs','progress','workers','errors'):(out/name).mkdir(exist_ok=True)
    if args.audit_only:
        result=audit_release(out,plan,plan_hash)
        print(json.dumps({k:result[k] for k in ('status','completed_families','counts','total_disk_bytes')}))
        return 0 if result['status']=='completed' else 1
    saved_plan=out/'plan.json'
    if saved_plan.exists() and saved_plan.read_bytes()!=args.plan.resolve().read_bytes():
        previous=json.loads(saved_plan.read_text('utf-8'))
        if (plan['routes'][:len(previous['routes'])]!=previous['routes']
                or plan.get('physics')!=previous.get('physics')
                or plan.get('target_families')!=previous.get('target_families')):
            raise ValueError('Existing route definitions or physics changed; create a new release')
        history=out/'plan_history';history.mkdir(exist_ok=True)
        from validation.common import sha256
        (history/(sha256(saved_plan)+'.json')).write_bytes(saved_plan.read_bytes())
    if not saved_plan.exists() or saved_plan.read_bytes()!=args.plan.resolve().read_bytes():
        saved_plan.write_bytes(args.plan.resolve().read_bytes())
    started=time.perf_counter();batches=[]
    for lo,hi in ((0,12),(12,32),(32,len(plan['routes']))):
        stop=min(args.stage_limit,len(plan['routes']))
        if lo>=stop:break
        hi=min(hi,stop)
        workers=[];handles=[]
        for slot in range(min(args.workers, hi-lo)):
            cmd=[sys.executable,str(Path(__file__).resolve()),'--plan',str(args.plan.resolve()),
                 '--output',str(out),'--worker-slot',str(slot),'--workers',str(args.workers),
                 '--batch-start',str(lo),'--batch-stop',str(hi),
                 '--backend',args.backend,'--threads',str(args.threads),'--samples-per-src',str(args.samples_per_src)]
            if args.windows_jit_compat:cmd.append('--windows-jit-compat')
            if args.resume:cmd.append('--resume')
            worker_name=f'slot_{slot:02d}'
            log=(out/'logs'/f'batch_{hi:03d}_{worker_name}.log').open('a' if args.resume else 'x',encoding='utf-8')
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            workers.append((worker_name,proc));handles.append(log)
        while any(p.poll() is None for _,p in workers):
            save_json(out/'run_status.json',dict(status='running',batch_target=hi,
                elapsed_sec=time.perf_counter()-started,
                published_families=len(list((out/'routes').rglob('*.h5'))) if (out/'routes').exists() else 0,
                workers=[dict(city=c,pid=p.pid,returncode=p.poll()) for c,p in workers]))
            time.sleep(10)
        for h in handles:h.close()
        stage=dict(batch_target=hi,workers=[dict(city=c,returncode=p.returncode) for c,p in workers])
        batches.append(stage)
        result=audit_release(out,plan,plan_hash)
        save_json(out/'run_status.json',dict(status='batch_complete',batches=batches,
            elapsed_sec=time.perf_counter()-started,completed_families=result['completed_families']))
        print('[batch complete] '+str(hi)+' published='+str(result['completed_families']),flush=True)
        if any(p.returncode not in (0,1) for _,p in workers):
            return 2
    result=audit_release(out,plan,plan_hash)
    save_json(out/'run_status.json',dict(status=result['status'],batches=batches,
        elapsed_sec=time.perf_counter()-started,completed_families=result['completed_families']))
    stage_succeeded=all(p['returncode']==0 for b in batches for p in b['workers'])
    return 0 if result['status']=='completed' or stage_succeeded else 1


if __name__=='__main__':
    raise SystemExit(main())

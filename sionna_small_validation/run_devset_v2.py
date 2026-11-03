"""Fixed-map-frame, canonical-object-ID development release generation.

The pilot runner and its loaded modules remain immutable. This entry point uses
an explicitly versioned RT adapter and a separate release directory.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',default='llvm_ad_mono_polarized')
    p.add_argument('--windows-jit-compat',action='store_true')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--samples-per-src',type=int,default=100000)
    p.add_argument('--retry-samples',type=int,default=300000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker-slot',type=int)
    p.add_argument('--batch-start',type=int,default=0)
    p.add_argument('--batch-stop',type=int,default=96)
    p.add_argument('--stage-limit',type=int,choices=[12,32,96],default=96)
    p.add_argument('--audit-only',action='store_true')
    return p.parse_args()


def worker(args):
    if args.windows_jit_compat:os.environ['SIONNA_WINDOWS_JIT_COMPAT']='1'
    os.environ['SIONNA_VALIDATION_CPU_THREADS']=str(args.threads)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    from validation.common import setup_backend,save_json,sha256
    backend=setup_backend(args.backend)
    from run_devset import read_plan
    from validation.devset_runtime_v2 import DevsetRTV2
    from validation.devset_generate import generate_route
    from validation.devset_storage import audit_route_file
    plan,plan_hash=read_plan(args.plan.resolve());out=args.output.resolve()
    selected=[r for j,r in enumerate(plan['routes'][args.batch_start:args.batch_stop]) if j%args.workers==args.worker_slot]
    name=f'slot_{args.worker_slot:02d}';results=[];contexts={}
    with ExitStack() as stack:
        for route in selected:
            family=route['route_family_id'];city=route['city']
            if city not in contexts:
                ctx=stack.enter_context(DevsetRTV2(city,samples=args.samples_per_src,retry_samples=args.retry_samples))
                ctx.run_provenance=dict(runner_sha256=sha256(__file__),plan_sha256=plan_hash,
                    backend=backend,threads=args.threads,workers=args.workers,
                    sample_effort=args.samples_per_src,retry_effort=args.retry_samples)
                contexts[city]=ctx
            context=contexts[city]
            shard=out/'routes'/route['split']/(family+'.h5');shard.parent.mkdir(parents=True,exist_ok=True)
            report_path=shard.with_suffix('.report.json')
            if shard.exists():
                if not args.resume:raise FileExistsError('Published shard exists; explicit --resume required')
                disk=audit_route_file(shard)
                if not disk['passed']:raise ValueError('Existing shard fails disk contract: '+family)
                row=json.loads(report_path.read_text('utf-8')) if report_path.exists() else disk
                row.update(route_family_id=family,reused=True);results.append(row);continue
            if shutil.disk_usage(out).free<20_000_000_000:
                raise OSError('Stop before a new route: less than20GB free; all existing data retained')
            try:
                print('[v2 route start] '+family+' '+route['intended_mechanism'],flush=True)
                row=generate_route(context,route,shard,resume=args.resume,
                    progress_path=out/'progress'/(name+'.json'),samples=args.samples_per_src)
                row.update(backend=backend,plan_sha256=plan_hash,numerical_search_effort=args.samples_per_src,
                    runtime_contract='fixed_survey_reference_and_canonical_geometry_ids.v2')
                save_json(report_path,row);results.append(row)
                print('[v2 route complete] '+family+' admitted='+str(row['admitted_tasks']),flush=True)
            except Exception:
                row=dict(route_family_id=family,passed=False,error=traceback.format_exc(),
                    shard=str(shard),diagnostic_partial=str(shard)+'.partial')
                save_json(out/'errors'/(family+'.json'),row);results.append(row)
                print('[v2 diagnostic retained] '+family,flush=True)
            save_json(out/'workers'/(name+'.json'),dict(worker=name,results=results,
                batch_start=args.batch_start,batch_stop=args.batch_stop,complete=False))
    save_json(out/'workers'/(name+'.json'),dict(worker=name,results=results,
        batch_start=args.batch_start,batch_stop=args.batch_stop,complete=True))
    return 0 if all(r.get('passed',False) for r in results) else 1


def main():
    a=arguments()
    if not 1<=a.workers<=16 or a.threads<1 or a.samples_per_src<1 or a.retry_samples<a.samples_per_src:
        raise ValueError('Invalid concurrency or numerical search budgets')
    if a.worker_slot is not None:return worker(a)
    from validation.common import save_json,sha256
    from run_devset import read_plan,audit_release
    plan,plan_hash=read_plan(a.plan.resolve());out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ('logs','progress','workers','errors','provenance'):(out/name).mkdir(exist_ok=True)
    fixed=out/'plan.json'
    if fixed.exists() and fixed.read_bytes()!=a.plan.resolve().read_bytes():
        raise ValueError('V2 uses one frozen plan; create a new release for changed routes')
    if not fixed.exists():shutil.copy2(a.plan,fixed)
    if a.audit_only:
        result=audit_release(out,plan,plan_hash)
        print(json.dumps({k:result[k] for k in ('status','completed_families','counts','total_disk_bytes')}))
        return 0 if result['status']=='completed' else 1
    save_json(out/'provenance'/'run_contract.json',dict(runner_sha256=sha256(__file__),
        plan_sha256=plan_hash,workers=a.workers,threads=a.threads,backend=a.backend,
        samples_per_src=a.samples_per_src,retry_samples=a.retry_samples,
        sample_scope='finite deterministic candidate search; no global convergence guarantee',
        coordinate_scope='fixed survey reference pose; actual online pose',
        object_id_scope='canonical mesh geometry and controlled vehicle role'))
    started=time.perf_counter();batches=[]
    for start,stop in ((0,12),(12,32),(32,96)):
        stop=min(stop,a.stage_limit,len(plan['routes']))
        if start>=stop:break
        processes=[];logs=[]
        for slot in range(min(a.workers,stop-start)):
            name=f'slot_{slot:02d}'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--plan',str(a.plan.resolve()),'--output',str(out),
                '--backend',a.backend,'--workers',str(a.workers),'--threads',str(a.threads),
                '--samples-per-src',str(a.samples_per_src),'--retry-samples',str(a.retry_samples),
                '--worker-slot',str(slot),'--batch-start',str(start),'--batch-stop',str(stop)]
            if a.resume:cmd.append('--resume')
            if a.windows_jit_compat:cmd.append('--windows-jit-compat')
            log=(out/'logs'/f'batch_{stop:03d}_{name}.log').open('a' if a.resume else 'x',encoding='utf-8')
            processes.append((name,subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)));logs.append(log)
        while any(p.poll() is None for _,p in processes):
            save_json(out/'run_status.json',dict(status='running',batch_target=stop,
                elapsed_sec=time.perf_counter()-started,published_families=len(list((out/'routes').rglob('*.h5'))) if (out/'routes').exists() else 0,
                workers=[dict(worker=n,pid=p.pid,returncode=p.poll()) for n,p in processes]))
            time.sleep(10)
        for log in logs:log.close()
        batches.append(dict(batch_target=stop,workers=[dict(worker=n,returncode=p.returncode) for n,p in processes]))
        result=audit_release(out,plan,plan_hash)
        save_json(out/'run_status.json',dict(status='batch_complete',batches=batches,
            elapsed_sec=time.perf_counter()-started,completed_families=result['completed_families']))
        print('[v2 batch complete] '+str(stop)+' published='+str(result['completed_families']),flush=True)
        if any(p.returncode!=0 for _,p in processes):return 1
    result=audit_release(out,plan,plan_hash)
    stage_passed=all(row['returncode']==0 for b in batches for row in b['workers'])
    save_json(out/'run_status.json',dict(status=result['status'],batches=batches,
        elapsed_sec=time.perf_counter()-started,completed_families=result['completed_families']))
    return 0 if result['status']=='completed' or stage_passed else 1


if __name__=='__main__':raise SystemExit(main())

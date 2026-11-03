"""Unique-specular-branch, fixed-map-frame development release generation.

The pilot runner and its loaded modules remain immutable. This entry point uses
an explicitly versioned RT adapter and a separate release directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent


def validate_qualification(plan, args):
    """Bind final route geometry and runtime sources to fresh screening evidence."""
    from validation.common import sha256
    if plan.get('v4_qualification_status')!='passed':
        raise ValueError('V4 requires fresh physical qualification evidence')
    parents={};locks={}
    for route in plan['routes']:
        proof=route.get('v4_qualification',{})
        for field in ('report_path','report_sha256','plan_path','plan_sha256','source_lock_path','source_lock_sha256'):
            if not proof.get(field):raise ValueError('Missing V4 qualification binding: '+field)
        report_path=Path(proof['report_path']);parent_path=Path(proof['plan_path']);lock_path=Path(proof['source_lock_path'])
        if sha256(report_path)!=proof['report_sha256'] or sha256(parent_path)!=proof['plan_sha256'] or sha256(lock_path)!=proof['source_lock_sha256']:
            raise ValueError('Qualification report, parent plan or source lock changed')
        if str(parent_path) not in parents:
            parents[str(parent_path)]={r['route_family_id']:r for r in json.loads(parent_path.read_text('utf-8'))['routes']}
        original=parents[str(parent_path)].get(route['route_family_id'])
        if original!={k:v for k,v in route.items() if k!='v4_qualification'}:
            raise ValueError('Final route differs from the freshly qualified geometry/motion/deployment')
        report=json.loads(report_path.read_text('utf-8'))
        if not (report.get('schema')=='kap_devset.native_selected_v4_screen.v1'
                and report.get('eligibility') is True and report.get('screen_completed') is True
                and report.get('plan_sha256')==proof['plan_sha256']
                and report.get('route_family_id')==route['route_family_id']):
            raise ValueError('Route screening is incomplete, unqualified or a different version')
        if str(lock_path) not in locks:
            lock=json.loads(lock_path.read_text('utf-8'))
            if lock['samples']!=args.samples_per_src or lock['retry_samples']!=args.retry_samples:
                raise ValueError('Final search budgets differ from the qualification settings')
            if any(sha256(ROOT/name)!=expected for name,expected in lock['source_sha256'].items()):
                raise ValueError('A source used for qualification has changed')
            locks[str(lock_path)]=lock
        if report.get('source_sha256')!=locks[str(lock_path)]['source_sha256']:
            raise ValueError('Screening report is not bound to the recorded runtime sources')
    return True


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
    from validation.devset_runtime_v4 import DevsetRTV4
    from validation.devset_generate_v4 import generate_route
    from validation.devset_storage_v4 import audit_route_file_v4 as audit_route_file
    plan,plan_hash=read_plan(args.plan.resolve());out=args.output.resolve()
    validate_qualification(plan,args)
    selected=[r for j,r in enumerate(plan['routes'][args.batch_start:args.batch_stop]) if j%args.workers==args.worker_slot]
    name=f'slot_{args.worker_slot:02d}';results=[];contexts={}
    with ExitStack() as stack:
        for route in selected:
            if (out/'STOP').exists():break
            family=route['route_family_id'];city=route['city']
            if city not in contexts:
                ctx=stack.enter_context(DevsetRTV4(city,samples=args.samples_per_src,retry_samples=args.retry_samples))
                ctx.run_provenance=dict(runner_sha256=sha256(__file__),plan_sha256=plan_hash,
                    backend=backend,threads=args.threads,workers=args.workers,
                    sample_effort=args.samples_per_src,retry_effort=args.retry_samples)
                ctx.stop_file=out/'STOP'
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
                print('[v4 route start] '+family+' '+route['intended_mechanism'],flush=True)
                row=generate_route(context,route,shard,resume=args.resume,
                    progress_path=out/'progress'/(name+'.json'),samples=args.samples_per_src)
                row.update(backend=backend,plan_sha256=plan_hash,numerical_search_effort=args.samples_per_src,
                    runtime_contract='certified_unique_specular_native_branches.v4')
                save_json(report_path,row);results.append(row)
                print('[v4 route complete] '+family+' admitted='+str(row['admitted_tasks']),flush=True)
            except Exception:
                row=dict(route_family_id=family,passed=False,error=traceback.format_exc(),
                    shard=str(shard),diagnostic_partial=str(shard)+'.partial')
                if (out/'STOP').exists():
                    row['stopped_by_control_file']=True
                save_json(out/'errors'/(family+'.json'),row);results.append(row)
                print('[v4 diagnostic retained] '+family,flush=True)
                if row.get('stopped_by_control_file'):break
            save_json(out/'workers'/(name+'.json'),dict(worker=name,results=results,
                batch_start=args.batch_start,batch_stop=args.batch_stop,complete=False))
    save_json(out/'workers'/(name+'.json'),dict(worker=name,results=results,
        batch_start=args.batch_start,batch_stop=args.batch_stop,complete=True))
    return 0 if all(r.get('passed',False) for r in results) else 1


def audit_release(out,plan,plan_hash):
    """Keep archive rows, physical snapshots and native solves distinct."""
    import h5py
    import numpy as np
    from validation.common import save_json,sha256
    from validation.devset_storage_v4 import audit_route_file_v4
    rows=[];totals=Counter()
    for route in plan['routes']:
        path=out/'routes'/route['split']/(route['route_family_id']+'.h5')
        row=dict(route_family_id=route['route_family_id'],city=route['city'],split=route['split'],
                 region_id=route['region_id'],intended_mechanism=route['intended_mechanism'])
        if not path.exists():
            rows.append(dict(row,passed=False,reason='not_published'));continue
        disk=audit_route_file_v4(path)
        with h5py.File(path,'r') as f:
            states=f['states'][:];lengths=np.diff(f['offsets'][:]);config=json.loads(f['config_json'][()].decode())
            primary=~np.isin(states['role'],[b'raw_native_candidate',b'quarantine_attempt'])
            good=primary & (states['quality_flags']==0);online=primary & (states['pass_index']>=0)
            survey=f['survey_index'][:]
            row.update(disk,physical_snapshot_records=int(primary.sum()),
                physical_path_records=int(lengths[primary].sum()),admitted_physical_states=int(good.sum()),
                admitted_physical_paths=int(lengths[good].sum()),
                rejected_physical_states=int(np.count_nonzero(primary&~good)),
                raw_candidate_records=int(np.count_nonzero(states['role']==b'raw_native_candidate')),
                quarantine_attempts=int(np.count_nonzero(states['role']==b'quarantine_attempt')),
                snapshot_quality_flags=dict(Counter(map(str,map(int,states['quality_flags'][primary])))),
                planned_survey_captures=512,
                admitted_surveys_per_design=np.bincount(survey['design_index'],minlength=4).tolist(),
                online_has_los_fraction=float(np.mean(states['has_los'][online])),
                base_anchors=int(np.count_nonzero(states['role']==b'online_anchor')),
                midpoint_states=int(np.count_nonzero(states['role']==b'online_midpoint')),
                nominal_anchor_step_m=config['anchor_step_m'],source_sha256=config['source_sha256'],
                runtime_adapter_version=config['runtime_adapter_version'],
                runtime_adapter_source_sha256=config['runtime_adapter_source_sha256'],
                survey_reference_pose_rad=config['survey_reference_pose_rad'],
                maximum_native_relative_error=float(np.nanmax(states['native_relative_error'][good])) if good.any() else None,
                bytes=path.stat().st_size,sha256=sha256(path),path=str(path))
            for key in ('states','path_records','physical_snapshot_records','physical_path_records',
                        'admitted_physical_states','admitted_physical_paths','rejected_physical_states',
                        'raw_candidate_records','raw_candidate_path_records','quarantine_attempts',
                        'tasks','admitted_tasks','rejected_tasks','surveys','dense_goldens',
                        'unsafe_leaf_intervals','committed_solver_calls','committed_primary_solver_calls',
                        'committed_revisit_solver_calls'):
                totals[key]+=int(row.get(key,0))
        rows.append(row)
    completed=sum(r['passed'] for r in rows)
    result=dict(schema='kap_devset.release_manifest.v4',plan_sha256=plan_hash,target_families=96,
        selected_families=len(plan['routes']),completed_families=completed,
        status='completed' if completed==96 else 'incomplete',counts=dict(totals),routes=rows,
        split_counts=dict(Counter(r['split'] for r in rows if r['passed'])),
        mechanism_counts=dict(Counter(r['intended_mechanism'] for r in rows if r['passed'])),
        total_disk_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        physical_admission='unique full-topology specular image certificates and finite native checks; not exact EM or globally exhaustive propagation',
        reference_kind='selected masked native Paths.cfr; original candidate coefficients/active geometry are archived separately',
        count_scope='states/path_records include diagnostic archive rows; physical_snapshot_records/physical_path_records exclude them; solver calls are a separate committed ledger',
        completed_utc=datetime.now(timezone.utc).isoformat())
    save_json(out/'release_manifest.json',result)
    return result


def main():
    a=arguments()
    if not 1<=a.workers<=16 or a.threads<1 or a.samples_per_src<1 or a.retry_samples<a.samples_per_src:
        raise ValueError('Invalid concurrency or numerical search budgets')
    if a.worker_slot is not None:return worker(a)
    from validation.common import save_json,sha256
    from run_devset import read_plan
    plan,plan_hash=read_plan(a.plan.resolve());out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    validate_qualification(plan,a)
    for name in ('logs','progress','workers','errors','provenance'):(out/name).mkdir(exist_ok=True)
    fixed=out/'plan.json'
    if fixed.exists() and fixed.read_bytes()!=a.plan.resolve().read_bytes():
        raise ValueError('V4 uses one frozen plan; create a new release for changed routes')
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
        object_id_scope='canonical mesh geometry and controlled vehicle role', native_reference='unique full-topology image certificate + masked native.cfr; raw compact/geometry archived'))
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
        print('[v4 batch complete] '+str(stop)+' published='+str(result['completed_families']),flush=True)
        if (out/'STOP').exists():
            save_json(out/'run_status.json',dict(status='stopped_control_file',batches=batches,
                completed_families=result['completed_families'],scope='controlled checkpoint stop, not a native crash'))
            return 75
        if any(p.returncode!=0 for _,p in processes):return 1
    result=audit_release(out,plan,plan_hash)
    stage_passed=all(row['returncode']==0 for b in batches for row in b['workers'])
    save_json(out/'run_status.json',dict(status=result['status'],batches=batches,
        elapsed_sec=time.perf_counter()-started,completed_families=result['completed_families']))
    return 0 if result['status']=='completed' or stage_passed else 1


if __name__=='__main__':raise SystemExit(main())

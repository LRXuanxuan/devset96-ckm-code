"""External receiver-grid checks and conservative legacy recertification.

Stored16-native-checked compact fields provide a *proxy* for legacy256 checks.
They never become independently native256 checked by relabeling. Fresh native256
references are supplied by the separate runtime extension.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
import numpy as np
from .cities import C,FC,reconstruct
from .devset_view import rotate_local_array

SCHEMA='kap_devset.external_receiver_grid_admission.v1'
FREQUENCIES_HZ=np.fft.fftfreq(256,1/30.72e6)
FREQUENCIES_SHA256=hashlib.sha256(FREQUENCIES_HZ.astype('<f8').tobytes()).hexdigest()


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def db_error(estimate,reference):
    norm=float(np.linalg.norm(reference))
    if not np.isfinite(norm) or norm<=0:return None
    value=float(np.linalg.norm(estimate-reference)/norm)
    return 20*math.log10(max(value,1e-150)) if np.isfinite(value) else None


def _basis(p,tx,rx,dt=0.,sign=1.):
    return (np.exp(2j*np.pi*FC/C*((rx-rx[0])@p['k_rx'].T)),
            np.exp(2j*np.pi*FC/C*((tx-tx[0])@p['k_tx'].T)),
            np.exp(-2j*np.pi*FREQUENCIES_HZ[:,None]*p['delay_sec'][None,:]),
            sign*p['b00'].astype(np.complex128)*np.exp(2j*np.pi*p['doppler_hz']*dt))


def _gram_norm(parts):
    if sum(len(p[3]) for p in parts)>512:return None
    rx,tx,f,c=[np.concatenate([p[j] for p in parts],axis=1 if j<3 else 0) for j in range(4)]
    g=(rx.conj().T@rx)*(tx.conj().T@tx)*(f.conj().T@f)
    terms=c.conj()[:,None]*c[None,:]*g;q=np.sum(terms,dtype=np.complex128)
    bound=128*np.finfo(float).eps*float(abs(terms).sum())
    if not np.isfinite(q) or q.real<=0 or abs(q.imag)>bound or bound>1e-5*q.real:return None
    return float(q.real)


def relative_arrays(rec,config):
    return (rotate_local_array(config['tx_array_local_m'],rec['fields']['tx_pose_rad']),
            rotate_local_array(config['rx_array_local_m'],rec['fields']['rx_pose_rad']))


def proxy_error(a,b,config,dt,*,force_dense=False):
    tx,rx=relative_arrays(a,config);btx,brx=relative_arrays(b,config)
    n=_gram_norm([_basis(a['compact'],tx,rx,dt),_basis(b['compact'],btx,brx,sign=-1.)])
    d=_gram_norm([_basis(b['compact'],btx,brx)])
    value=10*math.log10(n/d) if n is not None and d is not None else None
    dense=bool(force_dense or value is None or value>=-30.5)
    if dense:value=db_error(reconstruct(a['compact'],tx,rx,FREQUENCIES_HZ,delta_t=dt),
                            reconstruct(b['compact'],btx,brx,FREQUENCIES_HZ))
    return value,dense


def check_fresh_interval(a,b,c,config,receiver_references):
    tx,rx=relative_arrays(a,config);errors=[];proxy=[];sources=[];all_native=True
    for target in (b,c):
        dt=float(target['fields']['time_sec'])-float(a['fields']['time_sec'])
        record=receiver_references.get(int(target['index']))
        valid=bool(record is not None and record.get('schema')=='kap_devset.external_receiver_grid_native.v1'
            and record.get('frequencies_sha256')==FREQUENCIES_SHA256
            and record.get('compact_vs_native_relative_error') is not None
            and np.isfinite(record['compact_vs_native_relative_error'])
            and 0<=record['compact_vs_native_relative_error']<float(config['native_relative_error_tolerance']))
        if valid:
            estimate=reconstruct(a['compact'],tx,rx,FREQUENCIES_HZ,delta_t=dt)
            value=db_error(estimate,record['native'])
        else:value=None
        all_native &= valid
        pe,dense=proxy_error(a,target,config,dt)
        errors.append(value);proxy.append(pe)
        sources.append(dict(record_index=int(target['index']),native256_available=valid,
            compact_vs_native256_relative_error=_finite(record.get('compact_vs_native_relative_error')) if record else None,
            native256_sha256=record.get('native_sha256') if record else None,
            native_reference_trace_row=record.get('native_reference_trace_row') if record else None,
            source_revisit=record.get('source_revisit') if record else None,proxy_used_dense=dense))
    passed=bool(all_native and all(x is not None and np.isfinite(x) and x<=-30. for x in errors+proxy))
    return dict(schema=SCHEMA,reference_kind='fresh_native256_and_stored_compact256_proxy',
        frequencies_sha256=FREQUENCIES_SHA256,midpoint_nmse_db=errors[0],right_nmse_db=errors[1],
        proxy_midpoint_nmse_db=proxy[0],proxy_right_nmse_db=proxy[1],passed=passed,
        target_reference_sources=sources,threshold_db=-30.,
        scope='Frobenius norm over all16Rx x64Tx x256receiver FFT frequencies; no per-tone admission threshold')


def _plain(row):
    return {name:row[name].item() for name in row.dtype.names}


def _coverage(rows):
    result=[]
    for left,right in sorted((float(r['left_time_sec']),float(r['right_time_sec']))
                             for r in rows if r['passed'] and r['used_as_leaf']):
        if result and left<=result[-1][1]+1e-12:result[-1][1]=max(result[-1][1],right)
        else:result.append([left,right])
    return np.asarray(result,float).reshape(-1,2)


def recertify_tasks(writer,cache,intervals_by_pass,pass_index=None):
    rows=writer.handle['tasks'][:];by_index={int(r['index']):r for r in cache.values()}
    coverage={p:_coverage(items) for p,items in intervals_by_pass.items()}
    changed=0
    for i,row in enumerate(rows):
        p=int(row['pass_index'])
        if pass_index is not None and p!=int(pass_index):continue
        state=by_index[int(row['target_record_index'])];flags=0
        if state['fields']['quality_flags'] or not state['fields']['native_checked']:flags|=1
        spans=coverage[p]
        if not len(spans):flags|=2
        else:
            cutoff=float(row['cutoff_sec']);at=np.searchsorted(spans[:,0],cutoff-.004+1e-12,side='right')-1
            if at<0 or spans[at,1]<cutoff-1e-12:flags|=2
        admitted=flags==0
        if int(row['flags'])!=flags or bool(row['admitted'])!=admitted:
            writer.append_event('receiver_grid_task_recertified',dict(task_index=i,
                old_flags=int(row['flags']),old_admitted=bool(row['admitted']),
                new_flags=flags,new_admitted=admitted,task_sha256_before=hashlib.sha256(row.tobytes()).hexdigest(),
                pass_index=p,cutoff_sec=float(row['cutoff_sec']),target_record_index=int(row['target_record_index']),
                identity_time_seed_eval_and_target_reference_unchanged=True,
                reason='History coverage recomputed from receiver256-qualified leaf intervals; no target or task identity replacement'))
            writer.update_task(i,flags=flags,admitted=admitted);changed+=1
    return changed


def retrofit_committed_intervals(writer,cache,config,completed_intervals,intervals_by_pass,max_refinement):
    """Revoke only failing legacy256 proxies before any new native trace.

    Caller supplies an external writer with joint interval/task/state mutation
    checkpoints. All physical/raw rows stay unchanged; old certificates/events
    are preserved before mutation. This method never invokes a solver.
    """
    rows=writer.handle['intervals'][:];by_index={int(r['index']):r for r in cache.values()}
    evidence=[];revoked=[]
    for i in np.flatnonzero(rows['passed']&rows['used_as_leaf']):
        row=rows[i];a=by_index[int(row['left_index'])];values=[];modes=[]
        for field in ('midpoint_index','right_index'):
            b=by_index[int(row[field])];dt=float(b['fields']['time_sec'])-float(a['fields']['time_sec'])
            value,dense=proxy_error(a,b,config,dt);values.append(value);modes.append(dense)
        passed=all(x is not None and np.isfinite(x) and x<=-30. for x in values)
        evidence.append(dict(interval_index=int(i),midpoint_nmse_db=values[0],right_nmse_db=values[1],
            passed=passed,dense_endpoint_checks=modes))
        if not passed:
            old=_plain(row);new_leaf=int(row['level'])>=int(max_refinement)
            writer.append_event('receiver_grid_certificate_revoked',dict(interval_index=int(i),original_interval=old,
                original_interval_sha256=hashlib.sha256(row.tobytes()).hexdigest(),
                receiver256_midpoint_nmse_db=values[0],receiver256_right_nmse_db=values[1],
                new_passed=False,new_used_as_leaf=bool(new_leaf),original16_metrics_retained=True,
                reason='Legacy compact256 proxy exceeds unchanged minus30 threshold; prior certificate retained as evidence, not usable permission'))
            writer.update_interval(int(i),passed=False,used_as_leaf=bool(new_leaf))
            ident=(int(row['pass_index']),round(float(row['left_time_sec']),12),round(float(row['right_time_sec']),12),int(row['level']))
            completed_intervals[ident]['passed']=False;completed_intervals[ident]['used_as_leaf']=bool(new_leaf)
            revoked.append(int(i))
    writer.append_event('receiver_grid_legacy_validation',dict(schema=SCHEMA,
        reference_kind='legacy native16-checked compact256 proxy; not independently native256',
        frequencies_sha256=FREQUENCIES_SHA256,checked_intervals=len(evidence),revoked_intervals=revoked,
        threshold_db=-30.,intervals=evidence,actual_solver_calls=0))
    # Rebuild exact-left-state permission, including shrinkage to its own time.
    support=defaultdict(list)
    for i,row in enumerate(rows):
        if int(i) not in revoked and row['passed'] and row['used_as_leaf']:
            support[int(row['left_index'])].append((i,float(row['right_time_sec'])))
    for index,rec in by_index.items():
        if int(rec['fields']['pass_index'])<0:continue
        until=max([float(rec['fields']['time_sec'])]+[x[1] for x in support[index]])
        old=float(rec['fields']['receiver_validated_until_sec'])
        if abs(old-until)>1e-12:
            writer.append_event('receiver_grid_anchor_recertified',dict(record_index=index,
                old_receiver_validated_until_sec=old,new_receiver_validated_until_sec=until,
                supporting_interval_indices=[x[0] for x in support[index]],
                reason='Exact left_index support only; no valid passed256 leaf means no forward receiver permission'))
            rec['fields']['receiver_validated_until_sec']=until
            writer.update_snapshot(index,receiver_validated_until_sec=until)
    task_changes=recertify_tasks(writer,cache,intervals_by_pass)
    writer.flush(metadata=dict(stage='receiver256-legacy-recertification',receiver256_schema=SCHEMA,
        receiver256_revoked_intervals=len(revoked),receiver256_recertified_tasks=task_changes))
    return dict(checked_intervals=len(evidence),revoked_intervals=revoked,changed_tasks=task_changes)

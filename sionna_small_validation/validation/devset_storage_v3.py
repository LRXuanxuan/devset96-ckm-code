"""Transactional raw-candidate evidence alongside physically admitted V3 truth.

The frozen core schema remains readable. Raw native candidates are explicitly
unadmitted master states, with untouched compact128 coefficients and a 74B/path
compressed active-geometry table. Core and auxiliary tails share one rollback
checkpoint; raw evidence is never an implicit receiver/target path subset.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import warnings

import h5py
import numpy as np

from . import devset_storage as core
from .cities import COMPACT_DTYPE

VERSION='kap_devset.raw_native_candidates.v3'
BASE_SOURCE_SHA256='34b24b3bf66c6fa9dc0cbf1137743f64d9ecc14e7a4b4ff698393e35f958eead'
RAW_CANDIDATE_ONLY=1 << 30
UINT32_MAX=np.iinfo(np.uint32).max
GROUP_UNRESOLVED=1
GROUP_CERTIFIED=2
GROUP_GEOMETRY_ONLY=4
CANDIDATE_AGREES=8
GROUP_MULTIPLE=16
MISSING_CANDIDATE_METRICS=32

RAW_GEOMETRY_DTYPE=np.dtype([
    ('primitives','<u4',(3,)),('vertices_xyz_m','<f4',(3,3)),
    ('selected_mask','?'),('group_idx','<u4'),
    ('selection_score_m','<f4'),('max_point_error_m','<f4'),
    ('point_rms_error_m','<f4'),('delay_error_sec','<f4'),
    ('geometry_flags','<u2'),('reason_code','<u2'),('depth','u1'),
],align=False)
assert RAW_GEOMETRY_DTYPE.itemsize==74
RAW_INDEX_DTYPE=np.dtype([
    ('state_index','<i8'),('raw_path_count','<u4'),('group_count','<u4'),
    ('selected_path_count','<u4'),('full_report_sha256','u1',(32,)),
],align=False)
TRACE_ACCOUNTING_DTYPE=np.dtype([
    ('primary_record_index','<i8'),('actual_solver_calls','<u2'),('source_revisit','?'),
    ('final_candidate_count','<u4'),('candidate_capacity','<u4')],align=False)
assert TRACE_ACCOUNTING_DTYPE.itemsize==19
AUX_DATASETS=('raw_candidates/index','raw_candidates/geometry_offsets',
    'raw_candidates/geometry','raw_candidates/report_offsets','raw_candidates/report_bytes','trace_accounting')


def _assert_frozen_core():
    actual=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    if actual!=BASE_SOURCE_SHA256:
        raise RuntimeError('Frozen RouteWriter changed; review V3 joint checkpoint implementation before using it')


def _prepare_raw(compact,raw,report):
    compact=np.asarray(compact)
    if compact.dtype!=COMPACT_DTYPE or compact.ndim!=1:
        raise ValueError('Raw compact must use the unmodified128B source dtype')
    n=len(compact);primitives=np.asarray(raw['primitives']);vertices=np.asarray(raw['vertices_xyz_m'])
    types=np.asarray(raw.get('interaction_types',compact['interaction_types']))
    if primitives.shape!=(n,3) or vertices.shape!=(n,3,3) or types.shape!=(n,3):
        raise ValueError('Raw geometry must align with compact-valid path order and three interaction slots')
    if not np.array_equal(types,compact['interaction_types']):
        raise ValueError('Raw interaction types and compact rows differ; refuse an ambiguous archive')
    if 'native_path_index' in raw and not np.array_equal(np.asarray(raw['native_path_index']),compact['path_index']):
        raise ValueError('Raw native path permutation differs from compact.path_index')
    if np.any(compact['depth']>3):raise ValueError('Only three active interaction slots fit compact128')
    active=types!=0
    if np.any(primitives[active]<0) or np.any(primitives[active]>UINT32_MAX) or np.any(primitives[active]!=primitives[active].astype(np.uint32)):
        raise ValueError('Active primitive index is not an exact uint32')
    cast=vertices.astype(np.float32)
    if not np.array_equal(vertices[active].astype(float),cast[active].astype(float),equal_nan=True):
        raise ValueError('Native vertex values are not losslessly representable by the pinned float32 geometry table')
    values=np.zeros(n,RAW_GEOMETRY_DTYPE)
    values['primitives'].fill(UINT32_MAX)
    values['primitives'][active]=primitives[active].astype(np.uint32)
    values['vertices_xyz_m'][active]=cast[active]
    values['depth']=compact['depth'];values['group_idx'].fill(UINT32_MAX)
    for name in ('selection_score_m','max_point_error_m','point_rms_error_m','delay_error_sec'):
        values[name].fill(np.nan)
    kept=np.asarray(report.get('keep_compact_indices',[]),dtype=np.int64)
    if len(np.unique(kept))!=len(kept) or np.any(kept<0) or np.any(kept>=n):
        raise ValueError('Normalization keep_compact_indices are invalid')
    values['selected_mask'][kept]=True
    if report.get('raw_path_count',n)!=n or report.get('selected_path_count',len(kept))!=len(kept):
        raise ValueError('Normalization path counts disagree with stored raw/selected rows')
    groups=list(report.get('groups',[]));reasons=[''];reason_ids={'':0};status_counts=Counter();group_selected=[]
    for group_index,group in enumerate(groups):
        ids=np.asarray(group.get('path_indices',[]),dtype=np.int64)
        if not len(ids) or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=n):
            raise ValueError('Normalization group has invalid raw path indices')
        if np.any(values['group_idx'][ids]!=UINT32_MAX):raise ValueError('A raw path belongs to multiple normalization groups')
        topologies=[]
        for index in ids:
            used=active[index]
            topologies.append(tuple(zip(map(int,compact['object_ids'][index][used]),
                map(int,primitives[index][used]),map(int,types[index][used]))))
        if len(set(topologies))>1:
            raise ValueError('A normalization group merged different full active topologies')
        values['group_idx'][ids]=group_index
        status=str(group.get('status','unresolved'));status_counts[status]+=1
        flag=GROUP_CERTIFIED if status=='certified_unique' else GROUP_GEOMETRY_ONLY if status=='geometry_only' else GROUP_UNRESOLVED
        if len(ids)>1:flag|=GROUP_MULTIPLE
        values['geometry_flags'][ids]=flag|MISSING_CANDIDATE_METRICS
        reason=str(group.get('reason',''))
        if reason not in reason_ids:
            if len(reasons)>np.iinfo(np.uint16).max:raise ValueError('Reason catalog exceeds uint16 range')
            reason_ids[reason]=len(reasons);reasons.append(reason)
        values['reason_code'][ids]=reason_ids[reason]
        candidates=set()
        for candidate in group.get('candidates',[]):
            index=int(candidate['compact_index'])
            if index not in set(map(int,ids)) or index in candidates:raise ValueError('Candidate metrics do not match their raw group')
            candidates.add(index)
            values['geometry_flags'][index]&=np.uint16(0xffff ^ MISSING_CANDIDATE_METRICS)
            for name in ('selection_score_m','max_point_error_m','point_rms_error_m','delay_error_sec'):
                values[name][index]=candidate.get(name,np.nan)
            if candidate.get('agrees_with_unique_solution',False):values['geometry_flags'][index]|=CANDIDATE_AGREES
        selected=group.get('selected_compact_index')
        if selected is not None:
            if int(selected) not in set(map(int,ids)) or not values['selected_mask'][int(selected)]:
                raise ValueError('Group selected representative disagrees with the compact keep mask')
            group_selected.append(int(selected))
    if np.any(values['group_idx']==UINT32_MAX):
        raise ValueError('Every raw path must have an explicit normalization group, including unresolved paths')
    if set(group_selected)!=set(map(int,kept)):
        raise ValueError('Compact keep mask differs from the declared group representatives')
    # Preserve every reason, group assignment and per-path metric. Heavy image
    # solutions/triangle arrays are intentionally recomputable from archived
    # canonical mesh identity and primitive IDs, not copied per snapshot.
    summary=dict(schema=str(report.get('schema','unspecified')),admitted=bool(report.get('admitted',False)),
        raw_path_count=n,selected_path_count=len(kept),group_count=len(groups),
        group_status_counts=dict(status_counts),reason_catalog=reasons,
        inactive_slots_ignored=bool(report.get('inactive_slots_ignored',True)),
        different_topologies_merged=bool(report.get('different_topologies_merged',False)),
        occlusion_checked=bool(report.get('occlusion_checked',False)),
        detail_scope='Per-path metrics/selection/group/reason are stored; analytic plane/image arrays are reconstructable from canonical scene and primitive IDs')
    for name in ('unresolved_group_count','full_topology_groups','duplicate_topology_groups',
                 'selected_compact_vs_masked_native_relative_error','selected_compact_vs_masked_native_pass'):
        if name in report:summary[name]=report[name]
    for name in ('candidate_capacity','vehicle_world_bbox_guard'):
        if name in report:
            item=core._diagnostic_value(report[name]);encoded_item=core._json(item).encode('utf-8')
            if len(encoded_item)<=4096:summary[name]=item
            else:
                summary[name]=dict(summary_omitted=True,full_value_sha256=hashlib.sha256(encoded_item).hexdigest(),
                    detail_scope='large guard detail omitted from per-state JSON; full normalization report hash and canonical geometry retained')
                if isinstance(item,dict):
                    for key in ('passed','capacity','actual_count','saturated','status'):
                        if key in item and isinstance(item[key],(bool,int,float,str,type(None))):summary[name][key]=item[key]
    encoded=core._json(core._diagnostic_value(summary)).encode('utf-8')
    full_hash=hashlib.sha256(core._json(core._diagnostic_value(report)).encode('utf-8')).digest()
    return values,encoded,full_hash,len(groups),len(kept)


class RouteWriterV3(core.RouteWriter):
    """A core-compatible shard with jointly committed raw normalization evidence."""
    def __init__(self,*args,**kwargs):
        _assert_frozen_core()
        self._raw_geometry=[];self._raw_reports=[];self._raw_rows=[]
        self._trace_rows=[];self._pending_accounting_indices=set();self._awaiting_raw_accounting=False
        self._raw_state_indices=set();self._defer_auto_flush=False;self._flush_requested=False
        self._requires_resume=False;self._test_fault_hook=None
        super().__init__(*args,**kwargs)

    def _initialize(self):
        super()._initialize();f=self.handle
        f.attrs.update(raw_candidate_schema=VERSION,raw_candidate_quality_bit=RAW_CANDIDATE_ONLY,
            raw_candidate_role='raw_native_candidate; never a physical receiver anchor or target',
            raw_geometry_itemsize=RAW_GEOMETRY_DTYPE.itemsize,
            raw_geometry_scope='Only active native primitives/vertices; inactive vertices=0, primitives=uint32MAX; raw compact128 stays unchanged',
            raw_report_scope='All raw coefficients, active geometry, keep/group/status/reason and scalar candidate metrics retained; image/plane arrays recomputable from source scene',
            raw_runtime_handles='Canonical object IDs in compact/config replace process-local renderer handles',
            raw_checkpoint_scope='Core and auxiliary lengths are committed in one atomic sidecar; readable-HDF5 recovery only')
        self._dataset('raw_candidates/index',(0,),(None,),RAW_INDEX_DTYPE,'raw_snapshot','state_index;counts;SHA256',(128,))
        self._dataset('raw_candidates/geometry_offsets',(1,),(None,),'<i8','raw_snapshot_boundary','raw_geometry_index',(1024,))
        f['raw_candidates/geometry_offsets'][0]=0
        self._dataset('raw_candidates/geometry',(0,),(None,),RAW_GEOMETRY_DTYPE,'raw_path',
            'primitive:index;vertices:m;selected:bool;group:local_index;scores:m;delay_error:s;flags:bits',(4096,))
        f['raw_candidates/geometry'].attrs['geometry_flag_legend']=core._json(dict(
            unresolved=GROUP_UNRESOLVED,certified=GROUP_CERTIFIED,geometry_only=GROUP_GEOMETRY_ONLY,
            agrees=CANDIDATE_AGREES,multiple=GROUP_MULTIPLE,missing_metrics=MISSING_CANDIDATE_METRICS))
        self._dataset('raw_candidates/report_offsets',(1,),(None,),'<i8','raw_snapshot_boundary','UTF8_byte_index',(1024,))
        f['raw_candidates/report_offsets'][0]=0
        self._dataset('raw_candidates/report_bytes',(0,),(None,),'u1','normalization_report_UTF8_byte','UTF8',(16384,))
        self._dataset('trace_accounting',(0,),(None,),TRACE_ACCOUNTING_DTYPE,'trace_call',
            'primary_state_index;actual_solver_calls;source_revisit;raw_candidate_count;candidate_capacity',(1024,))
        f['trace_accounting'].attrs.update(itemsize_bytes=19,unknown_candidate_count=UINT32_MAX,
            scope='Solver calls associated with committed primary/revisit records; uncommitted pre-crash computation may be rerun and is not recoverable by this ledger')

    def _lengths(self):
        lengths=super()._lengths()
        lengths.update({name:len(self.handle[name]) for name in AUX_DATASETS})
        return lengths

    def _resume(self):
        if self.handle.attrs.get('raw_candidate_schema')!=VERSION:
            raise ValueError('Not a V3 raw-candidate shard; no silent schema upgrade')
        checkpoint=json.loads(self.checkpoint_path.read_text('utf-8'))
        if not set(AUX_DATASETS).issubset(checkpoint.get('lengths',{})):
            raise ValueError('Checkpoint lacks joint raw-candidate lengths')
        super()._resume()
        check=audit_raw_candidates_v3(self.partial_path,require_complete=False)
        if not check['passed']:raise ValueError('Resumed raw evidence is inconsistent: '+'; '.join(check['failures']))
        self._raw_state_indices=set(map(int,self.handle['raw_candidates/index']['state_index']))

    def append_snapshot(self,*args,**kwargs):
        if self._requires_resume:raise RuntimeError('Previous flush was interrupted; close and resume from the checkpoint')
        metadata=dict(args[1] if len(args)>1 and args[1] is not None else kwargs.get('metadata') or {})
        metadata.update({k:v for k,v in kwargs.items() if k not in ('metadata','native_golden')})
        physical=metadata.get('role','unspecified') not in ('raw_native_candidate','quarantine_attempt')
        required=bool(self.config.get('trace_accounting_required',False))
        if not required:return super().append_snapshot(*args,**kwargs)
        self._defer_auto_flush=True
        try:index=super().append_snapshot(*args,**kwargs)
        finally:self._defer_auto_flush=False
        if physical:self._pending_accounting_indices.add(index)
        else:self._awaiting_raw_accounting=True
        return index

    def append_raw_candidate(self,compact,metadata,raw_geometry,normalization_report,native_golden=None):
        if self._requires_resume:raise RuntimeError('Previous flush was interrupted; close and resume from the checkpoint')
        geometry,report_bytes,report_hash,groups,selected=_prepare_raw(compact,raw_geometry,normalization_report)
        fields=dict(metadata)
        fields.update(role='raw_native_candidate',receiver_anchor=False,
            quality_flags=int(fields.get('quality_flags',0))|RAW_CANDIDATE_ONLY,
            receiver_validated_until_sec=float(fields['time_sec']))
        # append_snapshot normally auto-flushes. Defer that single call until
        # this raw state's auxiliary evidence is in the same transaction.
        self._defer_auto_flush=True
        try:index=super().append_snapshot(compact,fields,native_golden=native_golden)
        finally:self._defer_auto_flush=False
        row=np.zeros((),RAW_INDEX_DTYPE)
        row['state_index']=index;row['raw_path_count']=len(compact);row['group_count']=groups;row['selected_path_count']=selected
        row['full_report_sha256']=np.frombuffer(report_hash,dtype='u1')
        self._raw_rows.append(row);self._raw_geometry.append(geometry);self._raw_reports.append(report_bytes)
        self._raw_state_indices.add(index)
        if self.config.get('trace_accounting_required',False):
            self._awaiting_raw_accounting=True
        elif self._flush_requested or len(self._states)>=self.batch_size:self.flush()
        return index

    def append_trace_accounting(self,primary_record_index,actual_solver_calls,source_revisit=False,*,
                                final_candidate_count=UINT32_MAX,candidate_capacity=4096):
        if self._requires_resume:raise RuntimeError('Previous flush was interrupted; close and resume')
        index=int(primary_record_index);calls=int(actual_solver_calls)
        if index!=primary_record_index or not 0<=index<self.state_count or index in self._raw_state_indices:
            raise ValueError('Trace accounting must reference an existing primary state, never a raw candidate')
        if calls!=actual_solver_calls or not 1<=calls<=np.iinfo(np.uint16).max:
            raise ValueError('Actual solver call count must be a positive uint16')
        count=UINT32_MAX if final_candidate_count is None else int(final_candidate_count)
        capacity=UINT32_MAX if candidate_capacity is None else int(candidate_capacity)
        if not 0<=count<=UINT32_MAX or not 1<=capacity<=UINT32_MAX:
            raise ValueError('Candidate count/capacity must fit the uint32 ledger')
        if (final_candidate_count is not None and count!=final_candidate_count) or (candidate_capacity is not None and capacity!=candidate_capacity):
            raise ValueError('Candidate count/capacity must be integral')
        row=np.zeros((),TRACE_ACCOUNTING_DTYPE)
        row['primary_record_index']=index;row['actual_solver_calls']=calls;row['source_revisit']=bool(source_revisit)
        row['final_candidate_count']=count;row['candidate_capacity']=capacity
        self._trace_rows.append(row)
        if not source_revisit:self._pending_accounting_indices.discard(index)
        self._awaiting_raw_accounting=False
        if self._flush_requested and not self._pending_accounting_indices:self.flush()

    def update_snapshot(self,index,**fields):
        if int(index) in self._raw_state_indices:
            if fields.get('receiver_anchor',False) or ('quality_flags' in fields and not int(fields['quality_flags'])&RAW_CANDIDATE_ONLY):
                raise ValueError('Raw native candidates cannot be promoted to physical receiver/target truth')
            if 'receiver_validated_until_sec' in fields:
                raise ValueError('A raw candidate never receives a local-propagation certificate')
        return super().update_snapshot(index,**fields)

    def _hook(self,stage):
        if self._test_fault_hook is not None:self._test_fault_hook(stage)

    def flush(self,metadata=None):
        if self._defer_auto_flush:
            self._flush_requested=True;return dict(deferred_until_raw_aux_ready=True)
        if self._closed:raise RuntimeError('Writer is closed')
        if self._requires_resume:raise RuntimeError('Previous flush was interrupted; close and resume')
        if self._pending_accounting_indices or self._awaiting_raw_accounting:
            raise ValueError('A trace is not fully accounted; append_trace_accounting must precede its joint checkpoint')
        if not len(self._raw_rows)==len(self._raw_geometry)==len(self._raw_reports):
            raise RuntimeError('Raw auxiliary buffers are unbalanced')
        f=self.handle;old=len(f['states'])
        if metadata is not None:self.checkpoint_metadata.update(metadata)
        certificates=dict(getattr(self,'_committed_certificates',{}))
        for index in self._updates:
            row=f['states'][index]
            certificates.setdefault(str(index),{key:row[key].item() for key in self._UPDATE_FIELDS})
        intent=dict(schema=core.SCHEMA+'.checkpoint',sequence=self._sequence,config_sha256=self.config_hash,
            lengths=self._lengths(),certificate_updates=certificates,metadata=self._committed_metadata)
        try:
            core._atomic_json(self.checkpoint_path,intent);self._hook('after_intent')
            if self._states:
                counts=np.asarray([len(item) for item in self._paths],dtype=np.int64);base=len(f['paths'])
                core._append(f['paths'],np.concatenate(self._paths))
                core._append(f['offsets'],base+np.cumsum(counts));core._append(f['states'],np.asarray(self._states,dtype=core.STATE_DTYPE))
            for index,fields in self._updates.items():
                row=f['states'][index]
                for name,value in fields.items():row[name]=value
                f['states'][index]=row
                certificates[str(index)]={key:row[key].item() for key in self._UPDATE_FIELDS}
            for name,rows,dtype in (('tasks',self._tasks,core.TASK_DTYPE),('intervals',self._intervals,core.INTERVAL_DTYPE),
                                    ('survey_index',self._survey_index,core.SURVEY_DTYPE)):
                if rows:core._append(f[name],np.asarray(rows,dtype=dtype))
            if self._survey_features:core._append(f['survey_features'],np.stack(self._survey_features))
            if self._goldens:
                core._append(f['golden_cfr/record_index'],[x[0] for x in self._goldens]);values=np.stack([x[1] for x in self._goldens])
                core._append(f['golden_cfr/real'],values.real);core._append(f['golden_cfr/imag'],values.imag)
            if self._events:core._append(f['events'],self._events)
            self._hook('after_master_append')
            if self._raw_rows:
                counts=np.asarray([len(x) for x in self._raw_geometry],dtype=np.int64)
                start=len(f['raw_candidates/geometry'])
                core._append(f['raw_candidates/geometry'],np.concatenate(self._raw_geometry))
                core._append(f['raw_candidates/geometry_offsets'],start+np.cumsum(counts))
                core._append(f['raw_candidates/index'],np.asarray(self._raw_rows,dtype=RAW_INDEX_DTYPE))
                sizes=np.asarray([len(x) for x in self._raw_reports],dtype=np.int64);start=len(f['raw_candidates/report_bytes'])
                core._append(f['raw_candidates/report_bytes'],np.frombuffer(b''.join(self._raw_reports),dtype='u1'))
                core._append(f['raw_candidates/report_offsets'],start+np.cumsum(sizes))
            if self._trace_rows:core._append(f['trace_accounting'],np.asarray(self._trace_rows,dtype=TRACE_ACCOUNTING_DTYPE))
            self._hook('after_aux_append');f.flush();self._hook('after_hdf5_flush_before_commit')
            self._sequence+=1
            cp=dict(schema=core.SCHEMA+'.checkpoint',sequence=self._sequence,config_sha256=self.config_hash,
                lengths=self._lengths(),certificate_updates=certificates,metadata=self.checkpoint_metadata)
            core._atomic_json(self.checkpoint_path,cp)
            self._committed_certificates=certificates;self._committed_metadata=dict(self.checkpoint_metadata)
            for buffer in (self._states,self._paths,self._goldens,self._intervals,self._tasks,self._survey_index,
                           self._survey_features,self._events,self._raw_rows,self._raw_geometry,self._raw_reports,self._trace_rows):buffer.clear()
            self._updates.clear();self._flush_requested=False;self._hook('after_commit')
        except BaseException:
            self._requires_resume=True
            raise
        disk_bytes=self.partial_path.stat().st_size
        if self.warning_bytes is not None and disk_bytes>self.warning_bytes and not self._warned:
            warnings.warn(f'Route shard grew to {disk_bytes}B; review capacity ledger',RuntimeWarning);self._warned=True
        if self.hard_budget_bytes is not None and disk_bytes>self.hard_budget_bytes:
            raise RuntimeError(f'Shard exceeds explicit byte budget ({disk_bytes}B); all committed data preserved')
        return dict(**cp,disk_bytes=disk_bytes,appended_states=len(f['states'])-old)

    def finish(self,*,require_tasks=True):
        self.flush();result=audit_route_file_v3(self.partial_path,require_complete=False,require_tasks=require_tasks)
        if not result['passed']:raise ValueError('V3 shard contract failed: '+'; '.join(result['failures']))
        self.handle.attrs['generation_complete']=True;self.handle.flush();self.handle.close();self._closed=True
        os.replace(self.partial_path,self.path);result.update(path=str(self.path),disk_bytes=self.path.stat().st_size)
        return result


def audit_raw_candidates_v3(path,*,require_complete=True):
    failures=[]
    with h5py.File(path,'r') as f:
        if f.attrs.get('raw_candidate_schema')!=VERSION:failures.append('missing V3 raw candidate schema')
        if require_complete and not bool(f.attrs.get('generation_complete',False)):failures.append('missing completion marker')
        if not all(name in f for name in AUX_DATASETS):
            return dict(passed=False,failures=failures+['missing raw auxiliary datasets'])
        index=f['raw_candidates/index'][:];geo=f['raw_candidates/geometry'][:]
        go=f['raw_candidates/geometry_offsets'][:];ro=f['raw_candidates/report_offsets'][:]
        states=f['states'][:];offsets=f['offsets'][:]
        if f['raw_candidates/geometry'].dtype!=RAW_GEOMETRY_DTYPE:failures.append('raw geometry dtype mismatch')
        if f['raw_candidates/index'].dtype!=RAW_INDEX_DTYPE:failures.append('raw index dtype mismatch')
        m=len(index)
        go_valid=len(go)==m+1 and go[0]==0 and go[-1]==len(geo) and not np.any(np.diff(go)<0)
        ro_valid=len(ro)==m+1 and ro[0]==0 and ro[-1]==len(f['raw_candidates/report_bytes']) and not np.any(np.diff(ro)<0)
        if not go_valid:failures.append('raw geometry offsets mismatch')
        if not ro_valid:failures.append('raw report offsets mismatch')
        refs=index['state_index'];valid=(refs>=0)&(refs<len(states))
        if not np.all(valid) or len(np.unique(refs))!=len(refs):failures.append('raw state references out of range or duplicate')
        raw_roles=np.flatnonzero(states['role']==b'raw_native_candidate')
        if not np.array_equal(np.sort(refs),raw_roles):failures.append('raw master role and auxiliary state mapping disagree')
        if np.all(valid):
            selected_states=states[refs]
            if (np.any(selected_states['receiver_anchor']) or np.any((selected_states['quality_flags']&RAW_CANDIDATE_ONLY)==0)
                or np.any(~np.isfinite(selected_states['receiver_validated_until_sec']))
                or np.any(selected_states['receiver_validated_until_sec']>selected_states['time_sec']+1e-13)):
                failures.append('raw native candidate was licensed as physical receiver truth')
            if go_valid:
                counts=np.diff(go);main_counts=offsets[refs+1]-offsets[refs]
                if not np.array_equal(counts,index['raw_path_count']) or not np.array_equal(counts,main_counts):
                    failures.append('raw geometry length differs from untouched compact path count')
                elif m:
                    path_rows=np.concatenate([np.arange(offsets[i],offsets[i+1],dtype=np.int64) for i in refs])
                    base_fields=f['paths'].fields(['depth','interaction_types'])[:][path_rows]
                    active=base_fields['interaction_types']!=0
                    if np.any(geo['primitives'][~active]!=UINT32_MAX) or np.any(geo['vertices_xyz_m'][~active]!=0):
                        failures.append('inactive raw geometry slots were not canonicalized')
                    if np.any(geo['depth']!=base_fields['depth']):failures.append('raw depth differs from source compact')
                    group_limit=np.repeat(index['group_count'],counts)
                    if np.any(geo['group_idx']>=group_limit):failures.append('raw group index out of range')
                    cs=np.r_[0,np.cumsum(geo['selected_mask'],dtype=np.int64)]
                    if not np.array_equal(cs[go[1:]]-cs[go[:-1]],index['selected_path_count']):
                        failures.append('raw selected count differs from persisted mask')
        if ro_valid and go_valid:
            blob=bytes(f['raw_candidates/report_bytes'][:])
            for i in range(m):
                try:
                    summary=json.loads(blob[ro[i]:ro[i+1]].decode('utf-8'))
                    if any(int(summary.get(key,-1))!=int(index[field][i]) for key,field in
                           (('raw_path_count','raw_path_count'),('group_count','group_count'),('selected_path_count','selected_path_count'))):
                        failures.append('raw normalization summary counts mismatch');break
                    catalog=summary['reason_catalog'];part=geo[go[i]:go[i+1]]
                    if not isinstance(catalog,list) or not catalog or catalog[0]!='' or np.any(part['reason_code']>=len(catalog)):
                        failures.append('raw normalization reason catalog mismatch');break
                    if len(np.unique(part['group_idx']))!=int(index['group_count'][i]):
                        failures.append('raw normalization groups are not a complete partition');break
                    proposed=part['selected_mask']
                    for group in np.unique(part['group_idx'][proposed]):
                        if np.count_nonzero(proposed & (part['group_idx']==group))>1:
                            failures.append('raw mask selects multiple representatives of one group');break
                except (ValueError,KeyError,TypeError,UnicodeDecodeError):
                    failures.append('raw normalization JSON is malformed');break
        accounting=f['trace_accounting'][:]
        if f['trace_accounting'].dtype!=TRACE_ACCOUNTING_DTYPE:failures.append('trace accounting dtype mismatch')
        accounting_refs=accounting['primary_record_index'];refs_ok=(accounting_refs>=0)&(accounting_refs<len(states))
        if not np.all(refs_ok) or np.any(accounting['actual_solver_calls']==0):failures.append('invalid trace accounting state/call reference')
        if np.all(refs_ok) and np.any(np.isin(states['role'][accounting_refs],[b'raw_native_candidate',b'quarantine_attempt'])):
            failures.append('trace accounting references a raw/quarantine state')
        baseline=accounting_refs[~accounting['source_revisit']]
        if len(np.unique(baseline))!=len(baseline):failures.append('primary trace counted more than once without revisit flag')
        primary_refs=np.flatnonzero(~np.isin(states['role'],[b'raw_native_candidate',b'quarantine_attempt']))
        accounting_complete=np.array_equal(np.sort(baseline),primary_refs)
        config=json.loads(f['config_json'][()].decode())
        if config.get('trace_accounting_required',False) and not accounting_complete:
            failures.append('required trace accounting lacks a unique baseline for every primary state')
        known=(accounting['final_candidate_count']!=UINT32_MAX)&(accounting['candidate_capacity']!=UINT32_MAX)
        if config.get('trace_accounting_required',False) and not np.all(known):
            failures.append('required trace accounting has unknown raw candidate count/capacity')
        if np.any(accounting['candidate_capacity']==0):failures.append('trace candidate capacity is zero')
        if np.all(refs_ok):
            reached=known&(accounting['final_candidate_count']>=accounting['candidate_capacity'])
            ref_states=states[accounting_refs]
            capacity_flagged=(ref_states['quality_flags']&32)!=0
            receiver_licensed=(ref_states['receiver_anchor'] &
                (ref_states['receiver_validated_until_sec']>ref_states['time_sec']+1e-13))
            if np.any(reached&(~capacity_flagged|receiver_licensed)):
                failures.append('raw pre-pruning search capacity reached without unadmitted capacity flag/certificate')
            baseline_known=known&~accounting['source_revisit']
            ref=accounting_refs[baseline_known]
            if np.any(accounting['final_candidate_count'][baseline_known]<(offsets[ref+1]-offsets[ref])):
                failures.append('raw candidate ledger count is smaller than its selected physical path count')
        aux_bytes=sum(int(f[name].size*f[name].dtype.itemsize) for name in AUX_DATASETS)
        report=dict(passed=not failures,failures=failures,raw_candidate_states=m,raw_candidate_path_records=len(geo),
            raw_selected_paths=int(np.count_nonzero(geo['selected_mask'])),raw_geometry_bytes_per_path=RAW_GEOMETRY_DTYPE.itemsize,
            raw_auxiliary_uncompressed_bytes=aux_bytes,raw_normalization_json_bytes=len(f['raw_candidates/report_bytes']))
        report.update(trace_accounting_rows=len(accounting),trace_accounting_complete=bool(accounting_complete),
            committed_solver_calls=int(accounting['actual_solver_calls'].sum()),
            committed_primary_solver_calls=int(accounting['actual_solver_calls'][~accounting['source_revisit']].sum()),
            committed_revisit_solver_calls=int(accounting['actual_solver_calls'][accounting['source_revisit']].sum()),
            trace_candidate_count_known_rows=int(np.count_nonzero(known)),
            trace_raw_capacity_reached_rows=int(np.count_nonzero(known&(accounting['final_candidate_count']>=accounting['candidate_capacity']))))
    return report


def audit_route_file_v3(path,*,require_complete=True,require_tasks=True):
    result=core.audit_route_file(path,require_complete=require_complete,require_tasks=require_tasks)
    raw=audit_raw_candidates_v3(path,require_complete=require_complete)
    result['passed']=bool(result['passed'] and raw['passed']);result['failures']+=raw['failures']
    result.update({k:v for k,v in raw.items() if k not in ('passed','failures')})
    return result

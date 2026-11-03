"""Small global task index and spawn-safe, on-demand training access.

No full H/Y cache is created. Inputs and supervision are separate return values.
Test access requires an explicit opt-in to help preserve the model-choice seal.
"""
from __future__ import annotations

from collections import defaultdict, OrderedDict
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .devset_view_v5 import DevsetViewV5 as DevsetView

INDEX_DTYPE = np.dtype([
    ('route_index','<u2'),('local_task_index','<u4'),('admitted','?'),
    ('eval_selected','?'),('event_target_los_change','?'),('cutoff_record_index','<u4'),
    ('cutoff_sec','<f8'),('target_time_sec','<f8')])


def build_index(release, output=None):
    release=Path(release).resolve()
    plan=json.loads((release/'plan.json').read_text('utf-8'))
    output=Path(output) if output else release/'task_index.h5'
    if output.exists():raise FileExistsError('Index exists; use a new output filename')
    temp=output.with_name(output.name+'.partial')
    table=[];rows=[]
    for route in plan['routes']:
        path=release/'routes'/route['split']/(route['route_family_id']+'.h5')
        if not path.exists():continue
        with h5py.File(path,'r') as f:
            if not f.attrs.get('generation_complete',False):raise ValueError('Partial shard in release plan')
            archived_config=json.loads(f['config_json'][()].decode())
            if archived_config.get('runtime_adapter_version')!='kap_devset.runtime.v5':
                raise ValueError('Final training index requires V5; earlier pilots remain diagnostic')
            archived=archived_config['route']
            if any(archived[k]!=route[k] for k in ('route_family_id','city','split','region_id')):
                raise ValueError('Index route identity differs from published shard')
            tasks=f['tasks'][:];states=f['states'][:]
        route_index=len(table)
        table.append(dict(route_family_id=route['route_family_id'],city=route['city'],
            region_id=route['region_id'],split=route['split'],mechanism=route['intended_mechanism'],
            shard=path.relative_to(release).as_posix()))
        refs=np.zeros(len(tasks),INDEX_DTYPE)
        refs['route_index']=route_index;refs['local_task_index']=np.arange(len(tasks))
        for field in ('admitted','eval_selected','cutoff_sec','target_time_sec'):refs[field]=tasks[field]
        primary=~np.isin(states['role'],[b'raw_native_candidate',b'quarantine_attempt'])
        for pass_index in np.unique(tasks['pass_index']):
            select=np.flatnonzero(tasks['pass_index']==pass_index)
            ids=np.flatnonzero(primary & (states['pass_index']==pass_index))
            ids=ids[np.argsort(states['time_sec'][ids])]
            times=states['time_sec'][ids]
            loc=np.searchsorted(times,tasks['cutoff_sec'][select]-1e-11,side='left')
            if np.any(loc>=len(ids)) or np.any(abs(times[np.minimum(loc,len(ids)-1)]-tasks['cutoff_sec'][select])>1e-10):
                raise ValueError('V5 cutoff labels require actual independently traced cutoff states')
            current=ids[loc];target_ids=tasks['target_record_index'][select]
            refs['cutoff_record_index'][select]=current
            refs['event_target_los_change'][select]=states['has_los'][current]!=states['has_los'][target_ids]
        rows.append(refs)
    if not rows:raise ValueError('No completed route shards')
    all_rows=np.concatenate(rows)
    with h5py.File(temp,'x') as f:
        f.attrs.update(schema='kap_devset.task_index.v2',generation_complete=False,
            release_relative_path=Path(os.path.relpath(release,output.parent)).as_posix(),independence_unit='route family / higher spatial region; windows overlap',
            dense_receiver_cache=False,target_families=plan.get('target_families',96),
            indexed_families=len(table),event_tag_scope='private offline stratified sampling and evaluation only; never a model input; exact cutoff vs direct target LoS',
            uniform_evaluation_unchanged=True)
        f.create_dataset('routes_json',data=json.dumps(table,ensure_ascii=False),dtype=h5py.string_dtype())
        f.create_dataset('task_refs',data=all_rows,chunks=True,compression='gzip',compression_opts=1,shuffle=True)
        f['task_refs'].attrs['axes']='task'
        f.attrs['generation_complete']=True
    temp.replace(output)
    return dict(path=str(output),families=len(table),task_indices=len(all_rows),
                admitted_tasks=int(np.count_nonzero(all_rows['admitted'])),bytes=output.stat().st_size)


class DevsetDataset:
    """Python sequence usable with a framework's own collation/DataLoader.

    ``condition`` returns public state/map inputs and a separate supervised CSI.
    ``observation`` additionally synthesizes every available FFT bin. Array and
    metadata shapes are retained, so the model adapter must explicitly choose
    its representation instead of silently dropping FFT bins or history.
    """
    def __init__(self,index_file,split='train',*,mode='condition',mapn=16,design=0,
                 beam='rotating',snr=10.,draw=0,eval_only=False,
                 families=None,allow_sealed_test=False,event_only=False):
        self.index_path=Path(index_file).resolve()
        if split not in ('train','validation','test'):raise ValueError('Unknown split')
        if split=='test' and not allow_sealed_test:raise PermissionError('Test is sealed for model choice; explicit evaluator opt-in required')
        if mode not in ('condition','observation'):raise ValueError('Unknown input mode')
        with h5py.File(self.index_path,'r') as f:
            if f.attrs.get('schema')!='kap_devset.task_index.v2' or not f.attrs.get('generation_complete',False):
                raise ValueError('Incomplete or unknown task index')
            self.release=(self.index_path.parent/str(f.attrs['release_relative_path'])).resolve()
            self.routes=json.loads(f['routes_json'][()].decode());refs=f['task_refs'][:]
        allowed={i for i,r in enumerate(self.routes) if r['split']==split and
                 (families is None or r['route_family_id'] in set(families))}
        keep=np.isin(refs['route_index'],list(allowed)) & refs['admitted']
        if eval_only:keep &= refs['eval_selected']
        if event_only:keep &= refs['event_target_los_change']
        self.refs=refs[keep]
        self.options=dict(mapn=mapn,design=design,beam=beam,snr=snr,draw=draw)
        self.mode=mode;self._views=OrderedDict()

    def __len__(self):return len(self.refs)

    def _view(self,i):
        if i not in self._views:self._views[i]=DevsetView(self.release/self.routes[i]['shard'])
        self._views.move_to_end(i)
        while len(self._views)>4:self._views.popitem(last=False)
        return self._views[i]

    def __getitem__(self,i):
        ref=self.refs[int(i)];view=self._view(int(ref['route_index']));task=int(ref['local_task_index'])
        if self.mode=='condition':
            inputs=view.load_condition(task,mapn=self.options['mapn'],design=self.options['design'])
        else:inputs=view.load(task,**self.options)
        label=view.load_evaluation(task)
        return dict(input=inputs,target_csi=np.asarray(label['target_csi'],np.complex64),
                    target_frequencies_hz=label['target_frequencies_hz'])

    def __getstate__(self):
        value=self.__dict__.copy();value['_views']=OrderedDict();return value


def nested_training_families(index_file,size,seed=20261027):
    """Balanced, nested8/16/32/64 family lists, independent of model errors."""
    if size not in (8,16,32,64):raise ValueError('Declared learning-curve sizes are8/16/32/64')
    with h5py.File(index_file,'r') as f:routes=json.loads(f['routes_json'][()].decode())
    buckets=defaultdict(list)
    for r in routes:
        if r['split']=='train':buckets[(r['city'],r['mechanism'])].append(r)
    for group in buckets.values():
        group.sort(key=lambda r:hashlib.sha256(f'{seed}/{r["route_family_id"]}'.encode()).digest())
    chosen=[];region_count=defaultdict(int)
    while len(chosen)<size:
        before=len(chosen)
        for key in sorted(buckets):
            if len(chosen)>=size:break
            group=buckets[key]
            if not group:continue
            at=min(range(len(group)),key=lambda j:region_count[group[j]['region_id']])
            r=group.pop(at);chosen.append(r['route_family_id']);region_count[r['region_id']]+=1
        if len(chosen)==before:raise ValueError('Insufficient completed training families')
    return chosen

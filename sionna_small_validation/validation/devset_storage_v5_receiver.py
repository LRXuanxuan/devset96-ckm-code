"""External V5 writer with joint interval/task revocation transactions.

No frozen source, physical path, state identity, task identity or array schema is
changed. Only admission fields may change in place. Their rollback/postimages
share the existing master/raw/phase atomic checkpoint, including automatic
flushes. Recovery requires readable HDF5, as in the frozen transaction engine.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np

from . import devset_storage as core
from .devset_storage_v5 import RouteWriterV5

VERSION='kap_devset.receiver_row_updates.v1'
METADATA_KEY='external_receiver_row_updates'
FIELDS={'intervals':('passed','used_as_leaf'),'tasks':('flags','admitted')}


def _plain(value):
    return value.item() if isinstance(value,np.generic) else value


class RouteWriterV5Receiver(RouteWriterV5):
    def __init__(self,*args,**kwargs):
        self._receiver_updates={name:{} for name in FIELDS}
        self._committed_receiver_images={name:{} for name in FIELDS}
        self._test_receiver_fault_hook=None
        super().__init__(*args,**kwargs)

    def _receiver_hook(self,stage):
        if self._test_receiver_fault_hook is not None:self._test_receiver_fault_hook(stage)

    def _resume(self):
        super()._resume()
        payload=self.checkpoint_metadata.get(METADATA_KEY)
        if payload is None:return
        if payload.get('schema')!=VERSION:raise ValueError('Unknown receiver row-update checkpoint schema')
        images=payload.get('rows',{})
        if set(images)!=set(FIELDS):raise ValueError('Receiver checkpoint row catalogs incomplete')
        for name,rows in images.items():
            for key,fields in rows.items():
                index=int(key)
                if str(index)!=key or not 0<=index<len(self.handle[name]) or set(fields)!=set(FIELDS[name]):
                    raise ValueError('Receiver checkpoint pre/postimage index or fields invalid')
                row=self.handle[name][index]
                for field,value in fields.items():row[field]=self._value(name,field,value)
                self._validate_row(name,row);self.handle[name][index]=row
        self._committed_receiver_images=copy.deepcopy(images)
        self.handle.flush()

    @staticmethod
    def _value(name,field,value):
        if field=='flags':
            if isinstance(value,(bool,np.bool_)) or int(value)!=value or not 0<=int(value)<=np.iinfo(np.uint32).max:
                raise ValueError('Task flags must be uint32')
            return int(value)
        if not isinstance(value,(bool,np.bool_)):raise ValueError(name+'.'+field+' requires bool')
        return bool(value)

    @staticmethod
    def _validate_row(name,row):
        if name=='tasks' and row['admitted'] and int(row['flags']):
            raise ValueError('A flagged task cannot be admitted')
        if name=='intervals' and row['passed'] and not np.isfinite([row['midpoint_nmse_db'],row['right_nmse_db']]).all():
            raise ValueError('A passed interval requires finite retained landmark checks')

    def _update_receiver_row(self,name,index,fields):
        if self._closed or self._requires_resume:raise RuntimeError('Writer is closed or requires checkpoint resume')
        if not fields or set(fields).difference(FIELDS[name]):raise ValueError('Only receiver admission fields may change')
        if isinstance(index,(bool,np.bool_)) or int(index)!=index:raise ValueError('Row index must be integral')
        index=int(index);buffer=self._intervals if name=='intervals' else self._tasks;stored=len(self.handle[name])
        if not 0<=index<stored+len(buffer):raise IndexError(index)
        row=self.handle[name][index] if index<stored else buffer[index-stored].copy()
        if index<stored:
            for field,value in self._receiver_updates[name].get(index,{}).items():row[field]=value
        original_passed=bool(row['passed']) if name=='intervals' else None
        for field,value in fields.items():row[field]=self._value(name,field,value)
        self._validate_row(name,row)
        if name=='intervals' and not original_passed and row['passed']:
            raise ValueError('A failed stored interval cannot be promoted; append a new independently checked interval')
        if index<stored:
            self._receiver_updates[name][index]={field:_plain(row[field]) for field in FIELDS[name]}
        else:buffer[index-stored]=row

    def update_interval(self,index,**fields):return self._update_receiver_row('intervals',index,fields)
    def update_task(self,index,**fields):return self._update_receiver_row('tasks',index,fields)

    def flush(self,metadata=None):
        if self._defer_auto_flush:return super().flush(metadata)
        if metadata is not None and METADATA_KEY in metadata:raise ValueError('Receiver transaction metadata is writer-owned')
        if self._closed or self._requires_resume:return super().flush(metadata)
        before=copy.deepcopy(self._committed_receiver_images);after=copy.deepcopy(before)
        for name,rows in self._receiver_updates.items():
            for index,fields in rows.items():
                key=str(index)
                if key not in before[name]:
                    original=self.handle[name][index]
                    before[name][key]={field:_plain(original[field]) for field in FIELDS[name]}
                after[name][key]=copy.deepcopy(fields)
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        def payload(rows):return dict(schema=VERSION,storage_relative_path='validation/devset_storage_v5_receiver.py',
            storage_source_sha256=source,rows=rows,scope='Only allowed interval/task admission fields; same joint atomic master/raw/phase checkpoint')
        # The frozen parent writes _committed_metadata in its intent and
        # checkpoint_metadata at commit. Thus rollback and postimages are in
        # the very same atomic sidecar as state certificates and append tails.
        self._committed_metadata=dict(self._committed_metadata,**{METADATA_KEY:payload(before)})
        self.checkpoint_metadata=dict(self.checkpoint_metadata,**{METADATA_KEY:payload(after)})
        try:
            result=super().flush(metadata)
        except BaseException:
            self._requires_resume=True
            raise
        if not result.get('deferred') and not result.get('deferred_until_raw_aux_ready'):
            self._committed_receiver_images=after
            for rows in self._receiver_updates.values():rows.clear()
        return result

    def _hook(self,stage):
        if stage=='after_intent':
            self._receiver_hook('after_joint_intent_before_row_updates')
            for name,rows in self._receiver_updates.items():
                for index,fields in rows.items():
                    row=self.handle[name][index]
                    for field,value in fields.items():row[field]=value
                    self.handle[name][index]=row
            self._receiver_hook('after_row_updates_before_master_append')
        super()._hook(stage)


RouteWriter=RouteWriterV5Receiver

"""Joint phase-reference ledger on unchanged V4 raw physical-surface evidence.

Primary paths use a separately declared Float64 image-delay propagation phase.
The original native compact rows remain intact in the V4 raw archive. This is
an explicit numerical model, not a claim that native cir() was unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from . import devset_storage as core
from . import devset_storage_v4 as v4

VERSION='kap_devset.image_delay_phase_reference.v5'
PHASE_DTYPE=np.dtype([('passband_b00','<c8'),('native_carrier_phasor','<c8'),('image_delay_sec','<f8')],align=False)
INDEX_DTYPE=np.dtype([('primary_record_index','<i8'),('raw_record_index','<i8'),
    ('path_offset','<u8'),('path_count','<u4'),('source_binding_sha256','u1',(32,))],align=False)
PHASE_DATASETS=('phase_reference/index','phase_reference/paths')
V4_SOURCE_SHA256='478da24deb89c76aa0d74f42a8dc40f9b69b6afc4f6848e66b0cdb22c4f5e00a'
STATE_BINDING_FIELDS=('time_sec','pass_index','rx_xyz_m','rx_velocity_mps','rx_pose_rad',
    'tx_xyz_m','tx_pose_rad','car_xyz_m','car_velocity_mps','car_pose_rad','has_dynamic_object')


def phase_source_binding_sha256(raw_selected,passband_b00,native_carrier_phasor,image_delay_sec):
    """Bind serialized source data, not authenticate its physical provenance."""
    digest=hashlib.sha256(b'kap_devset.phase_source_binding.v5\0')
    for array,dtype in ((raw_selected,None),(passband_b00,'<c8'),(native_carrier_phasor,'<c8'),(image_delay_sec,'<f8')):
        value=np.ascontiguousarray(array,dtype=dtype)
        digest.update(np.asarray([len(value)],dtype='<u8').tobytes());digest.update(value.tobytes())
    return digest.hexdigest()


def _phase_binding_failures(compact,original,physical_state,raw_state,a,phasor,tau,carrier,native_carrier):
    failures=[]
    if len(compact)!=len(a) or len(original)!=len(a):return ['phase source/physical path count mismatch']
    if (not np.isfinite(a).all() or not np.isfinite(phasor).all() or not np.isfinite(tau).all() or np.any(tau<=0)):
        return ['invalid passband/phasor/image-delay source']
    if not np.isfinite(carrier) or carrier<=0 or not np.isfinite(native_carrier) or native_carrier<=0:
        return ['invalid explicit physical/native carrier frequency']
    for field in compact.dtype.names:
        if field not in ('b00','delay_sec') and not np.array_equal(compact[field],original[field]):
            failures.append('phase correction changed a non-phase native compact field');break
    for field in STATE_BINDING_FIELDS:
        if not np.array_equal(physical_state[field],raw_state[field]):
            failures.append('phase source physical-state binding differs: '+field);break
    expected=(a.astype(np.complex128)*np.exp(-2j*np.pi*carrier*tau)).astype(np.complex64)
    if not np.array_equal(compact['b00'],expected) or not np.array_equal(compact['delay_sec'],tau):
        failures.append('primary carrier/delay values disagree with the declared phase reference')
    if not np.allclose(abs(phasor),1.,rtol=0,atol=2e-6):failures.append('native carrier phasor is not unit modulus')
    native_rebuilt=a.astype(np.complex128)*phasor.astype(np.complex128)
    scale=np.maximum(abs(original['b00'].astype(np.complex128)),np.finfo(np.float32).tiny)
    if np.any(abs(native_rebuilt-original['b00'])>5e-6*scale):
        failures.append('native passband/carrier factorization differs from original raw b00')
    # Pinned cir(): two real products then x-2pi*floor(x/(2pi)) in
    # Float32 (possibly fused). Eight unit-roundoffs times phase magnitude
    # cover product/constant/wrap errors, plus a small sincos rounding term.
    # This is a compatibility envelope, not a physical or Jones error bound.
    angle=-2*np.pi*native_carrier*original['delay_sec'];u=np.finfo(np.float32).eps/2
    bound=8*u*(np.abs(angle)+2*np.pi)+2e-6
    error=np.abs(np.angle(phasor.astype(np.complex128)*np.exp(-1j*angle)))
    if np.any(error>bound):failures.append('native carrier phasor incompatible with raw native delay/Float32 phase envelope')
    return failures


class RouteWriterV5(v4.RouteWriterV4):
    def __init__(self,*args,**kwargs):
        if hashlib.sha256(Path(v4.__file__).read_bytes()).hexdigest()!=V4_SOURCE_SHA256:
            raise RuntimeError('Pinned V4 raw transaction engine changed')
        self._phase_rows=[];self._phase_index=[];self._pending_phase=set()
        config=args[1] if len(args)>1 else kwargs.get('config',{})
        if not config.get('trace_accounting_required',False):
            raise ValueError('V5 requires trace_accounting_required=True for joint phase/raw/master commits')
        if not np.isfinite(config.get('carrier_hz',np.nan)) or float(config['carrier_hz'])<=0:
            raise ValueError('V5 requires an explicit finite positive carrier_hz')
        if not np.isfinite(config.get('native_carrier_frequency_hz',np.nan)) or float(config['native_carrier_frequency_hz'])<=0:
            raise ValueError('V5 requires the explicit actual native_carrier_frequency_hz')
        super().__init__(*args,**kwargs)

    def _initialize(self):
        super()._initialize()
        self.handle.attrs.update(phase_reference_schema=VERSION,phase_reference_path_itemsize=24,
            phase_reference_storage_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            phase_reference='Float64 image-delay propagation phase with native passband transport; original native baseband CIR remains raw',
            phase_reference_scope='Original native transport/Jones passband a00, with independently image-solved Float64 delay for carrier and baseband-frequency phase; original native CIR/compact remains raw; no exact electromagnetic field guarantee')
        self._dataset('phase_reference/index',(0,),(None,),INDEX_DTYPE,'physical_snapshot',
            'primary_state;original_raw_state;phase_path_offset;count',(1024,))
        self._dataset('phase_reference/paths',(0,),(None,),PHASE_DTYPE,'physical_path',
            'native_passband_a00:complex;original_native_carrier:unit_complex;Float64_image_delay:s',(4096,))

    def _lengths(self):
        value=super()._lengths();value.update({name:len(self.handle[name]) for name in PHASE_DATASETS});return value

    def _resume(self):
        checkpoint=json.loads(self.checkpoint_path.read_text('utf-8'))
        if not set(PHASE_DATASETS).issubset(checkpoint.get('lengths',{})):
            raise ValueError('Checkpoint lacks joint phase-reference lengths')
        super()._resume()
        result=audit_phase_reference(self.partial_path,require_complete=False)
        if not result['passed']:raise ValueError('Resumed phase evidence inconsistent: '+'; '.join(result['failures']))

    def append_snapshot(self,*args,**kwargs):
        index=super().append_snapshot(*args,**kwargs)
        metadata=dict(args[1] if len(args)>1 and args[1] is not None else kwargs.get('metadata') or {})
        metadata.update({k:v for k,v in kwargs.items() if k not in ('metadata','native_golden')})
        if metadata.get('role','unspecified') not in ('raw_native_candidate','quarantine_attempt'):
            self._pending_phase.add(index)
        return index

    def _snapshot_source(self,index):
        committed=len(self.handle['states'])
        if index<committed:
            offsets=self.handle['offsets'];return self.handle['states'][index],self.handle['paths'][int(offsets[index]):int(offsets[index+1])]
        local=index-committed
        return self._states[local],self._paths[local]

    def _raw_selected_source(self,index):
        for i,row in enumerate(self._raw_rows):
            if int(row['state_index'])==index:
                return self._snapshot_source(index)[1][self._raw_geometry[i]['selected_mask']]
        rows=self.handle['raw_candidates/index'][:];matches=np.flatnonzero(rows['state_index']==index)
        if len(matches)!=1:raise ValueError('Phase raw source has no unique archived geometry row')
        i=int(matches[0]);offsets=self.handle['raw_candidates/geometry_offsets']
        mask=self.handle['raw_candidates/geometry'][int(offsets[i]):int(offsets[i+1])]['selected_mask']
        return self._snapshot_source(index)[1][mask]

    def append_phase_reference(self,primary_record_index,raw_record_index,passband_b00,image_delay_sec,*,native_carrier_phasor,source_binding_sha256=None):
        index=int(primary_record_index);raw=int(raw_record_index)
        if index!=primary_record_index or raw!=raw_record_index or index not in self._pending_phase or raw not in self._raw_state_indices:
            raise ValueError('Phase reference must bind one new physical record to an original raw archive')
        a=np.asarray(passband_b00);phasor=np.asarray(native_carrier_phasor);tau=np.asarray(image_delay_sec,float)
        if a.ndim!=1 or tau.shape!=a.shape or phasor.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(phasor).all() or not np.isfinite(tau).all() or np.any(tau<=0):
            raise ValueError('Passband/image-delay ledger must be finite aligned positive-delay vectors')
        for values in (a,phasor):
            if not np.array_equal(values.astype(np.complex64).astype(np.complex128),values.astype(np.complex128)):
                raise ValueError('Native passband/phasor values must be losslessly representable by complex64')
        pstate,compact=self._snapshot_source(index);rstate,_=self._snapshot_source(raw);original=self._raw_selected_source(raw)
        failures=_phase_binding_failures(compact,original,pstate,rstate,a,phasor,tau,float(self.config['carrier_hz']),float(self.config['native_carrier_frequency_hz']))
        if failures:raise ValueError('; '.join(failures))
        digest=phase_source_binding_sha256(original,a,phasor,tau)
        if source_binding_sha256 is not None and source_binding_sha256!=digest:raise ValueError('Runtime phase-source binding SHA256 differs from archived source')
        row=np.zeros((),INDEX_DTYPE);row['primary_record_index']=index;row['raw_record_index']=raw
        row['path_offset']=len(self.handle['phase_reference/paths'])+sum(map(len,self._phase_rows));row['path_count']=len(a)
        row['source_binding_sha256']=np.frombuffer(bytes.fromhex(digest),'u1')
        values=np.zeros(len(a),PHASE_DTYPE);values['passband_b00']=a;values['native_carrier_phasor']=phasor;values['image_delay_sec']=tau
        self._phase_rows.append(values);self._phase_index.append(row);self._pending_phase.remove(index)

    def append_trace_accounting(self,primary_record_index,*args,**kwargs):
        if int(primary_record_index) in self._pending_phase:
            raise ValueError('Phase reference must be recorded before accounting/checkpoint')
        return super().append_trace_accounting(primary_record_index,*args,**kwargs)

    def flush(self,metadata=None):
        if self._defer_auto_flush:
            self._flush_requested=True;return dict(deferred=True)
        if self._pending_phase:raise ValueError('A physical snapshot lacks its phase-reference ledger')
        if len(self._phase_rows)!=len(self._phase_index):raise RuntimeError('Phase auxiliary buffers unbalanced')
        return super().flush(metadata)

    def _hook(self,stage):
        if stage=='after_aux_append':
            if self._phase_rows:core._append(self.handle['phase_reference/paths'],np.concatenate(self._phase_rows))
            if self._phase_index:core._append(self.handle['phase_reference/index'],np.asarray(self._phase_index,INDEX_DTYPE))
        if stage=='after_commit':self._phase_rows.clear();self._phase_index.clear()
        super()._hook(stage)

    def finish(self,*,require_tasks=True):
        self.flush();result=audit_route_file_v5(self.partial_path,require_complete=False,require_tasks=require_tasks)
        if not result['passed']:raise ValueError('V5 shard contract failed: '+'; '.join(result['failures']))
        self.handle.attrs['generation_complete']=True;self.handle.flush();self.handle.close();self._closed=True
        os.replace(self.partial_path,self.path);result.update(path=str(self.path),disk_bytes=self.path.stat().st_size);return result


def audit_phase_reference(path,*,require_complete=True):
    failures=[];count=0;path_count=0
    with h5py.File(path,'r') as f:
        if f.attrs.get('phase_reference_schema')!=VERSION or not all(n in f for n in PHASE_DATASETS):
            return dict(passed=False,failures=['missing explicit image-delay phase-reference ledger'])
        if require_complete and not f.attrs.get('generation_complete',False):failures.append('incomplete phase-reference shard')
        if f['phase_reference/index'].dtype!=INDEX_DTYPE or f['phase_reference/paths'].dtype!=PHASE_DTYPE:
            return dict(passed=False,failures=failures+['phase-reference dtype mismatch'])
        if f.attrs.get('phase_reference_storage_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            failures.append('phase-reference storage source hash mismatch')
        config=json.loads(f['config_json'][()].decode());carrier=float(config.get('carrier_hz',np.nan))
        native_carrier=float(config.get('native_carrier_frequency_hz',np.nan))
        states=f['states'][:];offsets=f['offsets'][:];paths=f['paths'][:]
        index=f['phase_reference/index'][:];values=f['phase_reference/paths'][:]
        physical=np.flatnonzero(~np.isin(states['role'],[b'raw_native_candidate',b'quarantine_attempt']))
        if not np.array_equal(np.sort(index['primary_record_index']),physical):failures.append('phase ledger does not cover every physical state exactly once')
        raw_index=f['raw_candidates/index'][:];go=f['raw_candidates/geometry_offsets'][:];geo=f['raw_candidates/geometry'][:]
        raw_map={int(row['state_index']):i for i,row in enumerate(raw_index)};cursor=0
        for row in index:
            i=int(row['primary_record_index']);raw=int(row['raw_record_index']);start=int(row['path_offset']);n=int(row['path_count']);end=start+n
            if start!=cursor or not 0<=start<=end<=len(values) or not 0<=i<len(states) or raw not in raw_map:
                failures.append('phase index bounds or original raw reference invalid');continue
            cursor=end;part=values[start:end];compact=paths[offsets[i]:offsets[i+1]];r=raw_map[raw]
            selected=geo[go[r]:go[r+1]]['selected_mask'];original=paths[offsets[raw]:offsets[raw+1]][selected]
            if len(compact)!=n or len(original)!=n:failures.append('phase source/physical path count mismatch');continue
            failures.extend(_phase_binding_failures(compact,original,states[i],states[raw],
                part['passband_b00'],part['native_carrier_phasor'],part['image_delay_sec'],carrier,native_carrier))
            digest=phase_source_binding_sha256(original,part['passband_b00'],part['native_carrier_phasor'],part['image_delay_sec'])
            if bytes(row['source_binding_sha256']).hex()!=digest:failures.append('phase source binding SHA256 mismatch')
            count+=1;path_count+=n
        if cursor!=len(values):failures.append('unindexed phase-reference path tail')
    return dict(passed=not failures,failures=failures,phase_reference_records=count,
        phase_reference_paths=path_count,phase_reference_bytes_per_path=24,
        phase_geometry_independently_recomputed=False)


def audit_route_file_v5(path,*,require_complete=True,require_tasks=True):
    result=v4.audit_route_file_v4(path,require_complete=require_complete,require_tasks=require_tasks)
    phase=audit_phase_reference(path,require_complete=require_complete)
    result['passed']=bool(result['passed'] and phase['passed']);result['failures']+=phase['failures']
    result.update({k:v for k,v in phase.items() if k not in ('passed','failures')});return result

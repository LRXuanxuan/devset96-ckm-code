"""V4 raw evidence: exact physical patches without rewriting native triangles.

The pinned V3 transaction engine remains unchanged. A 24B/raw-path mapping
indexes deduplicated patch identities and declared/physical face evidence.
Ideal image points are reconstructed from the pinned scene and actual endpoint
states; an original full-report hash is not a recalculated geometry proof.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from . import devset_storage as core
from . import devset_storage_v3 as v3
from .cities import COMPACT_DTYPE

VERSION='kap_devset.raw_native_candidates.v4'
V3_SOURCE_SHA256='b690b56b162028fdaa649abc7779091ca2ca296ad8b6a23a317953733b1d5512'
PATCH_SCHEMA='kap_devset.exact_coplanar_surface_patch.v1'
UINT64_MAX=np.iinfo(np.uint64).max
UINT32_MAX=v3.UINT32_MAX
RAW_CANDIDATE_ONLY=v3.RAW_CANDIDATE_ONLY
RAW_GEOMETRY_DTYPE=v3.RAW_GEOMETRY_DTYPE
RAW_INDEX_DTYPE=v3.RAW_INDEX_DTYPE
TRACE_ACCOUNTING_DTYPE=v3.TRACE_ACCOUNTING_DTYPE
MAPPING_CONTAINMENT=1
MAPPING_INTERNAL_SEAM=2
MAPPING_DECLARED_CHANGED=4
PATCH_IDENTITY_DTYPE=np.dtype([
    ('object_id','<u8'),('patch_sha256','u1',(32,)),
    ('triangle_geometry_multiset_sha256','u1',(32,)),('valid','?'),
    ('primitive_offset','<u8'),('primitive_count','<u4'),('boundary_edge_count','<u4'),
    ('material_offset','<u8'),('material_bytes','<u4')],align=False)
PATCH_MAPPING_DTYPE=np.dtype([
    ('patch_index','<u8'),('declared_primitive_id','<u4'),('physical_primitive_id','<u4'),
    ('containing_faces_offset','<u8'),('containing_faces_count','<u4'),('mapping_flags','u1')],align=False)
PATCH_DATASETS=('raw_candidates/physical_mapping','physical_patches/identities',
    'physical_patches/primitives','physical_patches/material_utf8','physical_patches/mappings',
    'physical_patches/containing_faces')
AUX_DATASETS=v3.AUX_DATASETS+PATCH_DATASETS


def _digest(value):
    if not isinstance(value,str) or len(value)!=64:raise ValueError('Physical patch identity must be a complete SHA256 hex string')
    try:return bytes.fromhex(value)
    except ValueError:raise ValueError('Invalid physical patch SHA256') from None


def _uint(value,maximum,label):
    result=int(value)
    if result!=value or not 0<=result<=maximum:raise ValueError(label+' must be an exact unsigned integer')
    return result


def _topology(value):
    return tuple((_uint(x['object_id'],UINT64_MAX,'object ID'),_digest(x['patch_id']),
                  int(x['interaction_type'])) for x in value)


def _identity_records(report):
    identities={}
    for item in report.get('patch_identities',[]):
        oid=_uint(item['object_id'],UINT64_MAX,'object ID');digest=_digest(item['patch_id'])
        primitive=tuple(_uint(x,UINT32_MAX,'patch primitive') for x in item['primitive_ids'])
        if not primitive or primitive!=tuple(sorted(set(primitive))):raise ValueError('Patch primitive membership must be nonempty, sorted and unique')
        if 'material_slot_key' not in item:raise ValueError('Patch identity must bind a material slot')
        material=core._json(core._diagnostic_value(item['material_slot_key'])).encode('utf-8')
        boundary=_uint(item['boundary_edge_count'],UINT32_MAX,'patch boundary edge count')
        component_hash=hashlib.sha256(np.asarray(primitive,dtype='<u4').tobytes()).hexdigest()
        if item.get('primitive_ids_sha256',component_hash)!=component_hash or item.get('triangle_count',len(primitive))!=len(primitive):
            raise ValueError('Patch component membership hash/count is inconsistent')
        geometry_hash=_digest(item['triangle_geometry_multiset_sha256'])
        if not isinstance(item.get('valid'),bool):raise ValueError('Patch identity needs an explicit validity flag')
        definition=(oid,digest,primitive,material,boundary,geometry_hash,item['valid'])
        key=(oid,digest)
        if key in identities and identities[key]!=definition:raise ValueError('One physical patch SHA has inconsistent membership/material metadata')
        identities[key]=definition
    return identities


def _prepare_raw_v4(compact,raw,report):
    compact=np.asarray(compact);n=len(compact)
    if compact.dtype!=COMPACT_DTYPE or compact.ndim!=1:raise ValueError('Raw compact must be unmodified128B source rows')
    primitives=np.asarray(raw['primitives']);vertices=np.asarray(raw['vertices_xyz_m'])
    types=np.asarray(raw.get('interaction_types',compact['interaction_types']))
    if primitives.shape!=(n,3) or vertices.shape!=(n,3,3) or types.shape!=(n,3):raise ValueError('Raw geometry must align with three original interaction slots')
    if not np.array_equal(types,compact['interaction_types']):raise ValueError('Original interaction metadata changed')
    if 'native_path_index' in raw and not np.array_equal(raw['native_path_index'],compact['path_index']):raise ValueError('Original native path permutation changed')
    if np.any(compact['depth']>3):raise ValueError('Raw compact depth exceeds three slots')
    active=types!=0
    if np.any(primitives[active]<0) or np.any(primitives[active]>UINT32_MAX) or np.any(primitives[active]!=primitives[active].astype(np.uint32)):
        raise ValueError('Original primitive is not an exact uint32')
    cast=vertices.astype(np.float32)
    if not np.array_equal(vertices[active].astype(float),cast[active].astype(float),equal_nan=True):raise ValueError('Native active vertices cannot be preserved losslessly as float32')
    identities=_identity_records(report);values=np.zeros(n,RAW_GEOMETRY_DTYPE)
    values['primitives'].fill(UINT32_MAX);values['primitives'][active]=primitives[active]
    values['vertices_xyz_m'][active]=cast[active];values['depth']=compact['depth'];values['group_idx'].fill(UINT32_MAX)
    for name in ('selection_score_m','max_point_error_m','point_rms_error_m','delay_error_sec'):values[name].fill(np.nan)
    kept=np.asarray(report.get('keep_compact_indices',[]),np.int64)
    if len(set(map(int,kept)))!=len(kept) or np.any(kept<0) or np.any(kept>=n):raise ValueError('Invalid raw keep mask')
    if report.get('raw_path_count',n)!=n or report.get('selected_path_count',len(kept))!=len(kept):raise ValueError('Raw/selected path count mismatch')
    values['selected_mask'][kept]=True
    groups=report.get('groups',[]);reasons=[''];reason_ids={'':0};status_counts=Counter();selected=[]
    # Per-slot logical mapping: identity key, declared face, physical face,
    # containing faces and explicit status. UINT64MAX will represent None.
    mappings=[[None]*3 for _ in range(n)];seen_topologies=set()
    for group_index,group in enumerate(groups):
        ids=np.asarray(group.get('path_indices',[]),np.int64);id_set=set(map(int,ids))
        if not len(ids) or len(ids)!=len(id_set) or np.any(ids<0) or np.any(ids>=n) or np.any(values['group_idx'][ids]!=UINT32_MAX):
            raise ValueError('Raw groups must partition the original candidate rows')
        values['group_idx'][ids]=group_index;status=str(group.get('status','unresolved'));status_counts[status]+=1
        flag=v3.GROUP_CERTIFIED if status=='certified_unique' else v3.GROUP_GEOMETRY_ONLY if status=='geometry_only' else v3.GROUP_UNRESOLVED
        if len(ids)>1:flag|=v3.GROUP_MULTIPLE
        values['geometry_flags'][ids]=flag|v3.MISSING_CANDIDATE_METRICS
        reason=str(group.get('reason',''))
        if reason not in reason_ids:
            if len(reasons)>np.iinfo(np.uint16).max:raise ValueError('Reason catalog exceeds uint16')
            reason_ids[reason]=len(reasons);reasons.append(reason)
        values['reason_code'][ids]=reason_ids[reason]
        known='physical_topology' in group
        physical=_topology(group['physical_topology']) if known else None
        if known:
            if physical in seen_topologies:raise ValueError('One complete physical topology was split across groups')
            seen_topologies.add(physical)
            if any(t!=1 for _,_,t in physical):raise ValueError('Physical patch certificate must be purely specular')
            declared={int(x['compact_index']):x['topology'] for x in group.get('declared_triangle_topologies',[])}
            if set(declared)!=id_set:raise ValueError('Every raw row requires its original declared triangle topology')
            for i in ids:
                used=active[i]
                if not np.array_equal(used,np.arange(3)<int(compact['depth'][i])) or len(physical)!=int(compact['depth'][i]):
                    raise ValueError('Physical topology depth/prefix differs from original candidate')
                original=tuple(zip(map(int,compact['object_ids'][i][used]),map(int,primitives[i][used]),map(int,types[i][used])))
                declaration=tuple((int(x['object_id']),int(x['primitive_id']),int(x['interaction_type'])) for x in declared[int(i)])
                if original!=declaration:raise ValueError('Declared triangle topology changed an original primitive')
                for slot,((oid,face,interaction),(poid,pid,pt)) in enumerate(zip(original,physical)):
                    if oid!=poid or interaction!=pt or (oid,pid) not in identities or face not in identities[(oid,pid)][2]:
                        raise ValueError('Raw declared face does not belong to its certified physical patch')
                    mappings[i][slot]=((oid,pid),face,UINT32_MAX,(),0)
        elif len(ids)>1 or group.get('selected_compact_index') is not None:
            raise ValueError('Unknown physical topology may only archive one unresolved unselected raw row')
        candidate_ids=set()
        for candidate in group.get('candidates',[]):
            i=int(candidate['compact_index'])
            if i not in id_set or i in candidate_ids:raise ValueError('Candidate metrics have an invalid raw row reference')
            candidate_ids.add(i);values['geometry_flags'][i]&=np.uint16(0xffff ^ v3.MISSING_CANDIDATE_METRICS)
            if int(candidate.get('native_path_index',compact['path_index'][i]))!=int(compact['path_index'][i]):raise ValueError('Candidate native path reference changed')
            for name in ('selection_score_m','max_point_error_m','point_rms_error_m','delay_error_sec'):values[name][i]=candidate.get(name,np.nan)
            if candidate.get('agrees_with_unique_solution',False):values['geometry_flags'][i]|=v3.CANDIDATE_AGREES
            correspondence=candidate.get('declared_to_physical',[])
            if not known or len(correspondence)!=len(physical):raise ValueError('Candidate requires a complete declared-to-physical correspondence')
            ideal_mappings=group.get('analytic_solution',{}).get('patch_mappings',[])
            if len(ideal_mappings)!=len(physical):raise ValueError('Candidate physical correspondence lacks its group image containment proof')
            for slot,point in enumerate(correspondence):
                oid,pid,_=physical[slot];declared_face=int(primitives[i,slot])
                if (int(point['object_id'])!=oid or _digest(point['patch_id'])!=pid or int(point['declared_primitive_id'])!=declared_face):
                    raise ValueError('Candidate physical mapping changed its declared face/object/patch')
                physical_face=_uint(point['physical_primitive_id'],UINT32_MAX,'physical primitive')
                faces=tuple(sorted(set((int(o),int(p)) for o,p in point['containing_faces'])))
                ideal=ideal_mappings[slot]
                if (int(ideal['object_id'])!=oid or _digest(ideal['patch_id'])!=pid or int(ideal['physical_primitive_id'])!=physical_face
                    or faces!=tuple(sorted(set((int(o),int(p)) for o,p in ideal['containing_faces'])))
                    or not np.array_equal(np.asarray(point['barycentric']),np.asarray(ideal['barycentric']))):
                    raise ValueError('Raw mapping differs from the same unique group image containment proof')
                membership=identities[(oid,pid)][2]
                if not identities[(oid,pid)][-1]:raise ValueError('Physical mapping references an invalid exact patch')
                if not faces or any(o!=oid or p not in membership for o,p in faces) or (oid,physical_face) not in faces:
                    raise ValueError('Containing physical faces are outside the same exact material patch')
                numerical=np.asarray([point['patch_boundary_distance_m'],point['plane_error_m'],point['float64_location_margin_m']],float)
                bary=np.asarray(point['barycentric'],float)
                if (not np.isfinite(numerical).all() or numerical[2]<=0 or numerical[0]<=numerical[2] or numerical[1]<0 or numerical[1]>numerical[2]
                    or bary.shape!=(3,) or not np.isfinite(bary).all() or abs(float(bary.sum())-1)>1e-7):
                    raise ValueError('Physical point containment certificate is incomplete or inconsistent')
                changed=physical_face!=declared_face
                if bool(point.get('raw_declared_primitive_changed'))!=changed:raise ValueError('Declared-to-physical change flag is inconsistent')
                flags=MAPPING_CONTAINMENT|(MAPPING_INTERNAL_SEAM if point.get('internal_mesh_seam',False) else 0)|(MAPPING_DECLARED_CHANGED if changed else 0)
                mappings[i][slot]=((oid,pid),declared_face,physical_face,tuple(p for _,p in faces),flags)
        chosen=group.get('selected_compact_index')
        if chosen is not None:
            chosen=int(chosen)
            if chosen not in id_set or not values['selected_mask'][chosen] or status!='certified_unique' or not values['geometry_flags'][chosen]&v3.CANDIDATE_AGREES:
                raise ValueError('Selected physical representative lacks a certified agreeing native candidate')
            if any(x is None or not x[-1]&MAPPING_CONTAINMENT for x in mappings[chosen][:int(compact['depth'][chosen])]):
                raise ValueError('Selected representative has missing physical containment mapping')
            selected.append(chosen)
    if np.any(values['group_idx']==UINT32_MAX) or set(selected)!=set(map(int,kept)):raise ValueError('Physical groups/representatives do not match the complete raw keep mask')
    if report.get('admitted') and (not len(kept) or any(g.get('status')!='certified_unique' for g in groups)):
        raise ValueError('Admitted report contains unresolved physical groups')
    if report.get('different_physical_topologies_merged',False):raise ValueError('Different physical patch topologies were merged')
    summary=dict(schema=str(report.get('schema','unspecified')),patch_schema=str(report.get('patch_schema',PATCH_SCHEMA)),
        admitted=bool(report.get('admitted',False)),raw_path_count=n,selected_path_count=len(kept),group_count=len(groups),
        group_status_counts=dict(status_counts),reason_catalog=reasons,inactive_slots_ignored=True,
        raw_declared_triangle_topologies_preserved=True,different_physical_topologies_merged=False,
        detail_scope='Original compact/active triangles retained; discrete patch/face correspondence stored; ideal points and full geometry proof require source+actual-state reconstruction')
    for key in ('unresolved_group_count','full_topology_groups','duplicate_topology_groups','raw_declared_triangle_group_count',
                'different_raw_triangle_topologies_merged','occlusion_checked','selected_compact_vs_masked_native_relative_error',
                'selected_compact_vs_masked_native_pass','candidate_capacity','vehicle_world_bbox_guard'):
        if key in report:
            value=core._diagnostic_value(report[key]);encoded=core._json(value).encode()
            summary[key]=value if len(encoded)<=4096 else dict(detail_omitted=True,sha256=hashlib.sha256(encoded).hexdigest())
    encoded=core._json(core._diagnostic_value(summary)).encode()
    full_hash=hashlib.sha256(core._json(core._diagnostic_value(report)).encode()).digest()
    return values,encoded,full_hash,len(groups),len(kept),identities,mappings


class RouteWriterV4(v3.RouteWriterV3):
    def __init__(self,*args,**kwargs):
        if hashlib.sha256(Path(v3.__file__).read_bytes()).hexdigest()!=V3_SOURCE_SHA256:raise RuntimeError('Pinned V3 transaction engine changed')
        self._physical_rows=[];self._identity_rows=[];self._primitive_rows=[];self._material_rows=[]
        self._mapping_rows=[];self._face_rows=[];self._identity_lookup={};self._identity_definitions={};self._mapping_lookup={}
        super().__init__(*args,**kwargs);self._load_catalog()

    def _initialize(self):
        super()._initialize();f=self.handle
        f.attrs.update(raw_candidate_schema=VERSION,raw_physical_mapping_bytes_per_path=24,physical_patch_schema=PATCH_SCHEMA,
            physical_mapping_scope='raw-path/slot->deduplicated exact patch and declared/physical face correspondence; uint64MAX=unavailable; no triangle or coefficient is rewritten',
            physical_containment_scope='Discrete containing faces retained; actual ideal point/continuous geometry proof requires zero-RT reconstruction from pinned source and actual state',
            physical_patch_source_sha256=core._json({name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('devset_surface_patches.py','devset_specular_audit_v4.py')}))
        self._dataset('raw_candidates/physical_mapping',(0,3),(None,3),'<u8','raw_path,interaction_slot','physical_patches/mappings row index',(4096,3))
        self._dataset('physical_patches/identities',(0,),(None,),PATCH_IDENTITY_DTYPE,'physical_patch','canonical_object;complete_SHA256;primitive/material membership',(128,))
        self._dataset('physical_patches/primitives',(0,),(None,),'<u4','patch_primitive_member','original_ordered_mesh_primitive_id',(4096,))
        self._dataset('physical_patches/material_utf8',(0,),(None,),'u1','material_slot_UTF8_byte','canonical_JSON',(4096,))
        self._dataset('physical_patches/mappings',(0,),(None,),PATCH_MAPPING_DTYPE,'declared_to_physical_mapping','patch_index;original/physical_face;containing_faces;status',(1024,))
        self._dataset('physical_patches/containing_faces',(0,),(None,),'<u4','physical_containing_face','primitive_id_in_same_canonical_object',(4096,))
        f['physical_patches/mappings'].attrs.update(flag_legend=core._json(dict(containment_available=1,internal_mesh_seam=2,declared_face_changed=4)),
            unknown_physical_primitive=UINT32_MAX)
        f['raw_candidates/physical_mapping'].attrs['unavailable_index']=UINT64_MAX

    def _lengths(self):
        result=super()._lengths();result.update({name:len(self.handle[name]) for name in PATCH_DATASETS});return result

    def _resume(self):
        if self.handle.attrs.get('raw_candidate_schema')!=VERSION:raise ValueError('Not a V4 patch shard; no implicit schema upgrade')
        cp=json.loads(self.checkpoint_path.read_text('utf-8'))
        if not set(AUX_DATASETS).issubset(cp.get('lengths',{})):raise ValueError('Checkpoint lacks joint V4 patch lengths')
        core.RouteWriter._resume(self)
        audit=audit_raw_candidates_v4(self.partial_path,require_complete=False)
        if not audit['passed']:raise ValueError('Resumed V4 evidence inconsistent: '+'; '.join(audit['failures']))
        self._raw_state_indices=set(map(int,self.handle['raw_candidates/index']['state_index']))

    def _load_catalog(self):
        f=self.handle;primitives=f['physical_patches/primitives'][:];material=bytes(f['physical_patches/material_utf8'][:]);faces=f['physical_patches/containing_faces'][:]
        for i,row in enumerate(f['physical_patches/identities'][:]):
            key=(int(row['object_id']),bytes(row['patch_sha256']));a=int(row['primitive_offset']);b=a+int(row['primitive_count'])
            c=int(row['material_offset']);d=c+int(row['material_bytes'])
            self._identity_lookup[key]=i;self._identity_definitions[key]=(key[0],key[1],tuple(map(int,primitives[a:b])),material[c:d],int(row['boundary_edge_count']),bytes(row['triangle_geometry_multiset_sha256']),bool(row['valid']))
        for i,row in enumerate(f['physical_patches/mappings'][:]):
            a=int(row['containing_faces_offset']);b=a+int(row['containing_faces_count'])
            key=(int(row['patch_index']),int(row['declared_primitive_id']),int(row['physical_primitive_id']),tuple(map(int,faces[a:b])),int(row['mapping_flags']))
            self._mapping_lookup[key]=i

    def append_raw_candidate(self,compact,metadata,raw_geometry,normalization_report,native_golden=None):
        if self._requires_resume:raise RuntimeError('Previous flush interrupted; close and resume')
        geo,encoded,full_hash,groups,selected,identities,mappings=_prepare_raw_v4(compact,raw_geometry,normalization_report)
        # Stage deduplicated catalog additions without modifying writer buffers
        # until core metadata validation has accepted the raw master state.
        new_ids={};new_maps={};id_rows=[];primitive_rows=[];material_rows=[];mapping_rows=[];face_rows=[]
        primitive_cursor=len(self.handle['physical_patches/primitives'])+sum(map(len,self._primitive_rows))
        material_cursor=len(self.handle['physical_patches/material_utf8'])+sum(map(len,self._material_rows))
        face_cursor=len(self.handle['physical_patches/containing_faces'])+sum(map(len,self._face_rows))
        for key,definition in identities.items():
            if key in self._identity_lookup:
                if self._identity_definitions[key]!=definition:raise ValueError('Persisted patch SHA membership/material changed')
                continue
            index=len(self.handle['physical_patches/identities'])+len(self._identity_rows)+len(id_rows);new_ids[key]=index
            oid,digest,primitive,material,boundary,geometry_hash,valid=definition;row=np.zeros((),PATCH_IDENTITY_DTYPE)
            row['object_id']=oid;row['patch_sha256']=np.frombuffer(digest,'u1');row['primitive_offset']=primitive_cursor;row['primitive_count']=len(primitive)
            row['triangle_geometry_multiset_sha256']=np.frombuffer(geometry_hash,'u1');row['valid']=valid
            row['boundary_edge_count']=boundary;row['material_offset']=material_cursor;row['material_bytes']=len(material)
            id_rows.append(row);primitive_rows.append(np.asarray(primitive,'<u4'));material_rows.append(material)
            primitive_cursor+=len(primitive);material_cursor+=len(material)
        indices=np.full((len(compact),3),UINT64_MAX,dtype='<u8')
        for i,path in enumerate(mappings):
            for slot,item in enumerate(path):
                if item is None:continue
                identity,declared,physical,faces,flags=item;patch_index=self._identity_lookup.get(identity,new_ids.get(identity))
                if patch_index is None:raise ValueError('Physical mapping references missing patch identity')
                key=(patch_index,declared,physical,faces,flags);index=self._mapping_lookup.get(key,new_maps.get(key))
                if index is None:
                    index=len(self.handle['physical_patches/mappings'])+len(self._mapping_rows)+len(mapping_rows);new_maps[key]=index
                    row=np.zeros((),PATCH_MAPPING_DTYPE);row['patch_index']=patch_index;row['declared_primitive_id']=declared;row['physical_primitive_id']=physical
                    row['containing_faces_offset']=face_cursor;row['containing_faces_count']=len(faces);row['mapping_flags']=flags
                    mapping_rows.append(row);face_rows.append(np.asarray(faces,'<u4'));face_cursor+=len(faces)
                indices[i,slot]=index
        fields=dict(metadata);fields.update(role='raw_native_candidate',receiver_anchor=False,
            quality_flags=int(fields.get('quality_flags',0))|RAW_CANDIDATE_ONLY,receiver_validated_until_sec=float(fields['time_sec']))
        self._defer_auto_flush=True
        try:index=core.RouteWriter.append_snapshot(self,compact,fields,native_golden=native_golden)
        finally:self._defer_auto_flush=False
        row=np.zeros((),RAW_INDEX_DTYPE);row['state_index']=index;row['raw_path_count']=len(compact);row['group_count']=groups;row['selected_path_count']=selected
        row['full_report_sha256']=np.frombuffer(full_hash,'u1')
        self._raw_rows.append(row);self._raw_geometry.append(geo);self._raw_reports.append(encoded);self._physical_rows.append(indices)
        self._identity_rows.extend(id_rows);self._primitive_rows.extend(primitive_rows);self._material_rows.extend(material_rows)
        self._mapping_rows.extend(mapping_rows);self._face_rows.extend(face_rows);self._identity_lookup.update(new_ids);self._mapping_lookup.update(new_maps)
        self._identity_definitions.update({key:identities[key] for key in new_ids});self._raw_state_indices.add(index)
        if self.config.get('trace_accounting_required',False):self._awaiting_raw_accounting=True
        elif self._flush_requested or len(self._states)>=self.batch_size:self.flush()
        return index

    def flush(self,metadata=None):
        if len(self._physical_rows)!=len(self._raw_rows):raise RuntimeError('V4 physical mapping/raw buffers unbalanced')
        return super().flush(metadata)

    def _hook(self,stage):
        if stage=='after_aux_append':
            f=self.handle
            if self._identity_rows:core._append(f['physical_patches/identities'],np.asarray(self._identity_rows,PATCH_IDENTITY_DTYPE))
            if self._primitive_rows:core._append(f['physical_patches/primitives'],np.concatenate(self._primitive_rows))
            if self._material_rows:core._append(f['physical_patches/material_utf8'],np.frombuffer(b''.join(self._material_rows),'u1'))
            if self._mapping_rows:core._append(f['physical_patches/mappings'],np.asarray(self._mapping_rows,PATCH_MAPPING_DTYPE))
            if self._face_rows:core._append(f['physical_patches/containing_faces'],np.concatenate(self._face_rows))
            if self._physical_rows:core._append(f['raw_candidates/physical_mapping'],np.concatenate(self._physical_rows))
        if stage=='after_commit':
            for rows in (self._physical_rows,self._identity_rows,self._primitive_rows,self._material_rows,self._mapping_rows,self._face_rows):rows.clear()
        super()._hook(stage)

    def finish(self,*,require_tasks=True):
        self.flush();result=audit_route_file_v4(self.partial_path,require_complete=False,require_tasks=require_tasks)
        if not result['passed']:raise ValueError('V4 shard contract failed: '+'; '.join(result['failures']))
        self.handle.attrs['generation_complete']=True;self.handle.flush();self.handle.close();self._closed=True
        os.replace(self.partial_path,self.path);result.update(path=str(self.path),disk_bytes=self.path.stat().st_size);return result


def audit_raw_candidates_v4(path,*,require_complete=True):
    # The frozen shared audit's only incompatible requirement is its version
    # tag. No file/attribute/global is patched to reuse its unchanged checks.
    result=v3.audit_raw_candidates_v3(path,require_complete=require_complete)
    failures=[x for x in result['failures'] if x!='missing V3 raw candidate schema']
    with h5py.File(path,'r') as f:
        if f.attrs.get('raw_candidate_schema')!=VERSION:failures.append('missing V4 raw candidate schema')
        if not all(name in f for name in PATCH_DATASETS):return dict(passed=False,failures=failures+['missing V4 physical patch datasets'])
        if failures:
            result.update(passed=False,failures=failures);return result
        ids=f['physical_patches/identities'][:];maps=f['physical_patches/mappings'][:];members=f['physical_patches/primitives'][:]
        material=f['physical_patches/material_utf8'][:];faces=f['physical_patches/containing_faces'][:];physical=f['raw_candidates/physical_mapping'][:]
        if f['physical_patches/identities'].dtype!=PATCH_IDENTITY_DTYPE or f['physical_patches/mappings'].dtype!=PATCH_MAPPING_DTYPE:failures.append('V4 catalog dtype mismatch')
        if f['raw_candidates/physical_mapping'].dtype!=np.dtype('<u8') or physical.shape!=(len(f['raw_candidates/geometry']),3):failures.append('V4 physical mapping is not24B per raw path')
        definitions=[];seen=set()
        for row in ids:
            a=int(row['primitive_offset']);b=a+int(row['primitive_count']);c=int(row['material_offset']);d=c+int(row['material_bytes'])
            primitive=tuple(map(int,members[a:b]));key=(int(row['object_id']),bytes(row['patch_sha256']))
            if b>len(members) or d>len(material) or not primitive or primitive!=tuple(sorted(set(primitive))) or key in seen or not any(key[1]):failures.append('invalid/duplicate physical patch identity or component membership')
            try:json.loads(bytes(material[c:d]))
            except (ValueError,UnicodeDecodeError):failures.append('invalid physical patch material slot JSON')
            seen.add(key);definitions.append((key[0],key[1],set(primitive)))
        for row in maps:
            patch=int(row['patch_index']);a=int(row['containing_faces_offset']);b=a+int(row['containing_faces_count']);flag=int(row['mapping_flags'])
            if patch>=len(ids) or b>len(faces):failures.append('physical mapping catalog reference out of range');continue
            membership=definitions[patch][2];contained=tuple(map(int,faces[a:b]));declared=int(row['declared_primitive_id']);actual=int(row['physical_primitive_id'])
            if declared not in membership or contained!=tuple(sorted(set(contained))) or any(x not in membership for x in contained):failures.append('physical face correspondence leaves its exact patch')
            if flag&~7:failures.append('unknown physical mapping status flags')
            if flag&MAPPING_CONTAINMENT:
                if not ids[patch]['valid'] or actual not in contained or not contained or bool(flag&MAPPING_DECLARED_CHANGED)!=(actual!=declared):failures.append('invalid certified physical primitive/containment flags')
            elif actual!=UINT32_MAX or contained or flag:failures.append('unresolved physical mapping lacks explicit missing sentinels')
        if physical.shape==(len(f['raw_candidates/geometry']),3):
            states=f['states'][:];offsets=f['offsets'][:];paths=f['paths'][:];geo=f['raw_candidates/geometry'][:];go=f['raw_candidates/geometry_offsets'][:]
            for k,raw in enumerate(f['raw_candidates/index'][:]):
                state=int(raw['state_index']);part=geo[go[k]:go[k+1]];pp=physical[go[k]:go[k+1]];compact=paths[offsets[state]:offsets[state+1]]
                topologies={};selected=Counter()
                for j,row in enumerate(compact):
                    active=row['interaction_types']!=0;top=[];complete=True
                    if np.any(pp[j,~active]!=UINT64_MAX):failures.append('inactive physical mappings are not unavailable')
                    for slot in np.flatnonzero(active):
                        ref=int(pp[j,slot])
                        if ref==UINT64_MAX:complete=False;continue
                        if ref>=len(maps):failures.append('raw physical mapping index out of range');complete=False;continue
                        m=maps[ref]
                        if int(m['patch_index'])>=len(definitions):complete=False;continue
                        oid,pid,_=definitions[int(m['patch_index'])]
                        if oid!=int(row['object_ids'][slot]) or int(m['declared_primitive_id'])!=int(part['primitives'][j,slot]):failures.append('physical mapping changed original object/primitive')
                        top.append((oid,pid,int(row['interaction_types'][slot])))
                        if part['selected_mask'][j] and not int(m['mapping_flags'])&MAPPING_CONTAINMENT:failures.append('selected physical path lacks certified point correspondence')
                    if part['selected_mask'][j] and not complete:failures.append('selected physical path has unavailable mapping')
                    if complete:
                        key=tuple(top);group=int(part['group_idx'][j]);topologies.setdefault(key,set()).add(group);selected[key]+=int(part['selected_mask'][j])
                if any(len(groups)!=1 for groups in topologies.values()):failures.append('one physical topology split across normalization groups')
                if any(count>1 for count in selected.values()):failures.append('multiple selected representatives of one physical topology')
                reverse={}
                for key,groups in topologies.items():
                    for group in groups:reverse.setdefault(group,set()).add(key)
                if any(len(keys)>1 for keys in reverse.values()):failures.append('different physical topologies merged into one group')
        extra=sum(int(f[name].size*f[name].dtype.itemsize) for name in PATCH_DATASETS)
        result.update(physical_patch_identities=len(ids),physical_patch_mapping_entries=len(maps),physical_mapping_bytes_per_raw_path=24,
            physical_patch_auxiliary_uncompressed_bytes=extra,physical_mapping_unavailable_slots=int(np.count_nonzero(physical==UINT64_MAX)),
            physical_patch_geometry_recomputed=False)
    result.update(passed=not failures,failures=failures);return result


def audit_route_file_v4(path,*,require_complete=True,require_tasks=True):
    result=core.audit_route_file(path,require_complete=require_complete,require_tasks=require_tasks)
    raw=audit_raw_candidates_v4(path,require_complete=require_complete)
    result['passed']=bool(result['passed'] and raw['passed']);result['failures']+=raw['failures']
    result.update({k:v for k,v in raw.items() if k not in ('passed','failures')});return result

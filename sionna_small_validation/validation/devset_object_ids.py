"""Canonical object identities independent of renderer names and handles.

Static city objects use an exact world-triangle multiset. Rigid moving actors
use declared semantic instance keys and immutable source mesh geometry. These
are object identities, never temporal propagation-path identities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

VERSION='kap_devset.canonical_objects.v1'
UINT64_MAX=np.iinfo(np.uint64).max


def triangle_multiset_fingerprint(vertices,faces,*,coordinate_frame='world_m'):
    """Exact numeric triangle multiset, invariant to vertex/face permutations.

    Triangle winding is excluded from geometric identity. Duplicate triangles
    retain multiplicity. IEEE signed zero is normalized; no coordinate rounding
    or tolerance merges distinct triangles. Nonfinite geometry is rejected.
    """
    vertices=np.asarray(vertices,dtype='<f8').reshape(-1,3)
    faces=np.asarray(faces,dtype=np.int64).reshape(-1,3)
    if not np.isfinite(vertices).all():raise ValueError('Nonfinite mesh geometry')
    if np.any(faces<0) or np.any(faces>=len(vertices)):raise ValueError('Invalid triangle indices')
    tri=vertices[faces].copy()
    tri[tri==0.]=0.
    if len(tri):
        order=np.lexsort((tri[:,:,2],tri[:,:,1],tri[:,:,0]),axis=1)
        tri=np.take_along_axis(tri,order[:,:,None],axis=1)
        flat=tri.reshape(-1,9)
        row_order=np.lexsort(tuple(flat[:,j] for j in range(8,-1,-1)))
        payload=np.ascontiguousarray(flat[row_order],dtype='<f8').tobytes()
    else:payload=b''
    header=json.dumps(dict(version=VERSION,coordinates=str(coordinate_frame),encoding='little_endian_float64',
                           triangle_count=len(faces),vertex_permutation_invariant=True,
                           triangle_permutation_invariant=True,winding_invariant=True,
                           signed_zero_normalized=True,rounding=False),sort_keys=True,separators=(',',':')).encode()
    digest=hashlib.sha256(header+b'\0'+payload).hexdigest()
    return dict(canonical_geometry_sha256=digest,triangle_count=int(len(faces)),
                coordinate_frame=str(coordinate_frame),canonicalization='exact triangle multiset; signed-zero normalized; no rounding; winding ignored')


def mesh_fingerprint(obj,*,coordinate_frame='world_m'):
    import mitsuba as mi
    mesh=obj.mi_mesh if hasattr(obj,'mi_mesh') else obj
    params=mi.traverse(mesh)
    vertices=np.asarray(params['vertex_positions']).copy()
    faces=np.asarray(params['faces']).copy()
    canonical=triangle_multiset_fingerprint(vertices,faces,coordinate_frame=coordinate_frame)
    diagnostic=dict(raw_vertex_positions_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
                    raw_triangle_indices_sha256=hashlib.sha256(faces.tobytes()).hexdigest(),
                    raw_vertex_dtype=str(vertices.dtype),raw_face_dtype=str(faces.dtype))
    return canonical,diagnostic


def _identity(city,role,geometry_sha256,instance_key):
    description=dict(schema=VERSION,city=str(city),semantic_role=str(role),
                     identity_geometry_sha256=str(geometry_sha256),instance_key=str(instance_key))
    full=hashlib.sha256(json.dumps(description,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    ident=int.from_bytes(bytes.fromhex(full)[:8],'little')
    if ident==UINT64_MAX:raise RuntimeError('Reserved canonical object ID; choose an explicitly revised protocol')
    return ident,full


class CanonicalObjectCatalog:
    """Cache city geometry once; runtime mappings only perform identity lookups.

    Runtime aliases are diagnostic and deliberately excluded from stable config
    records. Register each future moving actor once using a source mesh and a
    declared instance key. The same source can have many distinct instances.
    """
    def __init__(self,scene,city,*,exclude_objects=()):
        self.city=str(city);self._objects={};self._entries={};self._ids={};self._source_cache={}
        excluded={id(obj) for obj in exclude_objects}
        for obj in scene.objects.values():
            if id(obj) not in excluded:self.register_static_object(obj)

    def _register(self,obj,identity,diagnostic):
        ident=identity['stable_id'];key=id(obj)
        if key in self._entries:
            if self._entries[key]['identity_sha256']!=identity['identity_sha256']:
                raise ValueError('Attempt to change a registered object identity')
            return ident
        if ident in self._ids and self._ids[ident]!=key:
            raise ValueError('Coincident canonical object identities require distinct declared semantic instance keys; auto names cannot resolve this')
        self._objects[key]=obj
        self._entries[key]=dict(identity,order_diagnostic=dict(diagnostic))
        self._ids[ident]=key
        return ident

    def register_static_object(self,obj,*,role='static_scene_mesh',instance_key=''):
        key=id(obj)
        if key in self._entries:
            old=self._entries[key]
            if old['semantic_role']!=role or old['instance_key']!=str(instance_key):
                raise ValueError('Static object semantic identity cannot be changed after registration')
            return old['stable_id']
        geometry,diagnostic=mesh_fingerprint(obj)
        ident,full=_identity(self.city,role,geometry['canonical_geometry_sha256'],instance_key)
        return self._register(obj,dict(stable_id=ident,identity_sha256=full,semantic_role=role,
            instance_key=str(instance_key),identity_geometry_sha256=geometry['canonical_geometry_sha256'],
            triangle_count=geometry['triangle_count'],identity_frame='static world triangle multiset',
            name='static_mesh_'+full[:20],id_rule=VERSION,
            interpretation='object identity; not a propagation-path ID'),diagnostic)

    def register_rigid_object(self,obj,*,role,source_mesh_path,instance_key):
        if not role or not instance_key:raise ValueError('A rigid actor needs explicit role and instance key')
        source=Path(source_mesh_path).resolve()
        if not source.is_file():raise FileNotFoundError(source)
        source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if source_sha not in self._source_cache:
            import mitsuba as mi
            mesh=mi.load_dict(dict(type='ply',filename=str(source)))
            geometry,diagnostic=mesh_fingerprint(mesh,coordinate_frame='source_local_m')
            self._source_cache[source_sha]=(geometry,diagnostic)
        geometry,diagnostic=self._source_cache[source_sha]
        ident,full=_identity(self.city,role,geometry['canonical_geometry_sha256'],instance_key)
        # Source raw bytes are provenance only; source geometry defines identity
        # so a harmless vertex/face reordering does not change the actor ID.
        return self._register(obj,dict(stable_id=ident,identity_sha256=full,semantic_role=str(role),
            instance_key=str(instance_key),identity_geometry_sha256=geometry['canonical_geometry_sha256'],
            source_mesh_sha256=source_sha,triangle_count=geometry['triangle_count'],
            identity_frame='immutable source mesh plus declared actor instance; independent of world pose',
            name=str(role)+'_'+str(instance_key),id_rule=VERSION,
            interpretation='rigid object instance identity; not a propagation-path ID'),diagnostic)

    def id_for(self,obj):
        try:return self._entries[id(obj)]['stable_id']
        except KeyError:raise KeyError('Unregistered object; declare its canonical identity before tracing') from None

    def runtime_mapping(self,scene):
        result={}
        for obj in scene.objects.values():
            runtime=int(obj.object_id);ident=self.id_for(obj)
            if runtime in result and result[runtime]!=ident:raise RuntimeError('Ambiguous renderer handle')
            result[runtime]=ident
        return result

    def records_for(self,scene,*,extra_objects=()):
        objects=list(scene.objects.values())+list(extra_objects)
        keys={id(obj) for obj in objects}
        records=[]
        for key in keys:
            if key not in self._entries:raise KeyError('Unregistered object in config catalogue')
            records.append({name:value for name,value in self._entries[key].items() if name!='order_diagnostic'})
        return sorted(records,key=lambda r:r['stable_id'])

    def runtime_aliases(self,scene):
        """Capture aliases separately; never include this in stable config hash."""
        return [dict(runtime_name=str(name),runtime_id=int(obj.object_id),stable_id=self.id_for(obj),
                     **self._entries[id(obj)]['order_diagnostic'])
                for name,obj in scene.objects.items()]

    def resolve_name(self,scene,name):
        return self.id_for(scene.objects[name])

    def verify_static_geometry(self,scene):
        """Explicit expensive audit, not called by runtime_mapping or tracing."""
        for obj in scene.objects.values():
            entry=self._entries[id(obj)]
            if entry['identity_frame']!='static world triangle multiset':continue
            geometry,_=mesh_fingerprint(obj)
            if geometry['canonical_geometry_sha256']!=entry['identity_geometry_sha256']:
                raise ValueError('Static object geometry changed after identity registration')
        return True

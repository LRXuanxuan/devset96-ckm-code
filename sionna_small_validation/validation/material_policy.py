"""Deterministic geometric facade partition; installed scene assets are read-only.

The intervention is on vertical triangle subsets of an explicit source-material
allowlist. It is not a measured building-wall classification. Nonvertical faces,
named roofs and ground meshes retain their original material in both scenes.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid
from xml.etree import ElementTree as ET

import numpy as np

from .common import save_json, sha256

POLICY_VERSION='vertical-facades-v1'
SOURCE_MATERIALS=('marble','brick')
VERTICAL_COS_TOL=1e-4
FACADE_PREFIX='kap_facade_base_'
PLY_TYPES={'char':'i1','uchar':'u1','int8':'i1','uint8':'u1','short':'<i2','ushort':'<u2',
           'int16':'<i2','uint16':'<u2','int':'<i4','uint':'<u4','int32':'<i4',
           'uint32':'<u4','float':'<f4','float32':'<f4','double':'<f8','float64':'<f8'}


def read_triangular_ply(path):
    """Strict installed-scene PLY reader; rejects unsupported layouts."""
    with Path(path).open('rb') as f:
        lines=[]
        while True:
            line=f.readline()
            if not line:raise ValueError('Incomplete PLY header')
            lines.append(line.decode('ascii').rstrip('\r\n'))
            if lines[-1]=='end_header':break
        if lines[0]!='ply' or 'format binary_little_endian 1.0' not in lines:
            raise ValueError('Only audited binary-little-endian PLY is supported')
        vertices=None;faces=None;props=[];element=None;face_properties=[]
        for line in lines:
            words=line.split()
            if not words:continue
            if words[0]=='element':
                element=words[1]
                if element=='vertex':vertices=int(words[2])
                elif element=='face':faces=int(words[2])
                else:raise ValueError('Unsupported PLY element '+element)
            if words[0]=='property':
                if element=='vertex' and words[1]!='list':props.append((words[2],PLY_TYPES[words[1]]))
                elif element=='face':face_properties.append(words[1:])
                else:raise ValueError('Unsupported PLY property')
        if vertices is None or faces is None or face_properties!=[['list','uchar','int','vertex_indices']]:
            raise ValueError('Expected one triangular face-index list')
        dtype=np.dtype(props,align=False)
        raw=f.read(vertices*dtype.itemsize)
        vertex=np.frombuffer(raw,dtype=dtype)
        face=np.frombuffer(f.read(),dtype=np.dtype([('n','u1'),('v','<i4',(3,))]))
        if len(vertex)!=vertices or len(face)!=faces or np.any(face['n']!=3):
            raise ValueError('Unexpected PLY payload/triangulation')
        xyz=np.column_stack([vertex[k] for k in ('x','y','z')]).astype(np.float64)
        if np.any(face['v']<0) or np.any(face['v']>=vertices):raise ValueError('Invalid vertex index')
        return lines,raw,xyz,face.copy()


def write_partition(path,header,vertex_raw,faces):
    text='\n'.join(('element face '+str(len(faces))) if line.startswith('element face ') else line
                   for line in header)+'\n'
    with Path(path).open('xb') as f:
        f.write(text.encode('ascii'));f.write(vertex_raw);f.write(faces.tobytes())


def vertical_mask(xyz,faces,tolerance=VERTICAL_COS_TOL):
    tri=xyz[faces['v']]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1)
    valid=length>1e-12
    return valid & (np.abs(cross[:,2])<=tolerance*length)


def prepare_facade_scene(asset):
    asset=Path(asset).resolve();tree=ET.parse(asset);root=tree.getroot()
    shapes=list(root.findall('shape'));bsdfs={x.attrib['id']:x for x in root.findall('bsdf')}
    meshes={}
    for shape in shapes:
        filename=shape.find("string[@name='filename']")
        if filename is None or shape.attrib.get('type')!='ply':raise ValueError('Unsupported scene shape')
        if shape.find('transform') is not None:raise ValueError('Audit transformed assets before partitioning')
        p=(asset.parent/filename.attrib['value']).resolve();meshes[p]=sha256(p)
    signature=dict(version=POLICY_VERSION,xml=sha256(asset),meshes={str(p):h for p,h in sorted(meshes.items())},
                   materials=SOURCE_MATERIALS,vertical_cos_tolerance=VERTICAL_COS_TOL)
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    if os.name=='nt':
        base=Path(os.environ.get('LOCALAPPDATA',str(Path.home()/'.cache')))/'sionna-small-validation'/'facade-assets'
    else:base=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'sionna-small-validation'/'facade-assets'
    base.mkdir(parents=True,exist_ok=True);target=base/digest[:24]
    if target.exists():
        manifest=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
        if manifest['content_digest']!=digest:raise RuntimeError('Facade cache digest mismatch')
        for name,h in manifest['generated_files_sha256'].items():
            if sha256(target/name)!=h:raise RuntimeError('Facade cache payload changed')
        return str(target/'scene.xml'),manifest
    stage=base/('.stage-'+uuid.uuid4().hex);stage.mkdir()
    rows=[];used=set()
    for number,shape in enumerate(shapes):
        name=shape.attrib['id'];filename=shape.find("string[@name='filename']")
        source=(asset.parent/filename.attrib['value']).resolve()
        filename.set('value',source.as_posix())
        ref=shape.find("ref[@name='bsdf']");material=ref.attrib['id']
        protected=any(token in name.lower() for token in ('ground','plane','-roof','_roof'))
        if material not in SOURCE_MATERIALS or protected:continue
        header,raw,xyz,faces=read_triangular_ply(source);mask=vertical_mask(xyz,faces)
        if not mask.any():continue
        # Both partitions retain the exact original vertex bytes and original
        # triangle indices. Together they are the original triangle multiset.
        facade_name=f'facade_{number:04d}.ply';rest_name=f'other_{number:04d}.ply'
        write_partition(stage/facade_name,header,raw,faces[mask])
        if (~mask).any():write_partition(stage/rest_name,header,raw,faces[~mask])
        selected=copy.deepcopy(shape);selected.attrib['id']=f'mesh-kap_facade_{number:04d}'
        selected.find("string[@name='filename']").set('value',(target/facade_name).as_posix())
        selected.find("ref[@name='bsdf']").set('id',FACADE_PREFIX+material)
        root.append(selected)
        if (~mask).any():filename.set('value',(target/rest_name).as_posix())
        else:root.remove(shape)
        used.add(material)
        rows.append(dict(source_shape_id=name,source_material=material,source_mesh_sha256=meshes[source],
            source_faces=len(faces),selected_faces=int(mask.sum()),unchanged_faces=int((~mask).sum()),
            selected_face_indices_sha256=hashlib.sha256(np.flatnonzero(mask).astype('<i8').tobytes()).hexdigest(),
            unchanged_vertices_sha256=hashlib.sha256(raw).hexdigest(),
            triangle_partition_exact=bool(int(mask.sum())+int((~mask).sum())==len(faces)),
            selected_mesh=facade_name))
    if not rows:raise RuntimeError('No predeclared vertical facades; do not silently broaden intervention')
    for material in sorted(used):
        new=copy.deepcopy(bsdfs[material]);new.attrib['id']=FACADE_PREFIX+material;root.insert(0,new)
    ET.indent(root)
    (stage/'scene.xml').write_text(ET.tostring(root,encoding='unicode'),encoding='utf-8')
    manifest=dict(policy_version=POLICY_VERSION,content_digest=digest,source_asset=str(asset),
        source_asset_sha256=sha256(asset),source_material_allowlist=SOURCE_MATERIALS,
        protected_source_tokens=['ground','plane','-roof','_roof'],vertical_cos_tolerance=VERTICAL_COS_TOL,
        meaning='geometrically vertical triangle subsets; not measured architectural wall semantics',
        selected_partitions=rows,baseline_material_groups=[FACADE_PREFIX+m for m in sorted(used)],
        source_geometry_unchanged_as_triangle_multiset=True,
        generated_files_sha256={p.name:sha256(p) for p in stage.iterdir() if p.is_file()})
    save_json(stage/'manifest.json',manifest)
    try:stage.rename(target)
    except OSError:
        if not (target/'manifest.json').exists():raise
    return str(target/'scene.xml'),manifest


def apply_facade_intervention(scene,rt,manifest):
    """Fail closed if partitioned objects are not exactly as declared."""
    import mitsuba as mi
    names={name:obj.radio_material.name for name,obj in scene.objects.items()}
    targets=[name for name,m in names.items() if m in manifest['baseline_material_groups']]
    if not targets:raise RuntimeError('Expected explicit facade material groups are absent')
    before={name:mesh_geometry_hash(obj,mi) for name,obj in scene.objects.items()}
    material=rt.ITURadioMaterial(name='v2_vertical_facades_wood',itu_type='wood',thickness=.1)
    changes=[]
    for name in targets:
        obj=scene.objects[name];params=mi.traverse(obj.mi_mesh)
        xyz=np.asarray(params['vertex_positions']).reshape(-1,3)
        indices=np.asarray(params['faces']).reshape(-1,3)
        faces=np.zeros(len(indices),dtype=[('n','u1'),('v','<i4',(3,))]);faces['n']=3;faces['v']=indices
        if not vertical_mask(xyz.astype(float),faces).all():raise RuntimeError('Nonvertical geometry in facade group')
        old=obj.radio_material.name;obj.radio_material=material
        changes.append(dict(object=name,old_material=old,new_material=material.name,
                            triangle_count=len(indices),all_faces_vertical=True))
    after={name:obj.radio_material.name for name,obj in scene.objects.items()}
    changed={name for name in names if names[name]!=after[name]}
    if changed!=set(targets):raise RuntimeError('Intervention changed undeclared objects')
    if before!={name:mesh_geometry_hash(obj,mi) for name,obj in scene.objects.items()}:
        raise RuntimeError('Material intervention changed geometry')
    guard=dict(passed=True,changed_objects=changes,unchanged_object_count=len(names)-len(targets),
               only_explicit_facade_groups_changed=True,geometry_hashes_unchanged=True,
               baseline_materials=names,intervened_materials=after)
    return changes,guard


def mesh_geometry_hash(obj,mi):
    params=mi.traverse(obj.mi_mesh);h=hashlib.sha256()
    for key in ('vertex_positions','faces'):h.update(np.asarray(params[key]).tobytes())
    return h.hexdigest()

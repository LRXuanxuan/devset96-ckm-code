"""Exact coplanar mesh patches for the image-source topology of one snapshot.

The native backtracker deliberately offsets rays. Its actual hit primitive can
therefore differ from the triangle containing the unoffset image solution at an
internal triangulation seam. This module represents the physical planar surface
without changing any archived native primitive or bounce coordinate.

Coplanarity and mesh adjacency are exact predicates on the binary values of the
actual world vertices. They are not angle/delay thresholds. Patch IDs are scoped
to a snapshot geometry, material partition and canonical object; they are not
temporal path IDs. Float64 point location remains a finite numerical certificate.
"""
from __future__ import annotations

from collections import defaultdict
from functools import reduce
import hashlib
import json
import math

import numpy as np

from .devset_specular_audit import UnresolvedSpecularGroup
from .cities import C

VERSION='kap_devset.exact_coplanar_surface_patch.v1'
_UNSET=object()


def _vertex(value):
    result=tuple(float(x) if float(x)!=0. else 0. for x in value)
    if len(result)!=3 or not all(math.isfinite(x) for x in result):
        raise ValueError('Patch vertices must be finite XYZ binary floats')
    return result


def _integer_points(points):
    """One common power-of-two scale, hence no rational rounding at all."""
    ratios=[[float(x).as_integer_ratio() for x in point] for point in points]
    denominator=max(d for row in ratios for _,d in row)
    return [[n*(denominator//d) for n,d in row] for row in ratios],denominator


def exact_plane_key(triangle):
    """Primitive integer coefficients Ax+By+Cz+D=0, independent of winding."""
    vertices=[_vertex(p) for p in np.asarray(triangle).reshape(3,3)]
    points,scale=_integer_points(vertices)
    a=[points[1][k]-points[0][k] for k in range(3)]
    b=[points[2][k]-points[0][k] for k in range(3)]
    normal=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    if not any(normal):raise UnresolvedSpecularGroup('Exactly degenerate patch triangle')
    coefficients=[x*scale for x in normal]+[-sum(x*y for x,y in zip(normal,points[0]))]
    divisor=reduce(math.gcd,map(abs,coefficients))
    coefficients=[x//divisor for x in coefficients]
    if next(x for x in coefficients if x)!=abs(next(x for x in coefficients if x)):
        coefficients=[-x for x in coefficients]
    return tuple(coefficients)


def _orient2(a,b,c):
    points,_=_integer_points((a,b,c))
    u,v,w=points
    return (v[0]-u[0])*(w[1]-u[1])-(v[1]-u[1])*(w[0]-u[0])


def _on_segment(a,b,p):
    return _orient2(a,b,p)==0 and all(min(a[k],b[k])<=p[k]<=max(a[k],b[k]) for k in range(2))


def _strict_triangle_inside(p,triangle):
    signs=[_orient2(triangle[i],triangle[(i+1)%3],p) for i in range(3)]
    return all(v>0 for v in signs) or all(v<0 for v in signs)


def _pair_ambiguity(first,second,axis):
    """Exact interior overlap / nonconforming contact test on one plane."""
    a=[tuple(v[k] for k in range(3) if k!=axis) for v in first]
    b=[tuple(v[k] for k in range(3) if k!=axis) for v in second]
    if set(first)==set(second):return 'duplicate geometric triangles'
    if any(_strict_triangle_inside(p,b) for p in a) or any(_strict_triangle_inside(p,a) for p in b):
        return 'coplanar triangle interiors overlap'
    for i in range(3):
        x,y=a[i],a[(i+1)%3]
        for j in range(3):
            u,v=b[j],b[(j+1)%3]
            s1,s2,s3,s4=_orient2(x,y,u),_orient2(x,y,v),_orient2(u,v,x),_orient2(u,v,y)
            if s1*s2<0 and s3*s4<0:return 'coplanar triangle edges cross through their interiors'
            common=set((x,y))&set((u,v))
            if set((x,y))==set((u,v)):
                za=a[(i+2)%3];zb=b[(j+2)%3]
                if _orient2(x,y,za)*_orient2(x,y,zb)>=0:
                    return 'two triangles sharing a full edge occupy the same side'
                continue
            for p in (x,y):
                if p not in (u,v) and _on_segment(u,v,p):return 'nonconforming partial edge or T junction'
            for p in (u,v):
                if p not in (x,y) and _on_segment(x,y,p):return 'nonconforming partial edge or T junction'
            # A single shared endpoint is an allowed zero-area contact and does
            # not connect two otherwise separate surface patches.
    return None


def _material_json(key):
    return json.dumps(key,sort_keys=True,separators=(',',':'),ensure_ascii=True)


class SurfacePatchCatalog:
    """Cached exact connected planar patches over SpecularSceneGeometry trees.

    ``material_keys`` optionally maps ``(object_id,primitive_id)`` or object ID
    to a stable material-slot key. The default is one uniform radio-material slot
    per Sionna SceneObject, as used by the pinned mesh scene loader. Radio-material
    property changes do not change that slot or the geometry identity.
    """
    def __init__(self,geometry=None,material_keys=None):
        self.objects={};self.by_face={};self._tree_tokens={};self._material_signatures={}
        self._object_summaries={}
        self._material_keys=material_keys
        if geometry is not None:self.refresh(geometry,material_keys)

    def refresh(self,geometry,material_keys=_UNSET):
        if material_keys is _UNSET:material_keys=self._material_keys
        else:self._material_keys=material_keys
        keys={} if material_keys is None else material_keys
        present=set(geometry.trees)
        for object_id,tree in geometry.trees.items():
            if material_keys is None and self._tree_tokens.get(object_id) is tree and self._material_signatures.get(object_id)=='uniform_default':
                continue
            slots=[keys.get((int(object_id),int(p)),keys.get(int(object_id),'uniform_scene_object_material_slot')) for p in tree.primitives]
            signature=tuple(_material_json(x) for x in slots)
            cache_signature='uniform_default' if material_keys is None else signature
            if self._tree_tokens.get(object_id) is tree and self._material_signatures.get(object_id)==cache_signature:continue
            for key in [k for k in self.by_face if k[0]==object_id]:del self.by_face[key]
            self.objects[object_id]=self._build_object(int(object_id),tree,signature)
            patches=self.objects[object_id]
            serial=[dict(patch_id=p['patch_id'],valid=p['valid'],invalid_reasons=p['invalid_reasons'])
                    for p in sorted(patches,key=lambda p:p['patch_id'])]
            self._object_summaries[object_id]=dict(object_id=int(object_id),patch_count=len(patches),
                triangle_count=len(tree.primitives),invalid_patch_count=sum(not p['valid'] for p in patches),
                partition_sha256=hashlib.sha256(json.dumps(serial,sort_keys=True,separators=(',',':')).encode()).hexdigest())
            self._tree_tokens[object_id]=tree;self._material_signatures[object_id]=cache_signature
        for object_id in set(self.objects)-present:
            del self.objects[object_id];self._tree_tokens.pop(object_id,None);self._material_signatures.pop(object_id,None)
            self._object_summaries.pop(object_id,None)
            for key in [k for k in self.by_face if k[0]==object_id]:del self.by_face[key]

    def _build_object(self,object_id,tree,slots):
        triangles=[tuple(_vertex(v) for v in tri) for tri in tree.triangles]
        n=len(triangles);parent=list(range(n));plane_groups=defaultdict(list);face_edges=[];errors=defaultdict(list)
        planes=[]
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b:parent[max(a,b)]=min(a,b)
        for i,tri in enumerate(triangles):
            try:plane=exact_plane_key(tri)
            except UnresolvedSpecularGroup as error:
                plane=None;errors[i].append(str(error))
            planes.append(plane)
            edges=[tuple(sorted((tri[j],tri[(j+1)%3]))) for j in range(3)]
            face_edges.append(edges)
            if plane is not None:plane_groups[(slots[i],plane)].append(i)
        for (_,plane),ids in plane_groups.items():
            edge_faces=defaultdict(list)
            for i in ids:
                for edge in face_edges[i]:edge_faces[edge].append(i)
            for edge,incident in edge_faces.items():
                for i in incident[1:]:union(incident[0],i)
                if len(incident)>2:
                    for i in incident:errors[i].append('nonmanifold edge has more than two incident triangles')
            axis=max(range(3),key=lambda k:abs(plane[k]));axes=[k for k in range(3) if k!=axis]
            # Broad phase uses exact float comparisons; the narrow predicates
            # use dyadic integers. Test even separate components for overlap.
            boxes={i:([min(v[k] for v in triangles[i]) for k in axes],
                      [max(v[k] for v in triangles[i]) for k in axes]) for i in ids}
            ordered=sorted(ids,key=lambda i:boxes[i][0][0]);active=[]
            for i in ordered:
                lo,hi=boxes[i];active=[j for j in active if boxes[j][1][0]>=lo[0]]
                for j in active:
                    lj,hj=boxes[j]
                    if hj[1]<lo[1] or hi[1]<lj[1]:continue
                    problem=_pair_ambiguity(triangles[i],triangles[j],axis)
                    if problem:errors[i].append(problem);errors[j].append(problem)
                active.append(i)
        components=defaultdict(list)
        for i in range(n):components[find(i)].append(i)
        result=[]
        for indices in components.values():
            first=indices[0];plane=planes[first];counts=defaultdict(int)
            for i in indices:
                for edge in face_edges[i]:counts[edge]+=1
            boundary=[edge for edge,count in counts.items() if count==1]
            content=sorted(tuple(sorted(triangles[i])) for i in indices)
            digest=hashlib.sha256()
            digest.update(str(object_id).encode());digest.update(b'\0');digest.update(slots[first].encode());digest.update(b'\0')
            digest.update(np.asarray(content,dtype='<f8').tobytes())
            patch=dict(patch_id=digest.hexdigest(),object_id=object_id,plane_key=plane,
                primitive_ids=tuple(sorted(int(tree.primitives[i]) for i in indices)),
                triangles=np.asarray([triangles[i] for i in indices],float),
                triangle_primitives=np.asarray([tree.primitives[i] for i in indices],np.int64),
                boundary_edges=np.asarray(boundary,float).reshape(-1,2,3),material_slot_key=slots[first],
                valid=not any(errors[i] for i in indices),invalid_reasons=sorted(set(e for i in indices for e in errors[i])))
            patch['triangle_geometry_multiset_sha256']=hashlib.sha256(np.asarray(content,dtype='<f8').tobytes()).hexdigest()
            result.append(patch)
            for i in indices:self.by_face[(object_id,int(tree.primitives[i]))]=patch
        return result

    def patch_for(self,object_id,primitive_id):
        try:patch=self.by_face[(int(object_id),int(primitive_id))]
        except KeyError:raise UnresolvedSpecularGroup('Unknown actual object/primitive for physical patch') from None
        if not patch['valid']:raise UnresolvedSpecularGroup('Ambiguous physical patch: '+ '; '.join(patch['invalid_reasons']))
        return patch

    def locate(self,object_id,primitive_id,point):
        """Certify strict union interior; internal full-edge seams are allowed."""
        patch=self.patch_for(object_id,primitive_id);point=np.asarray(point,float).reshape(3)
        triangles=patch['triangles'];e1,e2=triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]
        normal=np.cross(e1[0],e2[0]);normal/=np.linalg.norm(normal)
        scale=max(1.,float(np.abs(triangles).max(initial=0.)),float(abs(point).max(initial=0.)))
        tolerance=4096*np.finfo(float).eps*scale
        plane_error=float(abs((point-triangles[0,0])@normal))
        if plane_error>tolerance:raise UnresolvedSpecularGroup('Image point is off the exact physical patch plane')
        q=point-triangles[:,0];aa=np.einsum('ij,ij->i',e1,e1);ab=np.einsum('ij,ij->i',e1,e2);bb=np.einsum('ij,ij->i',e2,e2)
        den=aa*bb-ab*ab
        if np.any(den<=512*np.finfo(float).eps*np.maximum(aa*bb,1.)):
            raise UnresolvedSpecularGroup('Ill-conditioned triangle in physical patch')
        av=np.einsum('ij,ij->i',e1,q);bv=np.einsum('ij,ij->i',e2,q)
        u=(bb*av-ab*bv)/den;v=(aa*bv-ab*av)/den;bary=np.column_stack((1-u-v,u,v))
        conditioning=np.sqrt((aa+bb)**2/den)
        bary_tolerance=4096*np.finfo(float).eps*np.maximum(conditioning,1.)
        contains=np.flatnonzero(np.min(bary,axis=1)>=-bary_tolerance)
        if not len(contains):raise UnresolvedSpecularGroup('Image bounce is outside the connected physical patch union')
        edges=patch['boundary_edges'];distance=np.inf
        if len(edges):
            direction=edges[:,1]-edges[:,0];length2=np.einsum('ij,ij->i',direction,direction)
            along=np.clip(np.einsum('ij,ij->i',point-edges[:,0],direction)/length2,0.,1.)
            distance=float(np.linalg.norm(point-edges[:,0]-along[:,None]*direction,axis=1).min())
        if distance<=tolerance:raise UnresolvedSpecularGroup('Image bounce touches physical patch outer boundary or hole edge')
        # A barycentrically uncertain point cannot be certified solely by a
        # negative tolerance. At a seam both opposite-side triangles must meet.
        strict=np.flatnonzero(np.min(bary,axis=1)>bary_tolerance)
        if not len(strict) and len(contains)<2:
            raise UnresolvedSpecularGroup('Patch containment has only one numerically uncertain boundary face')
        pick=int(strict[np.argmax(np.min(bary[strict],axis=1))]) if len(strict) else int(contains[0])
        hit_faces=[(int(object_id),int(patch['triangle_primitives'][i])) for i in contains]
        return dict(patch_id=patch['patch_id'],object_id=int(object_id),declared_primitive_id=int(primitive_id),
            physical_primitive_id=int(patch['triangle_primitives'][pick]),containing_faces=hit_faces,
            barycentric=bary[pick],patch_boundary_distance_m=distance,plane_error_m=plane_error,
            float64_location_margin_m=tolerance,internal_mesh_seam=not bool(len(strict)),
            raw_declared_primitive_changed=int(primitive_id)!=int(patch['triangle_primitives'][pick]),
            physical_topology_scope='exact coplanar full-edge-connected same-object/material-slot patch; snapshot scoped')

    def statistics(self):
        serial=[self._object_summaries[k] for k in sorted(self._object_summaries)]
        return dict(schema=VERSION,object_count=len(self.objects),patch_count=sum(r['patch_count'] for r in serial),
            triangle_count=len(self.by_face),invalid_patch_count=sum(r['invalid_patch_count'] for r in serial),
            snapshot_catalog_sha256=hashlib.sha256(json.dumps(serial,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            object_partitions=serial,
            exact_coplanarity=True,full_geometric_edge_adjacency=True,
            material_scope='stable source material slot, not changing radio-material property values/names; uniform per SceneObject unless explicit per-face slot keys supplied',
            identity_scope='snapshot geometry partition; rigid actor world patch IDs can change with pose; not temporal path IDs')

    def identities(self,faces):
        """Small identity/membership records for only the referenced raw faces.

        Invalid patches remain identifiable evidence; returning their identity
        does not grant image/visibility admission. Geometry is reconstructed from
        source canonical mesh identity and exact membership, not copied here.
        """
        patches={}
        for object_id,primitive_id in faces:
            patch=self.by_face.get((int(object_id),int(primitive_id)))
            if patch is not None:patches[(int(object_id),patch['patch_id'])]=patch
        return [dict(object_id=p['object_id'],patch_id=p['patch_id'],
            material_slot_key=json.loads(p['material_slot_key']),primitive_ids=list(p['primitive_ids']),
            primitive_ids_sha256=hashlib.sha256(np.asarray(p['primitive_ids'],dtype='<u4').tobytes()).hexdigest(),
            triangle_count=len(p['primitive_ids']),boundary_edge_count=len(p['boundary_edges']),
            triangle_geometry_multiset_sha256=p['triangle_geometry_multiset_sha256'],
            valid=p['valid'],invalid_reasons=p['invalid_reasons'],
            identity_rule='SHA256 canonical object decimal + NUL + stable source material-slot canonical JSON + NUL + sorted exact world triangles little-endian float64; vertex/winding/face order and signed zero normalized',
            scope='snapshot-scoped physical surface membership; not a temporal path identifier')
            for _,p in sorted(patches.items())]


def solve_patch_image(tx,rx,declared_topology,catalog,*,min_segment_m=.01):
    """Unique image solution for a full ordered connected physical patch list.

    ``declared_topology`` contains (canonical object, raw primitive, SPECULAR=1).
    The returned mapping is evidence; none of the input native rows is rewritten.
    """
    tx,rx=np.asarray(tx,float).reshape(3),np.asarray(rx,float).reshape(3)
    if not np.isfinite(np.r_[tx,rx]).all():raise UnresolvedSpecularGroup('Nonfinite physical endpoints')
    patches=[];triangles=[];normals=[];images=[tx.copy()]
    for object_id,primitive,kind in declared_topology:
        if kind!=1:raise UnresolvedSpecularGroup('Patch image solver only certifies SPECULAR interactions')
        patch=catalog.patch_for(object_id,primitive);triangle=patch['triangles'][0]
        edges=np.column_stack((triangle[1]-triangle[0],triangle[2]-triangle[0]))
        singular=np.linalg.svd(edges,compute_uv=False)
        if singular[-1]<=512*np.finfo(float).eps*max(singular[0],1.):
            raise UnresolvedSpecularGroup('Degenerate/ill-conditioned physical patch plane')
        normal=np.cross(edges[:,0],edges[:,1]);normal/=np.linalg.norm(normal)
        patches.append(patch);triangles.append(triangle);normals.append(normal)
        images.append(images[-1]-2*normal*np.dot(images[-1]-triangle[0],normal))
    points=np.zeros((len(patches),3));current=rx.copy();denominators=[]
    for i in range(len(patches)-1,-1,-1):
        direction=images[i+1]-current;denominator=float(normals[i]@direction)
        if abs(denominator)<=1e-12*max(np.linalg.norm(direction),1.):
            raise UnresolvedSpecularGroup('Parallel/grazing physical patch image intersection')
        fraction=float(normals[i]@(triangles[i][0]-current)/denominator)
        if not 1e-12<fraction<1-1e-12:raise UnresolvedSpecularGroup('Nonpositive/degenerate patch image intersection')
        points[i]=current+fraction*direction;current=points[i];denominators.append(denominator)
    nodes=np.vstack((tx,points,rx));segments=np.diff(nodes,axis=0);lengths=np.linalg.norm(segments,axis=1)
    if np.any(lengths<float(min_segment_m)):raise UnresolvedSpecularGroup('Zero/short patch-image segment outside certified native model')
    mappings=[];checks=[]
    for i,((object_id,primitive,_),normal) in enumerate(zip(declared_topology,normals)):
        mappings.append(catalog.locate(object_id,primitive,points[i]))
        incoming=segments[i]/lengths[i];outgoing=segments[i+1]/lengths[i+1]
        if min(abs(incoming@normal),abs(outgoing@normal))<=1e-10:raise UnresolvedSpecularGroup('Grazing physical patch reflection')
        if ((nodes[i]-points[i])@normal)*((nodes[i+2]-points[i])@normal)<=0:
            raise UnresolvedSpecularGroup('Patch reflection endpoints on opposite sides')
        residual=float(np.linalg.norm(outgoing-(incoming-2*(incoming@normal)*normal)))
        if residual>1e-9:raise UnresolvedSpecularGroup('Float64 physical patch reflection law residual exceeds certificate')
        checks.append(dict(plane_distance_m=mappings[-1]['plane_error_m'],reflection_vector_residual=residual))
    return dict(points=points,length_m=float(lengths.sum()),delay_sec=float(lengths.sum()/C),
        barycentric=np.asarray([m['barycentric'] for m in mappings]).reshape(-1,3),
        normals=np.asarray(normals).reshape(-1,3),segment_lengths_m=lengths,
        reflection_checks=checks,patch_mappings=mappings,image_backtrace_denominators=denominators,
        physical_topology=[dict(object_id=p['object_id'],patch_id=p['patch_id'],interaction_type=1) for p in patches],
        plane_reference_triangles_m=np.asarray(triangles).reshape(-1,3,3),
        uniqueness_scope='fixed ordered exact connected planar surface patches; strict union interior, non-grazing and nonzero segments; finite Float64 certificate')


def check_patch_segment(geometry,start,end,*,start_faces=(),end_faces=()):
    """Full-scene visibility, excluding only certified faces at their endpoints.

    An internal patch seam may have two endpoint faces. No other primitive of
    that patch, and no other face of its object, is ignored along the segment.
    This is the frozen V3 BVH narrow-phase with explicit endpoint face sets.
    """
    start,end=np.asarray(start,float),np.asarray(end,float);direction=end-start
    start_faces=set(map(tuple,start_faces));end_faces=set(map(tuple,end_faces))
    parameter_tol=1e-10;bary_tol=1e-10
    for tree in geometry.trees.values():
        if not tree.nodes:continue
        stack=[0]
        while stack:
            lo,hi,left,right,indices=tree.nodes[stack.pop()]
            near,far=0.,1.;miss=False
            for k in range(3):
                if abs(direction[k])<1e-15:
                    if start[k]<lo[k]-1e-10 or start[k]>hi[k]+1e-10:miss=True;break
                else:
                    a,b=(lo[k]-start[k])/direction[k],(hi[k]-start[k])/direction[k]
                    near=max(near,min(a,b));far=min(far,max(a,b))
                    if near>far+parameter_tol:miss=True;break
            if miss:continue
            if indices is None:stack.extend((left,right));continue
            tri=tree.triangles[indices];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
            cross=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,cross)
            area=np.linalg.norm(np.cross(e1,e2),axis=1)
            scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*np.linalg.norm(direction)
            parallel=abs(det)<=512*np.finfo(float).eps*np.maximum(scale,1.)
            normal=np.cross(e1,e2);normal_dist=np.einsum('ij,ij->i',start-tri[:,0],normal)
            coplanar=parallel&(area>1e-14)&(abs(normal_dist)<=1e-10*area)
            if coplanar.any():
                for ix in indices[coplanar]:
                    lower,upper=tree._lo[ix],tree._hi[ix];span=[]
                    for k in range(3):
                        if abs(direction[k])<1e-15:span.append((-np.inf,np.inf) if lower[k]-1e-10<=start[k]<=upper[k]+1e-10 else (1.,0.))
                        else:
                            a,b=(lower[k]-start[k])/direction[k],(upper[k]-start[k])/direction[k]
                            span.append((min(a,b),max(a,b)))
                    if max(0.,*(s[0] for s in span))<=min(1.,*(s[1] for s in span)):
                        return dict(status='unresolved',reason='coplanar segment/surface intersection',object_id=tree.object_id,primitive_id=int(tree.primitives[ix]))
            usable=~parallel
            if not usable.any():continue
            loc=np.flatnonzero(usable);inverse=1/det[usable];s=start-tri[usable,0]
            u=np.einsum('ij,ij->i',s,cross[usable])*inverse;q=np.cross(s,e1[usable])
            v=q@direction*inverse;t=np.einsum('ij,ij->i',e2[usable],q)*inverse
            hit=(u>=-bary_tol)&(v>=-bary_tol)&(u+v<=1+bary_tol)&(t>=-parameter_tol)&(t<=1+parameter_tol)
            for j in np.flatnonzero(hit):
                face=(tree.object_id,int(tree.primitives[indices[loc[j]]]))
                if t[j]<=parameter_tol and face in start_faces:continue
                if t[j]>=1-parameter_tol and face in end_faces:continue
                if parameter_tol<t[j]<1-parameter_tol:
                    if min(u[j],v[j],1-u[j]-v[j])<=bary_tol:
                        return dict(status='unresolved',reason='segment grazes triangle edge',object_id=face[0],primitive_id=face[1])
                    return dict(status='blocked',object_id=face[0],primitive_id=face[1],segment_fraction=float(t[j]))
                return dict(status='unresolved',reason='unexpected surface at a segment endpoint',object_id=face[0],primitive_id=face[1])
    return dict(status='clear',method='Float64 full-scene BVH; only certified containing primitive sets excluded at exact intended endpoints')

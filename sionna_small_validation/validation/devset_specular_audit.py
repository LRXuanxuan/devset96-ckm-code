"""Unique-image audit for fixed ordered pure-specular triangle sequences.

The exact-arithmetic uniqueness argument applies to a fixed planar sequence,
nondegenerate triangles, strict-interior non-grazing bounces and nonzero
segments. Float64 checks below are finite numerical certificates, not a theorem
covering all numerical scene configurations. Every unresolved group blocks
snapshot admission. Raw candidates are retained; different full topologies are
never merged by delay or angle proximity.
"""
from __future__ import annotations

from collections import defaultdict
import copy
from functools import lru_cache
import hashlib
from pathlib import Path
import time

import numpy as np

from .cities import C,extract_compact,native_cfr,reconstruct,xyz_rows
from .devset_rt import exact_metadata

VERSION='kap_devset.unique_specular_image.v1'
SOURCE_HASHES={
    'paths.py':'5ad56ec10f8587b39fa8db16be6b4264ebd2494ddee56b41224ee07d725b645c',
    'path_solver.py':'421c3926a2d9c8dc77c7323752f04d4f9bfba79c5d8a3faa6494806d51a6d71b',
    'paths_buffer.py':'450ae7436205af1f519654d5723e16e4edc4303c3361e0921c9023c8bda0f892',
}


class UnresolvedSpecularGroup(ValueError):
    pass


@lru_cache(maxsize=1)
def check_private_contract():
    import sionna.rt as rt
    if rt.__version__!='2.2.0':raise RuntimeError('Unique specular adapter is locked to Sionna RT2.2.0')
    base=Path(rt.__file__).parent/'path_solvers'
    actual={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in SOURCE_HASHES}
    if actual!=SOURCE_HASHES:raise RuntimeError('Pinned native paths/buffer/solver source changed')
    return dict(sionna_rt_version=rt.__version__,source_sha256=actual)


def active_topology(objects,primitives,types):
    objects,primitives,types=map(np.asarray,(objects,primitives,types))
    if objects.shape!=types.shape or primitives.shape!=types.shape:raise UnresolvedSpecularGroup('Interaction metadata shape mismatch')
    occupied=types!=0
    if np.any(np.diff(occupied.astype(int))>0):raise UnresolvedSpecularGroup('Non-prefix active interactions')
    if np.any(types[occupied]!=1):raise UnresolvedSpecularGroup('Only pure planar SPECULAR interactions are certified')
    return tuple((int(o),int(p),1) for o,p in zip(objects[occupied],primitives[occupied]))


def solve_specular_image(tx,rx,triangles,*,min_segment_m=.01):
    tx,rx=np.asarray(tx,float).reshape(3),np.asarray(rx,float).reshape(3)
    tri=np.asarray(triangles,float).reshape(-1,3,3)
    if not np.isfinite(np.r_[tx,rx,tri.ravel()]).all():raise UnresolvedSpecularGroup('Nonfinite geometry')
    images=[tx.copy()];normals=[];conditions=[]
    for face in tri:
        edges=np.column_stack((face[1]-face[0],face[2]-face[0]))
        singular=np.linalg.svd(edges,compute_uv=False)
        if singular[-1]<=512*np.finfo(float).eps*max(singular[0],1.):raise UnresolvedSpecularGroup('Degenerate/ill-conditioned triangle')
        normal=np.cross(edges[:,0],edges[:,1]);normal/=np.linalg.norm(normal)
        images.append(images[-1]-2*normal*np.dot(images[-1]-face[0],normal))
        normals.append(normal);conditions.append(float(singular[0]/singular[-1]))
    points=np.zeros((len(tri),3));current=rx.copy();denominators=[]
    for i in range(len(tri)-1,-1,-1):
        direction=images[i+1]-current;denominator=float(normals[i]@direction)
        if abs(denominator)<=1e-12*max(np.linalg.norm(direction),1.):raise UnresolvedSpecularGroup('Parallel/grazing image intersection')
        fraction=float(normals[i]@(tri[i,0]-current)/denominator)
        if not 1e-12<fraction<1-1e-12:raise UnresolvedSpecularGroup('Nonpositive/degenerate image intersection')
        points[i]=current+fraction*direction;current=points[i];denominators.append(denominator)
    vertices=np.vstack((tx,points,rx));segments=np.diff(vertices,axis=0);lengths=np.linalg.norm(segments,axis=1)
    if np.any(lengths<float(min_segment_m)):raise UnresolvedSpecularGroup('Zero/short segment outside certified native model')
    barycentrics=[];reflections=[]
    world_scale=max(float(np.abs(vertices).max(initial=0.)),1.)
    plane_tolerance=4096*np.finfo(float).eps*world_scale
    for i,(point,face,normal) in enumerate(zip(points,tri,normals)):
        e1,e2=face[1]-face[0],face[2]-face[0]
        uv=np.linalg.solve(np.array([[e1@e1,e1@e2],[e1@e2,e2@e2]]),np.array([e1@(point-face[0]),e2@(point-face[0])]))
        bary=np.r_[1-uv.sum(),uv]
        if np.min(bary)<=1e-8:raise UnresolvedSpecularGroup('Bounce on/outside triangle edge: certificate withheld')
        plane_error=float(abs((point-face[0])@normal))
        if plane_error>plane_tolerance:raise UnresolvedSpecularGroup('Float64 image point is off its actual triangle plane')
        incoming=segments[i]/lengths[i];outgoing=segments[i+1]/lengths[i+1]
        if min(abs(incoming@normal),abs(outgoing@normal))<=1e-10:raise UnresolvedSpecularGroup('Grazing reflection')
        if ((vertices[i]-point)@normal)*((vertices[i+2]-point)@normal)<=0:raise UnresolvedSpecularGroup('Reflection endpoints on opposite sides')
        reflected=incoming-2*(incoming@normal)*normal
        residual=float(np.linalg.norm(outgoing-reflected))
        if residual>1e-9:raise UnresolvedSpecularGroup('Float64 reflection law residual exceeds certificate')
        barycentrics.append(bary);reflections.append(dict(plane_distance_m=plane_error,reflection_vector_residual=residual))
    return dict(points=points,length_m=float(lengths.sum()),delay_sec=float(lengths.sum()/C),
        barycentric=np.asarray(barycentrics).reshape(-1,3),normals=np.asarray(normals).reshape(-1,3),
        segment_lengths_m=lengths,reflection_checks=reflections,triangle_condition_numbers=conditions,
        image_backtrace_denominators=denominators,plane_tolerance_m=plane_tolerance,
        uniqueness_scope='fixed ordered planar triangle sequence; strict-interior nongrazing nonzero-segment image solution')


class _TriangleBVH:
    def __init__(self,triangles,primitive_ids,object_id):
        self.triangles=np.asarray(triangles,float).reshape(-1,3,3)
        self.primitives=np.asarray(primitive_ids,np.int64);self.object_id=int(object_id)
        self.nodes=[];self.by_primitive={int(p):i for i,p in enumerate(self.primitives)}
        self._lo=self.triangles.min(axis=1) if len(self.triangles) else np.empty((0,3))
        self._hi=self.triangles.max(axis=1) if len(self.triangles) else np.empty((0,3))
        if len(self.triangles):self._build(np.arange(len(self.triangles)))

    def _build(self,indices):
        lo=self._lo[indices].min(axis=0);hi=self._hi[indices].max(axis=0)
        number=len(self.nodes);self.nodes.append(None)
        if len(indices)<=12:self.nodes[number]=(lo,hi,-1,-1,indices);return number
        centers=(self._lo[indices]+self._hi[indices])/2;axis=int(np.argmax(np.ptp(centers,axis=0)))
        middle=len(indices)//2;order=np.argpartition(centers[:,axis],middle)
        left=self._build(indices[order[:middle]]);right=self._build(indices[order[middle:]])
        self.nodes[number]=(lo,hi,left,right,None);return number

    def segment(self,start,end,start_face,end_face):
        start,end=np.asarray(start,float),np.asarray(end,float);direction=end-start
        if not self.nodes:return dict(status='clear')
        stack=[0];parameter_tol=1e-10;bary_tol=1e-10
        while stack:
            lo,hi,left,right,indices=self.nodes[stack.pop()]
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
            tri=self.triangles[indices];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
            cross=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,cross)
            area=np.linalg.norm(np.cross(e1,e2),axis=1)
            scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*np.linalg.norm(direction)
            parallel=abs(det)<=512*np.finfo(float).eps*np.maximum(scale,1.)
            normal=np.cross(e1,e2);normal_dist=np.einsum('ij,ij->i',start-tri[:,0],normal)
            coplanar=parallel & (area>1e-14) & (abs(normal_dist)<=1e-10*area)
            if coplanar.any():
                # Conservative only if its individual box intersects this segment.
                for ix in indices[coplanar]:
                    lower,upper=self._lo[ix],self._hi[ix]
                    span=[]
                    for k in range(3):
                        if abs(direction[k])<1e-15:span.append((-np.inf,np.inf) if lower[k]-1e-10<=start[k]<=upper[k]+1e-10 else (1.,0.))
                        else:
                            aa,bb=(lower[k]-start[k])/direction[k],(upper[k]-start[k])/direction[k]
                            span.append((min(aa,bb),max(aa,bb)))
                    if max(0.,*(s[0] for s in span))<=min(1.,*(s[1] for s in span)):
                        return dict(status='unresolved',reason='coplanar segment/surface intersection',object_id=self.object_id,primitive_id=int(self.primitives[ix]))
            usable=~parallel
            if not usable.any():continue
            loc=np.flatnonzero(usable);inverse=1/det[usable];s=start-tri[usable,0]
            u=np.einsum('ij,ij->i',s,cross[usable])*inverse;q=np.cross(s,e1[usable])
            v=q@direction*inverse;t=np.einsum('ij,ij->i',e2[usable],q)*inverse
            hit=(u>=-bary_tol)&(v>=-bary_tol)&(u+v<=1+bary_tol)&(t>=-parameter_tol)&(t<=1+parameter_tol)
            for j in np.flatnonzero(hit):
                face=(self.object_id,int(self.primitives[indices[loc[j]]]))
                if t[j]<=parameter_tol and face==start_face:continue
                if t[j]>=1-parameter_tol and face==end_face:continue
                if parameter_tol<t[j]<1-parameter_tol:
                    if min(u[j],v[j],1-u[j]-v[j])<=bary_tol:
                        return dict(status='unresolved',reason='segment grazes triangle edge',object_id=face[0],primitive_id=face[1])
                    return dict(status='blocked',object_id=face[0],primitive_id=face[1],segment_fraction=float(t[j]))
                return dict(status='unresolved',reason='unexpected surface at a segment endpoint',object_id=face[0],primitive_id=face[1])
        return dict(status='clear')


class SpecularSceneGeometry:
    """Cache static world triangles/BVHs; refresh declared rigid actors only."""
    def __init__(self,scene=None,object_map=None,dynamic_object_ids=()):
        self.trees={};self.scene=None;self._transform_signatures={}
        if scene is not None:self.begin_snapshot(scene,object_map,dynamic_object_ids)

    @classmethod
    def from_triangle_records(cls,records):
        result=cls();grouped=defaultdict(list)
        for row in records:grouped[int(row['object_id'])].append(row)
        for obj,rows in grouped.items():
            result.trees[obj]=_TriangleBVH([r['vertices_xyz_m'] for r in rows],[r['primitive_id'] for r in rows],obj)
        return result

    def begin_snapshot(self,scene,object_map,dynamic_object_ids=()):
        import mitsuba as mi
        self.scene=scene;dynamic=set(map(int,dynamic_object_ids));present=set()
        for obj in scene.objects.values():
            ident=int(object_map[int(obj.object_id)]);present.add(ident)
            signature=tuple(float(v) for attribute in ('position','orientation','scaling')
                            for v in np.asarray(getattr(obj,attribute)).reshape(-1))
            if ident in self.trees and ident not in dynamic and self._transform_signatures.get(ident)==signature:continue
            params=mi.traverse(obj.mi_mesh);vertices=np.asarray(params['vertex_positions'],float).reshape(-1,3)
            faces=np.asarray(params['faces'],np.int64).reshape(-1,3)
            self.trees[ident]=_TriangleBVH(vertices[faces],np.arange(len(faces)),ident)
            self._transform_signatures[ident]=signature
        for ident in set(self.trees)-present:
            del self.trees[ident];self._transform_signatures.pop(ident,None)

    def triangle(self,object_id,primitive_id):
        try:
            tree=self.trees[int(object_id)];return tree.triangles[tree.by_primitive[int(primitive_id)]]
        except KeyError:raise UnresolvedSpecularGroup('Unknown actual object/primitive triangle') from None

    def check_segment(self,start,end,start_face=None,end_face=None):
        for tree in self.trees.values():
            result=tree.segment(start,end,start_face,end_face)
            if result['status']!='clear':return result
        return dict(status='clear',method='Float64 BVH/Moller-Trumbore on actual full scene triangles; intended endpoints only excluded')


def audit_specular_candidates(compact,raw,tx,rx,geometry,*,validate_occlusion=True):
    compact=np.asarray(compact);vertices=np.asarray(raw['vertices_xyz_m'],float)
    primitives=np.asarray(raw['primitives']);types=np.asarray(raw.get('interaction_types',compact['interaction_types']))
    if vertices.shape!=(len(compact),3,3) or primitives.shape!=(len(compact),3) or types.shape!=(len(compact),3):
        raise ValueError('Pinned compact/raw metadata shape changed')
    groups=defaultdict(list);reports=[];selected=[]
    for i,row in enumerate(compact):
        try:key=active_topology(row['object_ids'],primitives[i],types[i])
        except UnresolvedSpecularGroup as error:
            reports.append(dict(status='unresolved',path_indices=[i],reason=str(error)));continue
        if int(row['depth'])!=len(key):
            reports.append(dict(status='unresolved',path_indices=[i],reason='Compact depth differs from active topology'));continue
        groups[key].append(i)
    for key,indices in groups.items():
        entry=dict(topology=[dict(object_id=o,primitive_id=p,interaction_type=t) for o,p,t in key],path_indices=indices,
                   multiplicity=len(indices),status='unresolved')
        try:
            triangles=[geometry.triangle(o,p) for o,p,_ in key]
            entry['triangles_world_m']=np.asarray(triangles).reshape(-1,3,3)
            solution=solve_specular_image(tx,rx,triangles)
            entry['analytic_solution']=solution
            nodes=np.vstack((np.asarray(tx,float),solution['points'],np.asarray(rx,float)))
            visibility=[]
            entry['segment_visibility']=visibility
            if validate_occlusion:
                for segment,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
                    result=geometry.check_segment(a,b,key[segment-1][:2] if segment else None,
                        key[segment][:2] if segment<len(key) else None)
                    visibility.append(result)
                    if result['status']!='clear':raise UnresolvedSpecularGroup('Physical segment visibility not certified: '+str(result))
            entry['segment_visibility']=visibility
            world_scale=max(1.,float(abs(nodes).max(initial=0.)))
            coordinate_ulp=float(abs(np.spacing(np.float32(world_scale))))
            plane_tolerance=max(1e-8,8*coordinate_ulp)
            delay_tolerance=max(1e-15,8*float(abs(np.spacing(np.float32(solution['delay_sec'])))))
            optical_tolerance=C*delay_tolerance
            length_roundoff=4096*np.finfo(float).eps*max(world_scale,solution['length_m'],1.)*(len(key)+1)
            candidates=[]
            for i in indices:
                active=vertices[i,:len(key)]
                point_errors=np.linalg.norm(active-solution['points'],axis=-1)
                max_point_error=float(point_errors.max(initial=0.));rms=float(np.sqrt(np.mean(point_errors**2))) if len(key) else 0.
                delay_error=float(compact['delay_sec'][i]-solution['delay_sec'])
                score=float(np.hypot(rms,C*delay_error/(len(key)+1)))
                signed_plane=np.einsum('ij,ij->i',active-np.asarray(triangles).reshape(-1,3,3)[:,0],solution['normals'])
                projected=active-signed_plane[:,None]*solution['normals']
                projected_nodes=np.vstack((np.asarray(tx,float),projected,np.asarray(rx,float)))
                projected_segments=np.diff(projected_nodes,axis=0)
                projected_lengths=np.linalg.norm(projected_segments,axis=1)
                polyline_length=float(projected_lengths.sum())
                optical_excess=polyline_length-solution['length_m']
                native_nodes=np.vstack((np.asarray(tx,float),active,np.asarray(rx,float)))
                native_polyline_length=float(np.linalg.norm(np.diff(native_nodes,axis=0),axis=1).sum())
                fermat=[];reflection_residuals=[];same_side=True
                if len(key) and np.all(projected_lengths>0):
                    for j,n in enumerate(solution['normals']):
                        incoming=projected_segments[j]/projected_lengths[j]
                        outgoing=projected_segments[j+1]/projected_lengths[j+1]
                        gradient=incoming-outgoing
                        fermat.append(float(np.linalg.norm(gradient-(gradient@n)*n)))
                        reflection_residuals.append(float(np.linalg.norm(outgoing-(incoming-2*(incoming@n)*n))))
                        same_side &= bool(((projected_nodes[j]-projected[j])@n)*((projected_nodes[j+2]-projected[j])@n)>0)
                max_plane_error=float(abs(signed_plane).max(initial=0.))
                agreement=bool(np.isfinite(active).all() and np.isfinite(score) and same_side
                    and np.all(projected_lengths>=.01) and max_plane_error<=plane_tolerance
                    and -length_roundoff<=optical_excess<=optical_tolerance+length_roundoff
                    and abs(delay_error)<=delay_tolerance)
                finite=lambda value:float(value) if np.isfinite(value) else None
                candidates.append(dict(compact_index=i,native_path_index=int(compact['path_index'][i]),selection_score_m=finite(score),
                    max_point_error_m=finite(max_point_error),point_rms_error_m=finite(rms),delay_error_sec=finite(delay_error),
                    max_plane_error_m=finite(max_plane_error),projected_polyline_length_m=finite(polyline_length),
                    projected_optical_excess_m=finite(optical_excess),native_polyline_length_m=finite(native_polyline_length),
                    native_delay_minus_native_polyline_sec=finite(float(compact['delay_sec'][i])-native_polyline_length/C),
                    max_projected_fermat_residual=max(fermat,default=0.),
                    max_projected_reflection_vector_residual=max(reflection_residuals,default=0.),
                    projected_same_side=bool(same_side),
                    agrees_with_unique_solution=agreement))
            entry.update(candidates=candidates,native_plane_agreement_tolerance_m=plane_tolerance,
                native_delay_agreement_tolerance_sec=delay_tolerance,
                native_optical_excess_tolerance_m=optical_tolerance,float64_length_roundoff_m=length_roundoff,
                point_distance_is_admission_bound=False,
                agreement_scope='unique actual-plane/visibility certificate +8-coordinate-ULP plane agreement +8-delay-ULP native delay and projected optical-length excess; point distance selects representative only; Fermat/reflection residuals reported; no exact EM field bound')
            eligible=sorted((c for c in candidates if c['agrees_with_unique_solution']),key=lambda c:(c['selection_score_m'],c['compact_index']))
            if not eligible:raise UnresolvedSpecularGroup('No native candidate agrees with the unique Float64 image solution')
            best=eligible[0];tie_tolerance=4096*np.finfo(float).eps*world_scale
            tied=[c for c in eligible if abs(c['selection_score_m']-best['selection_score_m'])<=tie_tolerance]
            if len(tied)>1:
                fields=('b00','delay_sec','doppler_hz','k_tx','k_rx')
                equivalent=all(all(np.array_equal(compact[f][c['compact_index']],compact[f][best['compact_index']]) for f in fields) for c in tied)
                if not equivalent:raise UnresolvedSpecularGroup('Numerically tied representatives have different channel fields')
            selected.append(best['compact_index']);entry.update(status='certified_unique' if validate_occlusion else 'geometry_only',
                selected_compact_index=best['compact_index'],selection_reason='closest joint bounce-geometry/analytic-delay score for a mathematically unique full topology',
                raw_candidates_removed_from_field_sum=len(indices)-1)
        except (UnresolvedSpecularGroup,np.linalg.LinAlgError,KeyError,IndexError,ValueError) as error:
            entry.update(status='unresolved',reason=str(error))
        reports.append(entry)
    selected=sorted(selected)
    return dict(schema=VERSION,admitted=bool(selected and reports and all(r['status']=='certified_unique' for r in reports)),
        keep_compact_indices=selected,raw_path_count=len(compact),selected_path_count=len(selected),
        groups=reports,unresolved_group_count=sum(r['status']=='unresolved' for r in reports),
        full_topology_groups=len(groups),duplicate_topology_groups=sum(len(v)>1 for v in groups.values()),
        inactive_slots_ignored=True,different_topologies_merged=False,occlusion_checked=bool(validate_occlusion),
        physical_scope='fixed ordered nondegenerate planar pure-specular image solution; finite Float64 checks; all unresolved groups block admission',
        candidate_storage='all raw candidates and their native coefficients remain separately archived')


def masked_native_cfr(paths,keep_native_mask):
    check_private_contract()
    import drjit as dr
    import mitsuba as mi
    if not paths.synthetic_array or int(paths._num_tx)!=1 or int(paths._num_rx)!=1:
        raise RuntimeError('Masked native adapter requires one synthetic-array TX/RX pair')
    mask=np.asarray(keep_native_mask,bool).reshape(-1);tau=np.asarray(paths._tau)
    if tau.size!=len(mask) or tau.shape[-1]!=len(mask):raise RuntimeError('Native path-axis/permutation contract changed')
    originals={name:np.array(np.asarray(getattr(paths,name)),copy=True) for name in ('_a_real','_a_imag','_valid','_tau')}
    masked=copy.copy(paths)
    if not len(mask):
        # Tensor/select cannot broadcast a scalar through a zero path axis in
        # the pinned Dr.Jit version. Explicit empty tensors preserve the native
        # zero-path behaviour without certifying a physical zero channel.
        masked._a_real=dr.zeros(mi.TensorXf,paths._a_real.shape)
        masked._a_imag=dr.zeros(mi.TensorXf,paths._a_imag.shape)
        masked._valid=dr.zeros(mi.TensorXb,paths._valid.shape)
        masked._tau=dr.zeros(mi.TensorXf,paths._tau.shape)
        native=native_cfr(masked)
        unchanged=all(np.array_equal(np.asarray(getattr(paths,name)),value,equal_nan=True) for name,value in originals.items())
        if not unchanged:raise RuntimeError('Empty native copy mutated original Paths')
        return native,dict(original_paths_unchanged=True,masked_path_count=0,original_native_path_axis=0,
            coefficient_zeroing=True,valid_masking=True,inactive_tau_minus_one=True,
            empty_snapshot_not_physical_zero_certified=True,reference='independent empty Paths copy evaluated by native .cfr'),masked
    for name in ('_a_real','_a_imag'):
        array=getattr(paths,name)
        if array.shape[-1]!=len(mask):raise RuntimeError('Native coefficient path axis changed')
        tensor_mask=mi.TensorXb(np.broadcast_to(mask,array.shape).copy())
        setattr(masked,name,dr.select(tensor_mask,array,0.))
    valid_mask=mi.TensorXb(np.broadcast_to(mask,paths._valid.shape).copy())
    masked._valid=paths._valid & valid_mask
    delay_mask=mi.TensorXb(np.broadcast_to(mask,paths._tau.shape).copy())
    masked._tau=dr.select(delay_mask,paths._tau,-1.)
    native=native_cfr(masked)
    unchanged=all(np.array_equal(np.asarray(getattr(paths,name)),value,equal_nan=True) for name,value in originals.items())
    if not unchanged:raise RuntimeError('Masked native evaluation mutated the original Paths')
    return native,dict(original_paths_unchanged=True,masked_path_count=int(mask.sum()),original_native_path_axis=len(mask),
        coefficient_zeroing=True,valid_masking=True,inactive_tau_minus_one=True,
        reference='independent shallow Paths copy with new a_real/a_imag/valid/tau tensors; native .cfr called directly'),masked


def audit_specular_paths(paths,scene,object_map,*,geometry_cache=None,dynamic_object_ids=(),tx_xyz=None,rx_xyz=None):
    source=check_private_contract()
    begun=time.perf_counter()
    raw_compact=extract_compact(paths,object_map);raw=exact_metadata(paths);raw_native=native_cfr(paths)
    exported=time.perf_counter()
    native_count=np.asarray(paths.tau).size
    if native_count and (not hasattr(paths,'_path_ind') or len(np.asarray(paths._path_ind).reshape(-1))!=native_count):
        raise RuntimeError('Single-pair raw permutation contract changed')
    tx=xyz_rows(paths.sources)[0] if tx_xyz is None else np.asarray(tx_xyz,float)
    rx=xyz_rows(paths.targets)[0] if rx_xyz is None else np.asarray(rx_xyz,float)
    geometry=geometry_cache if geometry_cache is not None else SpecularSceneGeometry()
    geometry.begin_snapshot(scene,object_map,dynamic_object_ids)
    cached=time.perf_counter()
    report=audit_specular_candidates(raw_compact,raw,tx,rx,geometry)
    audited=time.perf_counter()
    selected=np.asarray(report['keep_compact_indices'],int)
    keep=np.zeros(np.asarray(paths.tau).size,bool)
    native_indices=np.asarray(raw['native_path_index'],int)
    if len(native_indices)!=len(raw_compact) or len(np.unique(native_indices))!=len(native_indices):
        raise RuntimeError('Native/raw/compact selection permutation changed')
    keep[native_indices[selected]]=True
    native,mask_report,masked=masked_native_cfr(paths,keep)
    masked_at=time.perf_counter()
    compact=raw_compact[selected].copy()
    txrel=xyz_rows(scene.tx_array.rotate(scene.wavelength,next(iter(scene.transmitters.values())).orientation))
    rxrel=xyz_rows(scene.rx_array.rotate(scene.wavelength,next(iter(scene.receivers.values())).orientation))
    rebuilt=reconstruct(compact,txrel,rxrel);norm=float(np.linalg.norm(native))
    relative=float(np.linalg.norm(rebuilt-native)/norm) if np.isfinite(norm) and norm>0 else None
    report.update(private_source_contract=source,masked_native_control=mask_report,
        selected_compact_vs_masked_native_relative_error=relative,
        selected_compact_vs_masked_native_pass=bool(relative is not None and np.isfinite(relative) and relative<1e-4),
        timing_seconds=dict(raw_export=exported-begun,geometry_cache=cached-exported,
            image_and_full_scene_visibility=audited-cached,masked_native_reference=masked_at-audited,
            total=time.perf_counter()-begun))
    report['admitted']=bool(report['admitted'] and report['selected_compact_vs_masked_native_pass'])
    return dict(compact=compact,native=native,raw_compact=raw_compact,raw_native=raw_native,raw=raw,
        keep_native_mask=keep,report=report,masked_paths=masked,geometry_cache=geometry,
        tx_relative_m=txrel,rx_relative_m=rxrel)

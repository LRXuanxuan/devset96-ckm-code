"""Physical planar-patch uniqueness audit, preserving every raw triangle label.

This version fixes the distinction between a triangulation seam and a physical
surface boundary. It never merges distinct ordered connected physical surfaces.
Native coefficients are selected, not re-evaluated as ideal electromagnetic gain.
"""
from __future__ import annotations
from collections import defaultdict
import time

import numpy as np

from .cities import C,extract_compact,native_cfr,reconstruct,xyz_rows
from .devset_rt import exact_metadata
from .devset_specular_audit import (SpecularSceneGeometry,UnresolvedSpecularGroup,
    active_topology,check_private_contract,masked_native_cfr)
from .devset_surface_patches import SurfacePatchCatalog,solve_patch_image,check_patch_segment,VERSION as PATCH_VERSION

VERSION='kap_devset.unique_specular_physical_patch.v4'


def audit_patch_candidates(compact,raw,tx,rx,geometry,*,patch_catalog=None,validate_occlusion=True):
    compact=np.asarray(compact);vertices=np.asarray(raw['vertices_xyz_m'],float)
    primitives=np.asarray(raw['primitives']);types=np.asarray(raw.get('interaction_types',compact['interaction_types']))
    if vertices.shape!=(len(compact),3,3) or primitives.shape!=(len(compact),3) or types.shape!=(len(compact),3):
        raise ValueError('Pinned compact/raw metadata shape changed')
    catalog=SurfacePatchCatalog(geometry) if patch_catalog is None else patch_catalog
    catalog.refresh(geometry)
    groups=defaultdict(list);reports=[];selected=[];declared={};physical={}
    for i,row in enumerate(compact):
        try:
            top=active_topology(row['object_ids'],primitives[i],types[i]);declared[i]=top
            if int(row['depth'])!=len(top):raise UnresolvedSpecularGroup('Compact depth differs from active topology')
            key=tuple((o,catalog.patch_for(o,p)['patch_id'],t) for o,p,t in top);physical[i]=key
        except UnresolvedSpecularGroup as error:
            reports.append(dict(status='unresolved',path_indices=[i],reason=str(error),
                declared_triangle_topologies=[dict(compact_index=i,topology=[dict(object_id=o,primitive_id=p,interaction_type=t)
                    for o,p,t in declared.get(i,())])]))
            continue
        groups[key].append(i)
    for key,indices in groups.items():
        entry=dict(physical_topology=[dict(object_id=o,patch_id=p,interaction_type=t) for o,p,t in key],
            path_indices=indices,multiplicity=len(indices),status='unresolved',
            declared_triangle_topologies=[dict(compact_index=i,topology=[dict(object_id=o,primitive_id=p,interaction_type=t)
                for o,p,t in declared[i]]) for i in indices])
        try:
            solution=solve_patch_image(tx,rx,declared[indices[0]],catalog)
            entry['analytic_solution']=solution;triangles=solution['plane_reference_triangles_m']
            nodes=np.vstack((np.asarray(tx,float),solution['points'],np.asarray(rx,float)));mappings=solution['patch_mappings']
            visibility=[];entry['segment_visibility']=visibility
            if validate_occlusion:
                for j,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
                    result=check_patch_segment(geometry,a,b,
                        start_faces=mappings[j-1]['containing_faces'] if j else (),
                        end_faces=mappings[j]['containing_faces'] if j<len(mappings) else ())
                    visibility.append(result)
                    if result['status']!='clear':raise UnresolvedSpecularGroup('Physical segment visibility not certified: '+str(result))
            world_scale=max(1.,float(abs(nodes).max(initial=0.)))
            coordinate_ulp=float(abs(np.spacing(np.float32(world_scale))))
            plane_tolerance=max(1e-8,8*coordinate_ulp)
            delay_tolerance=max(1e-15,8*float(abs(np.spacing(np.float32(solution['delay_sec'])))))
            optical_tolerance=C*delay_tolerance
            length_roundoff=4096*np.finfo(float).eps*max(world_scale,solution['length_m'],1.)*(len(key)+1)
            candidates=[]
            for i in indices:
                active=vertices[i,:len(key)];point_errors=np.linalg.norm(active-solution['points'],axis=-1)
                max_point_error=float(point_errors.max(initial=0.));rms=float(np.sqrt(np.mean(point_errors**2))) if len(key) else 0.
                delay_error=float(compact['delay_sec'][i]-solution['delay_sec'])
                score=float(np.hypot(rms,C*delay_error/(len(key)+1)))
                signed_plane=np.einsum('ij,ij->i',active-triangles[:,0],solution['normals'])
                projected=active-signed_plane[:,None]*solution['normals']
                projected_nodes=np.vstack((np.asarray(tx,float),projected,np.asarray(rx,float)))
                projected_segments=np.diff(projected_nodes,axis=0);projected_lengths=np.linalg.norm(projected_segments,axis=1)
                polyline_length=float(projected_lengths.sum());optical_excess=polyline_length-solution['length_m']
                native_nodes=np.vstack((np.asarray(tx,float),active,np.asarray(rx,float)))
                native_polyline_length=float(np.linalg.norm(np.diff(native_nodes,axis=0),axis=1).sum())
                fermat=[];reflection_residuals=[];same_side=True
                if len(key) and np.all(projected_lengths>0):
                    for j,n in enumerate(solution['normals']):
                        incoming=projected_segments[j]/projected_lengths[j];outgoing=projected_segments[j+1]/projected_lengths[j+1]
                        gradient=incoming-outgoing;fermat.append(float(np.linalg.norm(gradient-(gradient@n)*n)))
                        reflection_residuals.append(float(np.linalg.norm(outgoing-(incoming-2*(incoming@n)*n))))
                        same_side &= bool(((projected_nodes[j]-projected[j])@n)*((projected_nodes[j+2]-projected[j])@n)>0)
                max_plane_error=float(abs(signed_plane).max(initial=0.))
                agreement=bool(np.isfinite(active).all() and np.isfinite(score) and same_side
                    and np.all(projected_lengths>=.01) and max_plane_error<=plane_tolerance
                    and -length_roundoff<=optical_excess<=optical_tolerance+length_roundoff and abs(delay_error)<=delay_tolerance)
                finite=lambda value:float(value) if np.isfinite(value) else None
                mapping=[catalog.locate(o,p,solution['points'][j]) for j,(o,p,_) in enumerate(declared[i])]
                candidates.append(dict(compact_index=i,native_path_index=int(compact['path_index'][i]),selection_score_m=finite(score),
                    max_point_error_m=finite(max_point_error),point_rms_error_m=finite(rms),delay_error_sec=finite(delay_error),
                    max_plane_error_m=finite(max_plane_error),projected_polyline_length_m=finite(polyline_length),
                    projected_optical_excess_m=finite(optical_excess),native_polyline_length_m=finite(native_polyline_length),
                    native_delay_minus_native_polyline_sec=finite(float(compact['delay_sec'][i])-native_polyline_length/C),
                    max_projected_fermat_residual=max(fermat,default=0.),max_projected_reflection_vector_residual=max(reflection_residuals,default=0.),
                    projected_same_side=bool(same_side),agrees_with_unique_solution=agreement,declared_to_physical=mapping))
            entry.update(candidates=candidates,native_plane_agreement_tolerance_m=plane_tolerance,
                native_delay_agreement_tolerance_sec=delay_tolerance,native_optical_excess_tolerance_m=optical_tolerance,
                float64_length_roundoff_m=length_roundoff,point_distance_is_admission_bound=False,
                agreement_scope='exact physical patch + strict union interior/full visibility + original V3 plane/delay/projected-optical-length consistency; numerical native representative, not an exact EM gain bound')
            eligible=sorted((c for c in candidates if c['agrees_with_unique_solution']),key=lambda c:(c['selection_score_m'],c['compact_index']))
            if not eligible:raise UnresolvedSpecularGroup('No native candidate agrees with the unique Float64 physical-patch image solution')
            best=eligible[0];tie_tolerance=4096*np.finfo(float).eps*world_scale
            tied=[c for c in eligible if abs(c['selection_score_m']-best['selection_score_m'])<=tie_tolerance]
            if len(tied)>1:
                fields=('b00','delay_sec','doppler_hz','k_tx','k_rx')
                equivalent=all(all(np.array_equal(compact[f][c['compact_index']],compact[f][best['compact_index']]) for f in fields) for c in tied)
                if not equivalent:raise UnresolvedSpecularGroup('Numerically tied representatives have different channel fields')
            selected.append(best['compact_index']);entry.update(status='certified_unique' if validate_occlusion else 'geometry_only',
                selected_compact_index=best['compact_index'],selection_reason='closest native bounce/delay representative of one mathematically unique ordered physical-patch topology',
                raw_candidates_removed_from_field_sum=len(indices)-1)
        except (UnresolvedSpecularGroup,np.linalg.LinAlgError,KeyError,IndexError,ValueError) as error:
            entry.update(status='unresolved',reason=str(error))
        reports.append(entry)
    selected=sorted(selected);raw_triangle_keys=defaultdict(list)
    for i,top in declared.items():raw_triangle_keys[top].append(i)
    return dict(schema=VERSION,patch_schema=PATCH_VERSION,
        admitted=bool(selected and reports and all(r['status']=='certified_unique' for r in reports)),
        keep_compact_indices=selected,raw_path_count=len(compact),selected_path_count=len(selected),groups=reports,
        unresolved_group_count=sum(r['status']=='unresolved' for r in reports),
        full_topology_groups=len(groups),duplicate_topology_groups=sum(len(v)>1 for v in groups.values()),
        raw_declared_triangle_group_count=len(raw_triangle_keys),
        raw_declared_triangle_topologies_preserved=True,
        patch_identities=catalog.identities([(o,p) for top in declared.values() for o,p,_ in top]),
        different_raw_triangle_topologies_merged=any(len(set(declared[i] for i in ids))>1 for ids in groups.values()),
        different_physical_topologies_merged=False,different_topologies_merged=False,
        inactive_slots_ignored=True,occlusion_checked=bool(validate_occlusion),patch_catalog_statistics=catalog.statistics(),
        physical_scope='complete ordered exact-coplanar full-edge-connected same-object/material-slot patch sequence; strict union interior, actual scene visibility; unresolved blocks admission',
        candidate_storage='all original coefficients, active vertices, primitive IDs and explicit raw-declared to physical patch/face mapping remain archived')


def audit_patch_paths(paths,scene,object_map,*,geometry_cache=None,patch_catalog=None,dynamic_object_ids=(),tx_xyz=None,rx_xyz=None):
    source=check_private_contract();begun=time.perf_counter()
    raw_compact=extract_compact(paths,object_map);raw=exact_metadata(paths);raw_native=native_cfr(paths)
    exported=time.perf_counter();native_count=np.asarray(paths.tau).size
    if native_count and (not hasattr(paths,'_path_ind') or len(np.asarray(paths._path_ind).reshape(-1))!=native_count):
        raise RuntimeError('Single-pair raw permutation contract changed')
    tx=xyz_rows(paths.sources)[0] if tx_xyz is None else np.asarray(tx_xyz,float)
    rx=xyz_rows(paths.targets)[0] if rx_xyz is None else np.asarray(rx_xyz,float)
    geometry=geometry_cache if geometry_cache is not None else SpecularSceneGeometry()
    geometry.begin_snapshot(scene,object_map,dynamic_object_ids)
    catalog=patch_catalog if patch_catalog is not None else SurfacePatchCatalog()
    catalog.refresh(geometry);cached=time.perf_counter()
    report=audit_patch_candidates(raw_compact,raw,tx,rx,geometry,patch_catalog=catalog)
    audited=time.perf_counter();selected=np.asarray(report['keep_compact_indices'],int)
    keep=np.zeros(np.asarray(paths.tau).size,bool);native_indices=np.asarray(raw['native_path_index'],int)
    if len(native_indices)!=len(raw_compact) or len(np.unique(native_indices))!=len(native_indices):
        raise RuntimeError('Native/raw/compact selection permutation changed')
    keep[native_indices[selected]]=True;native,mask_report,masked=masked_native_cfr(paths,keep);masked_at=time.perf_counter()
    compact=raw_compact[selected].copy()
    txrel=xyz_rows(scene.tx_array.rotate(scene.wavelength,next(iter(scene.transmitters.values())).orientation))
    rxrel=xyz_rows(scene.rx_array.rotate(scene.wavelength,next(iter(scene.receivers.values())).orientation))
    rebuilt=reconstruct(compact,txrel,rxrel);norm=float(np.linalg.norm(native))
    relative=float(np.linalg.norm(rebuilt-native)/norm) if np.isfinite(norm) and norm>0 else None
    report.update(private_source_contract=source,masked_native_control=mask_report,
        selected_compact_vs_masked_native_relative_error=relative,
        selected_compact_vs_masked_native_pass=bool(relative is not None and np.isfinite(relative) and relative<1e-4),
        timing_seconds=dict(raw_export=exported-begun,geometry_cache=cached-exported,
            image_and_full_scene_visibility=audited-cached,masked_native_reference=masked_at-audited,total=time.perf_counter()-begun))
    report['admitted']=bool(report['admitted'] and report['selected_compact_vs_masked_native_pass'])
    return dict(compact=compact,native=native,raw_compact=raw_compact,raw_native=raw_native,raw=raw,
        keep_native_mask=keep,report=report,masked_paths=masked,geometry_cache=geometry,patch_catalog=catalog,
        tx_relative_m=txrel,rx_relative_m=rxrel)

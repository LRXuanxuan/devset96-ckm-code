"""Native city RT, fixed-pose compact-path reconstruction and midpoint checks.

The selected corridors are geometric candidates, not surveyed driving routes.
No training set or measured-channel claim is made by this module.
"""
from __future__ import annotations

import math
import hashlib
import time
from pathlib import Path

import h5py
import numpy as np

from .common import portable_asset, save_json, sha256, nmse_db
from .scene_probe import probe_scene

C = 299792458.0
FC = 28e9
FREQUENCIES = np.linspace(-12e6, 12e6, 16, dtype=np.float32)
# Fixed 128-byte schema, with actual interaction metadata rather than padding.
# path_index is local to a snapshot, NOT a cross-snapshot physical path ID.
COMPACT_DTYPE = np.dtype([
    ('b00', '<c8'), ('delay_sec', '<f8'), ('doppler_hz', '<f8'),
    ('k_tx', '<f8', (3,)), ('k_rx', '<f8', (3,)),
    ('object_ids', '<u8', (3,)), ('interaction_types', '<u4', (3,)),
    ('depth', '<u4'), ('path_index', '<u8'), ('flags', '<u8'),
], align=False)
assert COMPACT_DTYPE.itemsize == 128


def complex_pair(pair):
    # Sionna out_type='numpy' returns a native complex ndarray, whereas the
    # default Dr.Jit output is a (real,imag) tuple.
    if isinstance(pair, (tuple,list)) and len(pair)==2:
        return np.asarray(pair[0]) + 1j*np.asarray(pair[1])
    return np.asarray(pair)


def xyz_rows(points):
    return np.column_stack([np.asarray(points.x).ravel(), np.asarray(points.y).ravel(),
                            np.asarray(points.z).ravel()]).astype(np.float64)


def extract_compact(paths, runtime_to_stable):
    import sionna.rt as rt
    if rt.__version__ != '2.2.0' or not paths.synthetic_array:
        raise RuntimeError('Compact adapter is pinned to 2.2.0 synthetic-array paths')
    pair, tau = paths.cir(normalize_delays=False, out_type='numpy')
    b = complex_pair(pair)[0, :, 0, :, :, 0]  # [rx_ant,tx_ant,path]
    tau = np.asarray(tau).reshape(-1)
    valid = np.asarray(paths.valid).reshape(-1) & (tau >= 0)
    index = np.flatnonzero(valid)
    compact = np.zeros(len(index), dtype=COMPACT_DTYPE)
    compact['b00'] = b[0, 0, index]
    compact['delay_sec'] = tau[index]
    compact['doppler_hz'] = np.asarray(paths.doppler).reshape(-1)[index]
    # Version-pinned internal vectors avoid angle singularities near poles.
    # Source/API hash is recorded; stop on shape/version changes.
    k_tx = np.asarray(paths._k_tx).reshape(-1, 3)
    k_rx = np.asarray(paths._k_rx).reshape(-1, 3)
    if len(k_tx) != len(tau) or len(k_rx) != len(tau):
        raise RuntimeError('Unexpected native direction-vector shape')
    compact['k_tx'], compact['k_rx'] = k_tx[index], k_rx[index]
    if not len(tau):
        return compact
    # Avoid the visualization-oriented lazy component assembly. On this local
    # Windows backend that extra scatter kernel has produced native failures.
    # Read the already-computed buffer and reproduce its documented path-index
    # permutation in NumPy. This adapter supports exactly one TX/RX source pair.
    buffer=paths._paths_buffer
    path_permutation=np.asarray(paths._path_ind).astype(np.int64)
    if not np.array_equal(np.sort(path_permutation),np.arange(len(tau))):
        raise RuntimeError('Raw metadata adapter requires one synthetic TX/RX pair')
    inverse=np.empty(len(tau),dtype=np.int64)
    inverse[path_permutation]=np.arange(len(tau))
    objects=np.array(buffer.shapes.numpy(),copy=True)[inverse].T
    types=np.array(buffer.interaction_types.numpy(),copy=True)[inverse].T
    if objects.shape[0] > 3:
        raise RuntimeError('128-byte schema supports at most three interaction slots')
    compact['object_ids'].fill(np.iinfo(np.uint64).max)
    native_ids=objects[:, index].T
    mapped=np.full(native_ids.shape,np.iinfo(np.uint64).max,dtype=np.uint64)
    for runtime_id,stable_id in runtime_to_stable.items():
        mapped[native_ids==runtime_id]=stable_id
    occupied=(types[:,index].T!=0)
    if np.any(occupied & (mapped==np.iinfo(np.uint64).max)):
        raise RuntimeError('Unmapped nonempty native object handle')
    compact['object_ids'][:, :objects.shape[0]] = mapped
    compact['interaction_types'][:, :types.shape[0]] = types[:, index].T
    compact['depth'] = np.count_nonzero(types[:, index], axis=0)
    compact['path_index'] = index
    compact['flags'] = 1  # bit 0 = valid; other bits reserved zero
    return compact


def reconstruct(compact, tx_relative, rx_relative, frequencies=FREQUENCIES,
                delta_t=0.0, carrier=FC):
    """Fixed pattern/polarization/pose; first antenna is the stored reference.

    Delay is frozen inside each local Doppler interval, matching native cir.
    Cross-anchor geometric evolution is checked separately at RT midpoints.
    """
    ptx = tx_relative - tx_relative[0]
    prx = rx_relative - rx_relative[0]
    phase = 2*np.pi*carrier/C * (prx @ compact['k_rx'].T)[:, None, :]
    phase = phase + 2*np.pi*carrier/C * (ptx @ compact['k_tx'].T)[None, :, :]
    a = compact['b00'][None, None, :] * np.exp(1j*phase)
    a *= np.exp(2j*np.pi*compact['doppler_hz']*delta_t)[None,None,:]
    freq_phase = np.exp(-2j*np.pi*compact['delay_sec'][:,None]*np.asarray(frequencies)[None,:])
    return np.einsum('rtp,pf->rtf', a, freq_phase, optimize=True)


def native_cfr(paths):
    import mitsuba as mi
    pair = paths.cfr(frequencies=mi.Float(FREQUENCIES), normalize_delays=False,
                     normalize=False, out_type='numpy')
    return complex_pair(pair)[0, :, 0, :, 0, :]


def solver_settings(samples):
    return dict(max_depth=2, samples_per_src=samples, max_num_paths_per_src=4096,
                synthetic_array=True, los=True, specular_reflection=True,
                diffuse_reflection=False, refraction=False, diffraction=False, seed=20261025)


def stable_objects(scene, city):
    mapping={};records=[]
    for name,obj in sorted(scene.objects.items()):
        stable_id=int.from_bytes(hashlib.sha256((city+'/'+name).encode()).digest()[:8],'little')
        if stable_id==np.iinfo(np.uint64).max or stable_id in mapping.values():
            raise RuntimeError('Stable object hash collision')
        runtime_id=int(obj.object_id)
        mapping[runtime_id]=stable_id
        records.append({'stable_id':stable_id,'runtime_id_at_capture':runtime_id,'name':name,
                        'id_rule':'first 64 little-endian bits of SHA256(scene/name)',
                        'geometry_provenance':'scene asset hashes and recorded object transforms'})
    return mapping,records


def defined_nmse(estimate,reference):
    # An empty/no-energy reference has no defined normalized estimation error.
    return nmse_db(estimate,reference) if np.linalg.norm(reference)>0 else None


def write_archive(path, snapshots, golden):
    temp = path.with_suffix('.h5.partial')
    offsets = np.r_[0, np.cumsum([len(s['compact']) for s in snapshots])].astype('<i8')
    all_paths = np.concatenate([s['compact'] for s in snapshots])
    with h5py.File(temp, 'x') as f:
        f.attrs['schema'] = 'sionna_small_validation.compact128.v1'
        f.attrs['description'] = 'Local validation archive; not a production training dataset'
        f.attrs['phase_reference'] = 'native baseband CIR at antenna rx0,tx0; no extra carrier phase'
        f.attrs['scope'] = 'synthetic_array=True; fixed single-polarization pattern and actual stored poses'
        f.attrs['carrier_hz'] = FC
        d = f.create_dataset('paths', data=all_paths, chunks=True)
        d.attrs['axes'] = 'flat_path'; d.attrs['itemsize_bytes'] = COMPACT_DTYPE.itemsize
        d.attrs['units'] = 'b00:linear complex; delay:s; doppler:Hz; k:unit GCS vectors; IDs:dimensionless'
        d = f.create_dataset('offsets', data=offsets); d.attrs['axes'] = 'snapshot_plus_one'
        for name, arr, axes, units in (
            ('time_sec', [s['time_sec'] for s in snapshots], 'snapshot', 's'),
            ('rx_xyz_m', [s['rx'] for s in snapshots], 'snapshot,xyz', 'm'),
            ('tx_xyz_m', [s['tx'] for s in snapshots], 'snapshot,xyz', 'm'),
            ('tx_relative_m', [s['tx_relative'] for s in snapshots], 'snapshot,tx_ant,xyz', 'm'),
            ('rx_relative_m', [s['rx_relative'] for s in snapshots], 'snapshot,rx_ant,xyz', 'm'),
            ('frequencies_hz', FREQUENCIES, 'frequency', 'Hz'),
        ):
            d = f.create_dataset(name, data=arr); d.attrs['axes'] = axes; d.attrs['units'] = units
        for field in ('scene','fragment','state','kind'):
            f.create_dataset(field, data=np.asarray([str(s[field]) for s in snapshots], dtype=h5py.string_dtype()))
        for field in ('tx_orientation_rad','rx_orientation_rad'):
            d=f.create_dataset(field,data=[s[field] for s in snapshots]);d.attrs['axes']='snapshot,alpha_beta_gamma';d.attrs['units']='rad'
        group=f.create_group('golden_cfr')
        group.attrs['axes']='snapshot,rx_ant,tx_ant,frequency'
        group.attrs['normalization']='none; physical native RT coefficients'
        group.create_dataset('snapshot_index',data=[g[0] for g in golden])
        H=np.stack([g[1] for g in golden]).astype(np.complex64)
        group.create_dataset('real',data=H.real,compression='gzip',compression_opts=1)
        group.create_dataset('imag',data=H.imag,compression='gzip',compression_opts=1)
    # Verify real on-disk values, including compound 128-byte records.
    with h5py.File(temp,'r') as f:
        if not np.array_equal(f['paths'][:],all_paths) or not np.array_equal(f['offsets'][:],offsets):
            raise RuntimeError('Compact archive roundtrip failed')
    temp.replace(path)
    return {'path':str(path.resolve()),'sha256':sha256(path),'bytes':path.stat().st_size,
            'path_records':len(all_paths),'snapshot_count':len(snapshots),'roundtrip_exact':True}


def run_cities(out_dir, fragments=8, anchors=8, samples=20000, dynamic=True,
               cities=('munich','florence','etoile')):
    import mitsuba as mi
    import sionna.rt as rt
    import matplotlib.pyplot as plt
    start=time.perf_counter()
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    solver=rt.PathSolver(deterministic=True)
    # Evaluated loops are slower but avoid a large symbolic CPU kernel in this
    # small Windows portability check; this is an official solver setting.
    solver.loop_mode='evaluated'
    snapshots=[];golden=[];metrics=[];midpoints=[];city_reports=[]
    settings=solver_settings(samples)
    max_rel_tolerance=1e-4
    # The midpoint bound is a screening engineering requirement, not achieved by definition.
    midpoint_nmse_limit_db=-30.0
    for city in cities:
        print(f'[city] loading {city}',flush=True)
        asset=portable_asset(getattr(rt.scene,city));scene=rt.load_scene(asset)
        scene.frequency=FC
        probe=probe_scene(scene,city,out_dir/'geometry')
        if not probe['routes']:
            raise RuntimeError(f'No sampled open corridor in {city}')
        scene.tx_array=rt.PlanarArray(num_rows=8,num_cols=8,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
        scene.rx_array=rt.PlanarArray(num_rows=4,num_cols=4,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
        tx=rt.Transmitter(name='tx',position=probe['suggested_tx_xyz'])
        rx=rt.Receiver(name='rx',position=probe['suggested_rx_xyz'])
        scene.add([tx,rx]);city_indices=[];dynamic_counts=[];object_records={}
        for frag in range(fragments):
            proposal=probe['routes'][frag%len(probe['routes'])]
            heading=proposal['heading_rad'];direction=np.array([math.cos(heading),math.sin(heading),0.])
            center=np.asarray(proposal['center_xyz'],float)+(frag//len(probe['routes'])-1)*6*direction
            tx.position=proposal['suggested_tx_xyz'];tx.look_at(mi.Point3f(center.tolist()))
            rx.orientation=[heading,0,0];rx.velocity=(30*direction).tolist()
            txpos=np.asarray(proposal['suggested_tx_xyz'],float)
            # Half of the fragments use a geometric NLoS candidate; native
            # nonzero reflected paths are checked rather than assumed.
            nlos_case=bool(frag%2 and probe.get('nlos_tx_candidate'))
            if nlos_case:
                candidate=probe['nlos_tx_candidate']
                center=np.asarray(candidate['rx_xyz'],float)+(frag//len(probe['routes'])-1)*.5*direction
                txpos=np.asarray(candidate['tx_xyz'],float)
                tx.position=txpos.tolist();tx.look_at(mi.Point3f(center.tolist()))
            ptx=xyz_rows(scene.tx_array.rotate(scene.wavelength,tx.orientation))
            prx=xyz_rows(scene.rx_array.rotate(scene.wavelength,rx.orientation))
            baseline={}
            # 8 anchors over 7cm; optional midpoint of every other gap.
            positions=np.arange(anchors)*.01-(anchors-1)*.005
            mid=((positions[:-1]+positions[1:])/2)[::2]
            for state in ('static','dynamic') if dynamic and frag<2 else ('static',):
                car=None;car0=None;carvelocity=None
                if state=='dynamic':
                    # Illustrative moving metallic van, not a calibrated traffic model.
                    material=rt.ITURadioMaterial(name=f'van_metal_{frag}',itu_type='metal',thickness=.01)
                    car=rt.SceneObject(fname=portable_asset(rt.scene.low_poly_car),name=f'van_{frag}',radio_material=material)
                    car.scaling=mi.Vector3f(1.,1.,1.5)
                    bb=car.mi_mesh.bbox();height=float(bb.max.z-bb.min.z)
                    car0=center-5*direction;car0[2]=height/2
                    car.orientation=[heading,0,0]
                    carvelocity=10*np.array([-direction[1],direction[0],0.])
                    car.position=car0.tolist();car.velocity=carvelocity.tolist();scene.add(car)
                object_map,object_list=stable_objects(scene,city)
                for obj in object_list:object_records[obj['name']]=obj
                previous=[]
                for kind,offsets in (('anchor',positions),('midpoint',mid)):
                    for q,offset in enumerate(offsets):
                        t=float((offset-positions[0])/30.)
                        rpos=center+offset*direction;rx.position=rpos.tolist()
                        if car is not None:
                            car.position=(car0+carvelocity*t).tolist()
                        effective_settings=settings.copy()
                        if nlos_case:
                            effective_settings.update(max_depth=2,diffraction=False)
                        tick=time.perf_counter();paths=solver(scene,**effective_settings);solve_s=time.perf_counter()-tick
                        compact=extract_compact(paths,object_map);native=native_cfr(paths)
                        reconstructed=reconstruct(compact,ptx,prx)
                        error=defined_nmse(reconstructed,native)
                        valid_h=bool(np.isfinite(native).all() and np.linalg.norm(native)>0)
                        if kind=='anchor':
                            idx=len(snapshots)
                            s={'scene':city,'fragment':f'{city}_{frag}','state':state,'kind':kind,
                               'time_sec':t,'tx':txpos.copy(),'rx':rpos.copy(),'compact':compact,
                               'tx_relative':ptx.copy(),'rx_relative':prx.copy(),
                               'tx_orientation_rad':xyz_rows(tx.orientation)[0],
                               'rx_orientation_rad':xyz_rows(rx.orientation)[0]}
                            snapshots.append(s);city_indices.append(idx)
                            met={'snapshot_index':idx,'scene':city,'fragment':frag,'state':state,'intended_nlos':nlos_case,
                                 'path_count':len(compact),'has_los':bool(np.any(compact['depth']==0)),
                                 'max_abs_doppler_hz':float(np.max(np.abs(compact['doppler_hz']))) if len(compact) else 0.,
                                 'compact_cfr_nmse_db':error,'nonzero_finite_cfr':valid_h,'solve_seconds':solve_s}
                            metrics.append(met);previous.append((offset,s,native))
                            if q==0:
                                golden.append((idx,native))
                            if state=='static': baseline[q]=native
                            elif q in baseline:
                                met['paired_dynamic_difference_nmse_db']=defined_nmse(native,baseline[q])
                        else:
                            left=max((item for item in previous if item[0]<offset),key=lambda item:item[0])
                            dt=float((offset-left[0])/30.)
                            estimate=reconstruct(left[1]['compact'],ptx,prx,delta_t=dt)
                            midpoint_error=defined_nmse(estimate,native)
                            midpoints.append({'scene':city,'fragment':frag,'state':state,
                                              'offset_m':float(offset),'dt_sec':dt,
                                              'native_midpoint_path_count':len(compact),
                                              'left_path_count':len(left[1]['compact']),
                                              'cfr_nmse_db':midpoint_error,
                                              'passes_screening_bound':midpoint_error is not None and midpoint_error<=midpoint_nmse_limit_db})
                if car is not None:
                    # Actual Doppler varies with scattering; do not infer car-path coverage from presence of a mesh.
                    dynamic_counts.append({'fragment':frag,'van_center_at_t0_m':car0.tolist(),
                                           'van_velocity_mps':carvelocity.tolist(),
                                           'scale_xyz':[1.,1.,1.5],'position_integrated_explicitly':True})
                    scene.remove(car)
            print(f'[city] {city} fragment {frag+1}/{fragments} complete',flush=True)
        asset_root=Path(asset).parent
        city_reports.append({'scene':city,'asset_xml':asset,'asset_files_sha256':
                            {str(p.relative_to(asset_root)):sha256(p) for p in sorted(asset_root.rglob('*')) if p.is_file()},
                            'probe':probe,'snapshot_count':len(city_indices),'dynamic_cases':dynamic_counts,
                            'object_id_map':list(object_records.values())})
    archive=write_archive(out_dir/'compact128.h5',snapshots,golden)
    errors=np.asarray([m['compact_cfr_nmse_db'] for m in metrics if m['compact_cfr_nmse_db'] is not None])
    counts=np.asarray([m['path_count'] for m in metrics])
    mid_errors=np.asarray([m['cfr_nmse_db'] for m in midpoints if m['cfr_nmse_db'] is not None])
    good_recon=all(m['nonzero_finite_cfr'] and m['compact_cfr_nmse_db'] is not None and m['compact_cfr_nmse_db']<=20*np.log10(max_rel_tolerance) for m in metrics)
    good_mid=all(m['passes_screening_bound'] for m in midpoints)
    report={'status':'passed' if good_recon and good_mid else 'needs_attention',
            'native_compact_reconstruction_passed':good_recon,'midpoint_screening_passed':good_mid,
            'solver_settings':settings,'solver_loop_mode':'evaluated',
            'metadata_adapter':'version-pinned raw PathsBuffer, one synthetic TX/RX pair, explicit path-index permutation',
            'nlos_solver_override':{'max_depth':2,'diffraction':False},
            'carrier_hz':FC,'frequencies_hz':FREQUENCIES,
            'array_config':{'tx':[8,8],'rx':[4,4],'spacing_wavelength':.5,'pattern':'iso','polarization':'V'},
            'anchor_step_m':.01,'ue_speed_mps':30.,'compact_record_bytes':128,
            'compact_relative_error_tolerance':max_rel_tolerance,'midpoint_screening_nmse_limit_db':midpoint_nmse_limit_db,
            'snapshot_count':len(snapshots),'midpoint_count':len(midpoints),
            'total_solve_states':len(snapshots)+len(midpoints),
            'path_count':{'median':float(np.median(counts)),'p95':float(np.percentile(counts,95)),
                          'max':int(counts.max()),'mean':float(counts.mean()),'hard_truncation_to_128':False},
            'empty_channel_anchor_count':sum(not m['nonzero_finite_cfr'] for m in metrics),
            'undefined_midpoint_nmse_count':sum(m['cfr_nmse_db'] is None for m in midpoints),
            'compact_nmse_db':{'worst':float(errors.max()) if len(errors) else None,'median':float(np.median(errors)) if len(errors) else None},
            'midpoint_nmse_db':{'worst':float(mid_errors.max()) if len(mid_errors) else None,'median':float(np.median(mid_errors)) if len(mid_errors) else None},
            'archive':archive,'cities':city_reports,'anchor_metrics':metrics,'midpoint_metrics':midpoints,
            'elapsed_seconds':time.perf_counter()-start,
            'limitations':[
                'Open-ground corridors are sampled geometric candidates, not verified semantic drivable lanes.',
                'This short-window suite does not certify long-route support birth/death or a complete LoS/NLoS/corner balance.',
                'Synthetic arrays assume local plane waves and common path visibility; compact reconstruction only validates against that native approximation.',
                'Single fixed polarization/pattern/actual pose only; no arbitrary-pattern or dual-polarization reuse guarantee.',
                'Depth=2 and limited candidate rays are validation settings, not production convergence settings.',
                'A mean path count from this pilot is not a guarantee that a full dataset averages 128 paths.',
                'Stored b00 inherits native single-precision geometric phase; float64 delay does not recover lost native precision.',
                'Doppler extrapolation freezes local amplitudes, directions and delays; RT midpoint errors are reported independently.'
            ]}
    save_json(out_dir/'cities_report.json',report)
    fig,axs=plt.subplots(1,3,figsize=(12,3.6))
    for city in cities:
        a=[x for x in metrics if x['scene']==city]
        axs[0].plot([x['compact_cfr_nmse_db'] for x in a],'.',label=city)
        axs[1].plot([x['cfr_nmse_db'] for x in midpoints if x['scene']==city],'.',label=city)
        axs[2].hist([x['path_count'] for x in a],bins=12,alpha=.4,label=city)
    axs[0].axhline(20*np.log10(max_rel_tolerance),color='k',ls='--');axs[0].set_ylabel('Compact/native CFR NMSE [dB]')
    axs[1].axhline(midpoint_nmse_limit_db,color='k',ls='--');axs[1].set_ylabel('Extrapolated/RT midpoint NMSE [dB]')
    axs[2].set_xlabel('Retained native path count');axs[2].set_ylabel('Anchors')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out_dir/'city_metrics.png',dpi=160);plt.close(fig)
    payload={'kind':'rt_path_records','provenance':{'source':'Sionna RT 2.2.0 native baseband antenna 0,0',
             'compact_archive':archive['path'],'not_measurement_data':True},
             'records':[{'trajectory_id':i,'time_sec':s['time_sec'],'gain':s['compact']['b00'],
                         'delay_sec':s['compact']['delay_sec'],'doppler_hz':s['compact']['doppler_hz'],
                         'path_id':s['compact']['path_index']} for i,s in enumerate(snapshots)]}
    return report,payload

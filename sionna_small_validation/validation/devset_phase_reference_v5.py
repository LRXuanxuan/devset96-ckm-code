"""High-precision image delay and propagation phase over native transport.

Original V4 native coefficients/delays/reports remain untouched. The new
reference is formed from every selected masked native array coefficient before
``Paths.cir`` applies its carrier rotation. It is independent of compact array
reconstruction and does not claim an exact electromagnetic error bound.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .cities import FC,FREQUENCIES,reconstruct
from .devset_specular_audit_v4 import audit_patch_paths

VERSION='kap_devset.image_delay_phase_reference.v5'


def phase_source_sha256(correction):
    """Integrity anchor of selected native source rows, not source attestation."""
    digest=hashlib.sha256((VERSION+'\0native-passband-source\0').encode())
    for name,dtype in (('selected_raw_compact_indices','<u8'),('selected_native_path_indices','<u8'),
                       ('selected_native_passband_b00','<c8'),('selected_native_carrier_phasor','<c8'),
                       ('native_delay_sec','<f8'),('ideal_delay_sec','<f8')):
        digest.update(name.encode()+b'\0');digest.update(np.asarray(correction[name],dtype=dtype).tobytes())
    return digest.hexdigest()


def reference_from_full_passband(passband,ideal_delay_sec,frequencies_hz=FREQUENCIES,carrier_hz=FC):
    """Independent full Rx×Tx per-path summation in complex128/Float64."""
    a=np.asarray(passband,dtype=np.complex128)
    delay=np.asarray(ideal_delay_sec,dtype=np.float64)
    frequencies=np.asarray(frequencies_hz,dtype=np.float64)
    if a.ndim!=3 or delay.shape!=(a.shape[2],) or frequencies.ndim!=1:
        raise ValueError('Expected native passband[rx,tx,path], image_delay[path], frequencies[frequency]')
    if (not np.isfinite(a).all() or not np.isfinite(delay).all() or np.any(delay<0)
            or not np.isfinite(frequencies).all() or not np.isfinite(carrier_hz)):
        raise ValueError('Nonfinite native passband/image delay/reference frequency')
    # One Float64 phase law includes carrier and baseband frequencies. This
    # does not call reconstruct or use its b00/steering-vector implementation.
    phase=np.exp(-2j*np.pi*delay[:,None]*(float(carrier_hz)+frequencies[None,:]))
    return np.einsum('rtp,pf->rtf',a,phase,optimize=True)


def audit_phase_paths(paths,scene,object_map,*,geometry_cache=None,patch_catalog=None,dynamic_object_ids=()):
    original_a=tuple(np.array(np.asarray(x),copy=True) for x in paths.a)
    original_tau=np.array(np.asarray(paths.tau),copy=True)
    base=audit_patch_paths(paths,scene,object_map,geometry_cache=geometry_cache,
        patch_catalog=patch_catalog,dynamic_object_ids=dynamic_object_ids)
    # Do not mutate this report or reinterpret its native.cfr error numbers.
    report=base['report'];indices=np.asarray(report['keep_compact_indices'],dtype=np.int64)
    source=np.asarray(base['raw']['native_path_index'],dtype=np.int64)[indices]
    delay_by_index={int(group['selected_compact_index']):float(group['analytic_solution']['delay_sec'])
        for group in report['groups'] if group.get('selected_compact_index') is not None}
    ideal=np.asarray([delay_by_index[int(index)] for index in indices],dtype=np.float64)
    old=base['compact'];corrected=old.copy()
    pair=base['masked_paths'].a
    full=np.asarray(pair[0],dtype=np.float64)+1j*np.asarray(pair[1],dtype=np.float64)
    if (full.ndim!=5 or full.shape[0]!=1 or full.shape[2]!=1
            or full.shape[-1]!=np.asarray(paths.tau).size):
        raise RuntimeError('Pinned one-Tx/Rx native passband array axes changed')
    selected_passband=full[0,:,0,:,:][:,:,source]
    a00=selected_passband[0,0,:].astype(np.complex64)
    # This is exactly the phase-only part of the pinned Paths.cir expression,
    # using its actual native frequency and tau tensors. It does no RT solve.
    import drjit as dr
    from sionna.rt.utils import map_angle_to_canonical_range
    if len(source):
        native_phase=-dr.two_pi*paths._frequency*paths.tau
        native_phase=map_angle_to_canonical_range(native_phase)
        sine,cosine=dr.sincos(native_phase)
        native_factor=(np.asarray(cosine).reshape(-1)+1j*np.asarray(sine).reshape(-1))[source].astype(np.complex64)
    else:native_factor=np.zeros(0,np.complex64)
    source_rebuilt=a00.astype(np.complex128)*native_factor.astype(np.complex128)
    source_norm=float(np.linalg.norm(old['b00']))
    source_error=float(np.linalg.norm(source_rebuilt-old['b00'])/source_norm) if source_norm>0 else 0.
    if not np.isfinite(source_error) or source_error>2e-6:
        raise RuntimeError('Selected native Paths.a and native cir carrier factor do not reproduce the original raw b00')
    corrected['delay_sec']=ideal
    corrected['b00']=a00.astype(np.complex128)*np.exp(-2j*np.pi*float(FC)*ideal)
    reference=reference_from_full_passband(selected_passband,ideal)
    norm=float(np.linalg.norm(reference))
    rebuilt=reconstruct(corrected,base['tx_relative_m'],base['rx_relative_m'])
    error=float(np.linalg.norm(rebuilt-reference)/norm) if np.isfinite(norm) and norm>0 else None
    unchanged=(all(np.array_equal(np.asarray(x),previous,equal_nan=True) for x,previous in zip(paths.a,original_a))
               and np.array_equal(np.asarray(paths.tau),original_tau,equal_nan=True))
    if not unchanged:raise RuntimeError('Phase reference changed original native passband/delay tensors')
    correction=dict(selected_raw_compact_indices=indices,selected_native_path_indices=source,
        selected_native_passband_b00=a00,selected_native_carrier_phasor=native_factor,
        native_delay_sec=old['delay_sec'].copy(),ideal_delay_sec=ideal,
        native_carrier_frequency_hz=float(np.asarray(paths._frequency).reshape(-1)[0]))
    correction['native_source_integrity_sha256']=phase_source_sha256(correction)
    phase_report=dict(schema=VERSION,selected_path_count=len(indices),
        corrected_compact_vs_independent_reference_relative_error=error,
        corrected_compact_vs_independent_reference_pass=bool(error is not None and np.isfinite(error) and 0<=error<1e-4),
        original_native_paths_unchanged=unchanged,
        native_cir_b00_factorization_relative_error=source_error,
        native_cir_b00_factorization_pass=bool(source_error<=2e-6),
        native_carrier_frequency_hz=correction['native_carrier_frequency_hz'],
        native_source_integrity_sha256=correction['native_source_integrity_sha256'],
        independent_reference='selected masked native Paths.a full Rx/Tx/path coefficients multiplied by Float64 exp[-j2pi(fc+f)image_delay] and summed; no compact reconstruction',
        coordinate_scope='actual native Float32 source/target endpoints stored as Float64; exact physical-patch image geometry evaluated in Float64',
        native_jones_transport_and_synthetic_array_retained=True,native_directions_and_doppler_retained=True,
        original_native_cir_cfr_and_delay_retained_in_raw_archive=True,
        continuous_manifest_endpoint_correction=False,exact_electromagnetic_field_error_bound=False,
        phase_correction_delta_tau_sec=(ideal-old['delay_sec']).copy())
    return dict(base=base,compact=corrected,native=reference,native_relative_error=error,
        phase_correction=correction,phase_reference_report=phase_report,
        original_selected_native_cfr=base['native'],original_selected_native_compact=old)

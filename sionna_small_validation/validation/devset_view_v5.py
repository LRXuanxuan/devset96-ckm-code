"""Public fixed-frame views of native transport / image-delay physical paths.

The raw primitive and physical-surface proof ledgers stay private. The public
observation contract is the existing complete CP-OFDM receiver interface.
"""
from __future__ import annotations

import json
import h5py
import numpy as np

from .devset_view_v2 import DevsetViewV2
from .devset_view_v4 import DevsetViewV4,open_devset_view as _open_legacy


class DevsetViewV5(DevsetViewV4):
    def __init__(self,route_file):
        # Run the fixed historical coordinate-frame checks, then the new
        # version's evidence contract. V3's version check remains unchanged.
        DevsetViewV2.__init__(self,route_file)
        config=self.config;normalization=config.get('specular_normalization',{})
        reference=str(config.get('native_reference_contract',''))
        if (config.get('runtime_adapter_version')!='kap_devset.runtime.v5'
                or config.get('physical_surface_schema')!='kap_devset.exact_coplanar_surface_patch.v1'
                or normalization.get('version')!='kap_devset.unique_specular_physical_patch.v4'
                or normalization.get('physical_surface_schema')!='kap_devset.exact_coplanar_surface_patch.v1'
                or normalization.get('unresolved_blocks_admission') is not True
                or normalization.get('raw_candidate_archive') is not True
                or normalization.get('exact_electromagnetic_field_error_bound') is not False
                or 'Paths.a' not in reference or 'Float64' not in reference
                or config.get('phase_reference_version')!='kap_devset.image_delay_phase_reference.v5'
                or config.get('phase_reference_contract',{}).get('exact_em_error_bound') is not False
                or not all(config.get('phase_reference_contract',{}).get(k) is True for k in
                    ('native_transport_unchanged','original_native_cir_archived','float64_image_delay','actual_quantized_endpoints'))):
            raise ValueError('V5 native-transport / Float64 image-delay reference is not explicitly declared')
        with h5py.File(self.path,'r') as f:
            required=('raw_candidates/index','raw_candidates/geometry','raw_candidates/report_bytes',
                      'trace_accounting','raw_candidates/physical_mapping',
                      'physical_patches/mappings','physical_patches/identities',
                      'physical_patches/primitives','physical_patches/material_utf8',
                      'physical_patches/containing_faces','phase_reference/index','phase_reference/paths')
            if (not all(name in f for name in required)
                    or f.attrs.get('raw_candidate_schema')!='kap_devset.raw_native_candidates.v4'
                    or int(f.attrs.get('raw_candidate_quality_bit',0))!=(1<<30)
                    or f.attrs.get('phase_reference_schema')!='kap_devset.image_delay_phase_reference.v5'
                    or f['phase_reference/paths'].dtype.itemsize!=24):
                raise ValueError('Missing V5 phase reference or V4 raw/physical mapping and accounting separation')
            mapping=f['raw_candidates/physical_mapping']
            if (mapping.ndim!=2 or mapping.shape!=(len(f['raw_candidates/geometry']),3)
                    or mapping.dtype.str not in ('<u8','=u8')):
                raise ValueError('Raw physical mapping is not a three-slot uint64 ledger')

    def _map(self,f,query,mapn,design):
        rows=f['survey_index'][:]
        refs=rows['record_index'][rows['design_index']==design]
        states=f['states'][:]
        if (np.any(refs<0) or np.any(refs>=len(states))
                or np.any(states['role'][refs]!=b'historical_survey')
                or np.any(states['survey_design'][refs]!=design)):
            raise ValueError('Historical map references do not identify physical captures of the selected survey design')
        return super()._map(f,query,mapn,design)

    def load_evaluation(self,task_index,frequencies_hz=None):
        result=DevsetViewV2.load_evaluation(self,task_index,frequencies_hz)
        result['validation_source']='direct RT native transport fields at the target state; one contribution per certified physical-patch topology; Float64 image-delay carrier and frequency phase'
        result['numerical_scope']='finite physical-patch/image/visibility checks; native transport coefficients retain geometry approximation; original native cir and image-delay phase reference are distinct; no exact electromagnetic field error bound'
        return result


def open_devset_view(route_file,*,force_v2=False):
    """Dispatch explicit V5 contracts, retaining old engineering diagnostics."""
    with h5py.File(route_file,'r') as f:
        config=json.loads(f['config_json'][()].decode())
    if config.get('runtime_adapter_version')=='kap_devset.runtime.v5':return DevsetViewV5(route_file)
    return _open_legacy(route_file,force_v2=force_v2)

"""Public views of selected, image-certified native contributions only.

Raw candidate evidence is private. This reader preserves the validated full-FFT
receiver interface and requires explicit V3 reference/archival declarations;
the independent release audit checks the full geometric evidence and lineage.
"""
from __future__ import annotations

import json

import h5py

from .devset_view_v2 import DevsetViewV2,open_devset_view as _open_legacy


class DevsetViewV3(DevsetViewV2):
    def __init__(self,route_file):
        super().__init__(route_file)
        config=self.config
        if config.get('runtime_adapter_version')!='kap_devset.runtime.v3':
            raise ValueError('V3 reader requires the selected-native runtime contract')
        normalization=config.get('specular_normalization',{})
        reference=str(config.get('native_reference_contract',''))
        if (normalization.get('version')!='kap_devset.unique_specular_image.v1'
                or normalization.get('unresolved_blocks_admission') is not True
                or normalization.get('raw_candidate_archive') is not True
                or normalization.get('exact_electromagnetic_field_error_bound') is not False
                or 'masked' not in reference or 'Paths.cfr' not in reference):
            raise ValueError('Selected native reference and finite numerical scope are not declared')
        with h5py.File(self.path,'r') as f:
            if (f.attrs.get('raw_candidate_schema')!='kap_devset.raw_native_candidates.v3'
                    or int(f.attrs.get('raw_candidate_quality_bit',0))!=(1<<30)
                    or not all(name in f for name in ('raw_candidates/index','raw_candidates/geometry',
                                                     'raw_candidates/report_bytes','trace_accounting'))):
                raise ValueError('Missing V3 raw-candidate separation and trace-accounting contract')

    def _snapshot(self,f,state_index,row=None):
        # Semantic separation is checked independently of the quality-bit
        # marker, so stripping that marker cannot turn raw evidence into truth.
        if row is None:row=f['states'][int(state_index)]
        if (row['role'] in (b'raw_native_candidate',b'quarantine_attempt')
                or int(row['quality_flags'])&(1<<30)):
            raise ValueError('Raw/quarantined native candidates cannot supply an observation or target')
        return super()._snapshot(f,state_index,row)

    def load_evaluation(self,task_index,frequencies_hz=None):
        result=super().load_evaluation(task_index,frequencies_hz)
        result['validation_source']='direct RT state; selected unique-topology native contributions, independently masked Paths.cfr reference'
        result['numerical_scope']='finite geometric/native reconstruction checks; no exact electromagnetic field error bound'
        return result


def open_devset_view(route_file,*,force_v2=False):
    """Choose V3 from its explicit version; legacy readers stay diagnostic."""
    with h5py.File(route_file,'r') as f:
        config=json.loads(f['config_json'][()].decode())
    if config.get('runtime_adapter_version')=='kap_devset.runtime.v3':return DevsetViewV3(route_file)
    return _open_legacy(route_file,force_v2=force_v2)

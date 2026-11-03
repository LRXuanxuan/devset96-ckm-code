"""Explicit historical angular-map frame on top of the frozen v1 receiver.

V2 changes map coordinate metadata and rejects undeclared mixtures of array
poses. The receive waveform and independent target contracts remain inherited
from the validated core reader. Historical orientation is known metadata of the
simulated acquisition; no future online pose or channel is used to obtain it.
"""
from __future__ import annotations

import json

import h5py
import numpy as np

from .devset_view import DevsetView


class DevsetViewV2(DevsetView):
    def __init__(self, route_file):
        super().__init__(route_file)
        declared=self.config.get('survey_reference_pose_rad')
        fallback=[float(self.route['heading_rad']),0.,0.]
        self.survey_reference_pose_rad=np.asarray(fallback if declared is None else declared,float)
        if self.survey_reference_pose_rad.shape!=(3,) or not np.isfinite(self.survey_reference_pose_rad).all():
            raise ValueError('Historical map requires a finite three-angle reference pose')
        with h5py.File(self.path,'r') as f:
            states=f['states'][:]
            selected=(states['pass_index']<0)&(states['survey_design']>=0)&(states['role']!=b'quarantine_attempt')
            delta=np.angle(np.exp(1j*(states['rx_pose_rad'][selected]-self.survey_reference_pose_rad)))
            if not delta.size or np.max(abs(delta))>1e-9:
                raise ValueError('Historical Rx-local angular powers mix array poses; a fixed-reference map cannot be declared')
        self.coordinate_scope=str(self.config.get('survey_coordinate_scope',
            'Rx4x4 local angular DFT and cyclic-delay power at the declared fixed historical reference pose; online array uses actual pose'))
        self.reference_pose_provenance=('explicit known fixed pose in the simulated historical acquisition' if declared is not None else
            'legacy straight/fixed-UE captures: all stored historical poses independently checked against route heading')

    def _map(self,f,query,mapn,design):
        prior,calibration=super()._map(f,query,mapn,design)
        prior.update(survey_reference_pose_rad=self.survey_reference_pose_rad.copy(),
                     coordinate_scope=self.coordinate_scope,
                     reference_pose_provenance=self.reference_pose_provenance)
        return prior,calibration


def open_devset_view(route_file, *, force_v2=False):
    """Auto-select the published runtime contract, with an explicit QA override."""
    with h5py.File(route_file,'r') as f:
        config=json.loads(f['config_json'][()].decode())
    version=str(config.get('runtime_adapter_version',''))
    implementation=DevsetViewV2 if force_v2 or config.get('runtime_v2') or version=='kap_devset.runtime.v2' else DevsetView
    return implementation(route_file)

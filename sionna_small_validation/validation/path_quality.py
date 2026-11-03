"""Private truth quality and safe local-propagation admission, never filtering paths."""
from __future__ import annotations
from enum import IntFlag
import numpy as np


class QualityFlag(IntFlag):
    EMPTY_UNRESOLVED=1
    NONFINITE=2
    DUPLICATE_GEOMETRY=4
    RECONSTRUCTION_FAILED=8
    INVALID_PATH_GEOMETRY=16
    SEARCH_CAPACITY_REACHED=32


class UncertifiedChannelError(ValueError):
    pass


def assess_snapshot(compact,native,raw,relative_error,max_paths=4096):
    flag=QualityFlag(0)
    if not len(compact) or np.linalg.norm(native)==0:flag|=QualityFlag.EMPTY_UNRESOLVED
    if not np.isfinite(native).all() or any(not np.isfinite(compact[k]).all() for k in
            ('b00','delay_sec','doppler_hz','k_tx','k_rx')):flag|=QualityFlag.NONFINITE
    if raw.get('exact_duplicate_geometry_pairs'):flag|=QualityFlag.DUPLICATE_GEOMETRY
    if relative_error is None or not np.isfinite(relative_error) or relative_error>=1e-4:
        flag|=QualityFlag.RECONSTRUCTION_FAILED
    if np.any(compact['delay_sec']<0) or any(np.any(abs(np.linalg.norm(compact[k],axis=-1)-1)>1e-4) for k in ('k_tx','k_rx')):
        flag|=QualityFlag.INVALID_PATH_GEOMETRY
    if len(compact)>=max_paths:flag|=QualityFlag.SEARCH_CAPACITY_REACHED
    return dict(flags=int(flag),admitted=not bool(flag),
                reasons=[q.name for q in QualityFlag if flag&q],
                exact_duplicate_geometry_pairs=raw.get('exact_duplicate_geometry_pairs',[]),
                physical_zero_certified=False)


def check_selected_snapshots(snapshots,times,selected):
    """Reject unqualified state/interval before any receive waveform is produced."""
    for index in np.unique(selected):
        snapshot=snapshots[int(index)]
        if int(snapshot.get('quality_flags',0)):
            raise UncertifiedChannelError('Receiver uses an unadmitted RT snapshot: '+str(snapshot.get('record_index',index)))
        if len(snapshot['compact'])==0:
            raise UncertifiedChannelError('Empty RT candidate is not a certified physical zero channel')
        if 'validated_until_sec' in snapshot:
            bound=float(snapshot['validated_until_sec'])
            if not np.isfinite(bound) or bound<float(snapshot['time_sec'])-1e-13:
                raise UncertifiedChannelError('Invalid local-propagation admission bound')
            used=np.asarray(times)[selected==index]
            if used.max()>bound+1e-13:
                raise UncertifiedChannelError('Receiver crosses an uncertified local-propagation interval')


def require_target_admitted(snapshot):
    if int(snapshot.get('quality_flags',0)) or not len(snapshot['compact']):
        raise UncertifiedChannelError('Target CSI is not an admitted nonempty RT state')

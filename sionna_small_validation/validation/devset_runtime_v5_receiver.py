"""External V5 extension: independent native reference on the receiver FFT grid.

The frozen configure/16-tone truth and source manifest remain byte-compatible.
Only the lifetime of the already selected masked Paths.a tensors is extended to
form a second reference; no additional ray solve or new propagation model.
"""
from __future__ import annotations
import ast
import copy
import hashlib
from pathlib import Path
import numpy as np
from . import devset_runtime_v5 as _base
from .devset_phase_reference_v5 import reference_from_full_passband
from .cities import reconstruct

VERSION='kap_devset.external_receiver_grid_native.v1'
BASELINE_SHA256='096908615db86611c0eec6ffa6cf20ac984b85151851dc19bfe84639b9dd9159'
FREQUENCIES_HZ=np.fft.fftfreq(256,1/30.72e6)
FREQUENCIES_SHA256=hashlib.sha256(FREQUENCIES_HZ.astype('<f8').tobytes()).hexdigest()


def receiver_grid_from_phase(phase,time_sec):
    if float(time_sec)<0:
        return None  # Historical static map retains exactly its frozen16 capture.
    base=phase['base'];source=np.asarray(phase['phase_correction']['selected_native_path_indices'],int)
    delay=np.asarray(phase['phase_correction']['ideal_delay_sec'],np.float64)
    a=base['masked_paths'].a
    full=np.asarray(a[0],np.float64)+1j*np.asarray(a[1],np.float64)
    if full.ndim!=5 or full.shape[0]!=1 or full.shape[2]!=1:
        raise RuntimeError('Pinned native passband one-Tx/Rx array axes changed')
    selected=full[0,:,0,:,:][:,:,source]
    native=reference_from_full_passband(selected,delay,FREQUENCIES_HZ)
    rebuilt=reconstruct(phase['compact'],base['tx_relative_m'],base['rx_relative_m'],FREQUENCIES_HZ)
    norm=float(np.linalg.norm(native))
    error=float(np.linalg.norm(rebuilt-native)/norm) if np.isfinite(norm) and norm>0 else None
    return dict(schema=VERSION,native=native,frequencies_sha256=FREQUENCIES_SHA256,
        compact_vs_native_relative_error=error,
        native_sha256=hashlib.sha256(np.asarray(native,dtype='<c16').tobytes()).hexdigest(),
        reference_kind='independent selected masked native Paths.a full Rx/Tx/path coefficients + Float64 image-delay phase on all256 actual receiver FFT bins',
        extra_solver_calls=0)


def _derive_trace():
    path=Path(_base.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=BASELINE_SHA256:
        raise RuntimeError('Frozen V5 runtime SHA changed')
    root=ast.parse(path.read_text('utf-8'))
    cls=next(x for x in root.body if isinstance(x,ast.ClassDef) and x.name=='DevsetRTV5')
    fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='trace'))
    assignments=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='phase' for t in n.targets)]
    if len(assignments)!=1:raise RuntimeError('Native phase lifetime source anchor changed')
    # Insert exactly one extra computation after phase is formed, inside the
    # actual attempt loop; all parent physics and quality16 code is untouched.
    anchor=assignments[0];found=0
    for node in ast.walk(fn):
        body=getattr(node,'body',None)
        if isinstance(body,list) and anchor in body:
            body.insert(body.index(anchor)+1,ast.parse("phase['receiver_grid']=receiver_grid_from_phase(phase,t)").body[0]);found+=1
    returns=[n for n in ast.walk(fn) if isinstance(n,ast.Return)]
    if found!=1 or len(returns)!=1 or not isinstance(returns[0].value,ast.Call):raise RuntimeError('Native trace return source anchor changed')
    returns[0].value.keywords.append(ast.keyword(arg='receiver_grid',value=ast.parse("phase['receiver_grid']",mode='eval').body))
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace=dict(vars(_base));namespace['receiver_grid_from_phase']=receiver_grid_from_phase
    exec(compile(module,str(Path(__file__).resolve())+'::<derived frozen trace>','exec'),namespace)
    return namespace['trace']


class DevsetRTV5ReceiverGrid(_base.DevsetRTV5):
    receiver_grid_adapter_version=VERSION
    trace=_derive_trace()


def source_provenance():
    path=Path(__file__)
    return dict(schema=VERSION,relative_path='validation/'+path.name,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frozen_runtime_sha256=BASELINE_SHA256,frequencies_sha256=FREQUENCIES_SHA256,
        nfft=256,fs_hz=30.72e6,configure_inherited_unchanged=True,extra_solver_calls=0)

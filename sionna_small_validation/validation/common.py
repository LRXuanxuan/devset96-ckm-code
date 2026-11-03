from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

import numpy as np

_DLL_HANDLES = []
_ASSET_CACHE = {}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=jsonable, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def setup_backend(requested='auto'):
    if os.environ.get('SIONNA_WINDOWS_JIT_COMPAT')=='1':
        from .windows_llvm import configure_gnu_llvm
        configure_gnu_llvm()
    # The runtime is project-local; no changes to system environment variables.
    runtime_root=Path(__file__).resolve().parents[1]/'.runtime'
    candidates=[runtime_root/'llvm18'/'bin'/'LLVM-C.dll',runtime_root/'bin'/'LLVM-C.dll']
    local_dll=next((p for p in candidates if p.exists()),candidates[0])
    if os.name == 'nt' and local_dll.exists() and not os.environ.get('DRJIT_LIBLLVM_PATH'):
        os.environ['DRJIT_LIBLLVM_PATH'] = str(local_dll)
        _DLL_HANDLES.append(os.add_dll_directory(str(local_dll.parent)))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if os.name=='nt':
        cache=Path(os.environ.get('LOCALAPPDATA',str(Path.home()/'.cache'))) / 'sionna-small-validation' / 'drjit-cache-llvm18-evaluated'
        cache.mkdir(parents=True,exist_ok=True)
        os.environ.setdefault('DRJIT_CACHE_DIR',str(cache))
    import mitsuba as mi
    import drjit as dr
    if os.name=='nt' and 'SIONNA_VALIDATION_CPU_THREADS' in os.environ:
        # Conservative local validation fallback; no server throughput claim.
        dr.set_thread_count(int(os.environ['SIONNA_VALIDATION_CPU_THREADS']))
    if requested == 'auto':
        requested = ('cuda_ad_mono_polarized' if dr.has_backend(dr.JitBackend.CUDA)
                     else 'llvm_ad_mono_polarized')
    mi.set_variant(requested)
    import sionna.rt as rt
    if rt.__version__ != '2.2.0':
        raise RuntimeError(f'This validation targets Sionna RT 2.2.0, got {rt.__version__}')
    return mi.variant()


def portable_asset(path):
    """Work around Mitsuba's Windows mesh loader rejecting non-ASCII paths.

    Copy the small installed asset folder, preserving relative XML/PLY paths.
    Original package files are never modified. Linux uses originals directly.
    """
    path = Path(path).resolve()
    if os.name != 'nt' or str(path).isascii():
        return str(path)
    source=path.parent
    if source in _ASSET_CACHE:
        return str(_ASSET_CACHE[source]/path.name)
    files=sorted(p for p in source.rglob('*') if p.is_file())
    file_hashes={p.relative_to(source).as_posix():sha256(p) for p in files}
    digest=hashlib.sha256(json.dumps(file_hashes,sort_keys=True).encode()).hexdigest()
    base = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.cache')))/'sionna-small-validation'/'assets'
    base.mkdir(parents=True,exist_ok=True)
    target = base / (source.name+'-'+digest[:16])
    if not str(target).isascii():
        raise RuntimeError('Set an ASCII LOCALAPPDATA/cache directory for Windows RT assets')
    ready=target/'.asset_ready.json'
    if ready.exists():
        if json.loads(ready.read_text(encoding='utf-8')).get('content_digest')!=digest:
            raise RuntimeError('Asset cache manifest mismatch')
    else:
        stage=base/('.stage-'+uuid.uuid4().hex)
        shutil.copytree(source,stage)
        copied={p.relative_to(stage).as_posix():sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        if copied!=file_hashes:
            raise RuntimeError('Asset changed during copy; incomplete stage retained for diagnosis')
        save_json(stage/'.asset_ready.json',{'content_digest':digest,'files':file_hashes,'source':str(source)})
        try:
            stage.rename(target)
        except OSError:
            # A concurrent process may have committed this same immutable
            # content. Never overwrite files another loader is reading.
            if not ready.exists() or json.loads(ready.read_text(encoding='utf-8')).get('content_digest')!=digest:
                raise
    _ASSET_CACHE[source]=target
    return str(target / path.name)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def environment(backend):
    import drjit as dr
    import sionna.rt as rt
    versions = {}
    for package in ('sionna-rt','mitsuba','drjit','numpy','scipy','h5py','matplotlib','psutil'):
        versions[package] = importlib.metadata.version(package)
    import psutil
    from .windows_llvm import state as windows_llvm_state
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'machine': platform.machine(),
            'versions': versions, 'backend': backend,
            'windows_llvm_compat':windows_llvm_state(),
            'rt_source_sha256':{str(p.relative_to(Path(rt.__file__).parent)):sha256(p)
                                for p in [Path(rt.__file__).parent/'path_solvers'/'paths.py',
                                          Path(rt.__file__).parent/'path_solvers'/'path_solver.py']},
            'cuda_backend_available': bool(dr.has_backend(dr.JitBackend.CUDA)),
            'llvm_backend_available': bool(dr.has_backend(dr.JitBackend.LLVM)),
            'logical_cpu_count': psutil.cpu_count(),
            'ram_bytes': psutil.virtual_memory().total,
            'drjit_libllvm_path': os.environ.get('DRJIT_LIBLLVM_PATH'),
            'drjit_cache_dir':os.environ.get('DRJIT_CACHE_DIR'),
            'drjit_thread_count':int(dr.thread_count()),
            'note': 'Local runtime snapshot, not the remote lrx server.'}


def rel_error(estimate, reference):
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    denominator = max(float(np.linalg.norm(reference.ravel())), np.finfo(float).tiny)
    return float(np.linalg.norm((estimate-reference).ravel()) / denominator)


def nmse_db(estimate, reference):
    return float(20*np.log10(max(rel_error(estimate, reference), 1e-150)))

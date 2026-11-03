"""Explicit Windows-only GNU LLVM compatibility profile.

Diagnostic copies are never treated as upstream releases. The normal Linux/
CUDA path does not import this module. No installed wheel is modified.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

_HANDLES=[]
_STATE=None


def configure_gnu_llvm():
    global _STATE
    if _STATE is not None:return _STATE
    if os.name!='nt':raise RuntimeError('GNU Windows LLVM profile is only for Windows')
    if 'drjit' in sys.modules or 'mitsuba' in sys.modules:
        raise RuntimeError('Select Windows LLVM profile before importing Dr.Jit/Mitsuba')
    root=Path(__file__).resolve().parents[1]
    # Explicit override permits an unmodified GNU-host LLVM distribution.
    override=os.environ.get('SIONNA_GNU_LLVM_PATH')
    dll=Path(override).resolve() if override else root/'.runtime/llvm18_gnu/bin/LLVM-C.dll'
    if not dll.exists():
        raise RuntimeError('GNU LLVM runtime missing; see the local debug report/bootstrap instructions')
    dll_hash=hashlib.sha256(dll.read_bytes()).hexdigest()
    cache=Path(os.environ.get('LOCALAPPDATA',str(Path.home()/'.cache')))/'sionna-small-validation'/('drjit-gnu-'+dll_hash[:16])
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('DRJIT_CACHE_DIR',str(cache))
    os.environ.setdefault('SIONNA_VALIDATION_CPU_THREADS','1')
    patched_provenance=None
    if not override:
        manifest=dll.parent.parent/'provenance.json'
        patched_provenance=json.loads(manifest.read_text(encoding='utf-8'))
        if dll_hash!=patched_provenance['patched_sha256']:
            raise RuntimeError('Diagnostic LLVM copy does not match its provenance hash')
    for candidate in (dll.parent,dll.parent.parent/'bin',dll.parent.parent/'lib'):
        if candidate.is_dir():_HANDLES.append(os.add_dll_directory(str(candidate)))
    os.environ['DRJIT_LIBLLVM_PATH']=str(dll)
    llvm=ctypes.CDLL(str(dll));_HANDLES.append(llvm)
    llvm.LLVMGetDefaultTargetTriple.argtypes=[]
    llvm.LLVMGetDefaultTargetTriple.restype=ctypes.c_void_p
    llvm.LLVMDisposeMessage.argtypes=[ctypes.c_void_p]
    llvm.LLVMDisposeMessage.restype=None
    pointer=llvm.LLVMGetDefaultTargetTriple()
    triple=ctypes.string_at(pointer).decode();llvm.LLVMDisposeMessage(pointer)
    if not any(x in triple.lower() for x in ('gnu','mingw')):
        raise RuntimeError(f'GNU host triple required, got {triple}')
    ntdll=ctypes.WinDLL('ntdll');_HANDLES.append(ntdll)
    probe=ctypes.cast(getattr(ntdll,'__chkstk'),ctypes.c_void_p).value
    llvm.LLVMAddSymbol.argtypes=[ctypes.c_char_p,ctypes.c_void_p]
    llvm.LLVMAddSymbol.restype=None
    for name in (b'___chkstk_ms',b'__chkstk'):
        llvm.LLVMAddSymbol(name,probe)
    _STATE={'enabled':True,'llvm_path':str(dll),'triple':triple,
            'llvm_sha256':dll_hash,'jit_cache_directory':os.environ['DRJIT_CACHE_DIR'],
            'stack_probe_aliases':['___chkstk_ms','__chkstk'],
            'distribution':'unsupported diagnostic copy' if patched_provenance else 'explicit external distribution; provenance in debug report',
            'original_installed_wheel_modified':False,'large_core_patch_used':False}
    return _STATE


def state():return _STATE or {'enabled':False}

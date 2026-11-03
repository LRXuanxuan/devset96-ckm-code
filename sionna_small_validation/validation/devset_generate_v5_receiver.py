"""SHA/AST-locked external V5 receiver256 admission and recovery adapter.

The original38 config and native16 truth remain unchanged. Existing accepted
intervals receive explicit legacy compact256 proxy evidence. New/refined checks
require independent native256 and compact256 admission, with the same -30 dB
whole-array Frobenius threshold. No completed shard is edited by this entry.
"""
from __future__ import annotations
import ast
from collections import OrderedDict
import hashlib
import inspect
from pathlib import Path
import textwrap
from . import devset_generate_v5_resume as _parent
from .devset_receiver_grid import (SCHEMA,FREQUENCIES_SHA256,check_fresh_interval,
    retrofit_committed_intervals,recertify_tasks)

VERSION='kap_devset.external_receiver_grid_generate.v1'
PARENT_SHA256='a26e82623b666e79ec8c6cea40003c7175ec3066e9fa7bd79ad7a6f6ce6b0232'
WORKER_BOOTSTRAP_PROVENANCE={}


def _writer_factory(*args,**kwargs):
    from .devset_storage_v5_receiver import RouteWriterV5Receiver
    return RouteWriterV5Receiver(*args,**kwargs)


def source_provenance():
    names=('devset_generate_v5_receiver.py','devset_receiver_grid.py',
           'devset_runtime_v5_receiver.py','devset_storage_v5_receiver.py')
    return dict(schema=VERSION,derivation_parent_relative_path='validation/devset_generate_v5_resume.py',
        derivation_parent_sha256=PARENT_SHA256,
        source_sha256={'validation/'+name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names},
        receiver_grid_schema=SCHEMA,frequencies_sha256=FREQUENCIES_SHA256,
        legacy_reference='native16-checked stored compact256 proxy; not independently native256',
        fresh_reference='masked native Paths.a +Float64 image delay on all256 receiver bins; unchanged native16 truth',
        old16_metrics_preserved=True,threshold_db=-30.,config_unchanged=True,
        bookkeeping_only_source_extension=False,propagation_model_unchanged=True,
        receiver_admission_extended=True)


_REMEMBER='''    def remember_receiver(index, result, source_revisit=False):
        if float(result['state_fields']['time_sec']) < 0:
            return
        grid = result.get('receiver_grid')
        if not isinstance(grid, dict):
            raise ValueError('Receiver-grid adapter requires independent native256 from its runtime extension')
        grid = dict(grid)
        grid['native_reference_trace_row'] = len(writer.handle['trace_accounting']) + len(writer._trace_rows) - 1
        grid['source_revisit'] = bool(source_revisit)
        source = result['phase_correction'].get('native_source_integrity_sha256')
        writer.append_event('receiver_grid_native_reference', dict(
            record_index=int(index), time_sec=float(result['state_fields']['time_sec']),
            pass_index=int(result['state_fields']['pass_index']),
            native_sha256=grid.get('native_sha256'), frequencies_sha256=grid.get('frequencies_sha256'),
            compact_vs_native256_relative_error=grid.get('compact_vs_native_relative_error'),
            trace_accounting_row_index=grid['native_reference_trace_row'], source_revisit=bool(source_revisit),
            native_source_integrity_sha256=source, phase_reference_report=result.get('phase_reference_report'),
            reference_kind=grid.get('reference_kind'), extra_solver_calls=0))
        receiver_references[int(index)] = grid
        receiver_references.move_to_end(int(index))
        while len(receiver_references) > 24:
            receiver_references.popitem(last=False)

'''

# Explicit source anchors make the permitted control-flow differences auditable.
# Frozen helpers and source_manifest retain their own original module globals;
# the derived function receives an isolated namespace, never patched globals.
_PATCHES=(
    ("adapter_relative_path='validation/devset_generate_v5_resume.py'", "adapter_relative_path='validation/devset_generate_v5_receiver.py'"),
    ("scope='External control-flow recovery only; original38 source manifest/config and propagation model unchanged; failed designed captures remain failed and cold bad references are denial-only proxies'",
     "scope='External recovery plus stricter receiver256 admission; original38 config/native16 truth and propagation model unchanged; failed captures remain failed and cold bad references denial-only'"),
    ('    native_cache = OrderedDict()','    native_cache = OrderedDict()\n    receiver_references = OrderedDict()'),
    ('    def account_native_call(index, result, revisit=False):',_REMEMBER+'    def account_native_call(index, result, revisit=False):'),
    ("                remember_native(rec['index'], h)","                remember_native(rec['index'], h)\n                remember_receiver(rec['index'], result, source_revisit=True)"),
    ('        cache[key] = rec; remember_native(index, h)','        cache[key] = rec; remember_native(index, h)\n        remember_receiver(index, result)'),
    ('    try:\n        context.set_environment(0)',
     "    try:\n        writer.append_event('receiver_grid_adapter_provenance', _receiver_source_provenance())\n        retrofit_committed_intervals(writer, cache, config, completed_intervals, intervals_by_pass, max_refinement)\n        context.set_environment(0)"),
    ("                passed = bool(good and np.isfinite(me) and np.isfinite(re) and max(me, re) <= -30.)",
     "                band = check_fresh_interval(a, b, c, config, receiver_references)\n                passed = bool(good and np.isfinite(me) and np.isfinite(re) and max(me, re) <= -30. and band['passed'])"),
    ("                writer.append_interval(row); intervals_by_pass[pass_index].append(row); completed_intervals[ident] = row",
     "                interval_index = len(writer.handle['intervals']) + len(writer._intervals)\n                writer.append_event('receiver_grid_interval', dict(interval_index=interval_index,\n                    left_index=a['index'], midpoint_index=b['index'], right_index=c['index'],\n                    pass_index=pass_index, left_time_sec=float(left), right_time_sec=float(right), level=level,\n                    stored16_midpoint_nmse_db=me, stored16_right_nmse_db=re, passed=passed, receiver256=band))\n                writer.append_interval(row); intervals_by_pass[pass_index].append(row); completed_intervals[ident] = row"),
    ('            coverage = _coverage(intervals_by_pass[pass_index])',
     '            coverage = _coverage(intervals_by_pass[pass_index])\n            recertify_tasks(writer, cache, intervals_by_pass, pass_index)'),
    ("        save_json(path.with_suffix('.report.json'), result)",
     "        result['external_receiver_grid_admission'] = _receiver_source_provenance()\n        save_json(path.with_suffix('.report.json'), result)"),
)


def _derive_generate():
    if hashlib.sha256(Path(_parent.__file__).read_bytes()).hexdigest()!=PARENT_SHA256:
        raise RuntimeError('Frozen a26 resume source changed')
    _parent.derivation_audit()
    source=textwrap.dedent(inspect.getsource(_parent.generate_route)).replace('\r\n','\n')
    for old,new in _PATCHES:
        if source.count(old)!=1:raise RuntimeError('Receiver adapter source anchor changed: '+old[:100])
        source=source.replace(old,new,1)
    tree=ast.parse(source)
    # Record generated AST identity; baseline/parent module bytes are pinned.
    ast_sha=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    namespace=dict(vars(_parent));namespace.update(__file__=__file__,RouteWriter=_writer_factory,
        check_fresh_interval=check_fresh_interval,retrofit_committed_intervals=retrofit_committed_intervals,
        recertify_tasks=recertify_tasks,_receiver_source_provenance=source_provenance,
        WORKER_BOOTSTRAP_PROVENANCE=WORKER_BOOTSTRAP_PROVENANCE)
    exec(compile(tree,str(Path(__file__).resolve())+'::<approved a26 derivative>','exec'),namespace)
    return namespace['generate_route'],ast_sha


_GENERATE_ROUTE,DERIVED_AST_SHA256=_derive_generate()


def generate_route(context,route,path,**kwargs):
    if getattr(context,'receiver_grid_adapter_version',None)!='kap_devset.external_receiver_grid_native.v1':
        raise ValueError('Use DevsetRTV5ReceiverGrid with the receiver-grid generator')
    # Bootstrap assigns the dictionary after import. Refresh only this isolated
    # derived namespace; no original module or physics global is mutated.
    _GENERATE_ROUTE.__globals__['WORKER_BOOTSTRAP_PROVENANCE']=dict(WORKER_BOOTSTRAP_PROVENANCE)
    return _GENERATE_ROUTE(context,route,path,**kwargs)


def derivation_audit():
    _,current=_derive_generate()
    if current!=DERIVED_AST_SHA256:raise RuntimeError('Receiver derivative AST changed')
    return dict(schema=VERSION,parent_source_sha256=PARENT_SHA256,
        original_generator_sha256=_parent._BASELINE_SHA256,approved_source_edits=len(_PATCHES),
        derived_ast_sha256=current,original_source_manifest_unchanged=True)

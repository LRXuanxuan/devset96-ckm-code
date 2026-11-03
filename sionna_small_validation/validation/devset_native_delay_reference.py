"""External correction of the native-delay compatibility reference.

Sionna RT 2.2 computes native tau from its retained native vertices. The V4
predicate compared this value to the analytic image length, although it also
allowed finite native off-plane displacement separately. This module keeps the
original 8-ULP tolerance and every physical geometry predicate, but compares
native tau to the native vertex polyline that its producer actually evaluated.

It does not change native coefficients, the Float64 image-delay phase model,
the old image-delay diagnostic or the numerical representative's selection
score. Native geometric validity and native arithmetic consistency remain
separate checks. No original source is edited.
"""
from __future__ import annotations
import ast
import copy
import hashlib
from pathlib import Path

from . import devset_specular_audit_v4 as _base

VERSION = 'kap_devset.native_polyline_delay_compatibility.v1'
BASE_SHA256 = '5192397743a104728a1e8b1d34cab151b79960f4e77b0885c767056bcebf9e7d'
REFERENCE_KIND = 'retained_native_vertex_polyline_length_over_c'


def source_provenance():
    path = Path(__file__)
    return dict(schema=VERSION, relative_path='validation/' + path.name,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frozen_geometry_auditor_sha256=BASE_SHA256,
        base_geometry_schema=_base.VERSION,
        native_delay_reference=REFERENCE_KIND,
        original8ulp_tolerance_expression_unchanged=True,
        ideal_image_diagnostic_and_selection_score_unchanged=True,
        physical_image_visibility_plane_optical_same_side_and_length_unchanged=True,
        native_jones_coefficients_and_image_phase_model_unchanged=True,
        scope='Compatibility reference correction, not an exact EM gain or global search-convergence guarantee')


def _target(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def _derive():
    source = Path(_base.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != BASE_SHA256:
        raise RuntimeError('Frozen V4 physical-patch auditor changed')
    root = ast.parse(source.read_text('utf-8'))
    fn = copy.deepcopy(next(n for n in root.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_patch_candidates'))
    counters = dict(native_polyline_reference=0, original_agreement_preserved=0,
                    one_delay_operand_replaced=0, candidate_metadata=0,
                    report_metadata=0)
    for parent in list(ast.walk(fn)):
        body = getattr(parent, 'body', None)
        if not isinstance(body, list):
            continue
        for node in list(body):
            if _target(node, 'native_polyline_length'):
                new = ast.parse("native_polyline_delay_error = float(compact['delay_sec'][i] - native_polyline_length/C)").body[0]
                body.insert(body.index(node) + 1, new)
                counters['native_polyline_reference'] += 1
            elif _target(node, 'agreement'):
                original = copy.deepcopy(node)
                original.targets = [ast.Name(id='legacy_policy_result', ctx=ast.Store())]
                body.insert(body.index(node), original)
                counters['original_agreement_preserved'] += 1
                # Replace only the abs(delay_error) operand inside the original
                # conjunction. In particular, the threshold expression,
                # optical defect and every preceding finite/geometry term stay.
                for value in ast.walk(node.value):
                    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                            and value.func.id == 'abs' and len(value.args) == 1
                            and isinstance(value.args[0], ast.Name)
                            and value.args[0].id == 'delay_error'):
                        value.args[0].id = 'native_polyline_delay_error'
                        counters['one_delay_operand_replaced'] += 1
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'candidates' and node.func.attr == 'append'):
            payload = node.args[0]
            if not isinstance(payload, ast.Call) or not isinstance(payload.func, ast.Name) or payload.func.id != 'dict':
                raise RuntimeError('Frozen candidate diagnostic anchor changed')
            extra = {
                'native_delay_reference': 'REFERENCE_KIND',
                'native_polyline_delay_reference_sec': 'float(native_polyline_length/C)',
                'native_polyline_delay_error_sec': 'finite(native_polyline_delay_error)',
                'native_delay_compatibility_tolerance_sec': 'delay_tolerance',
                'original8ulp_tolerance_unchanged': 'True',
                'legacy_policy_result': 'bool(legacy_policy_result)',
                'legacy_policy_scope': "'Original V4 candidate agreement, including its image-delay comparison; not a whole-report selection claim'",
            }
            payload.keywords.extend(ast.keyword(arg=key, value=ast.parse(value, mode='eval').body)
                                    for key, value in extra.items())
            counters['candidate_metadata'] += 1
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
        raise RuntimeError('Frozen report return anchor changed')
    returns[0].value.keywords.append(ast.keyword(arg='native_delay_compatibility',
                                               value=ast.parse('source_provenance()', mode='eval').body))
    returns[0].value.keywords.append(ast.keyword(arg='base_schema',
                                               value=ast.Constant(value=_base.VERSION)))
    counters['report_metadata'] += 1
    if any(value != 1 for value in counters.values()):
        raise RuntimeError('Native-delay derivation must have exactly the declared five anchors: ' + str(counters))
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(vars(_base))
    # VERSION occurs only in the report format label of the frozen function;
    # the physical-patch schema and all numerical constants remain unchanged.
    namespace.update(VERSION=VERSION, REFERENCE_KIND=REFERENCE_KIND,
                     source_provenance=source_provenance)
    exec(compile(module, str(Path(__file__).resolve()) + '::<one-reference correction>', 'exec'), namespace)
    return namespace['audit_patch_candidates'], counters


audit_patch_candidates, DERIVATION_ANCHORS = _derive()


def derivation_audit():
    return dict(passed=all(v == 1 for v in DERIVATION_ANCHORS.values()),
                anchors=dict(DERIVATION_ANCHORS), **source_provenance())

"""Explicit V5 extension for the corrected native-delay arithmetic reference.

The propagation model is still native Jones transport plus Float64 image-delay
phase. All frozen geometry/phase/receiver functions keep their code objects;
isolated namespaces replace only the separately versioned candidate predicate.
The original failed route is never opened or edited by this module.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import types

from . import devset_specular_audit_v4 as geometry_base
from . import devset_phase_reference_v5 as phase_base
from . import devset_runtime_v5_receiver as receiver_runtime
from . import devset_generate_v5_receiver as receiver_generator
from . import devset_storage as core
from .devset_storage_v5_receiver import RouteWriterV5Receiver
from . import devset_native_delay_reference as corrected_geometry

VERSION = 'kap_devset.native_polyline_runtime.v1'
PARENTS = {
    'devset_specular_audit_v4.py': '5192397743a104728a1e8b1d34cab151b79960f4e77b0885c767056bcebf9e7d',
    'devset_phase_reference_v5.py': '71f2c8a12c303f80e38f89e2bb070248f5a741d6abf464ca891abb4d8bc34892',
    'devset_runtime_v5_receiver.py': 'cd8217e06f0cc274feb79ef9e081f5b9aeb4aa2dd8f8cfdcf689f2fba23588c5',
    'devset_generate_v5_receiver.py': '2edfcd46c9763b90a12ac9d54bf7fd8c5ca24e257f8ce1248f2a614c50567d44',
    'devset_storage_v5_receiver.py': 'a4cbc7032fecbc3236cb59169d1047bbb06734a853f2e6a996a4d5210e966c25',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_provenance():
    folder = Path(__file__).parent
    return dict(schema=VERSION,
        source_sha256={
            'validation/devset_native_polyline_runtime.py': sha(__file__),
            'validation/devset_native_delay_reference.py': sha(corrected_geometry.__file__),
        },
        frozen_parent_source_sha256={'validation/' + name: digest for name, digest in PARENTS.items()},
        candidate_rule=corrected_geometry.source_provenance(),
        physical_jones_and_image_phase_terms_unchanged=True,
        original_native_delay_ulp_count=8,
        physical_plane_optical_visibility_limits_unchanged=True,
        original_failed_route_preserved_separately=True,
        scope='Native arithmetic consistency uses its actual native-vertex polyline; '
              'Float64 ideal image remains the physical phase/geometry reference. '
              'This is not a native-coefficient reevaluation or exact EM guarantee.')


def _clone(function, replacements, name):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    result = types.FunctionType(function.__code__, namespace, name,
                                function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    result.__annotations__ = dict(function.__annotations__)
    return result


for _name, _digest in PARENTS.items():
    if sha(Path(__file__).with_name(_name)) != _digest:
        raise RuntimeError('Frozen native-polyline extension parent changed: ' + _name)
corrected_geometry.derivation_audit()

_PATCH_PATHS = _clone(geometry_base.audit_patch_paths,
    {'audit_patch_candidates': corrected_geometry.audit_patch_candidates},
    'audit_patch_paths_native_polyline')
_PHASE_PATHS = _clone(phase_base.audit_phase_paths,
    {'audit_patch_paths': _PATCH_PATHS}, 'audit_phase_paths_native_polyline')
_TRACE = _clone(receiver_runtime.DevsetRTV5ReceiverGrid.trace,
    {'audit_phase_paths': _PHASE_PATHS}, 'trace_native_polyline')


class DevsetRTV5NativePolyline(receiver_runtime.DevsetRTV5ReceiverGrid):
    """Same native solver and V5 reference, with a source-correct predicate."""
    trace = _TRACE

    def configure(self, route):
        extras = super().configure(route)
        extras['native_delay_compatibility_extension'] = source_provenance()
        return extras


class RouteWriterNativePolyline(RouteWriterV5Receiver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handle.attrs['native_delay_compatibility_extension'] = VERSION
        self.append_event('native_polyline_adapter_provenance', source_provenance())

    def append_raw_candidate(self, compact, metadata, raw_geometry,
                             normalization_report, native_golden=None):
        policy = normalization_report.get('native_delay_compatibility')
        expected = corrected_geometry.source_provenance()
        if policy != expected:
            raise ValueError('Raw normalization lacks the actual corrected native-delay source')
        index = super().append_raw_candidate(compact, metadata, raw_geometry,
                                              normalization_report, native_golden=native_golden)
        # V4 deliberately stores a compact report summary. Persist the new
        # predicate payload separately; a full-report hash alone cannot prove
        # which arithmetic reference or numerical tolerance was checked.
        groups = []
        for group in normalization_report['groups']:
            candidates = []
            for candidate in group.get('candidates', []):
                names = ('compact_index', 'native_path_index',
                         'native_delay_reference', 'native_polyline_delay_reference_sec',
                         'native_polyline_delay_error_sec',
                         'native_delay_compatibility_tolerance_sec',
                         'original8ulp_tolerance_unchanged', 'legacy_policy_result',
                         'agrees_with_unique_solution', 'delay_error_sec',
                         'projected_optical_excess_m', 'max_plane_error_m',
                         'projected_same_side', 'native_polyline_length_m')
                candidates.append({name: candidate[name] for name in names if name in candidate})
            solution = group.get('analytic_solution', {})
            groups.append(dict(path_indices=group['path_indices'],
                status=group['status'], reason=group.get('reason'),
                selected_compact_index=group.get('selected_compact_index'),
                ideal_delay_sec=solution.get('delay_sec'),
                ideal_length_m=solution.get('length_m'),
                ideal_points_xyz_m=solution.get('points'),
                plane_reference_triangles_m=solution.get('plane_reference_triangles_m'),
                plane_normals=solution.get('normals'),
                native_delay_tolerance_sec=group.get('native_delay_agreement_tolerance_sec'),
                optical_tolerance_m=group.get('native_optical_excess_tolerance_m'),
                plane_tolerance_m=group.get('native_plane_agreement_tolerance_m'),
                float64_length_roundoff_m=group.get('float64_length_roundoff_m'),
                candidates=candidates))
        digest = hashlib.sha256(core._json(core._diagnostic_value(normalization_report)).encode()).hexdigest()
        self.append_event('native_polyline_raw_certificate', dict(
            raw_record_index=int(index), schema=normalization_report['schema'],
            base_report_schema='kap_devset.unique_specular_physical_patch.v4',
            source=expected, full_report_sha256=digest,
            admitted=bool(normalization_report['admitted']), groups=groups,
            raw_native_coefficients_vertices_and_primitive_ids_unchanged=True,
            scope='Versioned predicate evidence; physical model and original eight-ULP/optical/plane limits retained'))
        return index


def _writer_factory(*args, **kwargs):
    return RouteWriterNativePolyline(*args, **kwargs)


def _receiver_provenance():
    result = receiver_generator.source_provenance()
    result['native_delay_compatibility_extension'] = source_provenance()
    return result


_GENERATE = _clone(receiver_generator._GENERATE_ROUTE,
    {'RouteWriter': _writer_factory, '_receiver_source_provenance': _receiver_provenance},
    'generate_route_native_polyline')


def generate_route(context, route, path, **kwargs):
    if not isinstance(context, DevsetRTV5NativePolyline):
        raise TypeError('Corrected candidate provenance requires its declared runtime')
    return _GENERATE(context, route, path, **kwargs)


def binding_audit():
    return dict(passed=(
        _PATCH_PATHS.__code__ is geometry_base.audit_patch_paths.__code__
        and _PHASE_PATHS.__code__ is phase_base.audit_phase_paths.__code__
        and _TRACE.__code__ is receiver_runtime.DevsetRTV5ReceiverGrid.trace.__code__
        and _GENERATE.__code__ is receiver_generator._GENERATE_ROUTE.__code__
        and _TRACE.__globals__['audit_phase_paths'] is _PHASE_PATHS
        and _PHASE_PATHS.__globals__['audit_patch_paths'] is _PATCH_PATHS
        and _PATCH_PATHS.__globals__['audit_patch_candidates'] is corrected_geometry.audit_patch_candidates
        and _GENERATE.__globals__['RouteWriter'] is _writer_factory),
        original_code_objects_unchanged=True, isolated_globals_only=True,
        parent=source_provenance())

"""Canonical model-input entry point over the frozen on-demand dataset.

Raw views retain diagnostic task/family/environment bookkeeping. Their hashes
are identifiers, not encryption: a binary pass ID can be recovered by evaluating
its two public candidates. This adapter moves such identifiers out of ``input``
without changing any physical values, receiver arrays, noise parameters or CSI
labels. Feed only ``sample['input']`` to a model; ``metadata`` is for bookkeeping.
"""
from __future__ import annotations

from collections.abc import Mapping

from .devset_index import DevsetDataset as _IndexedDevsetDataset
from .devset_index import nested_training_families

VERSION='kap_devset.learning_input_without_bookkeeping.v1'

# Explicit semantic names, rather than a blanket ``*_id`` rule: burst_id is a
# physical receiver-schedule grouping and must remain available to the model.
BOOKKEEPING_KEYS=frozenset({
    'task_id','pass_id','route_family_id','family_id','route_id','candidate_id',
    'city','split','region_id','task_index','route_index','local_task_index',
    'record_index','target_record_index','pass_index','environment_id',
    'environment_version','env_id','env_version','survey_sample_ids',
    'survey_design_index',
})
_ABSENT=object()


def _separate(value):
    if isinstance(value,Mapping):
        clean={};removed={}
        for key,item in value.items():
            if key in BOOKKEEPING_KEYS:removed[key]=item
            else:
                kept,metadata=_separate(item);clean[key]=kept
                if metadata is not _ABSENT:removed[key]=metadata
        return clean,removed if removed else _ABSENT
    if isinstance(value,(list,tuple)):
        items=[_separate(item) for item in value]
        clean=[item[0] for item in items]
        if isinstance(value,tuple):clean=tuple(clean)
        if not any(item[1] is not _ABSENT for item in items):return clean,_ABSENT
        removed=[None if item[1] is _ABSENT else item[1] for item in items]
        return clean,tuple(removed) if isinstance(value,tuple) else removed
    # No copying/coercion of arrays, scalar values or descriptive physical
    # strings. Large full-FFT observations therefore incur no extra array copy.
    return value,_ABSENT


def split_input_metadata(inputs):
    """Return fresh containers and unchanged value objects in two subtrees.

    Removed fields retain their original dictionary path. Lists/tuples in the
    metadata subtree preserve positional alignment with None placeholders.
    This is a contract for the frozen view's named identifiers, not a promise
    that legitimate physical observations cannot reveal their environment.
    """
    clean,metadata=_separate(inputs)
    return clean,{} if metadata is _ABSENT else metadata


class DevsetDataset(_IndexedDevsetDataset):
    """Canonical condition/observation dataset with separate bookkeeping.

    Constructor, sealed-test policy, family filtering, worker-safe serialization
    and on-demand synthesis are inherited unchanged from the frozen index class.
    No RT solver is imported or called by this adapter.
    """
    def __getitem__(self,index):
        sample=super().__getitem__(index)
        inputs,metadata=split_input_metadata(sample['input'])
        result=dict(sample);result['input']=inputs;result['metadata']=metadata
        return result


__all__=['DevsetDataset','nested_training_families','split_input_metadata',
         'BOOKKEEPING_KEYS','VERSION']

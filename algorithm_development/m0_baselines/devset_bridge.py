"""Controlled simulator/evaluator access, isolated from estimator signatures.

Frozen-static controls are NEW diagnostic observations. They are not relabeled
release_05 observations, and they are not the original 2 ms prediction task.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sionna_small_validation"))
from validation.devset_learning import DevsetDataset, nested_training_families
from validation.devset_view_v5 import DevsetViewV5
from validation.devset_view import _seed
from validation.dataset_observations import _bursts
from validation.cities import reconstruct
from validation.mimo_observation import simulate_schedule
from linear_models import channel_matrix, shrink_second_moment, static_forward


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""): h.update(b)
    return h.hexdigest()


def fit_training_covariance(index, n_families=8, per_family=4, shrinkage=.05):
    families=nested_training_families(index,n_families)
    moments=[];captures=[]
    for family in families:
        ds=DevsetDataset(index,split="train",mode="condition",families=[family])
        ids=np.unique(np.minimum((np.arange(per_family)+.5)*len(ds)/per_family,len(ds)-1).astype(int))
        block=[]
        for i in ids:
            sample=ds[int(i)]
            h=channel_matrix(sample["target_csi"])
            block.append(h @ h.conj().T / h.shape[1])
        moments.append(np.mean(block,axis=0))
        captures.append(dict(family=family,split="train",dataset_indices=ids.tolist(),labels=len(ids)))
    r=shrink_second_moment(np.mean(moments,axis=0),shrinkage)
    return r,dict(families=captures,shrinkage=shrinkage,zero_mean_model=True,
                  normalization="absolute complex channel units; equal family weights; no validation/test power normalization",
                  covariance_scope="global Tx spatial second moment; correlated Rx/frequency vectors; no iid sample-count claim",
                  eigenvalues=np.linalg.eigvalsh(r).tolist())


def validation_cases(index):
    with h5py.File(index,"r") as f:
        routes=json.loads(f["routes_json"][()].decode())
    selected=[]
    for city in ("munich","florence"):
        for mechanism in ("stable_los","reflection_nlos","visibility_boundary","dynamic_event"):
            r=next(r for r in routes if r["split"]=="validation" and r["city"]==city and r["mechanism"]==mechanism)
            selected.append(r)
    return selected


def load_case(index, route):
    """Choose an admitted window with a separately admitted direct cutoff state.

    Selection is by closeness to the physical midpoint of pass 1,
    not by algorithm score, path count, or an expected trend.
    """
    ds=DevsetDataset(index,split="validation",mode="condition",families=[route["route_family_id"]])
    path=Path(index).parent / route["shard"]
    view=DevsetViewV5(path)
    with h5py.File(path,"r") as f:
        states=f["states"][:]
        all_tasks=f["tasks"][:]
        local=all_tasks[ds.refs["local_task_index"]]
        valid=np.flatnonzero((states["quality_flags"][ds.refs["cutoff_record_index"]]==0)
                            & (local["pass_index"]==1))
        if not len(valid): raise ValueError("No independently admitted cutoff for frozen control")
        midpoint=float(view.route["duration_sec"])/2
        at=int(valid[np.argmin(abs(ds.refs["cutoff_sec"][valid]-midpoint))])
        ref=ds.refs[at];task_index=int(ref["local_task_index"])
        task=f["tasks"][task_index]
        cutoff=view._snapshot(f,int(ref["cutoff_record_index"]),states[int(ref["cutoff_record_index"])])
        candidates=np.flatnonzero((states["pass_index"]==task["pass_index"]) & states["receiver_anchor"])
        candidates=candidates[np.argsort(states["time_sec"][candidates],kind="stable")]
        times=states["time_sec"][candidates]
        lo=np.searchsorted(times,float(task["history_start_sec"]),side="right")-1
        hi=np.searchsorted(times,float(task["cutoff_sec"]),side="right")
        assert lo>=0
        snapshots=[view._snapshot(f,int(i),states[int(i)]) for i in candidates[lo:hi]]
    condition,calibration=view._condition(task_index,16,0)
    cfo=float(np.random.default_rng(_seed(view.family,view.config.get("cfo_seed",20261027),"cfo")).normal(0,view.config["cfo_prior_std_hz"]))
    frozen_paths=cutoff["compact"].copy();frozen_paths["doppler_hz"]=0
    # Synthetic fixed channel: no original local-validity certificate is reused.
    frozen=dict(time_sec=condition["task"]["history_start_sec"],compact=frozen_paths,
                tx_relative_m=cutoff["tx_relative_m"],rx_relative_m=cutoff["rx_relative_m"])
    return dict(metadata=dict(family=view.family,city=route["city"],mechanism=route["mechanism"],
                              split="validation",task_index=task_index,cutoff_record_index=int(ref["cutoff_record_index"]),
                              cutoff_sec=float(task["cutoff_sec"]),shard_sha256=sha(path),
                              pass_index=1,selection="closest admitted direct cutoff to physical route midpoint in pass1"),
                public_condition=condition,calibration_power=calibration,cutoff=cutoff,
                frozen=frozen,dynamic_snapshots=snapshots,private_cfo_hz=cfo,
                tx_local_m=view.tx_local,carrier_hz=view.config["carrier_hz"],
                original_target_frequencies_hz=view.frequencies_hz,
                max_local_age_sec=view.config["max_local_age_sec"])


def make_control(case, beam_mode):
    schedule=_bursts(len(case["tx_local_m"]),case["public_condition"]["task"],beam_mode)
    frequencies=np.fft.fftfreq(schedule["nfft"],1/schedule["fs_hz"])
    target=reconstruct(case["frozen"]["compact"],case["frozen"]["tx_relative_m"],
                       case["frozen"]["rx_relative_m"],frequencies)
    generated=simulate_schedule([case["frozen"]],schedule,cfo_hz=0.,noise_variance=0.,max_local_age_sec=.0041)
    ideal=static_forward(target,schedule["tx_beams"],schedule["symbols"])
    return schedule,target,ideal,generated["noiseless_full_fft"]


def dynamic_diagnostic(case,schedule):
    """Noiseless real-path observation; private CFO belongs to simulator only."""
    outputs={}
    for key,cfo in (("dynamic_cfo_zero",0.),("dynamic_cfo_original",case["private_cfo_hz"])):
        outputs[key]=simulate_schedule(case["dynamic_snapshots"],schedule,cfo_hz=cfo,
                    noise_variance=0.,max_local_age_sec=case["max_local_age_sec"])["noiseless_full_fft"]
    return outputs

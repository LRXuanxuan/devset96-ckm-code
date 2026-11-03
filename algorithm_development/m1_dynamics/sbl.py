"""Fixed-grid complex Gaussian SBL with known noise and EM variance updates.

Generic SBL baseline, not the WCM hierarchical/MM algorithm of Bu et al.
No map, off-grid refinement, true support, true path count, or learned noise.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from path_model import Estimate, remove_cfo


@dataclass
class PreparedDictionary:
    normalized: np.ndarray
    gram: np.ndarray
    norms: np.ndarray
    indices: np.ndarray
    original_atoms: int


def prepare_dictionary(phi):
    phi=np.asarray(phi,complex)
    if phi.ndim!=2 or not np.isfinite(phi).all():
        raise ValueError('Dictionary must be a finite complex matrix')
    norms=np.linalg.norm(phi,axis=0)
    keep=np.flatnonzero(norms>max(norms.max(initial=0)*1e-10,np.finfo(float).tiny))
    if not len(keep):
        raise ValueError('No observable atom')
    psi=phi[:,keep]/norms[keep]
    gram=psi.conj().T@psi
    return PreparedDictionary(psi,(gram+gram.conj().T)/2,norms[keep],keep,phi.shape[1])


def posterior_from_statistics(gram,b,energy,n_observations,gamma,noise_variance):
    """Proper complex posterior, evaluated in atom space rather than sample space.

    u = diag(column_norms) g; prior E[uu^H]=diag(gamma).
    C = I + sqrt(Gamma) Psi^H Psi sqrt(Gamma) / noise_variance.
    """
    gamma=np.asarray(gamma,float)
    if not np.isfinite(gamma).all() or np.any(gamma<=0) or not np.isfinite(noise_variance) or noise_variance<=0:
        raise ValueError('Positive finite variances required')
    root=np.sqrt(gamma)
    system=np.eye(len(gamma))+root[:,None]*gram*root[None,:]/noise_variance
    system=(system+system.conj().T)/2
    factor=cho_factor(system,lower=True,check_finite=False)
    inverse=cho_solve(factor,np.eye(len(gamma)),check_finite=False)
    mean=root*cho_solve(factor,root*b,check_finite=False)/noise_variance
    covariance=root[:,None]*inverse*root[None,:]
    logdet=n_observations*np.log(noise_variance)+2*np.log(np.diag(factor[0]).real).sum()
    quadratic=(energy-float(np.vdot(b,mean).real))/noise_variance
    # Omitted constants are independent of gamma. This is negative log evidence.
    return mean,covariance,float(logdet+quadratic)


def fit_sbl(full_fft,schedule,atoms,prepared,*,noise_variance,known_cfo_hz=None,
            reference_time_sec=0.,max_iterations=300,relative_tolerance=1e-5):
    started=time.perf_counter()
    if not np.isfinite(noise_variance) or noise_variance<=0 or max_iterations<2 or relative_tolerance<=0:
        raise ValueError('SBL requires known positive noise, an iteration budget and a positive tolerance')
    if len(atoms)!=prepared.original_atoms:
        raise ValueError('Prepared dictionary and atoms disagree')
    y=np.asarray(full_fft,complex)
    if known_cfo_hz is not None:
        y=remove_cfo(y,schedule,known_cfo_hz)
    y=y.ravel()
    if len(y)!=len(prepared.normalized) or not np.isfinite(y).all():
        raise ValueError('Invalid observation')
    b=prepared.normalized.conj().T@y
    energy=float(np.vdot(y,y).real);n=len(y);g=len(prepared.indices)
    # Fixed data-only initialization; no oracle signal power or sparsity count.
    initial=max((energy-n*noise_variance)/g,energy/(100*g),noise_variance)
    gamma=np.full(g,initial)
    floor=initial*1e-12
    previous_mean=None;costs=[];converged=False;relative_change=float('inf')
    for iteration in range(max_iterations):
        mean,covariance,cost=posterior_from_statistics(prepared.gram,b,energy,n,gamma,noise_variance)
        costs.append(cost)
        if previous_mean is not None:
            relative_change=float(np.linalg.norm(mean-previous_mean)/max(np.linalg.norm(previous_mean),1e-30))
            cost_change=abs(costs[-1]-costs[-2])/max(abs(costs[-2]),1.)
            if relative_change<relative_tolerance and cost_change<relative_tolerance:
                converged=True
                break
        if iteration==max_iterations-1:
            break
        previous_mean=mean
        # EM M-step for proper complex Gaussian latent variables. The floor is
        # a declared numerical lower bound on each variance, not atom pruning.
        gamma=np.maximum(abs(mean)**2+np.maximum(np.diag(covariance).real,0.),floor)
    gains=mean/prepared.norms
    residual=y-prepared.normalized@mean
    changes=np.diff(costs)
    relative_increases=changes/np.maximum(np.abs(costs[:-1]),1.)
    return Estimate(tuple(atoms[i] for i in prepared.indices),gains,
                    'physical_known_cfo' if known_cfo_hz is not None else 'total_received',
                    known_cfo_hz,reference_time_sec,
                    dict(algorithm='complex_gaussian_em_sbl',iterations=len(costs),converged=converged,
                         termination='relative_mean_and_evidence' if converged else 'iteration_budget',
                         relative_mean_change=relative_change,negative_log_evidence=costs,
                         maximum_relative_evidence_increase=float(max(0.,relative_increases.max(initial=0.))),
                         residual_energy=float(np.vdot(residual,residual).real),observation_energy=energy,
                         gamma_normalized_min=float(gamma.min()),gamma_normalized_max=float(gamma.max()),
                         atoms_above_one_percent_max_gamma=int(np.sum(gamma>.01*gamma.max())),
                         posterior_variance_scope='conditional on fitted gamma and given noise/CFO; excludes hyperparameter and frequency uncertainty',
                         column_normalization=True,pruning=False,noise_variance=float(noise_variance),
                         elapsed_sec=time.perf_counter()-started))

"""Finite-CP angle-delay-frequency model with an explicit future map.

Fixed paths, directions, delays and frequencies over a short control window.
This module neither imports RT nor reads path truth. Candidate atoms are supplied
by the caller. Physical Doppler and total received frequency are distinct frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np

C_LIGHT = 299792458.0


@dataclass(frozen=True)
class Atom:
    delay_sec: float
    tx_direction: tuple[float, float, float]
    rx_direction: tuple[float, float, float]
    frequency_hz: float

    def __post_init__(self):
        if not np.isfinite([self.delay_sec, self.frequency_hz]).all() or self.delay_sec < 0:
            raise ValueError("Delay/frequency must be finite and delay nonnegative")
        for direction in (self.tx_direction, self.rx_direction):
            if np.shape(direction) != (3,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1)>1e-10:
                raise ValueError("Directions must be finite 3D unit vectors")


def direction_y(u):
    if abs(u)>1:
        raise ValueError("Invalid direction cosine")
    return (float(np.sqrt(1-u*u)), float(u), 0.)


def cartesian_grid(tx_cosines, rx_cosines, delays_sec, frequencies_hz):
    return tuple(Atom(float(d), direction_y(u), direction_y(v), float(f))
                 for u,v,d,f in product(tx_cosines,rx_cosines,delays_sec,frequencies_hz))


def ula(n, carrier_hz=28e9):
    return np.column_stack((np.zeros(n),np.arange(n)*C_LIGHT/(2*carrier_hz),np.zeros(n)))


def steering(positions, direction, carrier_hz):
    p=np.asarray(positions,float)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all():
        raise ValueError("Positions must be finite [antenna,3]")
    return np.exp(2j*np.pi*carrier_hz/C_LIGHT*((p-p[0])@np.asarray(direction)))


def layout(schedule):
    fs=float(schedule['fs_hz']); n=int(schedule['nfft']); cp=int(schedule['cp_samples'])
    starts=np.asarray(schedule['start_sample']); x=np.asarray(schedule['symbols'],complex)
    w=np.asarray(schedule['tx_beams'],complex)
    if not np.isfinite(fs) or fs<=0 or n<2 or not 0<=cp<=n:
        raise ValueError("Invalid sampling parameters")
    if starts.ndim!=1 or not len(starts) or not np.isfinite(starts).all() or np.any(starts!=starts.astype(np.int64)):
        raise ValueError("Starts must be integer sample indices")
    starts=starts.astype(np.int64)
    if np.any(np.diff(starts)<n+cp) or x.shape!=(len(starts),n) or w.ndim!=2 or len(w)!=len(starts):
        raise ValueError("Overlapping symbols or incompatible dimensions")
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("Nonfinite pilot or beam")
    numbers=starts[:,None]+cp+np.arange(n)
    return fs,n,cp,starts,x,w,numbers,numbers/fs


def observation_basis(atoms, schedule, tx_positions, rx_positions, *,
                      carrier_hz=28e9, reference_time_sec=0., cfo_hz=0.):
    """Phi: [S*Nr*N, G]; row order matches full_fft[S,Nr,N].

    Exact fractional delay under finite Fourier-polynomial CP-OFDM. Actual
    source-symbol supports account for ISI; samplewise frequency creates ICI.
    CFO phase uses absolute time, matching the release's oscillator convention.
    """
    atoms=tuple(atoms)
    fs,n,cp,starts,x,w,numbers,times=layout(schedule)
    if w.shape[1]!=len(tx_positions) or not np.isfinite([carrier_hz,reference_time_sec,cfo_hz]).all() or carrier_hz<=0:
        raise ValueError("Invalid array size or phase reference")
    num=numbers.ravel(); t=times.ravel(); frequencies=np.fft.fftfreq(n,1/fs)
    result=np.empty((len(starts)*len(rx_positions)*n,len(atoms)),complex)
    cache={}
    for j,atom in enumerate(atoms):
        key=(atom.delay_sec,atom.tx_direction)
        if key not in cache:
            tx=steering(tx_positions,atom.tx_direction,carrier_hz)
            source_sample=num-atom.delay_sec*fs
            source=np.searchsorted(starts,source_sample,side='right')-1
            valid=source>=0
            valid[valid]&=source_sample[valid]<starts[source[valid]]+n+cp
            scalar=np.zeros(len(t),complex)
            delayed=np.fft.ifft(x*np.exp(-2j*np.pi*frequencies*atom.delay_sec),axis=1,norm='ortho')
            for s in np.unique(source[valid]):
                at=valid&(source==s)
                indices=(num[at]-starts[s]-cp)%n
                scalar[at]=delayed[s,indices]*(w[s]@tx)
            cache[key]=scalar
        rx=steering(rx_positions,atom.rx_direction,carrier_hz)
        temporal=np.exp(2j*np.pi*(atom.frequency_hz*(t-reference_time_sec)+cfo_hz*t))
        samples=(cache[key]*temporal).reshape(len(starts),n)[:,None,:]*rx[None,:,None]
        result[:,j]=np.fft.fft(samples,axis=-1,norm='ortho').ravel()
    return result


def future_basis(atoms, tx_positions, rx_positions, frequencies_hz, target_time_sec, *,
                 carrier_hz=28e9, reference_time_sec=0.):
    """B*: [Nr*Nt*F, G], arbitrary frequencies; no nearest FFT-bin mapping.

    The frequency meaning (physical or total) follows the atom frame supplied.
    Carrier propagation phase is already part of the reference complex gain.
    """
    f=np.asarray(frequencies_hz,float)
    if f.ndim!=1 or not len(f) or not np.isfinite(f).all() or not np.isfinite([target_time_sec,reference_time_sec]).all():
        raise ValueError("Invalid target frequencies/time")
    columns=[]
    for atom in atoms:
        tx=steering(tx_positions,atom.tx_direction,carrier_hz)
        rx=steering(rx_positions,atom.rx_direction,carrier_hz)
        phase=np.exp(2j*np.pi*(atom.frequency_hz*(target_time_sec-reference_time_sec)-f*atom.delay_sec))
        columns.append((rx[:,None,None]*tx[None,:,None]*phase[None,None,:]).ravel())
    return np.column_stack(columns) if columns else np.empty((len(rx_positions)*len(tx_positions)*len(f),0),complex)


def remove_cfo(full_fft, schedule, cfo_hz):
    """Derotate at every receive sample, including within-symbol CFO/ICI."""
    *_,times=layout(schedule)
    y=np.asarray(full_fft,complex)
    if y.ndim!=3 or y.shape[0]!=len(times) or y.shape[-1]!=times.shape[-1] or not np.isfinite(cfo_hz):
        raise ValueError("Invalid observation/CFO")
    return np.fft.fft(np.fft.ifft(y,axis=-1,norm='ortho')*
                      np.exp(-2j*np.pi*cfo_hz*times)[:,None,:],axis=-1,norm='ortho')


@dataclass
class Estimate:
    atoms: tuple[Atom, ...]
    gains: np.ndarray
    frame: str
    known_cfo_hz: float | None
    reference_time_sec: float
    information: dict


def grid_omp(full_fft, schedule, atoms, phi_zero_cfo, *, noise_variance=0.,
             max_paths=3, known_cfo_hz=None, reference_time_sec=0.):
    """Fixed-grid scalar OMP over joint angle-delay-frequency waveform atoms.

    An optional externally known CFO is a declared diagnostic input. Without
    it, atoms represent total received frequency rho=nu+epsilon, not Doppler.
    Neither true support nor true path count is accepted by the estimator.
    """
    if noise_variance<0 or not np.isfinite(noise_variance) or max_paths<1:
        raise ValueError("Invalid noise/sparsity budget")
    y=np.asarray(full_fft,complex)
    if known_cfo_hz is not None:
        y=remove_cfo(y,schedule,known_cfo_hz)
    y=y.ravel(); phi=np.asarray(phi_zero_cfo,complex)
    if phi.shape!=(len(y),len(atoms)) or not np.isfinite(y).all():
        raise ValueError("Dictionary and observation dimensions disagree")
    norms=np.linalg.norm(phi,axis=0)
    visible=norms>max(norms.max(initial=0)*1e-10,np.finfo(float).tiny)
    selected=[]; residual=y.copy(); gains=np.empty(0,complex)
    energy=float(np.vdot(y,y).real)
    threshold=max(1.05*len(y)*noise_variance,energy*1e-20)
    for _ in range(min(max_paths,len(atoms))):
        if np.vdot(residual,residual).real<=threshold:
            break
        score=np.abs(phi.conj().T@residual)/np.maximum(norms,np.finfo(float).tiny)
        score[~visible]=-np.inf; score[selected]=-np.inf
        j=int(np.argmax(score))
        if not np.isfinite(score[j]) or score[j]<=0:
            break
        trial=selected+[j]
        if np.linalg.matrix_rank(phi[:,trial],tol=1e-9*norms.max())<len(trial):
            visible[j]=False
            continue
        selected=trial
        gains=np.linalg.lstsq(phi[:,selected],y,rcond=1e-10)[0]
        residual=y-phi[:,selected]@gains
    return Estimate(tuple(atoms[j] for j in selected),gains,
                    'physical_known_cfo' if known_cfo_hz is not None else 'total_received',
                    known_cfo_hz,reference_time_sec,
                    dict(selected_indices=selected,residual_energy=float(np.vdot(residual,residual).real),
                         observation_energy=energy,stop_energy=threshold,max_paths=max_paths))


def predict(estimate, tx_positions, rx_positions, frequencies_hz, target_time_sec, *,
            target='propagation', cfo_anchor_hz=None, carrier_hz=28e9):
    if target not in ('propagation','effective'):
        raise ValueError("Choose propagation or effective channel explicitly")
    b=future_basis(estimate.atoms,tx_positions,rx_positions,frequencies_hz,target_time_sec,
                   carrier_hz=carrier_hz,reference_time_sec=estimate.reference_time_sec)
    h=b@estimate.gains
    if estimate.frame=='total_received' and target=='propagation':
        if cfo_anchor_hz is None or not np.isfinite(cfo_anchor_hz):
            raise ValueError("Propagation CSI is not identified without an explicit CFO/physical-frequency anchor")
        h*=np.exp(-2j*np.pi*cfo_anchor_hz*target_time_sec)
    elif estimate.frame=='physical_known_cfo' and target=='effective':
        h*=np.exp(2j*np.pi*estimate.known_cfo_hz*target_time_sec)
    elif estimate.frame not in ('total_received','physical_known_cfo'):
        raise ValueError("Unknown frequency frame")
    return h.reshape(len(rx_positions),len(tx_positions),len(frequencies_hz))


def reference_path_cfo(estimate, associated_index, reference_doppler_hz):
    """Requires an externally associated path with known physical Doppler.

    The benchmark labels this as an oracle reference-path diagnostic. It is
    NOT an estimator of the required reference information from a real map.
    """
    if estimate.frame!='total_received' or not np.isfinite(reference_doppler_hz):
        raise ValueError("Requires a total-frequency estimate and finite anchor")
    return float(estimate.atoms[associated_index].frequency_hz-reference_doppler_hz)

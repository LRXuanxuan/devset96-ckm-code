"""M0 static beam-domain observation model and classical estimators.

Shapes: H[rx, tx, frequency], W[symbol, tx], X[symbol, frequency].
The physical forward product is H @ w, without conjugating the beam twice.
Only constant-modulus, nonzero pilots are supported by the white-noise wrappers.
"""
from __future__ import annotations

import numpy as np


def channel_matrix(channel):
    h = np.asarray(channel, np.complex128)
    if h.ndim != 3 or not np.isfinite(h).all():
        raise ValueError("H must be finite [rx,tx,frequency]")
    return h.transpose(1, 0, 2).reshape(h.shape[1], -1)


def channel_tensor(matrix, n_rx, n_frequency):
    c = np.asarray(matrix, np.complex128)
    return c.reshape(c.shape[0], n_rx, n_frequency).transpose(1, 0, 2)


def static_forward(channel, beams, symbols):
    h = np.asarray(channel, np.complex128)
    w = np.asarray(beams, np.complex128)
    x = np.asarray(symbols, np.complex128)
    if w.ndim != 2 or x.shape != (len(w), h.shape[2]) or w.shape[1] != h.shape[1]:
        raise ValueError("Incompatible static forward dimensions")
    return np.einsum("st,rtk->srk", w, h, optimize=True) * x[:, None, :]


def remove_known_pilots(received, symbols):
    """Y / X estimates beam-projected channels, not the full MIMO channel."""
    y = np.asarray(received, np.complex128)
    x = np.asarray(symbols, np.complex128)
    if y.ndim != 3 or x.shape != (y.shape[0], y.shape[2]):
        raise ValueError("Y and X axes differ")
    if not np.isfinite(y).all() or not np.isfinite(x).all() or not np.allclose(abs(x), 1., atol=1e-7):
        raise ValueError("This first model requires finite unit-modulus pilots")
    return (y / x[:, None, :]).reshape(len(y), -1)


def least_squares(beams, demodulated):
    """Minimum-Euclidean-norm LS. No full-column-rank assumption or inverse."""
    return np.linalg.lstsq(np.asarray(beams, np.complex128), demodulated, rcond=1e-10)[0]


def lmmse(beams, demodulated, covariance, noise_variance):
    """Zero-mean spatial LMMSE, shared over receive antennas and frequencies.

    R is the model's Tx-channel second moment, fitted on training labels only.
    This does not claim that the physical channel is Gaussian or separable.
    """
    w = np.asarray(beams, np.complex128)
    r = np.asarray(covariance, np.complex128)
    if r.shape != (w.shape[1], w.shape[1]) or noise_variance <= 0:
        raise ValueError("LMMSE requires matching covariance and positive noise")
    if not np.isfinite(r).all() or not np.allclose(r, r.conj().T, rtol=1e-8, atol=1e-30):
        raise ValueError("R must be finite and Hermitian")
    system = w @ r @ w.conj().T + noise_variance * np.eye(len(w))
    return r @ w.conj().T @ np.linalg.solve(system, demodulated)


def shrink_second_moment(moment, shrinkage=.05):
    r = np.asarray(moment, np.complex128)
    r = (r + r.conj().T) / 2
    power = float(np.trace(r).real / len(r))
    if r.ndim != 2 or r.shape[0] != r.shape[1] or power <= 0 or not 0 < shrinkage <= 1:
        raise ValueError("Invalid second moment or shrinkage")
    return (1-shrinkage)*r + shrinkage*power*np.eye(len(r))


def planar_dictionary(tx_local_m, carrier_hz, grid_size=16):
    """Unit-norm local plane-wave atoms on a fixed y/z direction-cosine grid.

    The shipped Tx is planar in local y/z. Both half spaces induce the same
    phase-only response on this array; no front/back identification is claimed.
    """
    position = np.asarray(tx_local_m, float)
    if position.ndim != 2 or position.shape[1] != 3 or np.ptp(position[:, 0]) > 1e-10:
        raise ValueError("This dictionary assumes a local y/z planar array")
    q = np.arange(grid_size)*2/grid_size - 1
    uv = np.array([(u, v) for u in q for v in q if u*u+v*v <= 1+1e-12])
    directions = np.column_stack((np.sqrt(np.maximum(0, 1-(uv*uv).sum(1))), uv))
    phase = 2*np.pi*carrier_hz/299792458. * ((position-position[0]) @ directions.T)
    return np.exp(1j*phase)/np.sqrt(len(position)), directions


def sensing_diagnostics(beams, dictionary):
    phi = beams @ dictionary
    norm = np.linalg.norm(phi, axis=0)
    visible = norm > max(float(norm.max())*1e-6, np.finfo(float).tiny)
    normalized = phi[:, visible] / norm[visible]
    gram = abs(normalized.conj().T @ normalized)
    np.fill_diagonal(gram, 0)
    return dict(beam_rank=int(np.linalg.matrix_rank(beams, tol=1e-8)),
                n_tx=beams.shape[1], null_dimension=beams.shape[1]-int(np.linalg.matrix_rank(beams, tol=1e-8)),
                dictionary_atoms=dictionary.shape[1], numerically_visible_atoms=int(visible.sum()),
                normalized_sensing_coherence=float(gram.max(initial=0)))


def mmv_omp(beams, demodulated, dictionary, noise_variance=0., max_atoms=8):
    """Shared-support complex OMP with l2 aggregation across measurement vectors.

    Selection uses the sufficient Gram Z Z^H, exactly matching the l2 residual
    correlation score, to avoid large repeated [atom,rx*frequency] products.
    It is a baseline variant of simultaneous pursuit, not a novel estimator.
    """
    w = np.asarray(beams, np.complex128)
    z = np.asarray(demodulated, np.complex128)
    a = np.asarray(dictionary, np.complex128)
    if noise_variance < 0 or max_atoms < 1:
        raise ValueError("Invalid noise or sparsity cap")
    phi = w @ a
    norms = np.linalg.norm(phi, axis=0)
    visible = norms > max(float(norms.max())*1e-6, np.finfo(float).tiny)
    psi = np.zeros_like(phi)
    psi[:, visible] = phi[:, visible] / norms[visible]
    gram_y = z @ z.conj().T
    total = float(np.trace(gram_y).real)
    noise_stop = 1.05*z.size*noise_variance
    chosen = []
    projection = np.eye(len(w), dtype=complex)
    residual_energy = total
    for _ in range(min(max_atoms, int(np.linalg.matrix_rank(w, tol=1e-8)))):
        if residual_energy <= max(noise_stop, total*1e-12):
            break
        residual_atoms = projection @ psi
        scores = np.real(np.sum(residual_atoms.conj()*(gram_y @ residual_atoms), axis=0))
        scores[~visible] = -np.inf
        scores[chosen] = -np.inf
        maximum = scores.max(initial=-np.inf)
        if not np.isfinite(maximum) or maximum <= 0:
            break
        tied = np.flatnonzero(scores >= maximum*(1-1e-12))
        # A deterministic gain tie-break is not physical support identification.
        j = int(tied[np.argmax(norms[tied])])
        trial = chosen + [j]
        if np.linalg.matrix_rank(phi[:, trial], tol=1e-9*norms.max()) < len(trial):
            visible[j] = False
            continue
        chosen = trial
        active = phi[:, chosen]
        projection = np.eye(len(w)) - active @ np.linalg.pinv(active, rcond=1e-10)
        residual_energy = max(0., float(np.trace(projection @ gram_y @ projection.conj().T).real))
    if chosen:
        coefficients = np.linalg.lstsq(phi[:, chosen], z, rcond=1e-10)[0]
        estimate = a[:, chosen] @ coefficients
    else:
        estimate = np.zeros((w.shape[1], z.shape[1]), complex)
    return estimate, dict(selected_atoms=chosen, residual_energy=residual_energy,
                          noise_stop_energy=noise_stop, max_atoms=max_atoms,
                          selection="normalized l2 MMV correlation; fixed grid; no off-grid refinement")


def relative_mse(estimate, truth):
    denom = float(np.vdot(truth, truth).real)
    if denom <= 0:
        raise ValueError("Relative MSE requires nonzero reference")
    return float(np.vdot(estimate-truth, estimate-truth).real / denom)


def db(value):
    return float(10*np.log10(max(float(value), 1e-30)))

"""CP-OFDM observations with exact fractional delay under a local path model.

This is a generic baseband OFDM experiment, not an NR receiver.  Each transmitted
CP-OFDM symbol is its finite Fourier polynomial on its CP-inclusive interval and
zero outside the scheduled intervals.  Evaluating that polynomial at t-tau is
exact for the declared model: no nearest-tap delay or sinc truncation is used.
Rectangular symbol transitions and an ideal DAC are declared assumptions.

At each RECEIVE sample a left RT anchor supplies delay/directions and the native
baseband coefficient; its Doppler advances that coefficient to the sample time.
No path identity, interpolation, or extra carrier-delay phase is invented.
Anchor transitions are not guaranteed continuous: their physical accuracy must
be admitted separately by true RT midpoint/convergence checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .cities import C, FC, COMPACT_DTYPE, reconstruct


def make_schedule(n_tx: int, n_symbols: int = 19, nfft: int = 256,
                  fs_hz: float = 30.72e6, cp_samples: int = 64,
                  start_time_sec: float = 0.0, beam_mode: str = 'rotating',
                  seed: int = 20261027) -> dict[str, Any]:
    """Known full-band QPSK pilots and unit-norm DFT transmit beams.

    ``start_sample`` is on a single Fs grid, not an RT-anchor index.  Users may
    replace symbols, beams, masks or starts; simulate_schedule recalculates the
    ledger from those actual arrays.  The default per-RE energy is one; actual
    CP energy is counted rather than assumed equal to CP/N times useful energy.
    """
    if n_tx < 1 or n_symbols < 1 or nfft < 2 or not 0 <= cp_samples <= nfft:
        raise ValueError('Invalid OFDM schedule sizes')
    first = int(round(start_time_sec * fs_hz))
    if abs(first / fs_hz - start_time_sec) > 1e-13:
        raise ValueError('Symbol start must lie on the declared baseband sample grid')
    rng = np.random.default_rng(seed)
    symbols = np.exp(.5j*np.pi*rng.integers(0, 4, (n_symbols, nfft)))
    if beam_mode not in ('fixed', 'rotating'):
        raise ValueError('beam_mode must be fixed or rotating')
    beam_index = np.zeros(n_symbols, int) if beam_mode == 'fixed' else np.arange(n_symbols) % n_tx
    beams = np.exp(2j*np.pi*beam_index[:, None]*np.arange(n_tx)[None, :]/n_tx)/np.sqrt(n_tx)
    return dict(fs_hz=float(fs_hz), nfft=int(nfft), cp_samples=int(cp_samples),
                start_sample=first+np.arange(n_symbols)*(nfft+cp_samples),
                tx_beams=beams, symbols=symbols, pilot_mask=np.ones(symbols.shape, bool),
                definition='known CP-OFDM Fourier polynomial; receive-time Doppler; absolute path delays',
                beam_mode=beam_mode, seed=int(seed))


def _validated_schedule(schedule):
    fs = float(schedule['fs_hz']); n = int(schedule['nfft']); cp = int(schedule['cp_samples'])
    x = np.asarray(schedule['symbols'], np.complex128)
    beams = np.asarray(schedule['tx_beams'], np.complex128)
    starts_raw = np.asarray(schedule['start_sample'])
    starts = starts_raw.astype(np.int64)
    mask = np.asarray(schedule.get('pilot_mask', np.abs(x) > 0), bool)
    if fs <= 0 or n < 2 or not 0 <= cp <= n or x.ndim != 2 or x.shape[1] != n:
        raise ValueError('Invalid OFDM dimensions')
    if beams.ndim != 2 or beams.shape[0] != len(x) or starts.shape != (len(x),) or mask.shape != x.shape:
        raise ValueError('Schedule arrays have inconsistent shapes')
    if np.any(starts != starts_raw) or np.any(np.diff(starts) < n+cp):
        raise ValueError('Integer start_sample values must be ordered and symbols cannot overlap')
    if not np.isfinite(x).all() or not np.isfinite(beams).all():
        raise ValueError('Nonfinite scheduled symbols or beams')
    return fs, n, cp, x, beams, starts, mask


def schedule_ledger(schedule) -> dict[str, Any]:
    fs, n, cp, symbols, beams, starts, mask = _validated_schedule(schedule)
    useful = np.fft.ifft(symbols, axis=-1, norm='ortho')
    beam_power = np.sum(abs(beams)**2, axis=-1)
    useful_energy = np.sum(abs(symbols)**2, axis=-1)*beam_power
    cp_energy = np.sum(abs(useful[:, -cp:])**2, axis=-1)*beam_power if cp else np.zeros(len(symbols))
    return dict(fs_hz=fs, bandwidth_grid_hz=fs, nfft=n, scs_hz=fs/n, cp_samples=cp,
                n_symbols=len(symbols), n_tx=beams.shape[1],
                occupied_resource_elements=int(np.count_nonzero(symbols)),
                known_pilot_resource_elements=int(np.count_nonzero(mask)),
                useful_tx_energy=float(useful_energy.sum()), cp_tx_energy=float(cp_energy.sum()),
                total_tx_energy=float((useful_energy+cp_energy).sum()),
                energy_units='sum of squared complex baseband samples over antennas; divide by Fs for normalized integrated energy',
                integrated_tx_energy_sec=float((useful_energy+cp_energy).sum()/fs),
                start_time_sec=float(starts[0]/fs), end_time_sec=float((starts[-1]+cp+n)/fs),
                elapsed_sec=float((starts[-1]+cp+n-starts[0])/fs),
                occupied_sample_count=int(len(symbols)*(n+cp)),
                first_useful_receive_sample_sec=float((starts[0]+cp)/fs),
                input_cutoff_sec=float((starts[-1]+cp+n-1)/fs),
                per_symbol_start_sec=(starts/fs).tolist(),
                per_symbol_useful_start_sec=((starts+cp)/fs).tolist(),
                per_symbol_tx_energy=(useful_energy+cp_energy).tolist(),
                per_symbol_beam_power=beam_power.tolist())


def path_spatial_coefficients(compact, tx_relative_m, rx_relative_m, carrier_hz=FC):
    p = np.asarray(compact)
    tx = np.asarray(tx_relative_m, float); rx = np.asarray(rx_relative_m, float)
    if tx.ndim != 2 or rx.ndim != 2 or tx.shape[1] != 3 or rx.shape[1] != 3:
        raise ValueError('Array positions require [antenna,3] in global coordinates')
    txphase = 2*np.pi*carrier_hz/C*((tx-tx[0]) @ p['k_tx'].T)
    rxphase = 2*np.pi*carrier_hz/C*((rx-rx[0]) @ p['k_rx'].T)
    # TX beam multiplication is H @ beam, without an extra conjugation.
    return np.exp(1j*txphase), np.exp(1j*rxphase)


def simulate_schedule(snapshots, schedule, carrier_hz=FC, cfo_hz=0.0,
                      noise_variance=0.0, noise_seed=20261027,
                      max_local_age_sec=0.001) -> dict[str, Any]:
    """Return all antenna FFT bins plus the declared pilot view and resource log.

    snapshots: ordered dictionaries with time_sec, compact, tx_relative_m and
    rx_relative_m.  A snapshot is selected by RECEIVE time, never nearest/future
    time.  The caller must explicitly allow the largest local extrapolation.
    Noise variance is E|n|^2 at each RX sample (and each unitary-FFT component).
    No target label, latent path data or future anchor is returned to the view.
    """
    fs, n, cp, symbols, beams, starts, mask = _validated_schedule(schedule)
    if (not snapshots or not np.isfinite(cfo_hz) or not np.isfinite(noise_variance)
            or noise_variance < 0 or not np.isfinite(max_local_age_sec) or max_local_age_sec < 0
            or not np.isfinite(carrier_hz) or carrier_hz <= 0):
        raise ValueError('Invalid channel or receiver parameters')
    anchor_times = np.asarray([s['time_sec'] for s in snapshots], float)
    if not np.isfinite(anchor_times).all() or np.any(np.diff(anchor_times) <= 0):
        raise ValueError('Snapshots require strictly increasing finite times')
    n_rx = len(snapshots[0]['rx_relative_m'])
    sample_numbers = starts[:, None]+cp+np.arange(n)[None, :]
    sample_times = sample_numbers/fs
    flat_times = sample_times.ravel(); flat_numbers = sample_numbers.ravel()
    selected = np.searchsorted(anchor_times, flat_times, side='right')-1
    if np.any(selected < 0):
        raise ValueError('No causal RT anchor at or before a receive sample')
    from .path_quality import check_selected_snapshots
    check_selected_snapshots(snapshots,flat_times,selected)
    ages = flat_times-anchor_times[selected]
    if np.max(ages, initial=0.) > max_local_age_sec+1e-13:
        raise ValueError(f'Local Doppler extrapolation {ages.max():.9g}s exceeds explicit bound {max_local_age_sec:.9g}s')
    y = np.zeros((n_rx, len(flat_times)), np.complex128)
    freqs = np.fft.fftfreq(n, 1/fs)
    max_delay = 0.; min_delay = float('inf'); used = []
    # At fixed anchor delay, exact polynomial fractional shift is an IFFT of
    # X[k]*exp(-j2pi f[k]tau).  Integer receive indices then select this periodic
    # polynomial only on the ACTUAL delayed source-symbol support.
    for ai in np.unique(selected):
        s = snapshots[int(ai)]; paths = np.asarray(s['compact'])
        if paths.dtype != COMPACT_DTYPE:
            paths = paths.astype(COMPACT_DTYPE)
        if len(s['rx_relative_m']) != n_rx or len(s['tx_relative_m']) != beams.shape[1]:
            raise ValueError('Antenna dimensions changed inside a schedule')
        if np.any(paths['delay_sec'] < 0) or not np.isfinite(paths['delay_sec']).all():
            raise ValueError('Physical path delays must be finite and nonnegative')
        for field in ('b00', 'doppler_hz', 'k_tx', 'k_rx'):
            if not np.isfinite(paths[field]).all():
                raise ValueError('Nonfinite physical path field: '+field)
        if any(np.any(abs(np.linalg.norm(paths[field],axis=-1)-1)>1e-4) for field in ('k_tx','k_rx')):
            raise ValueError('Path direction vectors must be unit global-coordinate vectors')
        indices = np.flatnonzero(selected == ai)
        times = flat_times[indices]; nums = flat_numbers[indices]
        txsteer, rxsteer = path_spatial_coefficients(paths, s['tx_relative_m'], s['rx_relative_m'], carrier_hz)
        used.append(int(ai))
        for pi, p in enumerate(paths):
            tau = float(p['delay_sec']); max_delay = max(max_delay, tau); min_delay = min(min_delay, tau)
            source_time_samples = nums-tau*fs
            source_symbol = np.searchsorted(starts, source_time_samples, side='right')-1
            valid = source_symbol >= 0
            valid[valid] &= source_time_samples[valid] < starts[source_symbol[valid]]+cp+n
            scalar = np.zeros(len(indices), np.complex128)
            delay_phase = np.exp(-2j*np.pi*freqs*tau)
            for sj in np.unique(source_symbol[valid]):
                at = valid & (source_symbol == sj)
                delayed_periodic = np.fft.ifft(symbols[sj]*delay_phase, norm='ortho')
                sample_index = (nums[at]-starts[sj]-cp) % n
                beam_gain = txsteer[:, pi] @ beams[sj]
                scalar[at] = delayed_periodic[sample_index]*beam_gain
            gain = complex(p['b00'])*np.exp(2j*np.pi*float(p['doppler_hz'])*(times-anchor_times[ai]))
            y[:, indices] += rxsteer[:, pi, None]*(gain*scalar)[None, :]
    y *= np.exp(2j*np.pi*float(cfo_hz)*flat_times)[None, :]
    noiseless = y.reshape(n_rx, len(symbols), n).transpose(1, 0, 2)
    rng = np.random.default_rng(noise_seed)
    noise = np.sqrt(noise_variance/2)*(rng.standard_normal(noiseless.shape)+1j*rng.standard_normal(noiseless.shape))
    full_fft = np.fft.fft(noiseless+noise, axis=-1, norm='ortho')
    pilot_y = full_fft[:, :, :].copy()
    pilot_y[~np.broadcast_to(mask[:, None, :], pilot_y.shape)] = 0
    ledger = schedule_ledger(schedule)
    ledger.update(receiver_antennas=n_rx, noise_E_abs_n_squared=float(noise_variance), noise_seed=int(noise_seed),
                  local_anchor_indices_used=used, max_local_age_sec=float(ages.max(initial=0.)),
                  cfo_hz=float(cfo_hz), minimum_absolute_delay_sec=None if not np.isfinite(min_delay) else min_delay,
                  maximum_absolute_delay_sec=max_delay, cp_covers_maximum_absolute_delay=bool(max_delay <= cp/fs),
                  actual_receiver_information='all RX antennas and all FFT bins including ICI; pilot_y mask is an additional indexed view, never the whole information claim',
                  phase_epoch_sec=0.0,
                  model='finite CP-OFDM Fourier polynomial, absolute fractional delays, local frozen geometry with receive-sample Doppler and independent common CFO')
    return dict(full_fft=full_fft, pilot_y=pilot_y, pilot_mask=mask.copy(),
                receive_time_sec=sample_times, receive_samples=noiseless+noise,
                noiseless_full_fft=np.fft.fft(noiseless, axis=-1, norm='ortho'), ledger=ledger)


def archive_snapshot(path: str | Path, snapshot_index: int = 0) -> dict[str, Any]:
    """Read one old or new compact archive snapshot without importing RT."""
    with h5py.File(path, 'r') as f:
        group = f['online'] if 'online' in f else f
        offsets = group['offsets']; i = int(snapshot_index)
        return dict(time_sec=float(group['time_sec'][i]), compact=group['paths'][offsets[i]:offsets[i+1]],
                    tx_relative_m=group['tx_relative_m'][i], rx_relative_m=group['rx_relative_m'][i])


def _relative(actual, reference):
    return float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference), np.finfo(float).tiny))


def _direct_polynomial_oracle(snapshot, schedule, cfo_hz=0.0):
    """Slow independent exp-sum evaluation, not fractional-shift IFFT."""
    fs, n, cp, symbols, beams, starts, mask = _validated_schedule(schedule)
    times = (starts[:, None]+cp+np.arange(n))/fs
    rx = np.asarray(snapshot['rx_relative_m']); tx = np.asarray(snapshot['tx_relative_m'])
    y = np.zeros((len(symbols), len(rx), n), complex)
    freqs = np.fft.fftfreq(n, 1/fs)
    for p in snapshot['compact']:
        ar = np.exp(2j*np.pi*FC/C*((rx-rx[0]) @ p['k_rx']))
        at = np.exp(2j*np.pi*FC/C*((tx-tx[0]) @ p['k_tx']))
        for si in range(len(symbols)):
            tau_times = times[si]-p['delay_sec']
            scalar = np.zeros(n, complex)
            for source in range(len(symbols)):
                valid = (tau_times >= starts[source]/fs) & (tau_times < (starts[source]+cp+n)/fs)
                if not valid.any():
                    continue
                phase = np.exp(2j*np.pi*freqs[:, None]*(tau_times[valid]-(starts[source]+cp)/fs)[None, :])
                scalar[valid] += (symbols[source] @ phase)/np.sqrt(n)*(at @ beams[source])
            gain = p['b00']*np.exp(2j*np.pi*(p['doppler_hz']*(times[si]-snapshot['time_sec'])+cfo_hz*times[si]))
            y[si] += ar[:, None]*(gain*scalar)[None, :]
    return np.fft.fft(y, axis=-1, norm='ortho')


def run_validation(out_dir: str | Path, archive_root: str | Path | None = None):
    from .common import save_json, sha256
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261027)
    p = np.zeros(2, COMPACT_DTYPE)
    p['b00'] = [1+.2j, -.3+.15j]; p['delay_sec'] = [7.37/30.72e6, 19.21/30.72e6]
    p['doppler_hz'] = [2317., -1821.]; p['k_tx'] = [[1, 0, 0], [0, 1, 0]]
    p['k_rx'] = [[0, 1, 0], [1, 0, 0]]; p['flags'] = 1
    tx = np.zeros((4, 3)); tx[:, 1] = np.arange(4)*C/FC/2
    rx = np.zeros((3, 3)); rx[:, 0] = np.arange(3)*C/FC/2
    fixture = dict(time_sec=0., compact=p, tx_relative_m=tx, rx_relative_m=rx)
    schedule = make_schedule(4, n_symbols=3)
    result = simulate_schedule([fixture], schedule, cfo_hz=713.)
    oracle = _direct_polynomial_oracle(fixture, schedule, cfo_hz=713.)
    error = _relative(result['full_fft'], oracle)
    static = p.copy(); static['doppler_hz'] = 0
    static_fixture = dict(fixture, compact=static)
    observed = simulate_schedule([static_fixture], schedule)
    freqs = np.fft.fftfreq(256, 1/30.72e6)
    h = reconstruct(static, tx, rx, freqs)
    expected = np.einsum('rtf,st,sf->srf', h, schedule['tx_beams'], schedule['symbols'])
    static_error = _relative(observed['full_fft'], expected)
    quantized = p.copy(); quantized['delay_sec'] = np.round(quantized['delay_sec']*30.72e6)/30.72e6
    quant = simulate_schedule([dict(fixture, compact=quantized)], schedule, cfo_hz=713.)
    quant_error = _relative(quant['full_fft'], result['full_fft'])
    frozen = np.einsum('rtf,st,sf->srf', reconstruct(p, tx, rx, freqs), schedule['tx_beams'], schedule['symbols'])
    frozen_error = _relative(frozen, result['full_fft'])
    # A continuous pure tone gives a closed-form FFT leakage pattern even when
    # its propagation delay is fractional.  This checks signs and absolute time.
    tone = make_schedule(1, n_symbols=1)
    tone['symbols'][:] = 0; tone['symbols'][0, 17] = np.sqrt(256)
    tone['pilot_mask'][:] = True
    one = p[:1].copy(); one['b00'] = 1.; one['delay_sec'] = 5.123/30.72e6; one['doppler_hz'] = 1234.
    one_snap = dict(time_sec=0.,compact=one,tx_relative_m=np.zeros((1,3)),rx_relative_m=np.zeros((1,3)))
    tone_result = simulate_schedule([one_snap],tone,cfo_hz=213.)
    tt=tone_result['receive_time_sec'][0]
    reference_samples = np.exp(2j*np.pi*(17*120000*(tt-64/30.72e6-one['delay_sec'][0])+(1234+213)*tt))
    tone_error = _relative(tone_result['full_fft'][0,0],np.fft.fft(reference_samples,norm='ortho'))
    # An intentionally short CP retains previous-symbol interference; oracle
    # equality verifies it was not silently forced into circular convolution.
    short = make_schedule(4,n_symbols=3,cp_samples=2)
    short_result = simulate_schedule([fixture],short,cfo_hz=713.)
    short_error = _relative(short_result['full_fft'],_direct_polynomial_oracle(fixture,short,cfo_hz=713.))
    # Physical continuation split into two anchors must leave the waveform
    # unchanged when the second anchor coefficient has the correct phase epoch.
    split_time=320/30.72e6
    next_p=p.copy(); next_p['b00']*=np.exp(2j*np.pi*p['doppler_hz']*split_time)
    split_result=simulate_schedule([fixture,dict(fixture,time_sec=split_time,compact=next_p)],schedule,cfo_hz=713.)
    split_error=_relative(split_result['full_fft'],result['full_fft'])
    # Complex64 storage incurs only its explicit coefficient quantization error.
    causal_rejection=False
    try:
        simulate_schedule([dict(fixture,time_sec=.1)],schedule)
    except ValueError:
        causal_rejection=True
    noise_result=simulate_schedule([static_fixture],make_schedule(4,n_symbols=64),
                                   noise_variance=.025,max_local_age_sec=.001)
    fft_noise=noise_result['full_fft']-noise_result['noiseless_full_fft']
    noise_power=float(np.mean(abs(fft_noise)**2))
    noise_tol=6/np.sqrt(fft_noise.size)
    controls=[]
    for nfft,n_symbols in ((1024,19),(512,38),(256,76)):
        ss=make_schedule(1,n_symbols=n_symbols,nfft=nfft,cp_samples=nfft//4,beam_mode='fixed')
        # Enforce exactly common CP-inclusive emitted energy, in addition to
        # common occupied RE and duration.  This is disclosed global scaling.
        ss['symbols'] *= np.sqrt(24320/schedule_ledger(ss)['total_tx_energy'])
        controls.append(schedule_ledger(ss))
    real=[]
    if archive_root is None:
        archive_root=Path(__file__).resolve().parents[1]/'artifacts'/'suite_20261026_gnu_01'
    for city in ('munich','florence','etoile'):
        new_archive=Path(archive_root)/city/'truth.h5'
        old_archive=Path(archive_root)/f'city_{city}'/'cities'/'compact128.h5'
        archive=new_archive if new_archive.exists() else old_archive
        if not archive.exists():
            real.append(dict(city=city,status='missing_archive',passed=False));continue
        with h5py.File(archive,'r') as f:
            golden=f['golden_cfr']
            ids=golden['record_index'][:] if 'record_index' in golden else golden['snapshot_index'][:]
            times=f['time_sec'][:]
            # New archives also retain an independently captured historical
            # survey at t=-86400s.  Waveform checks deliberately select an
            # ONLINE native golden, never extrapolate that old survey forward.
            online=times[ids]>=0
            if 'role' in f:
                online &= f['role'].asstr()[:][ids]!='historical_survey'
            candidates=np.flatnonzero(online)
            if not len(candidates):
                real.append(dict(city=city,archive=str(archive.resolve()),status='missing_online_native_golden',passed=False));continue
            golden_index=int(candidates[np.argmin(times[ids[candidates]])])
            gi=int(ids[golden_index]); snap=archive_snapshot(archive,gi)
            freqs_native=f['frequencies_hz'][:]
            values=golden['values'] if 'values' in golden else golden
            native=values['real'][golden_index]+1j*values['imag'][golden_index]
            archive_schema=str(f.attrs.get('schema','unspecified'))
        compact_h=reconstruct(snap['compact'],snap['tx_relative_m'],snap['rx_relative_m'],freqs_native)
        native_error=_relative(compact_h,native)
        ss=make_schedule(len(snap['tx_relative_m']),n_symbols=2,start_time_sec=round(snap['time_sec']*30.72e6)/30.72e6)
        # Anchor may be a fraction of a sampling interval after the grid origin:
        # put first CP on the first grid point at or after the anchor.
        ss['start_sample'] += int(np.ceil(snap['time_sec']*30.72e6))-ss['start_sample'][0]
        rr=simulate_schedule([snap],ss,max_local_age_sec=.001)
        oo=_direct_polynomial_oracle(snap,ss)
        true_error=_relative(rr['full_fft'],oo)
        np.savez_compressed(out_dir/f'{city}_fractional_mimo_observation.npz',full_fft=rr['full_fft'],
                            symbols=ss['symbols'],tx_beams=ss['tx_beams'],receive_time_sec=rr['receive_time_sec'])
        real.append(dict(city=city,archive=str(archive.resolve()),archive_sha256=sha256(archive),
                         archive_schema=archive_schema,snapshot_index=gi,snapshot_time_sec=snap['time_sec'],
                         native_golden_index=golden_index,native_cfr_relative_error=native_error,
                         fractional_mimo_oracle_relative_error=true_error,ledger=rr['ledger'],
                         paths=len(snap['compact']),passed=bool(native_error<1e-4 and true_error<2e-10)))
    checks=dict(two_path_fractional_mimo_independent_oracle=error<2e-10,
                static_fractional_mimo_equals_H_X=static_error<2e-10,
                single_tone_closed_form_doppler_cfo=tone_error<2e-10,
                insufficient_cp_includes_intersymbol_interference=short_error<2e-10,
                nearest_integer_delay_negative_control=quant_error>.05,
                frozen_H_negative_control=frozen_error>.05,
                fixed_bandwidth_same_Fs=all(c['fs_hz']==30.72e6 for c in controls),
                fixed_bandwidth_same_training_duration=np.ptp([c['elapsed_sec'] for c in controls])<1e-14,
                fixed_bandwidth_same_occupied_RE=len(set(c['occupied_resource_elements'] for c in controls))==1,
                fixed_bandwidth_same_CP_inclusive_energy=np.ptp([c['total_tx_energy'] for c in controls])<1e-8,
                coherent_multi_anchor_continuation=split_error<2e-7,
                future_anchor_is_rejected=causal_rejection,
                unitary_fft_complex_noise_variance=abs(noise_power/.025-1)<noise_tol,
                real_rt_fractional_MIMO_all_cities=bool(real and all(c['passed'] for c in real)))
    report=dict(schema='fractional_mimo_observation.validation.v1',checks=checks,passed=all(checks.values()),
                relative_errors=dict(fractional_mimo_oracle=error,static_mimo=static_error,tone=tone_error,
                                     short_cp=short_error,integer_tap_surrogate=quant_error,frozen_H=frozen_error,
                                     coherent_multi_anchor=split_error),
                noise_check=dict(specified_variance=.025,measured_variance=noise_power,relative_tolerance=noise_tol),
                fixed_bandwidth_controls=controls,real_RT_cases=real,
                limits=['No NR synchronizer/decoder/BER claim',
                        'Native archive CFR agreement validates the fixed-pose synthetic-array representation only',
                        'Full FFT oracle agreement validates the declared ideal-DAC fractional-delay receive model, not RT geometry accuracy',
                        'Local anchor and Doppler validity require independent adaptive RT checks; no long-term phase continuity is assumed',
                        'CFO is an explicit independent oscillator parameter; separation from common Doppler is not claimed'])
    save_json(out_dir/'fractional_mimo_report.json',report)
    return report

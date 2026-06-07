import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class NMOParams:
    """Parameters for NMO correction."""
    stretch_limit: float = 0.3
    mute_near_offsets: float = 0.0
    mute_far_offsets: Optional[float] = None


@dataclass
class VelocitySpectrumParams:
    """Parameters for velocity spectrum calculation."""
    v_min: float = 1000.0
    v_max: float = 5000.0
    dv: float = 50.0
    semblance_window: int = 5
    v_pick: Optional[float] = None


def compute_nmo_time(t0: np.ndarray, offset: float, velocity: float) -> np.ndarray:
    """
    Compute NMO-corrected travel times.
    
    Parameters:
    - t0: Zero-offset travel times (n_samples,)
    - offset: Offset distance
    - velocity: NMO velocity
    
    Returns:
    t_nmo: NMO-corrected travel times
    """
    return np.sqrt(t0**2 + (offset / velocity)**2)


def compute_stretch(t0: np.ndarray, offset: float, velocity: float) -> np.ndarray:
    """
    Compute NMO stretch factor.
    
    Parameters:
    - t0: Zero-offset travel times
    - offset: Offset distance
    - velocity: NMO velocity
    
    Returns:
    stretch: Stretch factor (1 + dt/dt0)
    """
    t_nmo = compute_nmo_time(t0, offset, velocity)
    stretch = np.zeros_like(t0, dtype=np.float64)
    valid = (t_nmo > 1e-10) & (t0 > 1e-10)
    stretch[valid] = t_nmo[valid] / t0[valid] - 1.0
    return stretch


def nmo_correct(trace: np.ndarray, dt: float, offset: float, velocity: float,
                params: NMOParams) -> Dict:
    """
    Apply NMO correction to a single trace.
    
    Parameters:
    - trace: Input trace (n_samples,)
    - dt: Sample interval in seconds
    - offset: Offset distance in meters
    - velocity: NMO velocity in m/s
    - params: NMO parameters
    
    Returns:
    dict with corrected trace and mute mask
    """
    n_samples = len(trace)
    t0 = np.arange(n_samples) * dt
    
    t_nmo = compute_nmo_time(t0, offset, velocity)
    stretch = compute_stretch(t0, offset, velocity)
    
    mute_mask = np.ones(n_samples, dtype=bool)
    
    if params.stretch_limit > 0:
        mute_mask = mute_mask & (stretch < params.stretch_limit)
    
    if params.mute_near_offsets > 0 and abs(offset) < params.mute_near_offsets:
        mute_mask[:] = False
    
    if params.mute_far_offsets is not None and abs(offset) > params.mute_far_offsets:
        mute_mask[:] = False
    
    corrected_trace = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        if not mute_mask[i]:
            continue
        
        t = t_nmo[i]
        sample_idx = t / dt
        
        if sample_idx < 0 or sample_idx >= n_samples - 1:
            continue
        
        idx0 = int(sample_idx)
        idx1 = idx0 + 1
        frac = sample_idx - idx0
        
        if idx1 < n_samples:
            corrected_trace[i] = (1 - frac) * trace[idx0] + frac * trace[idx1]
    
    return {
        'corrected_trace': corrected_trace,
        'mute_mask': mute_mask,
        't0': t0,
        't_nmo': t_nmo,
        'stretch': stretch
    }


def nmo_correct_gather(gather: np.ndarray, offsets: np.ndarray, dt: float,
                       velocities: Union[float, np.ndarray], params: NMOParams) -> Dict:
    """
    Apply NMO correction to a CDP gather.
    
    Parameters:
    - gather: Input gather (n_traces, n_samples)
    - offsets: Offset distances (n_traces,)
    - dt: Sample interval
    - velocities: NMO velocities (single value or n_samples,)
    - params: NMO parameters
    
    Returns:
    dict with corrected gather, mute masks, and stretch info
    """
    n_traces, n_samples = gather.shape
    
    if np.isscalar(velocities):
        velocities = np.full(n_samples, velocities)
    
    corrected_gather = np.zeros_like(gather)
    mute_masks = np.zeros((n_traces, n_samples), dtype=bool)
    stretch_data = np.zeros((n_traces, n_samples))
    
    for i in range(n_traces):
        result = nmo_correct(gather[i, :], dt, offsets[i],
                             velocities if np.isscalar(velocities) else velocities,
                             params)
        corrected_gather[i, :] = result['corrected_trace']
        mute_masks[i, :] = result['mute_mask']
        stretch_data[i, :] = result['stretch']
    
    return {
        'corrected_gather': corrected_gather,
        'mute_masks': mute_masks,
        'stretch': stretch_data,
        'offsets': offsets,
        'dt': dt,
        'velocities': velocities
    }


def compute_semblance(gather: np.ndarray, dt: float, offset: np.ndarray,
                      velocity: float, window: int = 5) -> np.ndarray:
    """
    Compute semblance for a given velocity.
    
    Parameters:
    - gather: Input gather (n_traces, n_samples)
    - dt: Sample interval
    - offset: Offset distances (n_traces,)
    - velocity: Test velocity
    - window: Window size for semblance calculation
    
    Returns:
    semblance: Semblance values (n_samples,)
    """
    n_traces, n_samples = gather.shape
    t0 = np.arange(n_samples) * dt
    
    nmo_gather = np.zeros_like(gather)
    
    for i in range(n_traces):
        t_nmo = compute_nmo_time(t0, offset[i], velocity)
        
        for j in range(n_samples):
            t = t_nmo[j]
            sample_idx = t / dt
            
            if sample_idx < 0 or sample_idx >= n_samples - 1:
                continue
            
            idx0 = int(sample_idx)
            idx1 = idx0 + 1
            frac = sample_idx - idx0
            
            if idx1 < n_samples:
                nmo_gather[i, j] = (1 - frac) * gather[i, idx0] + frac * gather[i, idx1]
    
    semblance = np.zeros(n_samples)
    
    half_win = window // 2
    
    for j in range(half_win, n_samples - half_win):
        window_slice = slice(j - half_win, j + half_win + 1)
        
        num = np.sum(np.sum(nmo_gather[:, window_slice], axis=0)**2)
        den = np.sum(np.sum(nmo_gather[:, window_slice]**2, axis=0))
        
        if den > 0 and n_traces > 0:
            semblance[j] = num / (den * n_traces)
    
    return semblance


def compute_velocity_spectrum(gather: np.ndarray, dt: float, offsets: np.ndarray,
                              params: VelocitySpectrumParams) -> Dict:
    """
    Compute velocity spectrum using constant velocity scan.
    
    Parameters:
    - gather: Input CDP gather (n_traces, n_samples)
    - dt: Sample interval
    - offsets: Offset distances
    - params: Velocity spectrum parameters
    
    Returns:
    dict with velocity spectrum data
    """
    n_traces, n_samples = gather.shape
    velocities = np.arange(params.v_min, params.v_max + params.dv, params.dv)
    n_velocities = len(velocities)
    
    spectrum = np.zeros((n_samples, n_velocities))
    
    for vi, v in enumerate(velocities):
        spectrum[:, vi] = compute_semblance(gather, dt, offsets, v, params.semblance_window)
    
    return {
        'spectrum': spectrum,
        'velocities': velocities,
        'times': np.arange(n_samples) * dt,
        'dt': dt,
        'offsets': offsets,
        'params': params
    }


def pick_velocity(spectrum_result: Dict, time_idx: int, vel_idx: int) -> Dict:
    """
    Pick velocity from spectrum at specified location.
    
    Parameters:
    - spectrum_result: Result from compute_velocity_spectrum
    - time_idx: Index at which to pick
    - vel_idx: Velocity index at which to pick
    
    Returns:
    dict with picked velocity and time
    """
    spectrum = spectrum_result['spectrum']
    velocities = spectrum_result['velocities']
    times = spectrum_result['times']
    
    time_idx = min(max(time_idx, 0), len(times) - 1)
    vel_idx = min(max(vel_idx, 0), len(velocities) - 1)
    
    return {
        'time': times[time_idx],
        'velocity': velocities[vel_idx],
        'semblance': spectrum[time_idx, vel_idx],
        'time_idx': time_idx,
        'vel_idx': vel_idx
    }


def build_velocity_function(picks: List[Dict], n_samples: int, dt: float) -> np.ndarray:
    """
    Build velocity function from picked velocities using linear interpolation.
    
    Parameters:
    - picks: List of velocity picks with 'time' and 'velocity'
    - n_samples: Number of samples
    - dt: Sample interval
    
    Returns:
    velocity_function: Velocity as function of time (n_samples,)
    """
    if not picks:
        return np.full(n_samples, 2000.0)
    
    picks_sorted = sorted(picks, key=lambda x: x['time'])
    
    times = np.arange(n_samples) * dt
    pick_times = [0.0] + [p['time'] for p in picks_sorted] + [times[-1]]
    pick_vels = [picks_sorted[0]['velocity']] + [p['velocity'] for p in picks_sorted] + [picks_sorted[-1]['velocity']]
    
    return np.interp(times, pick_times, pick_vels)


def cdp_stack(corrected_gather: np.ndarray, mute_masks: np.ndarray) -> np.ndarray:
    """
    Compute CDP stack from NMO-corrected gather.
    
    Parameters:
    - corrected_gather: NMO-corrected gather (n_traces, n_samples)
    - mute_masks: Mute masks for each trace
    
    Returns:
    stack: Stacked trace (n_samples,)
    """
    weighted_sum = np.zeros(corrected_gather.shape[1])
    weight_sum = np.zeros(corrected_gather.shape[1])
    
    for i in range(corrected_gather.shape[0]):
        valid = mute_masks[i, :]
        weighted_sum[valid] += corrected_gather[i, valid]
        weight_sum[valid] += 1.0
    
    stack = np.zeros_like(weighted_sum)
    valid = weight_sum > 0
    stack[valid] = weighted_sum[valid] / weight_sum[valid]
    
    return stack


def process_cdp_gather(gather: np.ndarray, offsets: np.ndarray, dt: float,
                       nmo_params: NMOParams,
                       spectrum_params: VelocitySpectrumParams,
                       velocity_picks: Optional[List[Dict]] = None) -> Dict:
    """
    Complete CDP processing workflow.
    
    Parameters:
    - gather: Input CDP gather (n_traces, n_samples)
    - offsets: Offset distances (n_traces,)
    - dt: Sample interval
    - nmo_params: NMO parameters
    - spectrum_params: Velocity spectrum parameters
    - velocity_picks: Optional velocity picks
    
    Returns:
    dict with all processing results
    """
    n_traces, n_samples = gather.shape
    
    spectrum_result = compute_velocity_spectrum(gather, dt, offsets, spectrum_params)
    
    if velocity_picks is None and spectrum_params.v_pick is not None:
        vel_idx = np.argmin(np.abs(spectrum_result['velocities'] - spectrum_params.v_pick))
        max_time_idx = np.argmax(spectrum_result['spectrum'][:, vel_idx])
        velocity_picks = [pick_velocity(spectrum_result, max_time_idx, vel_idx)]
    
    if velocity_picks is None:
        max_idx = np.unravel_index(np.argmax(spectrum_result['spectrum']),
                                   spectrum_result['spectrum'].shape)
        velocity_picks = [pick_velocity(spectrum_result, max_idx[0], max_idx[1])]
    
    velocity_function = build_velocity_function(velocity_picks, n_samples, dt)
    
    nmo_result = nmo_correct_gather(gather, offsets, dt, velocity_function, nmo_params)
    
    stacked_trace = cdp_stack(nmo_result['corrected_gather'], nmo_result['mute_masks'])
    
    return {
        'input_gather': gather,
        'offsets': offsets,
        'dt': dt,
        'velocity_spectrum': spectrum_result,
        'velocity_picks': velocity_picks,
        'velocity_function': velocity_function,
        'nmo_result': nmo_result,
        'stacked_trace': stacked_trace
    }

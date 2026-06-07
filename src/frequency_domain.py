import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FilterParams:
    """Parameters for frequency domain filtering."""
    lowcut: float = 5.0
    highcut: float = 50.0
    order: int = 4
    filter_type: str = 'bandpass'


@dataclass
class DeconvolutionParams:
    """Parameters for predictive deconvolution."""
    prediction_lag: int = 1
    operator_length: int = 60
    white_noise: float = 0.001


def compute_spectrum(trace: np.ndarray, dt: float) -> Dict:
    """
    Compute amplitude and phase spectrum of a trace.
    
    Parameters:
    - trace: Input seismic trace (n_samples,)
    - dt: Sample interval in seconds
    
    Returns:
    dict with frequencies, amplitude spectrum, phase spectrum
    """
    n_samples = len(trace)
    
    freqs = np.fft.fftfreq(n_samples, d=dt)
    positive_idx = freqs >= 0
    
    fft_result = np.fft.fft(trace)
    
    amplitude_spectrum = np.abs(fft_result)
    phase_spectrum = np.angle(fft_result)
    
    return {
        'frequencies': freqs[positive_idx],
        'amplitude': amplitude_spectrum[positive_idx],
        'phase': phase_spectrum[positive_idx],
        'full_frequencies': freqs,
        'full_amplitude': amplitude_spectrum,
        'full_phase': phase_spectrum,
        'fft': fft_result
    }


def compute_average_spectrum(traces: np.ndarray, dt: float) -> Dict:
    """
    Compute average spectrum over multiple traces.
    
    Parameters:
    - traces: Input traces (n_traces, n_samples)
    - dt: Sample interval in seconds
    
    Returns:
    dict with average spectrum
    """
    n_traces, n_samples = traces.shape
    
    all_amplitudes = np.zeros((n_traces, n_samples // 2 + 1 if n_samples % 2 == 0 else n_samples // 2 + 1))
    
    for i in range(n_traces):
        spec = compute_spectrum(traces[i], dt)
        if i == 0:
            freqs = spec['frequencies']
        all_amplitudes[i, :len(spec['amplitude'])] = spec['amplitude']
    
    return {
        'frequencies': freqs,
        'amplitude_mean': np.mean(all_amplitudes, axis=0),
        'amplitude_std': np.std(all_amplitudes, axis=0),
        'amplitude_max': np.max(all_amplitudes, axis=0),
        'amplitude_min': np.min(all_amplitudes, axis=0)
    }


def design_butterworth_filter(filter_params: FilterParams, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design Butterworth filter coefficients.
    
    Parameters:
    - filter_params: Filter parameters
    - fs: Sampling frequency in Hz
    
    Returns:
    b, a: Filter coefficients
    """
    nyq = 0.5 * fs
    low = filter_params.lowcut / nyq
    high = filter_params.highcut / nyq
    
    if filter_params.filter_type == 'lowpass':
        b, a = butter(filter_params.order, high, btype='low')
    elif filter_params.filter_type == 'highpass':
        b, a = butter(filter_params.order, low, btype='high')
    elif filter_params.filter_type == 'bandpass':
        b, a = butter(filter_params.order, [low, high], btype='band')
    elif filter_params.filter_type == 'bandstop':
        b, a = butter(filter_params.order, [low, high], btype='bandstop')
    else:
        raise ValueError(f"Unknown filter type: {filter_params.filter_type}")
    
    return b, a


def apply_filter(trace: np.ndarray, b: np.ndarray, a: np.ndarray,
                 filtfilt_mode: bool = True) -> np.ndarray:
    """
    Apply filter to a trace.
    
    Parameters:
    - trace: Input trace
    - b, a: Filter coefficients
    - filtfilt_mode: Use filtfilt (zero phase) if True
    
    Returns:
    Filtered trace
    """
    if filtfilt_mode:
        return filtfilt(b, a, trace)
    else:
        return lfilter(b, a, trace)


def apply_filter_to_traces(traces: np.ndarray, filter_params: FilterParams,
                           dt: float, filtfilt_mode: bool = True) -> Dict:
    """
    Apply filter to multiple traces.
    
    Parameters:
    - traces: Input traces (n_traces, n_samples) or (n_samples, n_traces)
    - filter_params: Filter parameters
    - dt: Sample interval in seconds
    - filtfilt_mode: Use filtfilt (zero phase) if True
    
    Returns:
    dict with filtered traces and filter info
    """
    fs = 1.0 / dt
    
    if traces.shape[0] > traces.shape[1]:
        traces_input = traces.T
        transposed = True
    else:
        traces_input = traces
        transposed = False
    
    n_traces, n_samples = traces_input.shape
    
    b, a = design_butterworth_filter(filter_params, fs)
    
    filtered_traces = np.zeros_like(traces_input)
    
    for i in range(n_traces):
        filtered_traces[i, :] = apply_filter(traces_input[i, :], b, a, filtfilt_mode)
    
    if transposed:
        filtered_traces = filtered_traces.T
    
    return {
        'filtered_traces': filtered_traces,
        'b': b,
        'a': a,
        'fs': fs,
        'filter_params': filter_params
    }


def predictive_deconvolution(trace: np.ndarray, params: DeconvolutionParams) -> np.ndarray:
    """
    Apply predictive deconvolution to a trace.
    
    Parameters:
    - trace: Input trace (n_samples,)
    - params: Deconvolution parameters
    
    Returns:
    Deconvolved trace
    """
    n_samples = len(trace)
    pred_lag = params.prediction_lag
    op_len = params.operator_length
    white_noise = params.white_noise
    
    max_lag = min(op_len + pred_lag, n_samples - 1)
    
    autocorr = np.correlate(trace, trace, mode='full')
    autocorr = autocorr[n_samples - 1:n_samples + max_lag]
    
    r0 = autocorr[0]
    noise_floor = white_noise * r0
    
    R = np.zeros((op_len, op_len))
    r = np.zeros(op_len)
    
    for i in range(op_len):
        r[i] = autocorr[pred_lag + i]
        for j in range(op_len):
            R[i, j] = autocorr[abs(i - j)] + (noise_floor if i == j else 0)
    
    try:
        f = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        f = np.linalg.lstsq(R, r, rcond=None)[0]
    
    error_filter = np.zeros(op_len + 1)
    error_filter[0] = 1
    error_filter[pred_lag:pred_lag + op_len] = -f
    
    deconvolved = np.convolve(trace, error_filter, mode='same')
    
    return deconvolved


def apply_deconvolution_to_traces(traces: np.ndarray, params: DeconvolutionParams) -> Dict:
    """
    Apply predictive deconvolution to multiple traces.
    
    Parameters:
    - traces: Input traces (n_traces, n_samples) or (n_samples, n_traces)
    - params: Deconvolution parameters
    
    Returns:
    dict with deconvolved traces
    """
    if traces.shape[0] > traces.shape[1]:
        traces_input = traces.T
        transposed = True
    else:
        traces_input = traces
        transposed = False
    
    n_traces, n_samples = traces_input.shape
    
    deconvolved_traces = np.zeros_like(traces_input)
    
    for i in range(n_traces):
        deconvolved_traces[i, :] = predictive_deconvolution(traces_input[i, :], params)
    
    if transposed:
        deconvolved_traces = deconvolved_traces.T
    
    return {
        'deconvolved_traces': deconvolved_traces,
        'params': params
    }


def process_traces(traces: np.ndarray, dt: float,
                   filter_params: Optional[FilterParams] = None,
                   deconv_params: Optional[DeconvolutionParams] = None) -> Dict:
    """
    Complete frequency domain processing pipeline.
    
    Parameters:
    - traces: Input traces
    - dt: Sample interval
    - filter_params: Optional filter parameters
    - deconv_params: Optional deconvolution parameters
    
    Returns:
    dict with processed traces and spectra for comparison
    """
    result = {
        'original_traces': traces,
        'dt': dt,
        'original_spectrum': compute_average_spectrum(
            traces if traces.shape[0] < traces.shape[1] else traces.T, dt
        )
    }
    
    current_traces = traces
    
    if filter_params is not None:
        filter_result = apply_filter_to_traces(current_traces, filter_params, dt)
        current_traces = filter_result['filtered_traces']
        result['filtered_traces'] = current_traces
        result['filter_result'] = filter_result
        result['filtered_spectrum'] = compute_average_spectrum(
            current_traces if current_traces.shape[0] < current_traces.shape[1] else current_traces.T, dt
        )
    
    if deconv_params is not None:
        deconv_result = apply_deconvolution_to_traces(current_traces, deconv_params)
        current_traces = deconv_result['deconvolved_traces']
        result['deconvolved_traces'] = current_traces
        result['deconv_result'] = deconv_result
        result['deconvolved_spectrum'] = compute_average_spectrum(
            current_traces if current_traces.shape[0] < current_traces.shape[1] else current_traces.T, dt
        )
    
    result['processed_traces'] = current_traces
    
    return result

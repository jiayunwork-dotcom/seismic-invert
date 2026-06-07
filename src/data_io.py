import struct
import numpy as np
import pandas as pd
from io import BytesIO
from typing import Tuple, Dict, Optional, Union


def _ibm_float_to_ieee(ibm_float: bytes) -> float:
    """Convert IBM 32-bit float to IEEE 32-bit float."""
    ibm_int = struct.unpack('>I', ibm_float)[0]
    sign = ibm_int >> 31
    exponent = (ibm_int >> 24) & 0x7F
    fraction = ibm_int & 0xFFFFFF
    
    if fraction == 0 and exponent == 0:
        return 0.0
    
    ieee_mantissa = fraction << 4
    ieee_exponent = exponent * 4 - 130 + 127
    
    while ieee_mantissa >= 0x1000000:
        ieee_mantissa >>= 1
        ieee_exponent += 1
    
    if ieee_exponent <= 0:
        return 0.0
    if ieee_exponent >= 255:
        return float('inf') if sign == 0 else float('-inf')
    
    ieee_mantissa &= 0x7FFFFF
    ieee_int = (sign << 31) | (ieee_exponent << 23) | ieee_mantissa
    
    return struct.unpack('>f', struct.pack('>I', ieee_int))[0]


def _detect_encoding(data: bytes, n_samples: int) -> str:
    """Automatically detect IBM vs IEEE float encoding."""
    try:
        ibm_values = []
        for i in range(min(100, n_samples)):
            val = _ibm_float_to_ieee(data[i*4:(i+1)*4])
            ibm_values.append(val)
        ibm_max = max(abs(v) for v in ibm_values)
        
        ieee_values = list(struct.unpack(f'>{min(100, n_samples)}f', data[:min(100, n_samples)*4]))
        ieee_max = max(abs(v) for v in ieee_values)
        
        if 0 < ibm_max < 1e10 and ibm_max > ieee_max * 1e-5:
            if ieee_max > 1e20 or np.isnan(ieee_max):
                return 'ibm'
        return 'ieee'
    except:
        return 'ieee'


def parse_segy(file_bytes: bytes) -> Dict:
    """Parse SEG-Y file and extract headers and trace data."""
    result = {
        'format': 'segy',
        'binary_header': {},
        'trace_headers': [],
        'traces': None,
        'n_traces': 0,
        'n_samples': 0,
        'sample_interval': 0.0,
        'record_length': 0.0,
        'encoding': 'ieee',
        'coordinates': {'sx': [], 'sy': [], 'gx': [], 'gy': [], 'offset': []}
    }
    
    if len(file_bytes) < 3600:
        raise ValueError("File too small to be a valid SEG-Y file")
    
    text_header = file_bytes[:3200].decode('latin-1', errors='ignore')
    result['text_header'] = text_header
    
    binary_header_data = file_bytes[3200:3600]
    bh = binary_header_data
    result['binary_header']['job_id'] = struct.unpack('>I', bh[0:4])[0]
    result['binary_header']['line_number'] = struct.unpack('>I', bh[4:8])[0]
    result['binary_header']['reel_number'] = struct.unpack('>I', bh[8:12])[0]
    result['binary_header']['n_traces_per_ensemble'] = struct.unpack('>H', bh[12:14])[0]
    result['binary_header']['n_aux_traces'] = struct.unpack('>H', bh[14:16])[0]
    result['binary_header']['sample_interval'] = struct.unpack('>H', bh[16:18])[0]
    result['binary_header']['sample_interval_orig'] = struct.unpack('>H', bh[18:20])[0]
    result['binary_header']['n_samples'] = struct.unpack('>H', bh[20:22])[0]
    result['binary_header']['n_samples_orig'] = struct.unpack('>H', bh[22:24])[0]
    result['binary_header']['data_format'] = struct.unpack('>H', bh[24:26])[0]
    result['binary_header']['ensemble_fold'] = struct.unpack('>H', bh[26:28])[0]
    result['binary_header']['sorting_code'] = struct.unpack('>H', bh[28:30])[0]
    
    n_samples = result['binary_header']['n_samples']
    sample_interval = result['binary_header']['sample_interval'] / 1e6
    result['n_samples'] = n_samples
    result['sample_interval'] = sample_interval
    result['record_length'] = n_samples * sample_interval
    
    bytes_per_trace = 240 + n_samples * 4
    file_size = len(file_bytes) - 3600
    n_traces = file_size // bytes_per_trace
    result['n_traces'] = n_traces
    
    first_trace_data = file_bytes[3600+240:3600+240+n_samples*4]
    result['encoding'] = _detect_encoding(first_trace_data, n_samples)
    
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    trace_headers = []
    
    for i in range(n_traces):
        offset = 3600 + i * bytes_per_trace
        th_data = file_bytes[offset:offset+240]
        td_data = file_bytes[offset+240:offset+240+n_samples*4]
        
        th = {}
        th['trace_sequence'] = struct.unpack('>I', th_data[0:4])[0]
        th['ensemble_number'] = struct.unpack('>I', th_data[8:12])[0]
        th['trace_number_in_ensemble'] = struct.unpack('>H', th_data[12:14])[0]
        th['cdp_number'] = struct.unpack('>I', th_data[20:24])[0]
        th['shot_number'] = struct.unpack('>I', th_data[16:20])[0]
        th['coordinate_units'] = struct.unpack('>H', th_data[88:90])[0]
        th['n_samples'] = struct.unpack('>H', th_data[114:116])[0]
        th['sample_interval'] = struct.unpack('>H', th_data[116:118])[0]
        
        sx = struct.unpack('>i', th_data[72:76])[0]
        sy = struct.unpack('>i', th_data[76:80])[0]
        gx = struct.unpack('>i', th_data[80:84])[0]
        gy = struct.unpack('>i', th_data[84:88])[0]
        offset_val = struct.unpack('>i', th_data[36:40])[0]
        
        th['sx'] = sx
        th['sy'] = sy
        th['gx'] = gx
        th['gy'] = gy
        th['offset'] = offset_val
        
        result['coordinates']['sx'].append(sx)
        result['coordinates']['sy'].append(sy)
        result['coordinates']['gx'].append(gx)
        result['coordinates']['gy'].append(gy)
        result['coordinates']['offset'].append(offset_val)
        
        trace_headers.append(th)
        
        if result['encoding'] == 'ibm':
            for j in range(n_samples):
                traces[i, j] = _ibm_float_to_ieee(td_data[j*4:(j+1)*4])
        else:
            traces[i, :] = struct.unpack(f'>{n_samples}f', td_data)
    
    result['trace_headers'] = trace_headers
    result['traces'] = traces
    result['shot_numbers'] = np.array([th['shot_number'] for th in trace_headers])
    result['cdp_numbers'] = np.array([th['cdp_number'] for th in trace_headers])
    result['offsets'] = np.array([th['offset'] for th in trace_headers])
    
    return result


def parse_csv(file_bytes: bytes) -> Dict:
    """Parse CSV file for simplified seismic data import."""
    result = {
        'format': 'csv',
        'traces': None,
        'n_traces': 0,
        'n_samples': 0,
        'sample_interval': 0.0,
        'record_length': 0.0,
        'time_samples': None
    }
    
    df = pd.read_csv(BytesIO(file_bytes))
    result['time_samples'] = df.iloc[:, 0].values
    result['n_samples'] = len(result['time_samples'])
    
    if len(df.columns) > 1:
        result['traces'] = df.iloc[:, 1:].values.T.astype(np.float32)
        result['n_traces'] = result['traces'].shape[0]
        
        if len(result['time_samples']) > 1:
            result['sample_interval'] = result['time_samples'][1] - result['time_samples'][0]
        result['record_length'] = result['n_samples'] * result['sample_interval']
    else:
        raise ValueError("CSV file must have at least two columns: time and trace data")
    
    result['shot_numbers'] = np.arange(result['n_traces'])
    result['cdp_numbers'] = np.arange(result['n_traces'])
    result['offsets'] = np.arange(result['n_traces'])
    
    return result


def compute_statistics(data: Dict) -> Dict:
    """Compute basic statistics for seismic data."""
    stats = {}
    traces = data['traces']
    
    stats['max_amplitude'] = float(np.max(np.abs(traces)))
    stats['mean_amplitude'] = float(np.mean(np.abs(traces)))
    stats['rms_amplitude'] = float(np.sqrt(np.mean(traces**2)))
    
    if data['sample_interval'] > 0:
        n_samples = data['n_samples']
        freqs = np.fft.fftfreq(n_samples, d=data['sample_interval'])
        positive_freq_idx = freqs > 0
        
        spec = np.abs(np.fft.fft(traces, axis=1))
        mean_spec = np.mean(spec, axis=0)[positive_freq_idx]
        positive_freqs = freqs[positive_freq_idx]
        
        if len(mean_spec) > 0:
            peak_idx = np.argmax(mean_spec)
            stats['dominant_frequency'] = float(positive_freqs[peak_idx])
            
            total_energy = np.sum(mean_spec)
            cumulative = np.cumsum(mean_spec)
            stats['average_frequency'] = float(np.sum(positive_freqs * mean_spec) / total_energy)
        else:
            stats['dominant_frequency'] = 0.0
            stats['average_frequency'] = 0.0
        
        noise_floor = np.percentile(spec, 10)
        signal_max = np.percentile(spec, 99)
        if noise_floor > 0:
            stats['snr_db'] = float(20 * np.log10(signal_max / noise_floor))
        else:
            stats['snr_db'] = np.inf
    else:
        stats['dominant_frequency'] = 0.0
        stats['average_frequency'] = 0.0
        stats['snr_db'] = 0.0
    
    data['statistics'] = stats
    return data


def reorder_gathers(data: Dict, gather_type: str = 'shot') -> np.ndarray:
    """
    Reorder traces into different gather types.
    
    Parameters:
    - gather_type: 'shot', 'cdp', or 'offset'
    """
    traces = data['traces']
    
    if gather_type == 'shot':
        keys = data['shot_numbers']
        unique_keys = np.unique(keys)
    elif gather_type == 'cdp':
        keys = data['cdp_numbers']
        unique_keys = np.unique(keys)
    elif gather_type == 'offset':
        offsets = data['offsets']
        offset_bins = np.linspace(np.min(offsets), np.max(offsets), min(20, len(unique_keys := np.unique(offsets))))
        keys = np.digitize(offsets, offset_bins)
        unique_keys = np.unique(keys)
    else:
        raise ValueError(f"Unknown gather type: {gather_type}")
    
    n_samples = data['n_samples']
    n_unique = len(unique_keys)
    max_traces_per_gather = max(np.sum(keys == k) for k in unique_keys)
    
    gathers = np.zeros((n_unique, max_traces_per_gather, n_samples), dtype=np.float32)
    gather_mask = np.zeros((n_unique, max_traces_per_gather), dtype=bool)
    
    for i, key in enumerate(unique_keys):
        idx = np.where(keys == key)[0]
        n_traces = len(idx)
        gathers[i, :n_traces, :] = traces[idx, :]
        gather_mask[i, :n_traces] = True
    
    return gathers, gather_mask, unique_keys

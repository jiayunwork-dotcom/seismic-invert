import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class ForwardParams:
    """Parameters for finite-difference forward modeling."""
    dt: float = 0.001
    nt: int = 1000
    dx: float = 10.0
    dz: float = 10.0
    source_frequency: float = 30.0
    source_type: str = 'ricker'
    boundary_type: str = 'pml'
    pml_width: int = 20
    source_x: int = 50
    source_z: int = 2
    receiver_z: int = 2
    receiver_x_start: int = 10
    receiver_x_end: int = 90
    receiver_spacing: int = 1
    warnings: List[str] = field(default_factory=list)


def ricker_wavelet(frequency: float, dt: float, nt: int, t0: Optional[float] = None) -> np.ndarray:
    """Generate Ricker wavelet."""
    if t0 is None:
        t0 = 2.5 / frequency
    
    t = np.arange(nt) * dt
    tau = np.pi * frequency * (t - t0)
    wavelet = (1 - 2 * tau**2) * np.exp(-tau**2)
    return wavelet.astype(np.float32)


def gauss_first_derivative(frequency: float, dt: float, nt: int, t0: Optional[float] = None) -> np.ndarray:
    """Generate first derivative of Gaussian wavelet."""
    if t0 is None:
        t0 = 3.0 / frequency
    
    t = np.arange(nt) * dt
    tau = (t - t0) * frequency
    wavelet = -2 * np.pi**2 * frequency**2 * tau * np.exp(-np.pi**2 * tau**2)
    return wavelet.astype(np.float32)


def get_wavelet(source_type: str, frequency: float, dt: float, nt: int) -> np.ndarray:
    """Get wavelet based on source type."""
    if source_type == 'ricker':
        return ricker_wavelet(frequency, dt, nt)
    elif source_type == 'gauss':
        return gauss_first_derivative(frequency, dt, nt)
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def check_stability(velocity: np.ndarray, dx: float, dz: float, dt: float,
                    f_max: float) -> Dict:
    """
    Check numerical stability (CFL condition) and dispersion.
    
    Returns:
        dict with stability check results and warnings
    """
    v_min = np.min(velocity)
    v_max = np.max(velocity)
    
    results = {
        'v_min': v_min,
        'v_max': v_max,
        'cfl_ok': True,
        'dispersion_ok': True,
        'max_dt_stable': 0.0,
        'max_dx_stable': 0.0,
        'warnings': []
    }
    
    cfl_number = v_max * dt / min(dx, dz)
    cfl_limit = 1 / np.sqrt(2)
    results['cfl_number'] = cfl_number
    results['cfl_limit'] = cfl_limit
    
    if cfl_number > cfl_limit:
        results['cfl_ok'] = False
        results['max_dt_stable'] = cfl_limit * min(dx, dz) / v_max
        results['warnings'].append(
            f"CFL condition violated: CFL = {cfl_number:.3f} > {cfl_limit:.3f}. "
            f"Maximum stable dt = {results['max_dt_stable']*1000:.2f} ms"
        )
    else:
        results['max_dt_stable'] = cfl_limit * min(dx, dz) / v_max
    
    wavelength_min = v_min / f_max
    points_per_wavelength = min(dx, dz) / wavelength_min
    results['points_per_wavelength'] = points_per_wavelength
    results['min_points_required'] = 10
    
    if points_per_wavelength > 1/10:
        results['dispersion_ok'] = False
        results['max_dx_stable'] = wavelength_min / 10
        results['warnings'].append(
            f"Dispersion condition violated: {1/points_per_wavelength:.1f} points per wavelength < 10. "
            f"Maximum stable grid spacing = {results['max_dx_stable']:.2f} m"
        )
    else:
        results['max_dx_stable'] = wavelength_min / 10
    
    return results


def _apply_mur_boundary(p: np.ndarray, p_prev: np.ndarray, v: np.ndarray,
                        dt: float, dx: float, dz: float, nx: int, nz: int) -> None:
    """Apply Mur's first-order absorbing boundary condition."""
    v_surface = v[0, 0]
    coeff_left = (v_surface * dt - dx) / (v_surface * dt + dx)
    coeff_top = (v_surface * dt - dz) / (v_surface * dt + dz)
    
    for i in range(1, nx - 1):
        p[0, i] = p_prev[1, i] + coeff_top * (p[1, i] - p_prev[0, i])
        p[nz - 1, i] = p_prev[nz - 2, i] + coeff_top * (p[nz - 2, i] - p_prev[nz - 1, i])
    
    for j in range(1, nz - 1):
        p[j, 0] = p_prev[j, 1] + coeff_left * (p[j, 1] - p_prev[j, 0])
        p[j, nx - 1] = p_prev[j, nx - 2] + coeff_left * (p[j, nx - 2] - p_prev[j, nx - 1])


def _build_pml_profile(nx: int, nz: int, pml_width: int, dx: float, dz: float,
                       v_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build PML damping profiles."""
    R = 1e-6
    damping_max = -3 * v_max * np.log(R) / (2 * pml_width * min(dx, dz))
    
    sigma_x = np.zeros(nx, dtype=np.float64)
    sigma_z = np.zeros(nz, dtype=np.float64)
    
    for i in range(pml_width):
        ratio = (pml_width - i) / pml_width
        profile = damping_max * (ratio**2)
        
        sigma_x[i] = profile
        sigma_x[nx - 1 - i] = profile
        
        sigma_z[i] = profile
        sigma_z[nz - 1 - i] = profile
    
    return sigma_x, sigma_z


def run_forward(velocity: np.ndarray, params: ForwardParams) -> Dict:
    """
    Run 2D acoustic finite-difference forward modeling.
    
    Parameters:
    - velocity: 2D velocity model (nz, nx)
    - params: Forward modeling parameters
    
    Returns:
    dict with wavefield snapshots and synthetic seismograms
    """
    nz, nx = velocity.shape
    dx, dz = params.dx, params.dz
    dt = params.dt
    nt = params.nt
    
    stability = check_stability(velocity, dx, dz, dt, params.source_frequency * 2)
    params.warnings.extend(stability['warnings'])
    
    wavelet = get_wavelet(params.source_type, params.source_frequency, dt, nt)
    
    p = np.zeros((nz, nx), dtype=np.float64)
    p_prev = np.zeros((nz, nx), dtype=np.float64)
    p_next = np.zeros((nz, nx), dtype=np.float64)
    
    v_sq = (velocity.astype(np.float64) * dt)**2
    
    receiver_x = np.arange(params.receiver_x_start, params.receiver_x_end + 1, params.receiver_spacing)
    receiver_x = receiver_x[(receiver_x >= 0) & (receiver_x < nx)]
    n_receivers = len(receiver_x)
    
    seismograms = np.zeros((nt, n_receivers), dtype=np.float32)
    
    snapshots = []
    snapshot_interval = max(1, nt // 50)
    
    if params.boundary_type == 'pml':
        sigma_x, sigma_z = _build_pml_profile(nx, nz, params.pml_width, dx, dz, np.max(velocity))
        
        psi_x = np.zeros((nz, nx), dtype=np.float64)
        psi_z = np.zeros((nz, nx), dtype=np.float64)
        phi_x = np.zeros((nz, nx), dtype=np.float64)
        phi_z = np.zeros((nz, nx), dtype=np.float64)
        
        exp_sx = np.exp(-sigma_x * dt)
        exp_sz = np.exp(-sigma_z * dt)
    
    for it in range(nt):
        if params.boundary_type == 'pml':
            d2p_dx2 = np.zeros((nz, nx), dtype=np.float64)
            d2p_dz2 = np.zeros((nz, nx), dtype=np.float64)
            
            for i in range(2, nx - 2):
                d2p_dx2[:, i] = (-p[:, i + 2] + 16 * p[:, i + 1] - 30 * p[:, i] +
                                 16 * p[:, i - 1] - p[:, i - 2]) / (12 * dx**2)
            
            for j in range(2, nz - 2):
                d2p_dz2[j, :] = (-p[j + 2, :] + 16 * p[j + 1, :] - 30 * p[j, :] +
                                 16 * p[j - 1, :] - p[j - 2, :]) / (12 * dz**2)
            
            dp_dx = np.zeros((nz, nx), dtype=np.float64)
            dp_dz = np.zeros((nz, nx), dtype=np.float64)
            
            for i in range(1, nx - 1):
                dp_dx[:, i] = (p[:, i + 1] - p[:, i - 1]) / (2 * dx)
            
            for j in range(1, nz - 1):
                dp_dz[j, :] = (p[j + 1, :] - p[j - 1, :]) / (2 * dz)
            
            sigma_sum = sigma_x[np.newaxis, :] + sigma_z[:, np.newaxis]
            
            for i in range(nx):
                for j in range(nz):
                    sx = sigma_x[i]
                    sz = sigma_z[j]
                    
                    psi_x[j, i] = exp_sx[i] * psi_x[j, i] + sx * dp_dx[j, i]
                    psi_z[j, i] = exp_sz[j] * psi_z[j, i] + sz * dp_dz[j, i]
                    
                    phi_x[j, i] = exp_sx[i] * phi_x[j, i] + sx * d2p_dx2[j, i]
                    phi_z[j, i] = exp_sz[j] * phi_z[j, i] + sz * d2p_dz2[j, i]
            
            laplacian = d2p_dx2 + d2p_dz2 + phi_x + phi_z
            
            numerator = (2 * p - p_prev + v_sq * laplacian -
                        dt * sigma_sum * p_prev)
            denominator = 1 + 0.5 * dt * sigma_sum
            p_next = numerator / denominator
            
        else:
            d2p_dx2 = np.zeros((nz, nx), dtype=np.float64)
            d2p_dz2 = np.zeros((nz, nx), dtype=np.float64)
            
            for i in range(2, nx - 2):
                d2p_dx2[:, i] = (-p[:, i + 2] + 16 * p[:, i + 1] - 30 * p[:, i] +
                                 16 * p[:, i - 1] - p[:, i - 2]) / (12 * dx**2)
            
            for j in range(2, nz - 2):
                d2p_dz2[j, :] = (-p[j + 2, :] + 16 * p[j + 1, :] - 30 * p[j, :] +
                                 16 * p[j - 1, :] - p[j - 2, :]) / (12 * dz**2)
            
            p_next = 2 * p - p_prev + v_sq * (d2p_dx2 + d2p_dz2)
            
            _apply_mur_boundary(p_next, p_prev, velocity, dt, dx, dz, nx, nz)
        
        sx = min(max(params.source_x, 0), nx - 1)
        sz = min(max(params.source_z, 0), nz - 1)
        p_next[sz, sx] += wavelet[it] * dt**2
        
        for ri, rx in enumerate(receiver_x):
            rz = min(max(params.receiver_z, 0), nz - 1)
            seismograms[it, ri] = p[rz, rx]
        
        if it % snapshot_interval == 0:
            snapshots.append(p.copy())
        
        p_prev = p.copy()
        p = p_next.copy()
    
    result = {
        'seismograms': seismograms,
        'receiver_x': receiver_x,
        'receiver_z': params.receiver_z,
        'snapshots': snapshots,
        'snapshot_times': np.arange(0, nt, snapshot_interval) * dt,
        'wavelet': wavelet,
        'stability': stability,
        'params': params,
        'time': np.arange(nt) * dt
    }
    
    return result


def compute_residual(observed: np.ndarray, synthetic: np.ndarray) -> Dict:
    """Compute residual between observed and synthetic data."""
    if observed.shape != synthetic.shape:
        min_samples = min(observed.shape[0], synthetic.shape[0])
        min_traces = min(observed.shape[1], synthetic.shape[1])
        observed = observed[:min_samples, :min_traces]
        synthetic = synthetic[:min_samples, :min_traces]
    
    residual = observed - synthetic
    misfit = 0.5 * np.sum(residual**2)
    
    result = {
        'residual': residual,
        'misfit': misfit,
        'rms_error': np.sqrt(np.mean(residual**2)),
        'correlation': np.array([np.corrcoef(observed[:, i], synthetic[:, i])[0, 1]
                                for i in range(observed.shape[1])])
    }
    
    return result

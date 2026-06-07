import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from scipy.optimize import lsq_linear, minimize
from scipy.sparse import lil_matrix, csr_matrix

from .forward_modeling import ForwardParams, run_forward, compute_residual
from .travel_time import FastMarching, trace_ray_grid


@dataclass
class InversionParams:
    """Parameters for inversion algorithms."""
    max_iterations: int = 50
    convergence_threshold: float = 1e-4
    regularization: float = 0.01
    inversion_type: str = 'traveltime'
    frequency_scales: List[float] = field(default_factory=lambda: [5, 10, 20, 30])
    update_callback: Optional[Callable] = None
    verbose: bool = True


@dataclass
class InversionResult:
    """Result of inversion process."""
    initial_model: np.ndarray
    inverted_model: np.ndarray
    true_model: Optional[np.ndarray]
    objective_history: List[float]
    model_update_history: List[float]
    iterations: int
    converged: bool
    final_objective: float


def lsqr_solve(A: np.ndarray, b: np.ndarray, lambda_reg: float = 0.01,
               max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    LSQR solver for linear systems with regularization.
    
    Parameters:
    - A: Design matrix (m, n)
    - b: Data vector (m,)
    - lambda_reg: Regularization parameter
    - max_iter: Maximum iterations
    - tol: Convergence tolerance
    
    Returns:
    x: Solution vector (n,)
    """
    m, n = A.shape
    
    A_reg = np.vstack([A, lambda_reg * np.eye(n)])
    b_reg = np.hstack([b, np.zeros(n)])
    
    x = np.zeros(n)
    
    r = b_reg - A_reg @ x
    p = A_reg.T @ r
    r_norm = np.dot(r, r)
    p_norm = np.dot(p, p)
    
    for i in range(max_iter):
        Ap = A_reg @ p
        alpha = p_norm / np.dot(Ap, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        
        r_new_norm = np.dot(r, r)
        beta = r_new_norm / r_norm
        p = A_reg.T @ r + beta * p
        
        r_norm = r_new_norm
        p_norm = np.dot(p, p)
        
        if np.sqrt(r_norm) < tol:
            break
    
    return x


def build_traveltime_sensitivity(travel_time: np.ndarray, velocity: np.ndarray,
                                 dx: float, dz: float,
                                 source_idx: Tuple[int, int],
                                 receiver_idx: List[Tuple[int, int]]) -> csr_matrix:
    """
    Build sensitivity matrix for travel time tomography.
    
    Uses ray paths to compute Fréchet derivatives.
    """
    nz, nx = velocity.shape
    n_data = len(receiver_idx)
    n_model = nz * nx
    
    A = lil_matrix((n_data, n_model))
    
    for data_idx, (rx, rz) in enumerate(receiver_idx):
        ray = trace_ray_grid(travel_time, source_idx[0], source_idx[1], rx, rz, dx, dz)
        
        if ray is None or len(ray.x) < 2:
            continue
        
        for i in range(len(ray.x) - 1):
            x1, z1 = ray.x[i], ray.z[i]
            x2, z2 = ray.x[i + 1], ray.z[i + 1]
            
            x_mid = (x1 + x2) / 2
            z_mid = (z1 + z2) / 2
            
            xi = int(round(x_mid / dx))
            zi = int(round(z_mid / dz))
            
            xi = min(max(xi, 0), nx - 1)
            zi = min(max(zi, 0), nz - 1)
            
            seg_length = np.sqrt((x2 - x1)**2 + (z2 - z1)**2)
            v = velocity[zi, xi]
            
            model_idx = zi * nx + xi
            A[data_idx, model_idx] += -seg_length / (v**2)
    
    return A.tocsr()


def invert_traveltime(initial_velocity: np.ndarray, dx: float, dz: float,
                      sources: List[Tuple[int, int]],
                      receivers: List[Tuple[int, int]],
                      observed_times: np.ndarray,
                      params: InversionParams) -> InversionResult:
    """
    Travel time tomography using LSQR.
    
    Parameters:
    - initial_velocity: Initial velocity model (nz, nx)
    - dx, dz: Grid spacing
    - sources: List of source locations
    - receivers: List of receiver locations
    - observed_times: Observed travel times (n_sources, n_receivers)
    - params: Inversion parameters
    
    Returns:
    InversionResult object
    """
    nz, nx = initial_velocity.shape
    n_model = nz * nx
    n_sources = len(sources)
    n_receivers = len(receivers)
    n_data = n_sources * n_receivers
    
    current_velocity = initial_velocity.copy().astype(np.float64)
    
    objective_history = []
    model_update_history = []
    
    converged = False
    prev_objective = np.inf
    
    for iteration in range(params.max_iterations):
        all_sensitivities = []
        all_residuals = []
        
        fmm = FastMarching(current_velocity, dx, dz)
        
        total_residual = 0.0
        
        for source_idx, (sx, sz) in enumerate(sources):
            tau = fmm.solve(sx, sz)
            
            A_source = build_traveltime_sensitivity(tau, current_velocity, dx, dz,
                                                    (sx, sz), receivers)
            
            predicted_times = np.array([tau[rz, rx] for rx, rz in receivers])
            observed_times_source = observed_times[source_idx, :]
            
            residual = observed_times_source - predicted_times
            valid_mask = np.isfinite(residual) & np.isfinite(predicted_times)
            
            if np.any(valid_mask):
                residual_valid = residual[valid_mask]
                A_valid = A_source[valid_mask, :]
                
                all_sensitivities.append(A_valid)
                all_residuals.append(residual_valid)
                
                total_residual += np.sum(residual_valid**2)
        
        if not all_sensitivities:
            break
        
        A = lil_matrix((0, n_model))
        b = np.array([])
        
        for A_s, res in zip(all_sensitivities, all_residuals):
            A = lil_matrix(np.vstack([A.toarray(), A_s.toarray()]))
            b = np.hstack([b, res])
        
        A = A.tocsr()
        
        objective = 0.5 * total_residual
        objective_history.append(objective)
        
        relative_change = abs(prev_objective - objective) / (abs(prev_objective) + 1e-10)
        prev_objective = objective
        
        if params.verbose:
            print(f"Iteration {iteration+1}: Objective = {objective:.6e}, "
                  f"Relative change = {relative_change:.2e}")
        
        if iteration > 0 and relative_change < params.convergence_threshold:
            converged = True
            break
        
        if params.update_callback is not None:
            params.update_callback(iteration, objective, relative_change, current_velocity)
        
        lambda_reg = params.regularization
        delta_v = lsqr_solve(A.toarray(), b, lambda_reg, max_iter=100)
        
        delta_v = delta_v.reshape((nz, nx))
        
        update_norm = np.linalg.norm(delta_v)
        model_update_history.append(update_norm)
        
        current_velocity = current_velocity + delta_v
        current_velocity = np.maximum(current_velocity, 1000)
    
    return InversionResult(
        initial_model=initial_velocity,
        inverted_model=current_velocity,
        true_model=None,
        objective_history=objective_history,
        model_update_history=model_update_history,
        iterations=len(objective_history),
        converged=converged,
        final_objective=objective_history[-1] if objective_history else np.inf
    )


def compute_adjoint_gradient(velocity: np.ndarray, observed: np.ndarray,
                             forward_params: ForwardParams) -> np.ndarray:
    """
    Compute gradient using adjoint state method for waveform inversion.
    
    Parameters:
    - velocity: Current velocity model (nz, nx)
    - observed: Observed data (nt, n_receivers)
    - forward_params: Forward modeling parameters
    
    Returns:
    gradient: Gradient of misfit with respect to velocity (nz, nx)
    """
    forward_result = run_forward(velocity, forward_params)
    synthetic = forward_result['seismograms']
    snapshots = forward_result['snapshots']
    snapshot_times = forward_result['snapshot_times']
    
    nt, n_receivers = synthetic.shape
    nz, nx = velocity.shape
    
    min_samples = min(observed.shape[0], nt)
    min_receivers = min(observed.shape[1], n_receivers)
    observed = observed[:min_samples, :min_receivers]
    synthetic = synthetic[:min_samples, :min_receivers]
    
    residual = synthetic - observed
    
    adjoint_source = residual[::-1, :]
    
    dt = forward_params.dt
    dx = forward_params.dx
    dz = forward_params.dz
    
    p_adj = np.zeros((nz, nx), dtype=np.float32)
    p_adj_prev = np.zeros((nz, nx), dtype=np.float32)
    p_adj_next = np.zeros((nz, nx), dtype=np.float32)
    
    v_sq = (velocity * dt)**2
    
    receiver_x = forward_result['receiver_x'][:min_receivers]
    receiver_z = forward_result['receiver_z']
    
    gradient = np.zeros_like(velocity, dtype=np.float64)
    
    for it in range(min_samples):
        d2p_dx2 = np.zeros((nz, nx), dtype=np.float32)
        d2p_dz2 = np.zeros((nz, nx), dtype=np.float32)
        
        for i in range(2, nx - 2):
            d2p_dx2[:, i] = (-p_adj[:, i + 2] + 16 * p_adj[:, i + 1] - 30 * p_adj[:, i] +
                             16 * p_adj[:, i - 1] - p_adj[:, i - 2]) / (12 * dx**2)
        
        for j in range(2, nz - 2):
            d2p_dz2[j, :] = (-p_adj[j + 2, :] + 16 * p_adj[j + 1, :] - 30 * p_adj[j, :] +
                             16 * p_adj[j - 1, :] - p_adj[j - 2, :]) / (12 * dz**2)
        
        p_adj_next = 2 * p_adj - p_adj_prev + v_sq * (d2p_dx2 + d2p_dz2)
        
        for ri, rx in enumerate(receiver_x):
            rz = min(max(receiver_z, 0), nz - 1)
            rx_idx = min(max(rx, 0), nx - 1)
            p_adj_next[rz, rx_idx] += adjoint_source[it, ri] * dt**2
        
        snapshot_idx = np.argmin(np.abs(snapshot_times - (min_samples - 1 - it) * dt))
        if snapshot_idx < len(snapshots):
            forward_wavefield = snapshots[snapshot_idx]
            
            d2p_dt2 = (forward_wavefield - 2 * p_adj_prev + p_adj_prev) / (dt**2) if it > 1 else np.zeros_like(forward_wavefield)
            
            gradient += 2 * velocity * d2p_dt2 * p_adj * dt
        
        p_adj_prev = p_adj.copy()
        p_adj = p_adj_next.copy()
    
    return gradient


def _objective_function(model_vec: np.ndarray, nz: int, nx: int,
                        observed: np.ndarray, forward_params: ForwardParams) -> Tuple[float, np.ndarray]:
    """
    Objective function for L-BFGS optimization.
    
    Returns:
    misfit, gradient
    """
    velocity = model_vec.reshape((nz, nx))
    
    forward_result = run_forward(velocity, forward_params)
    synthetic = forward_result['seismograms']
    
    min_samples = min(observed.shape[0], synthetic.shape[0])
    min_receivers = min(observed.shape[1], synthetic.shape[1])
    
    residual = synthetic[:min_samples, :min_receivers] - observed[:min_samples, :min_receivers]
    misfit = 0.5 * np.sum(residual**2)
    
    gradient = compute_adjoint_gradient(velocity, observed, forward_params)
    
    return misfit, gradient.flatten()


def invert_waveform(initial_velocity: np.ndarray, observed: np.ndarray,
                    forward_params: ForwardParams,
                    params: InversionParams) -> InversionResult:
    """
    Full waveform inversion using L-BFGS with multi-scale strategy.
    
    Parameters:
    - initial_velocity: Initial velocity model (nz, nx)
    - observed: Observed data (nt, n_receivers)
    - forward_params: Forward modeling parameters
    - params: Inversion parameters
    
    Returns:
    InversionResult object
    """
    nz, nx = initial_velocity.shape
    
    from scipy.signal import butter, filtfilt
    
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)
    
    current_velocity = initial_velocity.copy()
    
    objective_history = []
    model_update_history = []
    
    fs = 1.0 / forward_params.dt
    converged = False
    
    for scale_idx, freq_max in enumerate(params.frequency_scales):
        if params.verbose:
            print(f"\nMulti-scale iteration {scale_idx+1}: "
                  f"Frequency band 0 - {freq_max} Hz")
        
        if freq_max < fs / 2:
            observed_filtered = bandpass_filter(observed, 0.5, freq_max * 0.9, fs)
        else:
            observed_filtered = observed.copy()
        
        prev_misfit = np.inf
        
        def callback(xk):
            pass
        
        def obj_func(x):
            misfit, grad = _objective_function(x, nz, nx, observed_filtered, forward_params)
            return misfit, grad.astype(np.float64)
        
        options = {
            'maxiter': params.max_iterations // len(params.frequency_scales),
            'ftol': params.convergence_threshold,
            'gtol': 1e-8,
            'disp': params.verbose
        }
        
        result = minimize(
            obj_func,
            current_velocity.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(1000, 8000)] * (nz * nx),
            options=options,
            callback=callback
        )
        
        new_velocity = result.x.reshape((nz, nx))
        
        update_norm = np.linalg.norm(new_velocity - current_velocity)
        model_update_history.append(update_norm)
        objective_history.append(result.fun)
        
        current_velocity = new_velocity
        
        relative_change = abs(prev_misfit - result.fun) / (abs(prev_misfit) + 1e-10)
        prev_misfit = result.fun
        
        if params.update_callback is not None:
            params.update_callback(scale_idx, result.fun, relative_change, current_velocity)
        
        if relative_change < params.convergence_threshold:
            converged = True
            break
    
    return InversionResult(
        initial_model=initial_velocity,
        inverted_model=current_velocity,
        true_model=None,
        objective_history=objective_history,
        model_update_history=model_update_history,
        iterations=len(objective_history),
        converged=converged,
        final_objective=objective_history[-1] if objective_history else np.inf
    )


def run_inversion(initial_velocity: np.ndarray, params: InversionParams,
                  forward_params: Optional[ForwardParams] = None,
                  observed_data: Optional[Dict] = None,
                  true_model: Optional[np.ndarray] = None) -> InversionResult:
    """
    Run inversion with specified type.
    
    Parameters:
    - initial_velocity: Initial velocity model
    - params: Inversion parameters
    - forward_params: Forward modeling parameters (for waveform inversion)
    - observed_data: Dictionary with observed data
    - true_model: Optional true model for comparison
    
    Returns:
    InversionResult object
    """
    result = None
    
    if params.inversion_type == 'traveltime':
        if observed_data is None or 'observed_times' not in observed_data:
            raise ValueError("Observed travel times required for traveltime inversion")
        
        sources = observed_data['sources']
        receivers = observed_data['receivers']
        observed_times = observed_data['observed_times']
        dx = observed_data.get('dx', 10.0)
        dz = observed_data.get('dz', 10.0)
        
        result = invert_traveltime(initial_velocity, dx, dz, sources, receivers,
                                   observed_times, params)
    
    elif params.inversion_type == 'waveform':
        if forward_params is None or observed_data is None or 'traces' not in observed_data:
            raise ValueError("Forward params and observed traces required for waveform inversion")
        
        result = invert_waveform(initial_velocity, observed_data['traces'],
                                 forward_params, params)
    
    else:
        raise ValueError(f"Unknown inversion type: {params.inversion_type}")
    
    if true_model is not None:
        result.true_model = true_model
    
    return result

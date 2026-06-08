import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
from scipy.optimize import root_scalar

from .velocity_model import VelocityModel, LayeredModel, GradientModel, GridModel
from .travel_time import FastMarching, compute_travel_times
from .inversion import (
    InversionParams, InversionResult, run_inversion,
    invert_traveltime
)


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    name: str
    scenario_type: str
    passed: bool
    true_model: np.ndarray
    inverted_model: np.ndarray
    absolute_error: float
    relative_error: float
    error_map: np.ndarray
    parameters: Dict
    metrics: Dict
    timestamp: float
    message: str = ""


@dataclass
class ConvergenceDiagnostics:
    """Convergence analysis diagnostics."""
    monotonic_convergence: bool
    step_size_decaying: bool
    rms_residual_ok: bool
    model_bounds_ok: bool
    max_gradient_ok: bool
    objective_history: List[float]
    model_update_history: List[float]
    final_rms: float
    velocity_range: Tuple[float, float]
    max_adjacent_gradient: float
    details: Dict


@dataclass
class SensitivityResult:
    """Parameter sensitivity analysis result."""
    parameter_name: str
    parameter_values: List[float]
    objective_values: List[float]
    rms_values: List[float]
    best_value: float
    best_index: int


@dataclass
class RegressionSnapshot:
    """Regression test snapshot."""
    test_name: str
    scenario_type: str
    timestamp: float
    parameters: Dict
    metrics: Dict
    hash: str

    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'scenario_type': self.scenario_type,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'hash': self.hash
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RegressionSnapshot':
        return cls(
            test_name=data['test_name'],
            scenario_type=data['scenario_type'],
            timestamp=data['timestamp'],
            parameters=data['parameters'],
            metrics=data['metrics'],
            hash=data['hash']
        )


@dataclass
class JointSensitivityResult:
    """Joint parameter sensitivity analysis result."""
    param1_name: str
    param2_name: str
    param1_values: List[float]
    param2_values: List[float]
    rms_matrix: np.ndarray
    objective_matrix: np.ndarray
    best_param1_value: float
    best_param2_value: float
    best_rms: float
    best_index: Tuple[int, int]
    single_best_rms: float
    completed_count: int
    total_count: int
    resolution_labels: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {
            'param1_name': self.param1_name,
            'param2_name': self.param2_name,
            'param1_values': self.param1_values,
            'param2_values': self.param2_values,
            'rms_matrix': self.rms_matrix.tolist(),
            'objective_matrix': self.objective_matrix.tolist(),
            'best_param1_value': self.best_param1_value,
            'best_param2_value': self.best_param2_value,
            'best_rms': self.best_rms,
            'best_index': list(self.best_index),
            'single_best_rms': self.single_best_rms,
            'completed_count': self.completed_count,
            'total_count': self.total_count,
            'resolution_labels': self.resolution_labels
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'JointSensitivityResult':
        return cls(
            param1_name=data['param1_name'],
            param2_name=data['param2_name'],
            param1_values=data['param1_values'],
            param2_values=data['param2_values'],
            rms_matrix=np.array(data['rms_matrix']),
            objective_matrix=np.array(data['objective_matrix']),
            best_param1_value=data['best_param1_value'],
            best_param2_value=data['best_param2_value'],
            best_rms=data['best_rms'],
            best_index=tuple(data['best_index']),
            single_best_rms=data['single_best_rms'],
            completed_count=data['completed_count'],
            total_count=data['total_count'],
            resolution_labels=data.get('resolution_labels')
        )


@dataclass
class RegressionResult:
    """Result of regression test comparison."""
    test_name: str
    status: str
    current_metrics: Dict
    baseline_metrics: Dict
    deviations: Dict
    message: str


def analytical_homogeneous(v: float, source_x: float, source_z: float,
                          receiver_x: np.ndarray, receiver_z: np.ndarray) -> np.ndarray:
    """
    Analytical travel time for homogeneous medium.
    t = sqrt((x_r - x_s)^2 + (z_r - z_s)^2) / v
    """
    dx = receiver_x - source_x
    dz = receiver_z - source_z
    dist = np.sqrt(dx**2 + dz**2)
    return dist / v


def analytical_two_layer(v1: float, v2: float, z_interface: float,
                        source_x: float, source_z: float,
                        receiver_x: np.ndarray, receiver_z: np.ndarray) -> np.ndarray:
    """
    Analytical travel time for two-layer medium with head wave.
    Uses Snell's law for refracted waves.
    Supports receivers at different depths.
    """
    travel_times = np.zeros_like(receiver_x, dtype=np.float64)

    for i, (rx, rz) in enumerate(zip(receiver_x, receiver_z)):
        dx = abs(rx - source_x)
        dz = rz - source_z

        both_in_upper = (source_z < z_interface and rz < z_interface)
        both_in_lower = (source_z >= z_interface and rz >= z_interface)

        if both_in_lower:
            t_direct = np.sqrt(dx**2 + dz**2) / v2
            travel_times[i] = t_direct
            continue

        z1 = max(0, z_interface - source_z)
        z2 = max(0, z_interface - rz)
        sin_i = v1 / v2
        cos_i = np.sqrt(1 - sin_i**2)
        x_refract = z1 * sin_i / cos_i + z2 * sin_i / cos_i
        x_crossover = x_refract * np.sqrt((v2 + v1) / (v2 - v1))

        if both_in_upper:
            t_direct = np.sqrt(dx**2 + dz**2) / v1
        else:
            z1_cross = abs(z_interface - source_z)
            z2_cross = abs(z_interface - rz)
            sin_cross = v1 / v2
            cos_cross = np.sqrt(1 - sin_cross**2)
            x_cross = z1_cross * sin_cross / cos_cross + z2_cross * sin_cross / cos_cross
            if dx < x_cross:
                t_direct = np.sqrt(dx**2 + dz**2) / v1
            else:
                t_direct = (z1_cross / (v1 * cos_cross) + 
                           z2_cross / (v1 * cos_cross) + 
                           (dx - x_cross) / v2)

        x_horizontal = dx - x_refract
        t_head = z1 / (v1 * cos_i) + z2 / (v1 * cos_i) + x_horizontal / v2

        if dx < x_crossover:
            travel_times[i] = t_direct
        else:
            travel_times[i] = min(t_direct, t_head)

    return travel_times


def analytical_gradient(v0: float, grad: float,
                       source_x: float, source_z: float,
                       receiver_x: np.ndarray, receiver_z: np.ndarray) -> np.ndarray:
    """
    Analytical travel time for linear gradient medium.
    v(z) = v0 + grad * z
    Uses ray parameter integral formula.
    Supports receivers at different depths.
    """
    travel_times = np.zeros_like(receiver_x, dtype=np.float64)

    for i, (rx, rz) in enumerate(zip(receiver_x, receiver_z)):
        dx = rx - source_x
        dz = rz - source_z

        vs = v0 + grad * source_z
        vr = v0 + grad * rz

        if abs(dz) < 1e-6:
            v_avg = (vs + vr) / 2
            travel_times[i] = np.sqrt(dx**2 + dz**2) / v_avg
            continue

        def ray_eq(p):
            z1, z2 = sorted([source_z, rz])

            def integrand(z):
                v = v0 + grad * z
                val = 1.0 / np.sqrt(v**2 - p**2)
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val

            n = 1000
            h = (z2 - z1) / n
            x_pred = 0.0
            t_pred = 0.0

            for j in range(n):
                z_mid = z1 + (j + 0.5) * h
                v = v0 + grad * z_mid
                denom = np.sqrt(v**2 - p**2)
                x_pred += p * h / denom
                t_pred += v * h / (denom * v**2)

            return x_pred - abs(dx), t_pred

        p_min = 1e-10
        p_max = min(vs, vr) / 1.001

        try:
            def f(p):
                return ray_eq(p)[0]

            result = root_scalar(f, bracket=[p_min, p_max], method='brentq')
            _, t = ray_eq(result.root)
            travel_times[i] = t
        except:
            v_avg = (vs + vr) / 2
            travel_times[i] = np.sqrt(dx**2 + dz**2) / v_avg

    return travel_times


class BenchmarkScenarios:
    """Class containing analytical benchmark scenarios."""

    @staticmethod
    def create_homogeneous(v: float = 2000.0, nx: int = 25, nz: int = 20,
                          dx: float = 20.0, dz: float = 20.0,
                          n_receivers: int = 16) -> Dict:
        """
        Create homogeneous medium benchmark scenario.
        Scenario 1: Uniform velocity 2000 m/s.
        """
        true_model = GridModel(nx, nz, dx, dz, default_velocity=v)
        true_model_np = true_model.velocity.copy()

        n_sources = 6
        sources = [(2, int(nz / (n_sources + 1) * (i + 1))) for i in range(n_sources)]
        receivers = [(nx - 3, int(nz / (n_receivers + 1) * (i + 1))) for i in range(n_receivers)]

        observed_times = np.zeros((len(sources), len(receivers)))
        for s_idx, (sx, sz) in enumerate(sources):
            source_coord_x = sx * dx
            source_coord_z = sz * dz
            receiver_coords_x = np.array([r[0] * dx for r in receivers])
            receiver_coords_z = np.array([r[1] * dz for r in receivers])
            t = analytical_homogeneous(v, source_coord_x, source_coord_z, receiver_coords_x, receiver_coords_z)
            observed_times[s_idx, :] = t

        initial_model = np.ones_like(true_model_np) * (v * 0.985)

        return {
            'name': '均匀介质基准测试',
            'scenario_type': 'homogeneous',
            'true_model': true_model_np,
            'initial_model': initial_model,
            'sources': sources,
            'receivers': receivers,
            'observed_times': observed_times,
            'dx': dx,
            'dz': dz,
            'analytical_function': lambda model: analytical_homogeneous(
                v, source_coord_x, source_coord_z, receiver_coords_x, receiver_coords_z
            ),
            'expected_velocity': v,
            'tolerance': 0.001,
            'description': f'均匀介质 (v={v}m/s) 旅行时反演，要求速度恢复误差<0.1%'
        }

    @staticmethod
    def create_two_layer(v1: float = 1500.0, v2: float = 3000.0,
                        z_interface: float = 200.0, nx: int = 30, nz: int = 25,
                        dx: float = 20.0, dz: float = 20.0,
                        n_receivers: int = 16) -> Dict:
        """
        Create two-layer medium benchmark scenario.
        Scenario 2: Upper layer 1500 m/s (200m), lower layer 3000 m/s.
        """
        layers = [
            {'depth': z_interface, 'velocity': v1},
            {'depth': z_interface + 1, 'velocity': v2}
        ]
        true_model = LayeredModel(nx, nz, dx, dz, layers)
        true_model_np = true_model.velocity.copy()

        n_sources = 6
        sources = [(2, int(nz / (n_sources + 1) * (i + 1))) for i in range(n_sources)]
        receivers = [(nx - 3, int(nz / (n_receivers + 1) * (i + 1))) for i in range(n_receivers)]

        from .travel_time import FastMarching
        fmm = FastMarching(true_model_np, dx, dz)
        observed_times = np.zeros((len(sources), len(receivers)))
        for s_idx, (sx, sz) in enumerate(sources):
            tau = fmm.solve(sx, sz)
            for r_idx, (rx, rz) in enumerate(receivers):
                observed_times[s_idx, r_idx] = tau[rz, rx]

        interface_idx = int(z_interface / dz)
        initial_model = np.ones_like(true_model_np) * 2000
        initial_model[:interface_idx, :] = v1 * 0.985
        initial_model[interface_idx:, :] = v2 * 0.985

        return {
            'name': '两层水平介质基准测试',
            'scenario_type': 'two_layer',
            'true_model': true_model_np,
            'initial_model': initial_model,
            'sources': sources,
            'receivers': receivers,
            'observed_times': observed_times,
            'dx': dx,
            'dz': dz,
            'expected_velocities': {'upper': v1, 'lower': v2},
            'interface_depth': z_interface,
            'tolerance': 0.02,
            'description': f'两层介质 (v1={v1}m/s, v2={v2}m/s, 界面={z_interface}m)，各层速度误差<2%'
        }

    @staticmethod
    def create_gradient(v0: float = 1500.0, grad: float = 0.5,
                       nx: int = 30, nz: int = 30,
                       dx: float = 20.0, dz: float = 20.0,
                       n_receivers: int = 16) -> Dict:
        """
        Create linear gradient medium benchmark scenario.
        Scenario 3: v(z) = 1500 + 0.5*z m/s.
        """
        z_max = (nz - 1) * dz
        v_bottom = v0 + grad * z_max
        true_model = GradientModel(nx, nz, dx, dz, v0, v_bottom, 'linear')
        true_model_np = true_model.velocity.copy()

        n_sources = 6
        sources = [(2, int(nz / (n_sources + 1) * (i + 1))) for i in range(n_sources)]
        receivers = [(nx - 3, int(nz / (n_receivers + 1) * (i + 1))) for i in range(n_receivers)]

        from .travel_time import FastMarching
        fmm = FastMarching(true_model_np, dx, dz)
        observed_times = np.zeros((len(sources), len(receivers)))
        for s_idx, (sx, sz) in enumerate(sources):
            tau = fmm.solve(sx, sz)
            for r_idx, (rx, rz) in enumerate(receivers):
                observed_times[s_idx, r_idx] = tau[rz, rx]

        initial_model = true_model_np * 0.985

        return {
            'name': '线性梯度介质基准测试',
            'scenario_type': 'gradient',
            'true_model': true_model_np,
            'initial_model': initial_model,
            'sources': sources,
            'receivers': receivers,
            'observed_times': observed_times,
            'dx': dx,
            'dz': dz,
            'expected_gradient': grad,
            'v0': v0,
            'tolerance': 0.05,
            'description': f'线性梯度介质 (v0={v0}m/s, grad={grad}/s)，速度梯度误差<5%'
        }


def run_benchmark(scenario: Dict, params: Optional[InversionParams] = None) -> BenchmarkResult:
    """
    Run a benchmark scenario and return results.

    Parameters:
    - scenario: Scenario dictionary from BenchmarkScenarios
    - params: Optional inversion parameters

    Returns:
    - BenchmarkResult with error analysis
    """
    if params is None:
        params = InversionParams(
            max_iterations=300,
            convergence_threshold=1e-12,
            regularization=0.03,
            inversion_type='traveltime',
            verbose=False
        )

    observed_data = {
        'sources': scenario['sources'],
        'receivers': scenario['receivers'],
        'observed_times': scenario['observed_times'],
        'dx': scenario['dx'],
        'dz': scenario['dz']
    }

    result = run_inversion(
        scenario['initial_model'],
        params,
        observed_data=observed_data
    )

    inverted_model = result.inverted_model
    true_model = scenario['true_model']

    error_map = inverted_model - true_model
    absolute_error = float(np.mean(np.abs(error_map)))
    relative_error = float(np.mean(np.abs(error_map) / (np.abs(true_model) + 1e-10)))

    passed = False
    message = ""

    if scenario['scenario_type'] == 'homogeneous':
        avg_v = float(np.mean(inverted_model))
        expected_v = scenario['expected_velocity']
        rel_error = abs(avg_v - expected_v) / expected_v
        passed = rel_error < scenario['tolerance']
        message = f"平均速度: {avg_v:.2f} m/s, 预期: {expected_v:.2f} m/s, 相对误差: {rel_error*100:.4f}%"

    elif scenario['scenario_type'] == 'two_layer':
        interface_idx = int(scenario['interface_depth'] / scenario['dz'])
        v1_inv = float(np.mean(inverted_model[:interface_idx, :]))
        v2_inv = float(np.mean(inverted_model[interface_idx:, :]))
        v1_true = scenario['expected_velocities']['upper']
        v2_true = scenario['expected_velocities']['lower']
        err1 = abs(v1_inv - v1_true) / v1_true
        err2 = abs(v2_inv - v2_true) / v2_true
        passed = err1 < scenario['tolerance'] and err2 < scenario['tolerance']
        message = (f"上层速度: {v1_inv:.2f} m/s (预期: {v1_true:.2f}, 误差: {err1*100:.2f}%), "
                   f"下层速度: {v2_inv:.2f} m/s (预期: {v2_true:.2f}, 误差: {err2*100:.2f}%)")

    elif scenario['scenario_type'] == 'gradient':
        z = np.arange(scenario['dz'], inverted_model.shape[0] * scenario['dz'], scenario['dz'])
        v_profile = np.mean(inverted_model, axis=1)[1:]
        if len(v_profile) > 1:
            grad_inv = float(np.polyfit(z, v_profile, 1)[0])
        else:
            grad_inv = 0.0
        grad_true = scenario['expected_gradient']
        rel_error = abs(grad_inv - grad_true) / abs(grad_true + 1e-10)
        passed = rel_error < scenario['tolerance']
        message = f"速度梯度: {grad_inv:.4f}/s, 预期: {grad_true:.4f}/s, 相对误差: {rel_error*100:.2f}%"

    metrics = {
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'final_objective': result.final_objective,
        'iterations': result.iterations,
        'converged': result.converged
    }

    return BenchmarkResult(
        name=scenario['name'],
        scenario_type=scenario['scenario_type'],
        passed=passed,
        true_model=true_model,
        inverted_model=inverted_model,
        absolute_error=absolute_error,
        relative_error=relative_error,
        error_map=error_map,
        parameters=asdict(params),
        metrics=metrics,
        timestamp=time.time(),
        message=message
    )


def analyze_convergence(result: InversionResult,
                        rms_threshold: float = 0.01,
                        v_min: float = 1000.0,
                        v_max: float = 8000.0,
                        max_gradient: float = 500.0) -> ConvergenceDiagnostics:
    """
    Analyze convergence diagnostics from inversion result.

    Parameters:
    - result: InversionResult object
    - rms_threshold: Maximum allowed RMS residual
    - v_min: Minimum allowed velocity
    - v_max: Maximum allowed velocity
    - max_gradient: Maximum allowed adjacent grid velocity difference

    Returns:
    - ConvergenceDiagnostics with all checks
    """
    obj_history = result.objective_history
    update_history = result.model_update_history

    is_monotonic = True
    for i in range(1, len(obj_history)):
        if obj_history[i] > obj_history[i-1] * 1.001:
            is_monotonic = False
            break

    if len(update_history) > 3:
        first_half = np.mean(update_history[:len(update_history)//2])
        second_half = np.mean(update_history[len(update_history)//2:])
        is_decaying = second_half < first_half
    else:
        is_decaying = True

    inverted = result.inverted_model
    v_range = (float(np.min(inverted)), float(np.max(inverted)))
    bounds_ok = v_range[0] >= v_min and v_range[1] <= v_max

    grad_x = np.abs(np.diff(inverted, axis=1))
    grad_z = np.abs(np.diff(inverted, axis=0))
    max_adj_grad = float(max(np.max(grad_x) if grad_x.size > 0 else 0,
                              np.max(grad_z) if grad_z.size > 0 else 0))
    gradient_ok = max_adj_grad <= max_gradient

    final_rms = float(np.sqrt(result.final_objective * 2 / max(1, len(obj_history))))
    rms_ok = final_rms < rms_threshold

    details = {
        'non_monotonic_points': [],
        'update_ratio': 0.0
    }

    if not is_monotonic:
        for i in range(1, len(obj_history)):
            if obj_history[i] > obj_history[i-1] * 1.001:
                details['non_monotonic_points'].append(i)

    if len(update_history) > 3:
        details['update_ratio'] = float(second_half / (first_half + 1e-10))

    return ConvergenceDiagnostics(
        monotonic_convergence=is_monotonic,
        step_size_decaying=is_decaying,
        rms_residual_ok=rms_ok,
        model_bounds_ok=bounds_ok,
        max_gradient_ok=gradient_ok,
        objective_history=obj_history,
        model_update_history=update_history,
        final_rms=final_rms,
        velocity_range=v_range,
        max_adjacent_gradient=max_adj_grad,
        details=details
    )


def parameter_sensitivity(scenario: Dict,
                         param_name: str,
                         param_values: List[float],
                         base_params: Optional[InversionParams] = None) -> SensitivityResult:
    """
    Run parameter sensitivity analysis by scanning one parameter.

    Parameters:
    - scenario: Benchmark scenario dict
    - param_name: Name of parameter to scan ('regularization', 'max_iterations')
    - param_values: List of parameter values to test
    - base_params: Base inversion parameters

    Returns:
    - SensitivityResult with objective and RMS values for each parameter
    """
    if base_params is None:
        base_params = InversionParams(
            max_iterations=50,
            convergence_threshold=1e-5,
            regularization=0.01,
            inversion_type='traveltime',
            verbose=False
        )

    objectives = []
    rms_values = []

    for val in param_values:
        params = InversionParams(
            max_iterations=base_params.max_iterations,
            convergence_threshold=base_params.convergence_threshold,
            regularization=base_params.regularization,
            inversion_type=base_params.inversion_type,
            verbose=False
        )

        if param_name == 'regularization':
            params.regularization = float(val)
        elif param_name == 'max_iterations':
            params.max_iterations = int(val)

        result = run_benchmark(scenario, params)
        objectives.append(result.metrics['final_objective'])
        rms_values.append(float(np.sqrt(result.metrics['final_objective'] * 2 / max(1, result.metrics['iterations']))))

    best_idx = int(np.argmin(rms_values))

    return SensitivityResult(
        parameter_name=param_name,
        parameter_values=param_values,
        objective_values=objectives,
        rms_values=rms_values,
        best_value=param_values[best_idx],
        best_index=best_idx
    )


def grid_sensitivity(scenario_template: Dict,
                     resolutions: List[Tuple[int, int, float, float]],
                     base_params: Optional[InversionParams] = None) -> List[SensitivityResult]:
    """
    Run sensitivity analysis on grid resolution.

    Parameters:
    - scenario_template: Template scenario dict (used for scenario type)
    - resolutions: List of (nx, nz, dx, dz) tuples
    - base_params: Base inversion parameters

    Returns:
    - List of SensitivityResult for each resolution
    """
    scenario_type = scenario_template['scenario_type']
    results = []
    rms_values = []
    objectives = []

    for (nx, nz, dx, dz) in resolutions:
        if scenario_type == 'homogeneous':
            scenario = BenchmarkScenarios.create_homogeneous(nx=nx, nz=nz, dx=dx, dz=dz)
        elif scenario_type == 'two_layer':
            scenario = BenchmarkScenarios.create_two_layer(nx=nx, nz=nz, dx=dx, dz=dz)
        elif scenario_type == 'gradient':
            scenario = BenchmarkScenarios.create_gradient(nx=nx, nz=nz, dx=dx, dz=dz)
        else:
            continue

        result = parameter_sensitivity(scenario, 'regularization', [base_params.regularization if base_params else 0.01], base_params)
        objectives.append(result.objective_values[0])
        rms_values.append(result.rms_values[0])

    res_labels = [f"{r[0]}x{r[1]}" for r in resolutions]

    results.append(SensitivityResult(
        parameter_name='grid_resolution',
        parameter_values=[float(i) for i in range(len(resolutions))],
        objective_values=objectives,
        rms_values=rms_values,
        best_value=float(np.argmin(rms_values)),
        best_index=int(np.argmin(rms_values))
    ))

    results[-1].resolution_labels = res_labels

    return results


def full_sensitivity_analysis(scenario: Dict,
                              base_params: Optional[InversionParams] = None,
                              progress_callback: Optional[Callable] = None) -> Dict:
    """
    Run complete parameter sensitivity analysis with heatmap matrix.

    Parameters:
    - scenario: Base benchmark scenario
    - base_params: Base inversion parameters
    - progress_callback: Optional callback(progress, message)

    Returns:
    - Dict with all sensitivity results including heatmap data
    """
    if base_params is None:
        base_params = InversionParams(
            max_iterations=50,
            convergence_threshold=1e-5,
            regularization=0.01,
            inversion_type='traveltime',
            verbose=False
        )

    reg_values = np.logspace(-4, 0, 7).tolist()
    iterations = [10, 20, 50, 100, 200]
    resolutions = [
        (20, 15, 25.0, 25.0),
        (30, 24, 17.0, 17.0),
        (50, 40, 10.0, 10.0),
        (70, 56, 7.0, 7.0),
        (100, 80, 5.0, 5.0)
    ]

    n_total = len(reg_values) + len(iterations) + len(resolutions) * len(reg_values)
    current = 0

    reg_result = parameter_sensitivity(scenario, 'regularization', reg_values, base_params)
    current += len(reg_values)
    if progress_callback:
        progress_callback(current / n_total, "完成正则化系数扫描")

    iter_result = parameter_sensitivity(scenario, 'max_iterations', iterations, base_params)
    current += len(iterations)
    if progress_callback:
        progress_callback(current / n_total, "完成迭代次数扫描")

    heatmap_rms = np.zeros((len(reg_values), len(resolutions)))
    heatmap_obj = np.zeros((len(reg_values), len(resolutions)))

    for j, res in enumerate(resolutions):
        nx, nz, dx, dz = res
        if scenario['scenario_type'] == 'homogeneous':
            scenario_res = BenchmarkScenarios.create_homogeneous(nx=nx, nz=nz, dx=dx, dz=dz)
        elif scenario['scenario_type'] == 'two_layer':
            scenario_res = BenchmarkScenarios.create_two_layer(nx=nx, nz=nz, dx=dx, dz=dz)
        elif scenario['scenario_type'] == 'gradient':
            scenario_res = BenchmarkScenarios.create_gradient(nx=nx, nz=nz, dx=dx, dz=dz)
        else:
            continue

        for i, reg in enumerate(reg_values):
            params = InversionParams(
                max_iterations=base_params.max_iterations,
                convergence_threshold=base_params.convergence_threshold,
                regularization=float(reg),
                inversion_type='traveltime',
                verbose=False
            )
            result = run_benchmark(scenario_res, params)
            rms = np.sqrt(result.metrics['final_objective'] * 2 / max(1, result.metrics['iterations']))
            heatmap_rms[i, j] = rms
            heatmap_obj[i, j] = result.metrics['final_objective']

            current += 1
            if progress_callback:
                progress_callback(current / n_total,
                                f"热力图矩阵: 正则化 {reg:.1e}, 分辨率 {nx}x{nz}")

    best_idx = np.unravel_index(np.argmin(heatmap_rms), heatmap_rms.shape)
    best_params = {
        'regularization': reg_values[best_idx[0]],
        'resolution': resolutions[best_idx[1]],
        'rms': float(heatmap_rms[best_idx])
    }

    return {
        'regularization': reg_result,
        'iterations': iter_result,
        'heatmap_rms': heatmap_rms,
        'heatmap_obj': heatmap_obj,
        'reg_values': reg_values,
        'resolutions': resolutions,
        'resolution_labels': [f"{r[0]}x{r[1]}" for r in resolutions],
        'best_params': best_params
    }


class RegressionTestFramework:
    """Framework for managing regression tests and snapshots."""

    def __init__(self, snapshot_dir: str = './snapshots'):
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)

    def _get_snapshot_path(self, test_name: str) -> str:
        safe_name = test_name.replace(' ', '_').replace('/', '_')
        return os.path.join(self.snapshot_dir, f'{safe_name}.json')

    def save_snapshot(self, result: BenchmarkResult) -> RegressionSnapshot:
        """
        Save a benchmark result as a regression snapshot.

        Parameters:
        - result: BenchmarkResult to save

        Returns:
        - RegressionSnapshot saved
        """
        import hashlib
        metrics_str = json.dumps(result.metrics, sort_keys=True)
        params_str = json.dumps(result.parameters, sort_keys=True)
        hash_str = hashlib.md5((metrics_str + params_str).encode()).hexdigest()

        snapshot = RegressionSnapshot(
            test_name=result.name,
            scenario_type=result.scenario_type,
            timestamp=result.timestamp,
            parameters=result.parameters,
            metrics=result.metrics,
            hash=hash_str
        )

        path = self._get_snapshot_path(result.name)
        with open(path, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        return snapshot

    def load_snapshot(self, test_name: str) -> Optional[RegressionSnapshot]:
        """Load a saved snapshot."""
        path = self._get_snapshot_path(test_name)
        if not os.path.exists(path):
            return None

        with open(path, 'r') as f:
            data = json.load(f)

        return RegressionSnapshot.from_dict(data)

    def list_snapshots(self) -> List[str]:
        """List all saved snapshot test names."""
        if not os.path.exists(self.snapshot_dir):
            return []

        files = [f for f in os.listdir(self.snapshot_dir) if f.endswith('.json')]
        names = []
        for f in files:
            path = os.path.join(self.snapshot_dir, f)
            try:
                with open(path, 'r') as fp:
                    data = json.load(fp)
                    names.append(data['test_name'])
            except:
                continue
        return sorted(names)

    def compare_result(self, current: BenchmarkResult,
                       tolerance: float = 0.05) -> RegressionResult:
        """
        Compare current result with saved snapshot.

        Parameters:
        - current: Current BenchmarkResult
        - tolerance: Allowed relative deviation (default 5%)

        Returns:
        - RegressionResult with comparison
        """
        baseline = self.load_snapshot(current.name)

        if baseline is None:
            return RegressionResult(
                test_name=current.name,
                status='NO_BASELINE',
                current_metrics=current.metrics,
                baseline_metrics={},
                deviations={},
                message='无基准快照，首次运行请保存快照'
            )

        deviations = {}
        all_ok = True

        for key in ['absolute_error', 'relative_error', 'final_objective']:
            if key in baseline.metrics and key in current.metrics:
                base_val = baseline.metrics[key]
                curr_val = current.metrics[key]

                if abs(base_val) < 1e-10:
                    rel_dev = abs(curr_val - base_val) / (abs(base_val) + 1e-10)
                else:
                    rel_dev = abs(curr_val - base_val) / abs(base_val)

                deviations[key] = float(rel_dev)

                if rel_dev > tolerance:
                    all_ok = False

        if all_ok:
            status = 'PASSED'
            message = '所有指标偏差在容忍范围内'
        else:
            status = 'FAILED'
            message = '检测到回归问题，部分指标偏差超过容忍度'

        return RegressionResult(
            test_name=current.name,
            status=status,
            current_metrics=current.metrics,
            baseline_metrics=baseline.metrics,
            deviations=deviations,
            message=message
        )

    def run_all_tests(self,
                     scenarios: Optional[List[Dict]] = None,
                     params: Optional[InversionParams] = None,
                     tolerance: float = 0.05,
                     progress_callback: Optional[Callable] = None) -> List[RegressionResult]:
        """
        Run all regression tests.

        Parameters:
        - scenarios: Optional list of scenarios (if None, runs all saved scenarios)
        - params: Inversion parameters
        - tolerance: Deviation tolerance
        - progress_callback: Optional callback(progress, message)

        Returns:
        - List of RegressionResult for each test
        """
        if scenarios is None:
            saved_names = self.list_snapshots()
            scenarios = []
            for name in saved_names:
                if '均匀' in name:
                    scenarios.append(BenchmarkScenarios.create_homogeneous())
                elif '两层' in name:
                    scenarios.append(BenchmarkScenarios.create_two_layer())
                elif '梯度' in name:
                    scenarios.append(BenchmarkScenarios.create_gradient())

        results = []
        n_total = len(scenarios)

        for i, scenario in enumerate(scenarios):
            if progress_callback:
                progress_callback(i / n_total, f"正在运行: {scenario['name']}")

            bench_result = run_benchmark(scenario, params)
            reg_result = self.compare_result(bench_result, tolerance)
            results.append(reg_result)

            if progress_callback:
                progress_callback((i + 1) / n_total,
                                f"完成: {scenario['name']} - {reg_result.status}")

        return results

    def save_joint_sensitivity_snapshot(self, result: JointSensitivityResult, scenario_name: str) -> RegressionSnapshot:
        """
        Save a joint sensitivity analysis result as a regression snapshot.

        Parameters:
        - result: JointSensitivityResult to save
        - scenario_name: Name of the benchmark scenario

        Returns:
        - RegressionSnapshot saved
        """
        import hashlib

        test_name = f"{scenario_name}_joint_{result.param1_name}_{result.param2_name}"
        scenario_type = "sensitivity_joint"

        params_dict = {
            'param1_name': result.param1_name,
            'param2_name': result.param2_name,
            'param1_values': result.param1_values,
            'param2_values': result.param2_values,
            'best_param1_value': result.best_param1_value,
            'best_param2_value': result.best_param2_value,
            'best_rms': result.best_rms,
            'completed_count': result.completed_count,
            'total_count': result.total_count
        }

        metrics_dict = {
            'best_rms': result.best_rms,
            'single_best_rms': result.single_best_rms,
            'improvement_percent': float(
                (result.single_best_rms - result.best_rms) / max(1e-10, abs(result.single_best_rms)) * 100 if result.single_best_rms != np.inf else 0.0
            ),
            'completed_count': result.completed_count,
            'total_count': result.total_count,
            'rms_matrix_mean': float(np.mean(np.isfinite(result.rms_matrix))),
            'rms_matrix_min': float(np.min(np.isfinite(result.rms_matrix))),
            'rms_matrix_max': float(np.max(np.isfinite(result.rms_matrix)))
        }

        metrics_str = json.dumps(metrics_dict, sort_keys=True, default=str)
        params_str = json.dumps(params_dict, sort_keys=True, default=str)
        hash_str = hashlib.md5((metrics_str + params_str).encode()).hexdigest()

        snapshot = RegressionSnapshot(
            test_name=test_name,
            scenario_type=scenario_type,
            timestamp=time.time(),
            parameters=params_dict,
            metrics=metrics_dict,
            hash=hash_str
        )

        path = self._get_snapshot_path(test_name)
        with open(path, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)

        return snapshot

    def delete_snapshot(self, test_name: str) -> bool:
        """Delete a saved snapshot."""
        path = self._get_snapshot_path(test_name)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


def joint_sensitivity_analysis(scenario: Dict,
                                param1_name: str,
                                param2_name: str,
                                base_params: Optional[InversionParams] = None,
                                progress_callback: Optional[Callable] = None,
                                timeout_seconds: float = 120.0,
                                stop_flag: Optional[Callable[[], bool]] = None) -> JointSensitivityResult:
    """
    Run joint parameter sensitivity analysis by scanning two parameters simultaneously.

    Parameters:
    - scenario: Base benchmark scenario
    - param1_name: First parameter name ('regularization', 'grid_resolution', 'max_iterations')
    - param2_name: Second parameter name ('regularization', 'grid_resolution', 'max_iterations')
    - base_params: Base inversion parameters
    - progress_callback: Optional callback(progress, message)
    - timeout_seconds: Maximum time allowed for the analysis
    - stop_flag: Optional function that returns True to stop the analysis

    Returns:
    - JointSensitivityResult with RMS matrix and optimal parameters
    """
    if base_params is None:
        base_params = InversionParams(
            max_iterations=50,
            convergence_threshold=1e-5,
            regularization=0.01,
            inversion_type='traveltime',
            verbose=False
        )

    light_params = InversionParams(
        max_iterations=min(base_params.max_iterations, 30),
        convergence_threshold=max(base_params.convergence_threshold, 1e-4),
        regularization=base_params.regularization,
        inversion_type='traveltime',
        verbose=False
    )

    param_values = {
        'regularization': np.logspace(-3, 0, 5).tolist(),
        'max_iterations': [10, 20, 30, 50, 80],
        'grid_resolution': [
            (15, 12, 35.0, 35.0),
            (20, 16, 25.0, 25.0),
            (25, 20, 20.0, 20.0),
            (30, 24, 17.0, 17.0),
            (40, 32, 12.5, 12.5)
        ]
    }

    scenario_type = scenario['scenario_type']

    def get_scenario_for_resolution(res_tuple):
        nx, nz, dx, dz = res_tuple
        if scenario_type == 'homogeneous':
            return BenchmarkScenarios.create_homogeneous(nx=nx, nz=nz, dx=dx, dz=dz)
        elif scenario_type == 'two_layer':
            return BenchmarkScenarios.create_two_layer(nx=nx, nz=nz, dx=dx, dz=dz)
        elif scenario_type == 'gradient':
            return BenchmarkScenarios.create_gradient(nx=nx, nz=nz, dx=dx, dz=dz)
        return scenario

    def get_display_values(param_name, values):
        if param_name == 'grid_resolution':
            return [f"{v[0]}x{v[1]}" for v in values], [float(i) for i in range(len(values))]
        else:
            return [str(v) for v in values], values

    p1_values = param_values[param1_name]
    p2_values = param_values[param2_name]
    n1 = len(p1_values)
    n2 = len(p2_values)
    total = n1 * n2

    _, p1_display = get_display_values(param1_name, p1_values)
    _, p2_display = get_display_values(param2_name, p2_values)

    rms_matrix = np.full((n1, n2), np.inf)
    objective_matrix = np.full((n1, n2), np.inf)

    start_time = time.time()
    completed = 0

    single_best_rms = np.inf

    def should_stop():
        if stop_flag and stop_flag():
            return True
        if time.time() - start_time > timeout_seconds:
            return True
        return False

    for i in range(n1):
        for j in range(n2):
            if should_stop():
                break

            p1_val = p1_values[i]
            p2_val = p2_values[j]

            current_scenario = scenario
            params = InversionParams(
                max_iterations=light_params.max_iterations,
                convergence_threshold=light_params.convergence_threshold,
                regularization=light_params.regularization,
                inversion_type='traveltime',
                verbose=False
            )

            if param1_name == 'regularization':
                params.regularization = float(p1_val)
            elif param1_name == 'max_iterations':
                params.max_iterations = int(p1_val)
            elif param1_name == 'grid_resolution':
                current_scenario = get_scenario_for_resolution(p1_val)

            if param2_name == 'regularization':
                params.regularization = float(p2_val)
            elif param2_name == 'max_iterations':
                params.max_iterations = int(p2_val)
            elif param2_name == 'grid_resolution':
                current_scenario = get_scenario_for_resolution(p2_val)

            try:
                result = run_benchmark(current_scenario, params)
                rms = float(np.sqrt(result.metrics['final_objective'] * 2 / max(1, result.metrics['iterations'])))
                rms_matrix[i, j] = rms
                objective_matrix[i, j] = result.metrics['final_objective']

                if rms < single_best_rms:
                    single_best_rms = rms
            except Exception:
                rms_matrix[i, j] = np.inf
                objective_matrix[i, j] = np.inf

            completed += 1

            if progress_callback:
                p1_label = f"{p1_val}" if param1_name != 'grid_resolution' else f"{p1_val[0]}x{p1_val[1]}"
                p2_label = f"{p2_val}" if param2_name != 'grid_resolution' else f"{p2_val[0]}x{p2_val[1]}"
                progress_callback(
                    completed / total,
                    f"当前第 {completed}/{total} 组: {param1_name}={p1_label}, {param2_name}={p2_label}"
                )

        if should_stop():
            break

    valid_mask = np.isfinite(rms_matrix)
    if np.any(valid_mask):
        masked_rms = np.where(valid_mask, rms_matrix, np.inf)
        best_idx = np.unravel_index(np.argmin(masked_rms), rms_matrix.shape)
        best_rms = float(rms_matrix[best_idx])
        best_p1 = p1_values[best_idx[0]]
        best_p2 = p2_values[best_idx[1]]
    else:
        best_idx = (0, 0)
        best_rms = np.inf
        best_p1 = p1_values[0]
        best_p2 = p2_values[0]

    resolution_labels = None
    if param1_name == 'grid_resolution':
        resolution_labels = [f"{r[0]}x{r[1]}" for r in p1_values]
    elif param2_name == 'grid_resolution':
        resolution_labels = [f"{r[0]}x{r[1]}" for r in p2_values]

    return JointSensitivityResult(
        param1_name=param1_name,
        param2_name=param2_name,
        param1_values=p1_display,
        param2_values=p2_display,
        rms_matrix=rms_matrix,
        objective_matrix=objective_matrix,
        best_param1_value=best_p1 if not isinstance(best_p1, tuple) else float(best_idx[0]),
        best_param2_value=best_p2 if not isinstance(best_p2, tuple) else float(best_idx[1]),
        best_rms=best_rms,
        best_index=best_idx,
        single_best_rms=single_best_rms,
        completed_count=completed,
        total_count=total,
        resolution_labels=resolution_labels
    )


def get_all_benchmark_scenarios() -> List[Dict]:
    """Get all three benchmark scenarios."""
    return [
        BenchmarkScenarios.create_homogeneous(),
        BenchmarkScenarios.create_two_layer(),
        BenchmarkScenarios.create_gradient()
    ]

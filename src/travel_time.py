import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RayPath:
    """Dataclass to store ray path information."""
    x: np.ndarray
    z: np.ndarray
    travel_time: float
    segments: List[Dict]


class FastMarching:
    """Fast Marching Method for solving the Eikonal equation."""
    
    def __init__(self, velocity: np.ndarray, dx: float, dz: float):
        self.velocity = velocity.astype(np.float64)
        self.nz, self.nx = velocity.shape
        self.dx = dx
        self.dz = dz
        self.slowness = 1.0 / self.velocity
    
    def _update_point(self, tau: np.ndarray, i: int, j: int) -> float:
        """Update travel time at a grid point using first-order scheme."""
        candidates = []
        
        if i > 0:
            t1 = tau[j, i - 1]
            s1 = self.slowness[j, i]
            candidates.append((t1, self.dx, s1))
        if i < self.nx - 1:
            t2 = tau[j, i + 1]
            s2 = self.slowness[j, i]
            candidates.append((t2, self.dx, s2))
        if j > 0:
            t3 = tau[j - 1, i]
            s3 = self.slowness[j, i]
            candidates.append((t3, self.dz, s3))
        if j < self.nz - 1:
            t4 = tau[j + 1, i]
            s4 = self.slowness[j, i]
            candidates.append((t4, self.dz, s4))
        
        if len(candidates) < 1:
            return np.inf
        
        candidates.sort(key=lambda x: x[0])
        
        s_current = self.slowness[j, i]
        
        if len(candidates) >= 2:
            t1, d1, s1 = candidates[0]
            t2, d2, s2 = candidates[1]
            
            a = 1 / d1**2 + 1 / d2**2
            b = -2 * (t1 / d1**2 + t2 / d2**2)
            c = t1**2 / d1**2 + t2**2 / d2**2 - s_current**2
            
            with np.errstate(invalid='ignore'):
                discriminant = b**2 - 4 * a * c
            
            if not np.isnan(discriminant) and discriminant >= -1e-15:
                discriminant = max(discriminant, 0.0)
                t_new = (-b + np.sqrt(discriminant)) / (2 * a)
                if not np.isnan(t_new) and t_new > max(t1, t2) - 1e-15:
                    return t_new
        
        t1, d1, s1 = candidates[0]
        return t1 + s1 * d1
    
    def solve(self, source_x: int, source_z: int) -> np.ndarray:
        """
        Solve Eikonal equation from source location.
        
        Parameters:
        - source_x: x-index of source
        - source_z: z-index of source
        
        Returns:
        tau: Travel time field (nz, nx)
        """
        source_x = min(max(source_x, 0), self.nx - 1)
        source_z = min(max(source_z, 0), self.nz - 1)
        
        tau = np.full((self.nz, self.nx), np.inf, dtype=np.float64)
        tau[source_z, source_x] = 0.0
        
        alive = np.zeros((self.nz, self.nx), dtype=bool)
        narrow_band = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                i, j = source_x + di, source_z + dj
                if 0 <= i < self.nx and 0 <= j < self.nz:
                    dist = np.sqrt((di * self.dx)**2 + (dj * self.dz)**2)
                    t = self.slowness[j, i] * dist
                    tau[j, i] = t
                    heapq.heappush(narrow_band, (t, i, j))
        
        while narrow_band:
            t_min, i, j = heapq.heappop(narrow_band)
            
            if alive[j, i]:
                continue
            
            alive[j, i] = True
            
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.nx and 0 <= nj < self.nz and not alive[nj, ni]:
                    t_new = self._update_point(tau, ni, nj)
                    if t_new < tau[nj, ni]:
                        tau[nj, ni] = t_new
                        heapq.heappush(narrow_band, (t_new, ni, nj))
        
        return tau


def trace_ray_layered(layers: List[Dict], source_x: float, source_z: float,
                      receiver_x: float, receiver_z: float,
                      dx: float, dz: float) -> Optional[RayPath]:
    """
    Trace a ray through layered media using Snell's law.
    
    Parameters:
    - layers: List of layer dicts with 'depth' and 'velocity'
    - source_x, source_z: Source position (in meters)
    - receiver_x, receiver_z: Receiver position (in meters)
    
    Returns:
    RayPath object or None if no valid path
    """
    layers_sorted = sorted(layers, key=lambda x: x['depth'])
    
    v_layers = [layers_sorted[0]['velocity']] + [l['velocity'] for l in layers_sorted]
    z_interfaces = [0] + [l['depth'] for l in layers_sorted]
    
    x_source = source_x
    z_source = source_z
    x_receiver = receiver_x
    z_receiver = receiver_z
    
    source_layer = np.searchsorted(z_interfaces, z_source, side='right') - 1
    receiver_layer = np.searchsorted(z_interfaces, z_receiver, side='right') - 1
    
    if source_layer == receiver_layer:
        dx_dist = x_receiver - x_source
        dz_dist = z_receiver - z_source
        dist = np.sqrt(dx_dist**2 + dz_dist**2)
        travel_time = dist / v_layers[source_layer]
        
        return RayPath(
            x=np.array([x_source, x_receiver]),
            z=np.array([z_source, z_receiver]),
            travel_time=travel_time,
            segments=[{'x1': x_source, 'z1': z_source, 'x2': x_receiver, 'z2': z_receiver,
                       'velocity': v_layers[source_layer], 'time': travel_time}]
        )
    
    x_offset = abs(x_receiver - x_source)
    direction = 1 if x_receiver >= x_source else -1
    
    def compute_trajectory(p0: float) -> Tuple[float, List[Tuple[float, float]]]:
        """Compute trajectory given initial ray parameter p = sin(theta)/v."""
        points = [(x_source, z_source)]
        total_time = 0.0
        
        x_current = x_source
        z_current = z_source
        
        if source_layer < receiver_layer:
            layer_range = range(source_layer, receiver_layer + 1)
        else:
            layer_range = range(source_layer, receiver_layer - 1, -1)
        
        for layer_idx in layer_range:
            if layer_idx == source_layer:
                continue
            
            if layer_idx > source_layer:
                z_target = z_interfaces[layer_idx]
            else:
                z_target = z_interfaces[layer_idx + 1]
            
            v_current = v_layers[layer_idx if layer_idx < receiver_layer else layer_idx]
            
            sin_theta = p0 * v_current
            if abs(sin_theta) > 1:
                return np.inf, []
            
            cos_theta = np.sqrt(1 - sin_theta**2)
            
            dz_ray = z_target - z_current
            dx_ray = dz_ray * sin_theta / cos_theta * direction
            
            x_current += dx_ray
            z_current = z_target
            
            segment_dist = np.sqrt(dx_ray**2 + dz_ray**2)
            total_time += segment_dist / v_current
            
            points.append((x_current, z_current))
        
        v_final = v_layers[receiver_layer]
        sin_theta_final = p0 * v_final
        if abs(sin_theta_final) > 1:
            return np.inf, []
        
        cos_theta_final = np.sqrt(1 - sin_theta_final**2)
        
        dz_final = z_receiver - z_current
        dx_final = dz_final * sin_theta_final / cos_theta_final * direction
        
        x_final = x_current + dx_final
        points.append((x_final, z_receiver))
        
        segment_dist = np.sqrt(dx_final**2 + dz_final**2)
        total_time += segment_dist / v_final
        
        x_error = abs(x_final - x_receiver)
        return x_error, points
    
    v_min = min(v_layers)
    p_max = 1.0 / v_min
    
    n_iterations = 100
    p_left, p_right = 0.0, p_max * 0.99
    
    for _ in range(n_iterations):
        p_mid = (p_left + p_right) / 2
        error_left, _ = compute_trajectory(p_left)
        error_mid, _ = compute_trajectory(p_mid)
        error_right, _ = compute_trajectory(p_right)
        
        if error_mid < 1e-3:
            break
        
        if error_left * error_mid < 0:
            p_right = p_mid
        else:
            p_left = p_mid
    
    p_optimal = (p_left + p_right) / 2
    _, points = compute_trajectory(p_optimal)
    
    if not points:
        return None
    
    x_coords = np.array([p[0] for p in points])
    z_coords = np.array([p[1] for p in points])
    
    dist = np.sqrt(np.diff(x_coords)**2 + np.diff(z_coords)**2)
    v_interp = np.interp((z_coords[:-1] + z_coords[1:]) / 2, z_interfaces, v_layers)
    times = dist / v_interp
    total_time = np.sum(times)
    
    segments = []
    for i in range(len(x_coords) - 1):
        segments.append({
            'x1': x_coords[i], 'z1': z_coords[i],
            'x2': x_coords[i + 1], 'z2': z_coords[i + 1],
            'velocity': v_interp[i], 'time': times[i]
        })
    
    return RayPath(x=x_coords, z=z_coords, travel_time=total_time, segments=segments)


def trace_ray_grid(travel_time: np.ndarray, source_x: int, source_z: int,
                   receiver_x: int, receiver_z: int,
                   dx: float, dz: float) -> Optional[RayPath]:
    """
    Trace ray from receiver to source by following gradient of travel time field.
    
    Parameters:
    - travel_time: Travel time field from FMM
    - source_x, source_z: Source indices
    - receiver_x, receiver_z: Receiver indices
    
    Returns:
    RayPath object
    """
    receiver_x = min(max(receiver_x, 0), travel_time.shape[1] - 1)
    receiver_z = min(max(receiver_z, 0), travel_time.shape[0] - 1)
    
    x = float(receiver_x * dx)
    z = float(receiver_z * dz)
    
    path_x = [x]
    path_z = [z]
    
    step_size = min(dx, dz) * 0.5
    max_steps = int(np.sqrt(travel_time.shape[0]**2 + travel_time.shape[1]**2) * 2)
    
    for _ in range(max_steps):
        xi = int(round(x / dx))
        zi = int(round(z / dz))
        
        xi = min(max(xi, 1), travel_time.shape[1] - 2)
        zi = min(max(zi, 1), travel_time.shape[0] - 2)
        
        dt_dx = (travel_time[zi, xi + 1] - travel_time[zi, xi - 1]) / (2 * dx)
        dt_dz = (travel_time[zi + 1, xi] - travel_time[zi - 1, xi]) / (2 * dz)
        
        grad_mag = np.sqrt(dt_dx**2 + dt_dz**2)
        
        if grad_mag < 1e-10:
            break
        
        x -= dt_dx / grad_mag * step_size
        z -= dt_dz / grad_mag * step_size
        
        path_x.append(x)
        path_z.append(z)
        
        dist_to_source = np.sqrt((x - source_x * dx)**2 + (z - source_z * dz)**2)
        if dist_to_source < step_size:
            path_x.append(source_x * dx)
            path_z.append(source_z * dz)
            break
    
    x_coords = np.array(path_x)
    z_coords = np.array(path_z)
    
    total_time = travel_time[receiver_z, receiver_x]
    
    segments = []
    for i in range(len(x_coords) - 1):
        segments.append({
            'x1': x_coords[i], 'z1': z_coords[i],
            'x2': x_coords[i + 1], 'z2': z_coords[i + 1],
            'velocity': 0.0, 'time': 0.0
        })
    
    return RayPath(x=x_coords, z=z_coords, travel_time=total_time, segments=segments)


def compute_travel_times(velocity: np.ndarray, dx: float, dz: float,
                         sources: List[Tuple[int, int]],
                         receivers: Optional[List[Tuple[int, int]]] = None) -> Dict:
    """
    Compute travel times for multiple sources using FMM.
    
    Parameters:
    - velocity: Velocity model (nz, nx)
    - dx, dz: Grid spacing
    - sources: List of (source_x, source_z) index tuples
    - receivers: Optional list of (receiver_x, receiver_z) index tuples
    
    Returns:
    Dictionary with travel time fields and ray paths
    """
    fmm = FastMarching(velocity, dx, dz)
    
    result = {
        'travel_time_fields': [],
        'source_locations': sources,
        'dx': dx,
        'dz': dz
    }
    
    for sx, sz in sources:
        tau = fmm.solve(sx, sz)
        result['travel_time_fields'].append(tau)
    
    if receivers is not None:
        result['receiver_locations'] = receivers
        result['ray_paths'] = []
        
        for (sx, sz), tau in zip(sources, result['travel_time_fields']):
            source_rays = []
            for rx, rz in receivers:
                ray = trace_ray_grid(tau, sx, sz, rx, rz, dx, dz)
                source_rays.append(ray)
            result['ray_paths'].append(source_rays)
    
    return result

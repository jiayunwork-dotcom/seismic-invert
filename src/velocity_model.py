import numpy as np
import json
from typing import Dict, List, Tuple, Optional


class VelocityModel:
    """Base class for velocity models."""
    
    def __init__(self, nx: int, nz: int, dx: float, dz: float):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.x_max = (nx - 1) * dx
        self.z_max = (nz - 1) * dz
        self.velocity = np.zeros((nz, nx), dtype=np.float32)
        self.model_type = 'base'
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.arange(self.nx) * self.dx
        z = np.arange(self.nz) * self.dz
        return x, z
    
    def to_grid(self) -> np.ndarray:
        return self.velocity.copy()
    
    def save(self, filepath: str) -> None:
        data = {
            'model_type': self.model_type,
            'nx': self.nx,
            'nz': self.nz,
            'dx': self.dx,
            'dz': self.dz,
            'velocity': self.velocity.tolist()
        }
        if hasattr(self, 'layers'):
            data['layers'] = self.layers
        if hasattr(self, 'gradient_type'):
            data['gradient_type'] = self.gradient_type
            data['v_top'] = self.v_top
            data['v_bottom'] = self.v_bottom
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'VelocityModel':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data['model_type'] == 'layered':
            model = LayeredModel(
                data['nx'], data['nz'], data['dx'], data['dz'],
                data.get('layers', [])
            )
        elif data['model_type'] == 'gradient':
            model = GradientModel(
                data['nx'], data['nz'], data['dx'], data['dz'],
                data.get('v_top', 1500),
                data.get('v_bottom', 4000),
                data.get('gradient_type', 'linear')
            )
        else:
            model = GridModel(data['nx'], data['nz'], data['dx'], data['dz'])
        
        model.velocity = np.array(data['velocity'], dtype=np.float32)
        return model


class LayeredModel(VelocityModel):
    """Layered velocity model with step-like velocity profile."""
    
    def __init__(self, nx: int, nz: int, dx: float, dz: float,
                 layers: Optional[List[Dict]] = None):
        super().__init__(nx, nz, dx, dz)
        self.model_type = 'layered'
        self.layers = layers or []
        self._build_model()
    
    def add_layer(self, depth: float, velocity: float) -> None:
        self.layers.append({'depth': depth, 'velocity': velocity})
        self.layers.sort(key=lambda x: x['depth'])
        self._build_model()
    
    def remove_layer(self, index: int) -> None:
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            self._build_model()
    
    def _build_model(self) -> None:
        if not self.layers:
            self.velocity[:] = 1500
            return
        
        z = np.arange(self.nz) * self.dz
        
        layers_sorted = sorted(self.layers, key=lambda x: x['depth'])
        layer_depths = [0] + [l['depth'] for l in layers_sorted]
        layer_velocities = [l['velocity'] for l in layers_sorted]
        
        for i in range(len(layer_velocities)):
            z_mask = (z >= layer_depths[i]) & (z < layer_depths[i + 1])
            self.velocity[z_mask, :] = layer_velocities[i]
        
        z_mask = z >= layer_depths[-1]
        self.velocity[z_mask, :] = layer_velocities[-1]
    
    def update_layer(self, index: int, depth: float, velocity: float) -> None:
        if 0 <= index < len(self.layers):
            self.layers[index] = {'depth': depth, 'velocity': velocity}
            self.layers.sort(key=lambda x: x['depth'])
            self._build_model()
    
    def to_grid_model(self) -> 'GridModel':
        grid_model = GridModel(self.nx, self.nz, self.dx, self.dz)
        grid_model.velocity = self.velocity.copy()
        return grid_model


class GradientModel(VelocityModel):
    """Gradient velocity model with linear or exponential velocity increase."""
    
    def __init__(self, nx: int, nz: int, dx: float, dz: float,
                 v_top: float = 1500, v_bottom: float = 4000,
                 gradient_type: str = 'linear'):
        super().__init__(nx, nz, dx, dz)
        self.model_type = 'gradient'
        self.v_top = v_top
        self.v_bottom = v_bottom
        self.gradient_type = gradient_type
        self._build_model()
    
    def _build_model(self) -> None:
        z = np.arange(self.nz) * self.dz
        z_norm = z / self.z_max if self.z_max > 0 else z
        
        if self.gradient_type == 'linear':
            velocities = self.v_top + (self.v_bottom - self.v_top) * z_norm
        elif self.gradient_type == 'exponential':
            velocities = self.v_top * np.exp(np.log(self.v_bottom / self.v_top) * z_norm)
        else:
            raise ValueError(f"Unknown gradient type: {self.gradient_type}")
        
        self.velocity = velocities[:, np.newaxis].repeat(self.nx, axis=1).astype(np.float32)
    
    def update_parameters(self, v_top: float, v_bottom: float, gradient_type: str) -> None:
        self.v_top = v_top
        self.v_bottom = v_bottom
        self.gradient_type = gradient_type
        self._build_model()
    
    def to_grid_model(self) -> 'GridModel':
        grid_model = GridModel(self.nx, self.nz, self.dx, self.dz)
        grid_model.velocity = self.velocity.copy()
        return grid_model


class GridModel(VelocityModel):
    """Custom grid-based velocity model with pixel editing capabilities."""
    
    def __init__(self, nx: int, nz: int, dx: float, dz: float,
                 default_velocity: float = 2000):
        super().__init__(nx, nz, dx, dz)
        self.model_type = 'grid'
        self.velocity[:] = default_velocity
    
    def paint_velocity(self, x: int, z: int, velocity: float, radius: int = 1) -> None:
        for dz_idx in range(-radius, radius + 1):
            for dx_idx in range(-radius, radius + 1):
                xi, zi = x + dx_idx, z + dz_idx
                if 0 <= xi < self.nx and 0 <= zi < self.nz:
                    dist = np.sqrt(dx_idx**2 + dz_idx**2)
                    if dist <= radius:
                        self.velocity[zi, xi] = velocity
    
    def fill_region(self, x_start: int, x_end: int, z_start: int, z_end: int,
                    velocity: float) -> None:
        x1, x2 = sorted([max(0, min(self.nx - 1, x_start)), max(0, min(self.nx - 1, x_end))])
        z1, z2 = sorted([max(0, min(self.nz - 1, z_start)), max(0, min(self.nz - 1, z_end))])
        self.velocity[z1:z2 + 1, x1:x2 + 1] = velocity
    
    def set_velocity(self, x: int, z: int, velocity: float) -> None:
        if 0 <= x < self.nx and 0 <= z < self.nz:
            self.velocity[z, x] = velocity
    
    def fill_between_velocities(self, v_top: float, v_bottom: float, gradient_type: str = 'linear') -> None:
        z = np.arange(self.nz) * self.dz
        z_norm = z / self.z_max if self.z_max > 0 else z
        
        if gradient_type == 'linear':
            velocities = v_top + (v_bottom - v_top) * z_norm
        elif gradient_type == 'exponential':
            velocities = v_top * np.exp(np.log(v_bottom / v_top) * z_norm)
        
        self.velocity = velocities[:, np.newaxis].repeat(self.nx, axis=1).astype(np.float32)
    
    def to_layered_model(self, n_layers: int = 5) -> LayeredModel:
        z = np.arange(self.nz) * self.dz
        avg_velocity = np.mean(self.velocity, axis=1)
        
        depth_bins = np.linspace(0, self.z_max, n_layers + 1)
        layers = []
        
        for i in range(1, n_layers + 1):
            z_mask = (z >= depth_bins[i - 1]) & (z < depth_bins[i])
            if np.any(z_mask):
                layer_vel = np.mean(avg_velocity[z_mask])
                layers.append({'depth': depth_bins[i], 'velocity': float(layer_vel)})
        
        return LayeredModel(self.nx, self.nz, self.dx, self.dz, layers)
    
    def smooth(self, radius: int = 2) -> None:
        from scipy.ndimage import gaussian_filter
        self.velocity = gaussian_filter(self.velocity, sigma=radius).astype(np.float32)


def create_default_model(model_type: str = 'layered', **kwargs) -> VelocityModel:
    """Create a default velocity model for demonstration purposes."""
    nx = kwargs.get('nx', 100)
    nz = kwargs.get('nz', 80)
    dx = kwargs.get('dx', 10.0)
    dz = kwargs.get('dz', 10.0)
    
    if model_type == 'layered':
        layers = kwargs.get('layers', [
            {'depth': 200, 'velocity': 1500},
            {'depth': 500, 'velocity': 2200},
            {'depth': 800, 'velocity': 3000},
            {'depth': 1200, 'velocity': 3800}
        ])
        return LayeredModel(nx, nz, dx, dz, layers)
    
    elif model_type == 'gradient':
        v_top = kwargs.get('v_top', 1500)
        v_bottom = kwargs.get('v_bottom', 4000)
        gradient_type = kwargs.get('gradient_type', 'linear')
        return GradientModel(nx, nz, dx, dz, v_top, v_bottom, gradient_type)
    
    else:
        default_v = kwargs.get('default_velocity', 2000)
        return GridModel(nx, nz, dx, dz, default_v)


def convert_model(model: VelocityModel, target_type: str, **kwargs) -> VelocityModel:
    """Convert between different velocity model types."""
    if isinstance(model, GridModel):
        if target_type == 'layered':
            n_layers = kwargs.get('n_layers', 5)
            return model.to_layered_model(n_layers)
        elif target_type == 'grid':
            return model
    
    if isinstance(model, (LayeredModel, GradientModel)):
        if target_type == 'grid':
            return model.to_grid_model()
        elif target_type == 'layered' and isinstance(model, LayeredModel):
            return model
    
    raise ValueError(f"Cannot convert {model.model_type} to {target_type}")

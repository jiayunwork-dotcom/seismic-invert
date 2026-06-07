import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from typing import Dict, List, Tuple, Optional, Union
from io import BytesIO
import platform
import os

_system = platform.system()
if _system == 'Darwin':
    _font_candidates = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
    ]
elif _system == 'Windows':
    _font_candidates = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
    ]
else:
    _font_candidates = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]

for _font_path in _font_candidates:
    if os.path.exists(_font_path):
        try:
            matplotlib.font_manager.fontManager.addfont(_font_path)
            _font_name = matplotlib.font_manager.FontProperties(fname=_font_path).get_name()
            plt.rcParams['font.sans-serif'] = [_font_name, 'DejaVu Sans']
            break
        except:
            continue

plt.rcParams['axes.unicode_minus'] = False


def create_figure(figsize: Tuple[float, float] = (10, 6), dpi: int = 100) -> Figure:
    """Create a matplotlib figure."""
    return Figure(figsize=figsize, dpi=dpi)


def plot_velocity(velocity: np.ndarray, dx: float, dz: float,
                  cmap: str = 'viridis', ax=None,
                  add_contours: bool = False, contour_levels: int = 10,
                  x_label: str = 'Distance (m)', z_label: str = 'Depth (m)',
                  title: str = 'Velocity Model',
                  vmin: Optional[float] = None, vmax: Optional[float] = None) -> Figure:
    """
    Plot velocity model as colored image.
    
    Parameters:
    - velocity: Velocity model (nz, nx)
    - dx, dz: Grid spacing
    - cmap: Colormap name ('jet', 'viridis', 'seismic')
    - ax: Optional matplotlib axes
    - add_contours: Whether to add contour lines
    - contour_levels: Number of contour levels
    - x_label, z_label: Axis labels
    - title: Plot title
    - vmin, vmax: Color scale limits
    
    Returns:
    Matplotlib figure
    """
    if ax is None:
        fig = create_figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    nz, nx = velocity.shape
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    
    if vmin is None:
        vmin = np.min(velocity)
    if vmax is None:
        vmax = np.max(velocity)
    
    im = ax.imshow(velocity, extent=[x[0], x[-1], z[-1], z[0]],
                   cmap=cmap, aspect='auto', norm=Normalize(vmin=vmin, vmax=vmax))
    
    if add_contours:
        levels = np.linspace(vmin, vmax, contour_levels)
        ax.contour(x, z, velocity, levels=levels, colors='black',
                   linewidths=0.5, alpha=0.7)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(z_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Velocity (m/s)', fontsize=12)
    
    scale_x = dx * 10
    scale_z = dz * 10
    ax.plot([x[-1] - scale_x, x[-1]], [z[-1] * 0.95, z[-1] * 0.95], 'k-', linewidth=2)
    ax.text(x[-1] - scale_x / 2, z[-1] * 0.92, f'{int(scale_x)} m',
            ha='center', va='top', fontsize=10)
    
    return fig


def plot_seismic_wiggle(traces: np.ndarray, time: np.ndarray,
                        trace_spacing: Optional[float] = None,
                        ax=None, fill_positive: bool = True,
                        x_label: str = 'Trace', y_label: str = 'Time (s)',
                        title: str = 'Seismic Record',
                        amplitude_scale: float = 1.0) -> Figure:
    """
    Plot seismic traces in wiggle style.
    
    Parameters:
    - traces: Seismic traces (n_traces, n_samples) or (n_samples, n_traces)
    - time: Time axis (n_samples,)
    - trace_spacing: Spacing between traces (default: auto)
    - ax: Optional matplotlib axes
    - fill_positive: Fill positive amplitudes with black
    - x_label, y_label: Axis labels
    - title: Plot title
    - amplitude_scale: Scaling factor for trace amplitudes
    
    Returns:
    Matplotlib figure
    """
    if traces.shape[0] == len(time):
        traces = traces.T
    
    n_traces, n_samples = traces.shape
    
    if trace_spacing is None:
        trace_spacing = 1.0
    
    if ax is None:
        fig = create_figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    max_amp = np.max(np.abs(traces)) * amplitude_scale
    if max_amp == 0:
        max_amp = 1.0
    
    for i in range(n_traces):
        trace = traces[i, :] / max_amp * trace_spacing * 0.4
        x_pos = i * trace_spacing + trace
        
        ax.plot(x_pos, time, 'k-', linewidth=0.5)
        
        if fill_positive:
            ax.fill_betweenx(time, i * trace_spacing, x_pos,
                             where=x_pos >= i * trace_spacing,
                             color='black', alpha=0.7)
    
    ax.set_ylim(time[-1], time[0])
    ax.set_xlim(-trace_spacing, n_traces * trace_spacing)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    trace_ticks = np.arange(0, n_traces, max(1, n_traces // 10))
    ax.set_xticks(trace_ticks * trace_spacing)
    ax.set_xticklabels([str(int(t)) for t in trace_ticks])
    
    return fig


def plot_seismic_image(traces: np.ndarray, time: np.ndarray,
                       cmap: str = 'seismic', ax=None,
                       x_label: str = 'Trace', y_label: str = 'Time (s)',
                       title: str = 'Seismic Record',
                       clip_percentile: float = 99.0) -> Figure:
    """
    Plot seismic traces as variable area image.
    
    Parameters:
    - traces: Seismic traces (n_traces, n_samples) or (n_samples, n_traces)
    - time: Time axis (n_samples,)
    - cmap: Colormap name
    - ax: Optional matplotlib axes
    - x_label, y_label: Axis labels
    - title: Plot title
    - clip_percentile: Percentile for amplitude clipping
    
    Returns:
    Matplotlib figure
    """
    if traces.shape[0] == len(time):
        traces = traces.T
    
    n_traces, n_samples = traces.shape
    
    if ax is None:
        fig = create_figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    clip_val = np.percentile(np.abs(traces), clip_percentile)
    
    im = ax.imshow(traces.T, extent=[0, n_traces, time[-1], time[0]],
                   cmap=cmap, aspect='auto',
                   norm=Normalize(vmin=-clip_val, vmax=clip_val))
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Amplitude', fontsize=12)
    
    return fig


def plot_spectrum(freqs: np.ndarray, amplitude: np.ndarray,
                  phase: Optional[np.ndarray] = None,
                  ax=None, title: str = 'Frequency Spectrum') -> Figure:
    """
    Plot amplitude and phase spectrum.
    
    Parameters:
    - freqs: Frequency axis
    - amplitude: Amplitude spectrum
    - phase: Optional phase spectrum
    - ax: Optional matplotlib axes
    - title: Plot title
    
    Returns:
    Matplotlib figure
    """
    if phase is not None:
        if ax is None:
            fig = create_figure(figsize=(12, 8))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = fig.add_subplot(212, sharex=ax1)
        
        ax1.plot(freqs, amplitude, 'b-', linewidth=1)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(freqs, phase, 'r-', linewidth=1)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Phase (rad)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.setp(ax1.get_xticklabels(), visible=False)
    else:
        if ax is None:
            fig = create_figure(figsize=(12, 5))
            ax1 = fig.add_subplot(111)
        else:
            fig = ax.figure
            ax1 = ax
        
        ax1.plot(freqs, amplitude, 'b-', linewidth=1)
        ax1.set_xlabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    return fig


def plot_travel_time_contours(travel_time: np.ndarray, velocity: np.ndarray,
                              dx: float, dz: float, rays: Optional[List] = None,
                              ax=None, cmap: str = 'viridis',
                              title: str = 'Travel Time Field',
                              contour_interval: float = 0.05) -> Figure:
    """
    Plot travel time field with contours and optional ray paths.
    
    Parameters:
    - travel_time: Travel time field (nz, nx)
    - velocity: Velocity model for background
    - dx, dz: Grid spacing
    - rays: Optional list of RayPath objects
    - ax: Optional matplotlib axes
    - cmap: Colormap for velocity background
    - title: Plot title
    - contour_interval: Travel time contour interval in seconds
    
    Returns:
    Matplotlib figure
    """
    if ax is None:
        fig = create_figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    nz, nx = travel_time.shape
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    
    v_min, v_max = np.min(velocity), np.max(velocity)
    ax.imshow(velocity, extent=[x[0], x[-1], z[-1], z[0]],
              cmap=cmap, aspect='auto', alpha=0.3,
              norm=Normalize(vmin=v_min, vmax=v_max))
    
    tt_valid = np.ma.masked_invalid(travel_time)
    tt_min, tt_max = np.min(tt_valid), np.max(tt_valid)
    levels = np.arange(tt_min, tt_max + contour_interval, contour_interval)
    
    if len(levels) > 2:
        cs = ax.contour(x, z, tt_valid, levels=levels,
                        colors='black', linewidths=1)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    
    if rays is not None:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(rays)))
        for i, ray in enumerate(rays):
            if ray is not None and len(ray.x) > 1:
                ax.plot(ray.x, ray.z, color=colors[i], linewidth=2,
                        label=f'Receiver {i+1}')
        
        if len(rays) > 0:
            ax.legend(loc='upper right', fontsize=8)
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig


def plot_wavefield_snapshot(wavefield: np.ndarray, dx: float, dz: float,
                            time: float, ax=None, cmap: str = 'seismic',
                            title: str = None, clip_percentile: float = 99.0) -> Figure:
    """
    Plot wavefield snapshot.
    
    Parameters:
    - wavefield: Wavefield snapshot (nz, nx)
    - dx, dz: Grid spacing
    - time: Time of snapshot in seconds
    - ax: Optional matplotlib axes
    - cmap: Colormap
    - title: Plot title
    - clip_percentile: Percentile for amplitude clipping
    
    Returns:
    Matplotlib figure
    """
    if ax is None:
        fig = create_figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    nz, nx = wavefield.shape
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    
    clip_val = np.percentile(np.abs(wavefield), clip_percentile)
    
    im = ax.imshow(wavefield, extent=[x[0], x[-1], z[-1], z[0]],
                   cmap=cmap, aspect='auto',
                   norm=Normalize(vmin=-clip_val, vmax=clip_val))
    
    if title is None:
        title = f'Wavefield at t = {time*1000:.1f} ms'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pressure', fontsize=12)
    
    return fig


def plot_inversion_result(initial: np.ndarray, inverted: np.ndarray,
                          dx: float, dz: float,
                          true_model: Optional[np.ndarray] = None,
                          objective_history: Optional[List[float]] = None,
                          cmap: str = 'viridis') -> Figure:
    """
    Plot inversion results with model comparison and convergence curve.
    
    Parameters:
    - initial: Initial velocity model
    - inverted: Inverted velocity model
    - dx, dz: Grid spacing
    - true_model: Optional true velocity model
    - objective_history: Optional objective function history
    - cmap: Colormap
    
    Returns:
    Matplotlib figure
    """
    if true_model is not None:
        n_cols = 3
    else:
        n_cols = 2
    
    if objective_history is not None:
        fig = create_figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, n_cols, height_ratios=[3, 1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        vmin = min(np.min(initial), np.min(inverted))
        vmax = max(np.max(initial), np.max(inverted))
        if true_model is not None:
            vmin = min(vmin, np.min(true_model))
            vmax = max(vmax, np.max(true_model))
        
        plot_velocity(initial, dx, dz, cmap=cmap, ax=ax1,
                      title='Initial Model', vmin=vmin, vmax=vmax)
        
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        plot_velocity(inverted, dx, dz, cmap=cmap, ax=ax2,
                      title='Inverted Model', vmin=vmin, vmax=vmax)
        ax2.set_ylabel('')
        
        if true_model is not None:
            ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
            plot_velocity(true_model, dx, dz, cmap=cmap, ax=ax3,
                          title='True Model', vmin=vmin, vmax=vmax)
            ax3.set_ylabel('')
        
        ax_conv = fig.add_subplot(gs[1, :])
        iterations = np.arange(1, len(objective_history) + 1)
        ax_conv.semilogy(iterations, objective_history, 'b-o', linewidth=1, markersize=3)
        ax_conv.set_xlabel('Iteration', fontsize=12)
        ax_conv.set_ylabel('Objective Function', fontsize=12)
        ax_conv.set_title('Convergence Curve', fontsize=14, fontweight='bold')
        ax_conv.grid(True, alpha=0.3)
    else:
        fig = create_figure(figsize=(15, 5))
        vmin = min(np.min(initial), np.min(inverted))
        vmax = max(np.max(initial), np.max(inverted))
        if true_model is not None:
            vmin = min(vmin, np.min(true_model))
            vmax = max(vmax, np.max(true_model))
        
        ax1 = fig.add_subplot(1, n_cols, 1)
        plot_velocity(initial, dx, dz, cmap=cmap, ax=ax1,
                      title='Initial Model', vmin=vmin, vmax=vmax)
        
        ax2 = fig.add_subplot(1, n_cols, 2, sharey=ax1)
        plot_velocity(inverted, dx, dz, cmap=cmap, ax=ax2,
                      title='Inverted Model', vmin=vmin, vmax=vmax)
        ax2.set_ylabel('')
        
        if true_model is not None:
            ax3 = fig.add_subplot(1, n_cols, 3, sharey=ax1)
            plot_velocity(true_model, dx, dz, cmap=cmap, ax=ax3,
                          title='True Model', vmin=vmin, vmax=vmax)
            ax3.set_ylabel('')
    
    fig.tight_layout()
    return fig


def plot_velocity_spectrum(spectrum: np.ndarray, velocities: np.ndarray,
                           times: np.ndarray, picks: Optional[List[Dict]] = None,
                           ax=None, title: str = 'Velocity Spectrum') -> Figure:
    """
    Plot velocity spectrum.
    
    Parameters:
    - spectrum: Velocity spectrum (n_times, n_velocities)
    - velocities: Velocity axis
    - times: Time axis
    - picks: Optional velocity picks
    - ax: Optional matplotlib axes
    - title: Plot title
    
    Returns:
    Matplotlib figure
    """
    if ax is None:
        fig = create_figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    max_val = np.max(spectrum)
    if max_val == 0:
        max_val = 1
    
    im = ax.imshow(spectrum / max_val,
                   extent=[velocities[0], velocities[-1], times[-1], times[0]],
                   cmap='jet', aspect='auto', norm=Normalize(vmin=0, vmax=1))
    
    if picks is not None:
        for pick in picks:
            ax.plot(pick['velocity'], pick['time'], 'wo', markersize=8, markeredgecolor='k')
            ax.annotate(f'{pick["velocity"]:.0f} m/s',
                        (pick['velocity'], pick['time']),
                        xytext=(5, 5), textcoords='offset points',
                        color='white', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Velocity (m/s)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Normalized Semblance', fontsize=12)
    
    return fig


def plot_comparison(observed: np.ndarray, synthetic: np.ndarray,
                    time: np.ndarray, trace_indices: Optional[List[int]] = None,
                    title: str = 'Observed vs Synthetic') -> Figure:
    """
    Plot comparison between observed and synthetic data.
    
    Parameters:
    - observed: Observed data (n_samples, n_traces)
    - synthetic: Synthetic data (n_samples, n_traces)
    - time: Time axis
    - trace_indices: Indices of traces to plot (default: all)
    - title: Plot title
    
    Returns:
    Matplotlib figure
    """
    if trace_indices is None:
        trace_indices = [0, observed.shape[1] // 2, observed.shape[1] - 1]
        trace_indices = [i for i in trace_indices if i < observed.shape[1]]
    
    n_plots = len(trace_indices)
    
    fig = create_figure(figsize=(15, 4 * n_plots))
    
    for i, idx in enumerate(trace_indices):
        ax = fig.add_subplot(n_plots, 1, i + 1)
        
        min_samples = min(len(observed), len(synthetic))
        obs = observed[:min_samples, idx]
        syn = synthetic[:min_samples, idx]
        t = time[:min_samples]
        
        ax.plot(t, obs, 'b-', label='Observed', linewidth=1, alpha=0.7)
        ax.plot(t, syn, 'r--', label='Synthetic', linewidth=1, alpha=0.7)
        ax.fill_between(t, obs - syn, alpha=0.3, color='g', label='Residual')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(f'Trace {idx + 1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    return fig


def save_figure(fig: Figure, filepath: str, dpi: int = 150) -> None:
    """Save figure to file."""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


def figure_to_bytes(fig: Figure, format: str = 'png', dpi: int = 150) -> BytesIO:
    """Convert figure to bytes for Streamlit display."""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf

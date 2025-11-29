"""
Visualization functions for generated samples
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import (
    KDE_GRID_SIZE, KDE_PADDING, MODE_CONFIGS, NUM_MODES, data_sampling_mode,
    n, p, TIME_SCHEDULE, N_gen, L_t, m_z, d, get_A0
)
from test_function import h

# Constants
SNAPSHOT_COLS = 3
SNAPSHOT_ROW_HEIGHT = 5
MODE_COUNT_RADIUS = 0.5
KDE_LEVELS = 15
KDE_CMAP = 'Oranges'
KDE_COLOR = 'orange'
FIGURE_DPI = 300


def kde2d_grid(samples, grid_size=250, padding=0.1):
    """
    Simple 2D Gaussian KDE on a grid (Scott's rule, isotropic bandwidth).
    Returns (X, Y, D) with X,Y meshgrids and D the density grid.
    """
    n_samples, d_ = samples.shape
    assert d_ == 2, "kde2d_grid expects samples with shape (N, 2)."

    # Scott's rule (scalar bandwidth)
    std = samples.std(axis=0, ddof=1)
    bandwidth = n_samples ** (-1.0 / (d_ + 4.0)) * float(std.mean())

    mins = samples.min(axis=0)
    maxs = samples.max(axis=0)
    span = maxs - mins
    mins = mins - padding * span
    maxs = maxs + padding * span

    xs = np.linspace(mins[0], maxs[0], grid_size)
    ys = np.linspace(mins[1], maxs[1], grid_size)
    X, Y = np.meshgrid(xs, ys)

    XY = np.stack([X.ravel(), Y.ravel()], axis=1)   # (G^2, 2)
    diffs = XY[:, None, :] - samples[None, :, :]    # (G^2, N, 2)
    sqnorm = np.sum(diffs * diffs, axis=2)          # (G^2, N)

    coef = 1.0 / (2.0 * np.pi * (bandwidth ** 2))
    D = (coef * np.mean(np.exp(-0.5 * sqnorm / (bandwidth ** 2)), axis=1)).reshape(grid_size, grid_size)
    return X, Y, D


def kde1d(values, grid_size=500, padding=0.1, bandwidth_factor=1.0):
    """
    Compute 1D Gaussian KDE for a set of values.
    
    Parameters:
        values: (N,) array of 1D values
        grid_size: Number of grid points for KDE evaluation
        padding: Padding factor for grid range
        bandwidth_factor: Factor to multiply the bandwidth (smaller = sharper peaks)
    
    Returns:
        x_grid: (grid_size,) array of grid points
        density: (grid_size,) array of density values
    """
    values = np.asarray(values).flatten()
    
    # Compute bandwidth using Scott's rule, then scale by bandwidth_factor
    n_samples = len(values)
    std = np.std(values, ddof=1)
    h = n_samples ** (-1.0 / 5.0) * std * bandwidth_factor  # Scott's rule for 1D, scaled
    
    # Create grid
    min_val = values.min()
    max_val = values.max()
    span = max_val - min_val
    if span == 0:
        span = 1.0  # Handle degenerate case
    min_val = min_val - padding * span
    max_val = max_val + padding * span
    
    x_grid = np.linspace(min_val, max_val, grid_size)
    
    # Compute KDE using Gaussian kernel
    diffs = x_grid[:, None] - values[None, :]  # (grid_size, N)
    density = np.mean(np.exp(-0.5 * (diffs / h) ** 2), axis=1) / (np.sqrt(2 * np.pi) * h)
    
    return x_grid, density


def _resolve_config_value(value, config_value):
    """Helper to resolve config value (handles both scalar and list cases)."""
    if value is not None:
        return value
    if isinstance(config_value, list):
        return config_value[0]
    return config_value


def _resolve_experiment_params(p_val=None, m_z_val=None, num_modes_val=None,
                                n_gen_val=None, l_t_val=None):
    """Resolve experiment parameters, falling back to config values."""
    return {
        'p_val': _resolve_config_value(p_val, p),
        'm_z_val': _resolve_config_value(m_z_val, m_z),
        'num_modes_val': _resolve_config_value(num_modes_val, NUM_MODES),
        'n_gen_val': _resolve_config_value(n_gen_val, N_gen),
        'l_t_val': _resolve_config_value(l_t_val, L_t),
    }


def _get_A0_init_identifier(p_val, a0_type=None):
    """Get a string identifier for the A0 initialization type."""
    if a0_type is not None:
        if a0_type == "zero":
            return "A0_zero"
        elif a0_type == "ones":
            return "A0_ones"
        elif a0_type == "random":
            return "A0_random"
        else:
            return f"A0_{a0_type}"
    
    # Fallback: try to detect from config
    A0 = get_A0(d, p_val)
    if np.allclose(A0, 0):
        return "A0_zero"
    elif np.allclose(A0, np.ones((d, p_val)) / np.sqrt(p_val)):
        return "A0_ones"
    else:
        # For other initialization types (like random), create a simple identifier
        norm = np.linalg.norm(A0)
        return f"A0_norm{norm:.2e}".replace('.', 'p').replace('+', '').replace('-', 'm')


def _generate_filename(prefix, params, script_source=None):
    """Generate filename from parameters."""
    filename = f"{prefix}_{data_sampling_mode}_m_z{params['m_z_val']}_p{params['p_val']}_{TIME_SCHEDULE}_modes{params['num_modes_val']}_N_gen{params['n_gen_val']}_L_t{params['l_t_val']}_{params['a0_init']}"
    if script_source:
        filename += f"_{script_source}"
    filename += ".png"
    return filename


def _setup_subplot_axes(n_snapshots, n_cols=SNAPSHOT_COLS):
    """Set up subplot axes for snapshot visualization."""
    n_rows = (n_snapshots + n_cols - 1) // n_cols
    fig_height = SNAPSHOT_ROW_HEIGHT * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, fig_height))
    
    if n_snapshots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    return fig, axes, n_rows, n_cols


def _hide_empty_subplots(axes, n_used, n_total, n_rows, n_cols):
    """Hide unused subplots."""
    for i in range(n_used, n_total):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)


def _parse_snapshot_data(snapshot_data):
    """Parse snapshot data, handling both old and new formats."""
    if len(snapshot_data) == 3:
        step_name, samples, time_val = snapshot_data
    else:
        step_name, samples = snapshot_data
        time_val = None
    return step_name, samples, time_val


def _plot_kde_contours(ax, X, Y, D):
    """Plot KDE contours on an axis."""
    ax.contourf(X, Y, D, levels=KDE_LEVELS, cmap=KDE_CMAP, alpha=0.7)
    ax.contour(X, Y, D, levels=KDE_LEVELS, colors=KDE_COLOR, alpha=0.3, linewidths=0.5)


def _count_samples_per_mode(samples, centers, radius=MODE_COUNT_RADIUS):
    """Count samples within radius of each mode center."""
    counts = []
    for center in centers:
        distances = np.linalg.norm(samples - center, axis=1)
        counts.append(np.sum(distances < radius))
    return counts


def plot_samples_and_kde(samples, title="Generated samples with KDE contours", 
                          p_val=None, m_z_val=None, num_modes_val=None, 
                          n_gen_val=None, l_t_val=None, script_source=None, a0_type=None):
    """Plot generated samples with KDE contours and true centers."""
    # Resolve parameters
    params = _resolve_experiment_params(p_val, m_z_val, num_modes_val, n_gen_val, l_t_val)
    params['a0_init'] = _get_A0_init_identifier(params['p_val'], a0_type)
    
    true_centers = MODE_CONFIGS[params['num_modes_val']]['centers']
    
    print("Computing KDE grid...")
    with tqdm(total=3, desc="Visualization", unit="step", leave=False, position=0, dynamic_ncols=True) as pbar:
        # Compute KDE
        Xg, Yg, Dg = kde2d_grid(samples, grid_size=KDE_GRID_SIZE, padding=KDE_PADDING)
        pbar.update(1)
        
        # Create plot
        print("Creating plot...")
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_kde_contours(ax, Xg, Yg, Dg)
        ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.6, color='blue', label='Generated')
        
        # Count samples per mode
        mode_counts = _count_samples_per_mode(samples, true_centers)
        
        # Plot true centers with annotations
        ax.scatter(true_centers[:, 0], true_centers[:, 1], marker='x', s=100, color='red', label='True centers')
        for center, count in zip(true_centers, mode_counts):
            ax.text(center[0] + 0.3, center[1] + 0.3, f'n={count}', 
                    fontsize=10, color='darkred', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='darkred'))
        
        ax.set_title(title)
        ax.legend()
        ax.axis('equal')
        plt.tight_layout()
        pbar.update(1)
        
        # Save plot
        filename = _generate_filename("final_samples", params, script_source)
        print(f"Saving plot to {filename}...")
        fig.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        pbar.update(1)


def plot_snapshots(snapshots, title_prefix="SDE Evolution", m_z=None, script_source=None,
                    p_val=None, num_modes_val=None, n_gen_val=None, l_t_val=None, a0_type=None):
    """Plot snapshots of the SDE evolution at different time steps with KDE contours."""
    if not snapshots:
        print("No snapshots to display")
        return
    
    # Resolve parameters
    params = _resolve_experiment_params(p_val, m_z, num_modes_val, n_gen_val, l_t_val)
    params['a0_init'] = _get_A0_init_identifier(params['p_val'], a0_type)
    
    true_centers = MODE_CONFIGS[params['num_modes_val']]['centers']
    n_snapshots = len(snapshots)
    
    # Setup subplots
    fig, axes, n_rows, n_cols = _setup_subplot_axes(n_snapshots)
    
    print("Computing KDE contours for snapshots...")
    with tqdm(total=n_snapshots, desc="Processing snapshots", unit="snapshot", leave=False, position=0, dynamic_ncols=True) as pbar:
        for i, snapshot_data in enumerate(snapshots):
            step_name, samples, time_val = _parse_snapshot_data(snapshot_data)
            
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Compute and plot KDE
            try:
                Xg, Yg, Dg = kde2d_grid(samples, grid_size=KDE_GRID_SIZE, padding=KDE_PADDING)
                _plot_kde_contours(ax, Xg, Yg, Dg)
            except Exception as e:
                print(f"Warning: Could not compute KDE for {step_name}: {e}")
            
            # Plot samples and centers
            ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.6, color='blue', label='Generated')
            ax.scatter(true_centers[:, 0], true_centers[:, 1], marker='x', s=100, color='red', label='True centers')
            
            # Set title
            title_parts = [f"{title_prefix}: {step_name}"]
            if time_val is not None:
                title_parts.append(f"t = {time_val:.6f}")
            if m_z is not None:
                title_parts.append(f"m_z = {m_z}")
            if script_source is not None:
                title_parts.append(f"from {script_source}")
            ax.set_title("\n".join(title_parts), fontsize=10, pad=20)
            
            ax.set_xlabel('x1', fontsize=9)
            ax.set_ylabel('x2', fontsize=9)
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(fontsize=8)
            
            pbar.update(1)
    
    # Hide empty subplots
    _hide_empty_subplots(axes, n_snapshots, n_rows * n_cols, n_rows, n_cols)
    
    # Adjust spacing and save
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    filename = _generate_filename("snapshots", params, script_source)
    print(f"Saving snapshots to {filename}...")
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def get_expected_h_dirac_locations(num_modes=None):
    """
    Get the expected Dirac delta locations for h(x) based on the mode centers.
    
    Parameters:
        num_modes: Number of modes (uses NUM_MODES from config if None)
    
    Returns:
        dirac_locations: List of h(x) values at the mode centers
        probabilities: List of probabilities for each mode
    """
    if num_modes is None:
        num_modes = NUM_MODES if not isinstance(NUM_MODES, list) else NUM_MODES[0]
    
    config = MODE_CONFIGS[num_modes]
    centers = config['centers']  # (K, 2)
    probabilities = config['probabilities']  # (K,)
    
    # Compute h(x) at each center
    dirac_locations = np.array([h(center) for center in centers])
    
    return dirac_locations, probabilities


def plot_snapshots_h_kde(snapshots, title_prefix="SDE Evolution h(x) KDE", m_z=None, script_source=None,
                         p_val=None, num_modes_val=None, n_gen_val=None, l_t_val=None, a0_type=None):
    """Plot KDE of h(x) for each snapshot, where h(x) is the test function."""
    if not snapshots:
        print("No snapshots to display for h(x) KDE")
        return
    
    # Resolve parameters
    params = _resolve_experiment_params(p_val, m_z, num_modes_val, n_gen_val, l_t_val)
    params['a0_init'] = _get_A0_init_identifier(params['p_val'], a0_type)
    
    n_snapshots = len(snapshots)
    dirac_locations, dirac_probs = get_expected_h_dirac_locations(params['num_modes_val'])
    
    # Setup subplots
    fig, axes, n_rows, n_cols = _setup_subplot_axes(n_snapshots)
    
    print("Computing h(x) and KDE for snapshots...")
    with tqdm(total=n_snapshots, desc="Processing h(x) KDE", unit="snapshot", leave=False, position=0, dynamic_ncols=True) as pbar:
        for i, snapshot_data in enumerate(snapshots):
            step_name, samples, time_val = _parse_snapshot_data(snapshot_data)
            
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Compute h(x) and KDE
            try:
                h_values = h(samples)  # (N,) array
                
                # Compute 1D KDE with smaller bandwidth for sharper peaks
                x_grid, density = kde1d(h_values, grid_size=500, padding=0.1, bandwidth_factor=0.3)
                
                # Plot KDE
                ax.plot(x_grid, density, 'b-', linewidth=2, label='KDE of h(x)')
                ax.fill_between(x_grid, density, alpha=0.3, color='blue')
                
                # Add histogram
                ax.hist(h_values, bins=100, density=True, alpha=0.2, color='gray', 
                       label='Histogram', edgecolor='black', linewidth=0.5)
                
                # Add vertical lines at expected Dirac delta locations
                max_density = density.max()
                dirac_label_added = False
                for dirac_loc, prob in zip(dirac_locations, dirac_probs):
                    label = 'Expected Dirac' if not dirac_label_added else ''
                    if not dirac_label_added:
                        dirac_label_added = True
                    ax.axvline(dirac_loc, color='red', linestyle='--', linewidth=2, alpha=0.7, label=label)
                    ax.text(dirac_loc, max_density * 0.9, f'  δ({dirac_loc:.2f})\n  p={prob:.2f}', 
                           rotation=90, verticalalignment='top', fontsize=7, color='red')
                
            except Exception as e:
                print(f"Warning: Could not compute h(x) KDE for {step_name}: {e}")
            
            # Set title
            title_parts = [f"{title_prefix}: {step_name}"]
            if time_val is not None:
                title_parts.append(f"t = {time_val:.6f}")
            if m_z is not None:
                title_parts.append(f"m_z = {m_z}")
            if script_source is not None:
                title_parts.append(f"from {script_source}")
            ax.set_title("\n".join(title_parts), fontsize=10, pad=20)
            
            ax.set_xlabel('h(x)', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(fontsize=8)
            
            pbar.update(1)
    
    # Hide empty subplots
    _hide_empty_subplots(axes, n_snapshots, n_rows * n_cols, n_rows, n_cols)
    
    # Adjust spacing and save
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    filename = _generate_filename("snapshots_h_kde", params, script_source)
    print(f"Saving h(x) KDE snapshots to {filename}...")
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

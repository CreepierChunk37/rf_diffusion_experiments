"""
Utility functions for RF Score experiment
"""
import numpy as np
import math
import time
from tqdm import tqdm

def print_progress(current, total, section_name, start_time):
    """Print progress percentage and estimated time remaining"""
    elapsed = time.time() - start_time
    if current > 0:
        progress = current / total
        eta = elapsed * (total - current) / current
        print(f"\r{section_name}: {progress:.1%} complete ({current}/{total}) - "
              f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s", end="", flush=True)
    else:
        print(f"\r{section_name}: Starting...", end="", flush=True)

def create_progress_bar(total, desc, start_time):
    """Create a tqdm progress bar with custom formatting"""
    return tqdm(
        total=total,
        desc=desc,
        unit="step",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        ncols=100,
        leave=False,
        position=0,  # Always use position 0 for single line
        dynamic_ncols=True  # Adjust width dynamically
    )

def show_overall_progress(section_name):
    """Show overall progress across all sections"""
    sections = ["UV computation", "A(∞) computation", "Reverse SDE", "Visualization"]
    current_idx = sections.index(section_name) if section_name in sections else 0
    progress = (current_idx + 1) / len(sections)
    print(f"\r🔄 Overall Progress: {progress:.1%} - {section_name}", end="", flush=True)

def create_time_schedule(schedule_type, t_min, t_max, L_sde):
    """Create time schedule for reverse SDE based on specified type"""
    if schedule_type == 'linear':
        # Linear schedule: uniform steps from t_max to t_min
        ts_sde = np.linspace(t_max, t_min, L_sde + 1)
        print(f"   Linear schedule: uniform steps from {t_max:.2f} to {t_min:.4f}")
        
    elif schedule_type == 'cosine':
        # Cosine schedule: cosine annealing from t_max to t_min
        # Uses cosine function to create smooth transition
        i = np.arange(L_sde + 1)
        ts_sde = t_min + (t_max - t_min) * 0.5 * (1 + np.cos(np.pi * i / L_sde))
        print(f"   Cosine schedule: smooth transition from {t_max:.2f} to {t_min:.4f}")
        
    elif schedule_type == 'exponential':
        # Exponential decay: each step is q times smaller than previous
        q = (t_max / t_min) ** (1.0 / L_sde)
        ts_sde = np.array([t_max * (q ** (-i)) for i in range(L_sde + 1)])
        #ts_sde[-1] = t_min  # Ensure exact t_min for numerical stability
        print(f"   Exponential decay: q = {q:.4f}, t_min = {t_min:.4f}")
        
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Must be 'linear', 'cosine', or 'exponential'")
    
    print(f"   Time range: [{ts_sde[0]:.2f}, {ts_sde[-1]:.4f}]")
    return ts_sde

def switch_to_mode(num_modes):
    """Helper function to switch between different mode configurations"""
    from config import NUM_MODES, MODE_CONFIGS
    if num_modes not in MODE_CONFIGS:
        raise ValueError(f"NUM_MODES must be 1, 2, 3, or 4, got {num_modes}")
    # Note: This would need to be called from main script to update global
    print(f"Switched to {MODE_CONFIGS[num_modes]['description']}")
    return MODE_CONFIGS[num_modes]

def switch_to_schedule(schedule_type):
    """Helper function to switch between different time schedules"""
    valid_schedules = ['linear', 'cosine', 'exponential']
    if schedule_type not in valid_schedules:
        raise ValueError(f"TIME_SCHEDULE must be one of {valid_schedules}, got {schedule_type}")
    print(f"Switched to {schedule_type} time schedule")
    return schedule_type

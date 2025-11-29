"""
Reverse SDE simulation functions
"""
import numpy as np
import math
import time
from config import N_gen, d, p, K, ridge, drift_scale, bias_switch
from schedules import h_t, a_t, phi_time
from utils import create_progress_bar, create_time_schedule
from config import TIME_SCHEDULE, T, L_sde, t_max, t_min

def simulate_reverse_sde(A_inf, W_x, W_tau, b, rng, capture_snapshots=True, snapshot_interval=200):
    """Simulate reverse SDE with configurable time schedule and optional snapshots"""
    # Create time schedule based on configuration
    ts_sde = create_time_schedule(TIME_SCHEDULE, t_min, t_max, L_sde)

    # Initialize at t = T
    x = rng.standard_normal((N_gen, d))
    
    # Debug: Check initial conditions
    print(f"   Initial x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"   Initial x mean: {x.mean(axis=0)}")
    print(f"   Initial x std: {x.std(axis=0)}")

    # Initialize snapshots list if capturing
    snapshots = []
    if capture_snapshots:
        snapshots.append(('Initial', x.copy(), ts_sde[0]))

    # Create progress bar for reverse SDE
    pbar = create_progress_bar(L_sde, "Reverse SDE", time.time())

    for l in range(L_sde):
        t = ts_sde[l]             # reverse-time clock (configurable schedule)
        dt = - ts_sde[l+1] + ts_sde[l]  # variable step size
        ht = h_t(t)
        at = a_t(t)

        phi_vec = phi_time(t, K, T)               # (2K+1,)
        Wphi = (W_tau @ phi_vec)[:, None] / math.sqrt(2 * K + 1)  # (p, 1) with √K normalization
        #Zx = (W_x @ (x.T / math.sqrt(d * ht)))     # (p, N_gen)
        Zx = (W_x @ (x.T / math.sqrt(d)))     # (p, N_gen) remove normalization root(ht);
        if bias_switch:
            H = np.maximum(0.0, Zx + Wphi + b[:, None])  # (p, N_gen)
        else:
            H = np.maximum(0.0, Zx + Wphi)  # (p, N_gen)

        g = A_inf @ H                        # (d, N_gen) posterior-mean approx
        
        drift =  (1 - 2.0 / ht) * x + (2.0 / ht) * g.T  # (N_gen, d)

        noise = rng.standard_normal((N_gen, d))
        x = x + drift * dt + noise * np.sqrt(2 * dt)
        
        # Capture snapshot every snapshot_interval steps
        if capture_snapshots and (l + 1) % snapshot_interval == 0:
            snapshots.append((f'Step {l+1}', x.copy(), t))
        
        # Debug: Check mode counts every 10% of steps
        if l % max(1, L_sde // 10) == 0:
            # Calculate mode distances
            dist_to_origin = np.linalg.norm(x, axis=1)
            dist_to_mode2 = np.linalg.norm(x - np.array([5.0, 0.0]), axis=1)
            
            # Count samples near each mode
            near_origin = np.sum(dist_to_origin < 1.0)
            near_mode2 = np.sum(dist_to_mode2 < 1.0)
            
            print(f"   Step {l}: Near (0,0): {near_origin}, Near (5,0): {near_mode2}")
        
        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()
    
    # Add final snapshot
    if capture_snapshots:
        snapshots.append(('Final', x.copy(), ts_sde[-1]))
    
    # Debug: Check final results
    print(f"   Final x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"   Final x mean: {x.mean(axis=0)}")
    print(f"   Final x std: {x.std(axis=0)}")
    
    return x, snapshots

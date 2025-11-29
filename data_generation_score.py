"""
Data generation and sampling functions
"""
import numpy as np
import math
import time
from config import NUM_MODES, MODE_CONFIGS, n, d, p, K, T, ridge, m_z, bias_switch, data_sampling_mode
from schedules import a_t, h_t, phi_time
from utils import create_progress_bar

def generate_data_samples(rng):
    """Generate data samples based on data_sampling_mode"""
    config = MODE_CONFIGS[NUM_MODES]
    centers = config['centers']
    probabilities = config['probabilities']
    
    if data_sampling_mode == "centers":
        # Use the centers directly (one occurrence of each center)
        X0 = centers  # (n_centers, d) where n_centers = len(centers)
    elif data_sampling_mode == "samples":
        # Sample n points from the mixture using probabilities
        choices = rng.choice(len(centers), size=n, p=probabilities)
        X0 = centers[choices]  # (n, d)
    else:
        raise ValueError(f"Unknown data_sampling_mode: {data_sampling_mode}. Use 'centers' or 'samples'.")
    
    return X0

def compute_UV_unweighted(X0, W_x, W_tau, b, ts, w, rng):
    """Compute U and V matrices based on data_sampling_mode"""
    from config import batch_size
    
    U_acc = np.zeros((p, p))
    V_acc = np.zeros((d, p))

    # Determine data size based on sampling mode
    if data_sampling_mode == "centers":
        n_data = X0.shape[0]  # Number of centers (NUM_MODES)
    elif data_sampling_mode == "samples":
        n_data = n  # Number of sampled data points
    else:
        raise ValueError(f"Unknown data_sampling_mode: {data_sampling_mode}")

    # Vectorize over time: compute all time-dependent quantities at once
    at_values = np.array([a_t(t) for t in ts])  # (L_t+1,)
    ht_values = np.array([h_t(t) for t in ts])  # (L_t+1,)
    phi_vectors = np.array([phi_time(t, K, T) for t in ts])  # (L_t+1, 2K+1)
    Wphi_values = phi_vectors @ W_tau.T / math.sqrt(2 * K + 1)  # (L_t+1, p) with √K normalization

    # Create progress bar for UV computation
    pbar = create_progress_bar(len(ts), "UV computation", time.time())
    
    for i, (t, wt) in enumerate(zip(ts, w)):
        at, ht = at_values[i], ht_values[i]
        Wphi = Wphi_values[i]  # (p,)

        # Initialize accumulators for this time step
        U_t_hat = np.zeros((p, p))
        V_t_hat = np.zeros((d, p))
        
        if data_sampling_mode == "centers":
            # Process all centers at once (no batching needed for small n_data)
            WX_X0T = W_x @ X0.T  # (p, n_data)
            
            # Batch over m_z noise samples
            m_z_batches = (m_z + batch_size - 1) // batch_size
            
            for m_batch_idx in range(m_z_batches):
                start_m = m_batch_idx * batch_size
                end_m = min((m_batch_idx + 1) * batch_size, m_z)
                m_batch_size = end_m - start_m
                
                # Generate noise for this batch of m_z samples
                Z_batch = rng.standard_normal((n_data, m_batch_size, d))  # (n_data, m_batch_size, d)
                WZ_over_sqrt_d_batch = np.einsum('pd,nmd->pnm', W_x, Z_batch) / math.sqrt(d)  # (p, n_data, m_batch_size)
                
                # mean term for all centers
                mean_i = WX_X0T * (at / math.sqrt(d + ridge))  # (p, n_data)
                # preactivations for this batch
                if bias_switch:
                    Z_all_batch = mean_i[:, :, None] + WZ_over_sqrt_d_batch + b[:, None, None] + Wphi[:, None, None]  # (p, n_data, m_batch_size)
                else:
                    Z_all_batch = mean_i[:, :, None] + WZ_over_sqrt_d_batch + Wphi[:, None, None]  # (p, n_data, m_batch_size)
                H_batch = np.maximum(0.0, Z_all_batch)  # (p, n_data, m_batch_size)
                
                # Accumulate U_t contribution from this batch
                H_flat_batch = H_batch.reshape(p, -1)  # (p, n_data * m_batch_size)
                U_t_hat += ht * (H_flat_batch @ H_flat_batch.T) / (n_data * m_z * p)
                
                # Accumulate V_t contribution from this batch
                # New formula: sqrt(ht) * (1/n_data) * sum_{i=1}^n_data * (1/mz) * sum_{m=1}^mz * z_m * sigma^T / p
                for i in range(n_data):
                    for m in range(m_batch_size):
                        z_m = Z_batch[i, m, :]  # (d,) - noise sample
                        sigma_m = H_batch[:, i, m]  # (p,) - activation for this noise sample
                        V_t_hat += math.sqrt(ht) * (1.0 / n_data) * (1.0 / m_z) * np.outer(z_m, sigma_m) / p  # (d, p)
        
        elif data_sampling_mode == "samples":
            # Batch over data samples (traditional approach)
            data_batches = (n_data + batch_size - 1) // batch_size
            
            for data_batch_idx in range(data_batches):
                start_idx = data_batch_idx * batch_size
                end_idx = min((data_batch_idx + 1) * batch_size, n_data)
                batch_size_actual = end_idx - start_idx
                X0_batch = X0[start_idx:end_idx]  # (batch_size_actual, d)
                
                # Process all m_z noise samples for this data batch
                WX_X0T_batch = W_x @ X0_batch.T  # (p, batch_size_actual)
                
                # Generate noise for all m_z samples
                Z_batch = rng.standard_normal((batch_size_actual, m_z, d))  # (batch_size_actual, m_z, d)
                WZ_over_sqrt_d_batch = np.einsum('pd,nmd->pnm', W_x, Z_batch) / math.sqrt(d)  # (p, batch_size_actual, m_z)
                
                # mean term for this data batch
                mean_i = WX_X0T_batch * (at / math.sqrt(d + ridge))  # (p, batch_size_actual)
                # preactivations for this batch
                if bias_switch:
                    Z_all_batch = mean_i[:, :, None] + WZ_over_sqrt_d_batch + b[:, None, None] + Wphi[:, None, None]  # (p, batch_size_actual, m_z)
                else:
                    Z_all_batch = mean_i[:, :, None] + WZ_over_sqrt_d_batch + Wphi[:, None, None]  # (p, batch_size_actual, m_z)
                H_batch = np.maximum(0.0, Z_all_batch)  # (p, batch_size_actual, m_z)
                
                # Accumulate U_t contribution from this batch
                H_flat_batch = H_batch.reshape(p, -1)  # (p, batch_size_actual * m_z)
                U_t_hat += ht * (H_flat_batch @ H_flat_batch.T) / (n_data * m_z * p)
                
                # Accumulate V_t contribution from this batch
                for i in range(n_data):
                    for m in range(m_batch_size):
                        z_m = Z_batch[i, m, :]  # (d,) - noise sample
                        sigma_m = H_batch[:, i, m]  # (p,) - activation for this noise sample
                        V_t_hat += math.sqrt(ht) * (1.0 / n_data) * (1.0 / m_z) * np.outer(z_m, sigma_m) / p  # (d, p)
        
        # Aggregate over time (trapezoid)
        U_acc += wt * U_t_hat
        V_acc += wt * V_t_hat
        
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()

    return U_acc, V_acc

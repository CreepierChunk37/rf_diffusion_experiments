import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from tqdm import tqdm
import copy
from tqdm.auto import tqdm
import numpy as np


# 线性时间
def sample(model, num_samples=1000, dim=2, num_steps=5000, T=40.0, device='cuda', num_snapshots=10):
    dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    dt = (T - 0.01) / num_steps  
    g = math.sqrt(2.0)
    
    snapshots = []
    if num_snapshots > 0:
        snapshot_indices = torch.linspace(0, num_steps, num_snapshots, dtype=torch.int64)
        snapshot_indices_set = set(snapshot_indices.tolist())
    else:
        snapshot_indices_set = set()

    y = torch.randn(num_samples, dim, device=device, dtype=dtype)
    
    if 0 in snapshot_indices_set:
        snapshots.append(y.cpu().clone())

    t_val = torch.zeros(num_samples, device=device, dtype=dtype)
    with torch.no_grad():
        for i in range(num_steps):
            T_minus_t = T - t_val
            h_T_minus_t = 1 - torch.exp(-2 * T_minus_t)
            h_T_minus_t = h_T_minus_t.view(-1, 1)  # Shape: (num_samples, 1)
            model_output = model(y, T_minus_t)
            drift = y - (2 / h_T_minus_t) * y + (2 / h_T_minus_t) * model_output
            diffusion_term = g * math.sqrt(dt) * torch.randn_like(y)
            y = y + drift * dt + diffusion_term
            t_val += dt
            current_step = i + 1
            if current_step in snapshot_indices_set:
                snapshots.append(y.cpu().clone())
    
    if num_snapshots > 0:
        return snapshots
    else:
        return y 

# 指数时间
def sample_exponential(model, num_samples=1000, dim=2, num_steps=10000, T=40.0, device='cuda',
                       num_snapshots=0):
    dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    g = math.sqrt(2.0)
    snapshots = []
    if num_snapshots > 0:
        snapshot_indices = torch.linspace(0, num_steps, num_snapshots, dtype=torch.int64)
        snapshot_indices_set = set(snapshot_indices.tolist())
    else:
        snapshot_indices_set = set()

    y = torch.randn(num_samples, dim, device=device, dtype=dtype)

    if 0 in snapshot_indices_set:
        snapshots.append(y.cpu().clone())
    
    t_min = 0.0001  
    t_max = T     
    L_sde = num_steps 

    if t_min <= 0:
        raise ValueError("t_min 必须大于 0 以进行指数调度。")
        
    q = (t_max / t_min) ** (1.0 / L_sde)

    time_points_np = np.array([t_min * (q ** i) for i in range(L_sde + 1)])

    time_points = torch.tensor(time_points_np, device=device, dtype=dtype)

    with torch.no_grad():
        for i in range(num_steps):
            t_i = time_points[i]
            t_i_plus_1 = time_points[i+1]
            dt = t_i_plus_1 - t_i  
            t_val = torch.full((num_samples,), t_i, device=device, dtype=dtype)
            T_minus_t = T - t_val
            h_T_minus_t = 1 - torch.exp(-2 * T_minus_t)
            h_T_minus_t = h_T_minus_t.view(-1, 1)  
            model_output = model(y, T_minus_t)
            drift = y - (2 / h_T_minus_t) * y + (2 / h_T_minus_t) * model_output
            diffusion_term = g * math.sqrt(dt) * torch.randn_like(y)
            y = y + drift * dt + diffusion_term
            
            current_step = i + 1
            if current_step in snapshot_indices_set:
                snapshots.append(y.cpu().clone())
                
    if num_snapshots > 0:
        return snapshots
    else:
        return y
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from tqdm import tqdm
import copy
from tqdm.auto import tqdm
import numpy as np

def train(model, data, z_grid, epochs=20000, lr=0.001, T=10.0, eps=1e-5, device='cuda', 
          t_grid_size=200, z_grid_size=500, t_batch_size=200): 
    
    dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    data = data.to(device=device, dtype=dtype)
    z_grid = z_grid.to(device=device, dtype=dtype)
    t_grid = torch.linspace(eps, T, t_grid_size, device=device, dtype=dtype)

    optimizer = optim.Adam(model.parameters(), lr=lr) 
    num_data = data.shape[0] 
    data_dim = data.shape[1:]

    loss_history = []
    
    x0 = data
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        epoch_total_loss = 0.0 
        for t_batch in torch.split(t_grid, t_batch_size):
            current_t_batch_size = t_batch.shape[0]
            x0_expanded = x0.repeat_interleave(current_t_batch_size * z_grid_size, dim=0)
            t_expanded = t_batch.repeat_interleave(z_grid_size).repeat(num_data) 
            z_expanded = z_grid.repeat(num_data * current_t_batch_size, *([1] * len(data_dim))) 
            exp_neg_t = torch.exp(-t_expanded)
            a_t = exp_neg_t
            sigma = torch.sqrt(1 - torch.exp(-2 * t_expanded))
            xt = exp_neg_t.unsqueeze(1) * x0_expanded + sigma.unsqueeze(1) * z_expanded
            pred = model(xt, t_expanded) 
            target = a_t.unsqueeze(1) * x0_expanded

            batch_loss = ((pred - target) ** 2).mean()
            
            scaled_batch_loss = batch_loss * (current_t_batch_size / t_grid_size)
            scaled_batch_loss.backward() 
            epoch_total_loss += batch_loss.item() * current_t_batch_size

        optimizer.step()
        avg_epoch_loss = epoch_total_loss / t_grid_size
        loss_history.append(avg_epoch_loss)

    return loss_history, model

def train_random(model, data, epochs=100000, lr=0.0001, T=40.0, eps=1e-6, device='cuda'):
    dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    data = data.to(device=device, dtype=dtype)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0) 
    num_data = data.shape[0]
    loss_history = []
    best_loss = float('inf')
    
    for epoch in tqdm(range(epochs)):
        x0 = data
        t = torch.rand(data.shape[0], device=device, dtype=dtype) * (T - eps) + eps
        a_t = torch.exp(-t)
        h_t = 1 - torch.exp(-2 * t)
        z = torch.randn_like(x0) 
        sqrt_h_t = torch.sqrt(h_t)
        xt = a_t.unsqueeze(1) * x0 + sqrt_h_t.unsqueeze(1) * z
        pred = model(xt, t)
        target = a_t * x0
        loss = ((pred - target) ** 2).mean()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.load_state_dict(best_model_wts)
    return loss_history, model
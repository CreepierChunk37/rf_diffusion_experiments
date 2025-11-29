import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from tqdm import tqdm
import copy
from tqdm.auto import tqdm
import numpy as np

def train_random_noise(model, data, epochs=100000, lr=0.001, T=40.0, eps=1e-4, device='cuda'):
    optimizer = optim.SGD(model.parameters(), lr=lr)  
    num_data = data.shape[0]

    loss_history = []
    best_loss = float('inf')
    
    for epoch in tqdm(range(epochs)):
        x0 = data
        t = torch.rand(num_data, device=device) * (T - eps) + eps
        
        exp_neg_t = torch.exp(-t)
        sigma = torch.sqrt(1 - torch.exp(-2 * t))
        
        z = torch.randn_like(x0)
        xt = exp_neg_t.unsqueeze(1) * x0 + sigma.unsqueeze(1) * z
        
        pred_noise = model(xt, t)
        loss = ((pred_noise - z) ** 2).mean()
        
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

def sample_noise(model, num_samples=1000, dim=2, num_steps=5000, T=40.0, device='cuda', dtype=torch.float32):
    dt = (T-0.001) / num_steps 
    y = torch.randn(num_samples, dim, device=device, dtype=dtype)
    
    with torch.no_grad():
        for i in range(num_steps):
            t_val_current = T - (i * dt)
            t_val_prev = T - ((i + 1) * dt)

            t_tensor = torch.full((num_samples,), t_val_current, device=device, dtype=dtype)
            a_t = torch.exp(-t_tensor).view(-1, 1)
            bar_alpha_t = a_t.pow(2)

            t_prev_tensor = torch.full((num_samples,), t_val_prev, device=device, dtype=dtype)
            a_t_prev = torch.exp(-t_prev_tensor).view(-1, 1)
            bar_alpha_t_prev = a_t_prev.pow(2)

            epsilon_pred = model(y, t_tensor) 

            x0_pred = (y - torch.sqrt(1 - bar_alpha_t) * epsilon_pred) / a_t

            y = a_t_prev * x0_pred + torch.sqrt(1 - bar_alpha_t_prev) * epsilon_pred
    return y

def train_random_score(model, data, epochs=100000, batch_size=3000, lr=0.001, T=10.0, eps=1e-5, device='cuda'):
    optimizer = optim.SGD(model.parameters(), lr=lr)  
    num_data = data.shape[0]

    loss_history = []
    best_loss = float('inf')
    
    for epoch in tqdm(range(epochs)):
        x0 = data
        
        t = torch.rand(num_data, device=device) * (T - eps) + eps
        
        exp_neg_t = torch.exp(-t)
        sigma = torch.sqrt(1 - torch.exp(-2 * t))
        
        z = torch.randn_like(x0)
        xt = exp_neg_t.unsqueeze(1) * x0 + sigma.unsqueeze(1) * z
        
        pred_noise = model(xt, t)
        loss = ((sigma.unsqueeze(1) * pred_noise + z) ** 2).mean()
        
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

def sample_score(model, num_samples=1000, dim=2, num_steps=2000, T=10.0, device='cuda'):
    dt = (-T) / num_steps
    g = math.sqrt(2.0)
    x = torch.randn(num_samples, dim, device=device)
    t_val = T * torch.ones(num_samples, device=device)
    
    with torch.no_grad():
        for _ in range(num_steps):
            pred_noise = model(x, t_val)
            score = pred_noise 
            f = -x
            drift = f - (g ** 2) * score
            diffusion_term = g * math.sqrt(abs(dt)) * torch.randn_like(x)
            x = x + drift * dt + diffusion_term
            t_val += dt
    
    return x
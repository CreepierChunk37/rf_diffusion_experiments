import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
import copy
from tqdm.auto import tqdm
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def compute_A_infty(U_tilde, V_tilde, p):
    U_sym = 0.5 * (U_tilde + U_tilde.T) 
    eigvals, Q = np.linalg.eigh(U_sym)
    lam = np.clip(eigvals, 0.0, None)      
    Qt = Q.T
    invlam = np.zeros_like(lam)
    mask = lam > 1/ (p * 10000)
    invlam[mask] = 1.0 / lam[mask]
    U_pinv = (Q * invlam) @ Qt 

    lam_truncated_diag = np.zeros_like(lam)
    
    lam_truncated_diag[mask] = lam[mask]
    

    U_truncated = (Q * lam_truncated_diag) @ Qt

    A_inf = V_tilde @ U_pinv               
    
    return A_inf, U_truncated

class RandomFeatureModel(nn.Module):
    def __init__(self, input_dim, feature_dim, K_t, T=1.0, activation='relu'):
        super().__init__()
        self.input_dim = input_dim  
        self.feature_dim = feature_dim  
        self.K_t = K_t
        self.T = T
        self.activation_name = activation
        self.A = nn.Parameter(torch.randn(input_dim, feature_dim))
        self.register_buffer('W_x', torch.randn(feature_dim, input_dim) / math.sqrt(input_dim))
        self.register_buffer('W_t', torch.randn(feature_dim, 2 * K_t + 1) / math.sqrt(2 * K_t + 1))
        self.register_buffer('b', torch.randn(feature_dim))
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            self.activation = F.relu  # Default to ReLU
            
    def time_parametrization(self, t):
        t_normalized = t / self.T
        batch_size = t.shape[0]
        phi = torch.zeros(batch_size, 2 * self.K_t + 1, device=t.device)
        phi[:, 0] = 1
        for k in range(1, self.K_t + 1):
            phi[:, 2*k-1] = torch.sin(2 * math.pi * k * t_normalized)
            phi[:, 2*k] = torch.cos(2 * math.pi * k * t_normalized)
            
        return phi
    
    def forward(self, x, t):
        batch_size = x.shape[0]
        phi = self.time_parametrization(t)  
        x_proj = torch.matmul(x, self.W_x.t())   
        t_proj = torch.matmul(phi, self.W_t.t()) 
        pre_activation = x_proj + t_proj + self.b.unsqueeze(0).expand(batch_size, -1)
        activated_features = self.activation(pre_activation)
        output = torch.matmul(activated_features, self.A.t()) / math.sqrt(self.feature_dim)
        
        return output

def a_t(t):
    """Forward schedule function"""
    return np.exp(-t)

def h_t(t):
    """Variance schedule function"""
    return 1 - np.exp(-2 * t)

def phi_time(t, K, T):
    """Fourier time embedding of length 2K+1 at absolute time t in [0,T]."""
    tau = t / T
    ks = np.arange(1, K + 1)  # ks from 1 to K (not 0 to K-1)
    ang = 2.0 * math.pi * ks * tau
    # Include constant term (1) + sin/cos terms
    return np.concatenate([[1.0], np.sin(ang), np.cos(ang)], axis=0)  # (2K+1,)

def compute_UV_unweighted_consistent(X0, W_x, W_tau, b, ts, w, rng, p, d, n, K_t, T, 
                                    z_grid_np, ridge=1e-9, bias_switch=1):
    """
    Consistent implementation of compute_UV_unweighted using Simpson's rule for time integration
    and the same Monte Carlo approach as train_simpson
    """
    n_data = 3
    at_values = np.array([a_t(t) for t in ts])  # (L_t+1,)
    ht_values = np.array([h_t(t) for t in ts])  # (L_t+1,)
    phi_vectors = np.array([phi_time(t, K_t, T) for t in ts])  # (L_t+1, 2K+1)
    Wphi_values = phi_vectors @ W_tau.T / math.sqrt(2 * K_t + 1)  # (L_t+1, p) with √K normalization

    U_acc = np.zeros((p, p))
    V_acc = np.zeros((d, p))
    
    # Use the same time integration weights as train_simpson (Simpson's rule)
    t_grid_size = len(ts)
    simpson_weights = np.ones(t_grid_size)
    simpson_weights[1:-1:2] = 4.0  # Odd indices get weight 4
    simpson_weights[2:-1:2] = 2.0  # Even indices get weight 2
    simpson_weights /= (3.0 * (t_grid_size - 1))
    simpson_weights *= T # Normalize by 3*(n-1)
    
    
    for i, (t, wt) in tqdm(enumerate(zip(ts, simpson_weights))):
        at, ht = at_values[i], ht_values[i]
        Wphi = Wphi_values[i]  # (p,)

        U_t_hat = np.zeros((p, p))
        V_t_hat = np.zeros((d, p))

        z_grid_size = 4000  # Same as train_simpson
        Z_batch = rng.standard_normal((z_grid_size, d)) 
        
        # Use the same Monte Carlo approach as train_simpson
        # For each data point, use the same z_grid (repeated pattern like train_simpson)
        WZ_over_sqrt_d = np.einsum('pd,md->pm', W_x, Z_batch) / math.sqrt(d)  # (p, z_grid_size)
        
        WX_X0T = W_x @ X0.T  # (p, n_data)
        
        # mean term for all centers
        mean_i = WX_X0T * (at / math.sqrt(d + ridge))  # (p, n_data)
        
        # preactivations for this batch - same pattern as train_simpson
        if bias_switch:
            Z_all = mean_i[:, :, None] + WZ_over_sqrt_d[:, None, :] + b[:, None, None] + Wphi[:, None, None]  # (p, n_data, z_grid_size)
        else:
            Z_all = mean_i[:, :, None] + WZ_over_sqrt_d[:, None, :] + Wphi[:, None, None]  # (p, n_data, z_grid_size)
        
        H = np.maximum(0.0, Z_all)  # (p, n_data, z_grid_size)
        
        # Accumulate U_t contribution - same formula as train_simpson
        H_flat = H.reshape(p, -1)  # (p, n_data * z_grid_size)
        U_t_hat += (H_flat @ H_flat.T) / (n_data * z_grid_size * p)
        
        # Accumulate V_t contribution - same formula as train_simpson
        for i_data in range(n_data):
            target_i = at * X0[i_data, :]
            for m in range(z_grid_size):
                sigma_m = H[:, i_data, m]  # (p,) - activation for this noise sample
                contribution = np.outer(target_i, sigma_m)
                V_t_hat += contribution / (n_data * z_grid_size * p)

        
        U_acc += wt * U_t_hat
        V_acc += wt * V_t_hat

    return U_acc, V_acc

def train_simpson_consistent(
    model, U_tilde_np, V_tilde_np, # <-- 新增：预先计算的 U 和 V
    epochs=10000, lr=0.01, T=10.0, eps=1e-5, device='cuda', A_ref=None
): 
    A_ref = A_ref.to(device=device) if A_ref is not None else None
    A_diff_history = []
    
    dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    
    # 将 U, V 转换为 PyTorch Tensors
    U_tilde_tensor = torch.tensor(U_tilde_np, device=device, dtype=dtype)
    V_tilde_tensor = torch.tensor(V_tilde_np, device=device, dtype=dtype)
    
    # 从模型中获取 p 和 d
    p = model.feature_dim
    d = model.input_dim
    sqrt_p = math.sqrt(p)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []
    
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        A = model.A  # 维度 (d, p)
        
        term1 = (1.0 / d) * torch.trace(A.T @ A @ U_tilde_tensor)
        
        term2 = (2.0 * sqrt_p / d) * torch.trace(V_tilde_tensor @ A.T)
        
        loss = term1 - term2
        

        loss.backward()
        optimizer.step()

        if A_ref is not None:
            A_current = model.A / math.sqrt(model.feature_dim)
            diff = torch.norm(A_current - A_ref, p='fro').item()
            A_diff_history.append(diff)
        
        avg_epoch_loss = loss.item()
        loss_history.append(avg_epoch_loss)

    return loss_history, model, A_diff_history

def evaluate_model_loss_consistent(model, data, z_grid, T=10.0, eps=1e-5, device='cuda',
                        t_grid_size=201, z_grid_size=700, t_batch_size=70):
    """
    Consistent implementation of evaluate_model_loss using Simpson's rule
    """
    assert t_grid_size % 2 == 1, "t_grid_size must be odd for Simpson's rule"

    dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    data = data.to(device=device, dtype=dtype)
    z_grid = z_grid.to(device=device, dtype=dtype)
    t_grid = torch.linspace(eps, T, t_grid_size, device=device, dtype=dtype)

    # Simpson's rule weights
    simpson_avg_weights = torch.ones(t_grid_size, device=device, dtype=dtype)
    simpson_avg_weights[1:-1:2] = 4.0  # Odd indices get weight 4
    simpson_avg_weights[2:-1:2] = 2.0  # Even indices get weight 2
    simpson_avg_weights /= (3.0 * (t_grid_size - 1))  # Normalize by 3*(n-1)

    num_data = data.shape[0]
    data_dim = data.shape[1:]
    data_flat_size = 1
    if len(data_dim) > 0:
        data_flat_size = torch.prod(torch.tensor(data_dim)).item()

    x0 = data
    total_loss = 0.0
    t_start_index = 0 

    with torch.no_grad():
        for t_batch in torch.split(t_grid, t_batch_size):
            current_t_batch_size = t_batch.shape[0]
            
            x0_expanded = x0.repeat_interleave(current_t_batch_size * z_grid_size, dim=0)
            t_expanded = t_batch.repeat_interleave(z_grid_size).repeat(num_data)
            
            z_expanded = z_grid.repeat(num_data * current_t_batch_size, *([1] * len(data_dim)))
            
            exp_neg_t = torch.exp(-t_expanded)
            a_t = exp_neg_t
            sigma = torch.sqrt(1 - torch.exp(-2 * t_expanded))
            view_shape = (-1,) + (1,) * len(data_dim)
            xt = exp_neg_t.view(*view_shape) * x0_expanded + sigma.view(*view_shape) * z_expanded

            # Use trained model for prediction
            pred = model(xt, t_expanded)
            target = a_t.view(*view_shape) * x0_expanded
            squared_error = (pred - target) ** 2
            squared_error_view = squared_error.view(num_data, current_t_batch_size, z_grid_size, data_flat_size)

            loss_per_t = squared_error_view.mean(dim=(0, 2, 3)) 
            
            t_end_index = t_start_index + current_t_batch_size
            batch_weights = simpson_avg_weights[t_start_index:t_end_index]
            t_start_index = t_end_index 
            
            scaled_batch_loss = (loss_per_t * batch_weights).sum()
            total_loss += scaled_batch_loss.item() 

    return total_loss

def sample_exponential(model, num_samples=1000, dim=2, num_steps=10000, T=40.0, device='cuda',
                       num_snapshots=0):
    """
    Sampling function - unchanged as it doesn't involve the M C or time integration issues
    """
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
    
    t_min = 0.0001  # Simulation start time (epsilon)
    t_max = T      # Simulation end time
    L_sde = num_steps # Number of steps

    # Calculate q for exponential scheduling
    if t_min <= 0:
        raise ValueError("t_min must be greater than 0 for exponential scheduling.")
        
    q = (t_max / t_min) ** (1.0 / L_sde)

    time_points_np = np.array([t_min * (q ** i) for i in range(L_sde + 1)])
    
    time_points = torch.tensor(time_points_np, device=device, dtype=dtype)

    with torch.no_grad():
        for i in range(num_steps):
            t_i = time_points[i]
            t_i_plus_1 = time_points[i+1]
            dt = t_i_plus_1 - t_i  
            
            # t_val is current time t_i
            t_val = torch.full((num_samples,), t_i, device=device, dtype=dtype)
            T_minus_t = T - t_val
            h_T_minus_t = 1 - torch.exp(-2 * T_minus_t)
            h_T_minus_t = h_T_minus_t.view(-1, 1)  
            
            h_T_minus_t = torch.clamp(h_T_minus_t, min=1e-9)
            
            model_output = model(y, T_minus_t)

            drift = y - (2 / h_T_minus_t) * y + (2 / h_T_minus_t) * model_output
            
            diffusion_term = g * math.sqrt(dt) * torch.randn_like(y)
            
            # Euler-Maruyama update
            y = y + drift * dt + diffusion_term
            
            # Snapshot logic
            current_step = i + 1
            if current_step in snapshot_indices_set:
                snapshots.append(y.cpu().clone())
                
    if num_snapshots > 0:
        return snapshots
    else:
        # Return final sampling result at time T
        return y

# Algorithm calling code - referencing the original notebook pattern
def run_consistent_experiment():
    """
    Main experiment function that replicates the notebook's algorithm calling pattern
    using the consistent implementations
    """
    # Data setup - same as notebook
    data = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], device=device)
    numpy_array = data.cpu().numpy()
    
    # Model parameters - same as notebook
    input_dim = 2
    feature_dim = 2048  # p
    K_t = 64
    T = 40.0
    L_t = 1000  # Number of time points for integration
    
    # Monte Carlo parameters
    d = input_dim
    n = 3  # Number of data points
    p = feature_dim
    
    # Create z_grid for training - same as notebook
    z_grid = torch.randn(700, *data.shape[1:], device=device)
    z_grid_np = z_grid.detach().cpu().numpy()
    
    # Initialize model with zero parameters - same as notebook
    torch.manual_seed(114514)
    model = RandomFeatureModel(input_dim=input_dim, feature_dim=feature_dim, K_t=K_t, T=T).to(device)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    
    # Extract model parameters for theoretical calculation
    W_x_np = model.W_x.cpu().numpy()
    W_t_np = model.W_t.cpu().numpy()
    b_np = model.b.cpu().numpy()
    
    # Setup time grid and weights for theoretical calculation
    ts = np.linspace(0.0, T, L_t + 1)
    
    # Use consistent time integration (Simpson's rule)
    t_grid_size = len(ts)
    w = np.ones(t_grid_size)
    w[1:-1:2] = 4.0  # Odd indices get weight 4
    w[2:-1:2] = 2.0  # Even indices get weight 2
    w /= (3.0 * (t_grid_size - 1))  # Normalize by 3*(n-1)
    
    # Calculate theoretical equilibrium point using consistent method
    rng = np.random.default_rng(42)
    U_tilde, V_tilde = compute_UV_unweighted_consistent(
        numpy_array, W_x_np, W_t_np, b_np, ts, w, rng, p, d, n, K_t, T, z_grid_np
    )
    
    A_ref, U_truncated = compute_A_infty(U_tilde, V_tilde, feature_dim)
    A_ref_tensor = torch.tensor(A_ref, device=device, dtype=torch.float32)
        
        # Train model
    z_grid = torch.randn(700, *data.shape[1:], device=device)
    loss_history, trained_model, A_diff_history = train_simpson_consistent(
            model, U_tilde_np=U_tilde, V_tilde_np=V_tilde, epochs=400000, lr=0.05, T=T,A_ref=A_ref_tensor  # Reduced for faster comparison
        )
    
    # Evaluate final model performance
    final_loss = 0.0
    
    # Generate samples
    print("Generating samples...")
    samples = sample_exponential(
        trained_model, num_samples=1000, T=T, device=device
    )
    
    # Calculate final difference between trained A and theoretical equilibrium
    final_A_diff = torch.norm(trained_model.A - A_ref_tensor, p='fro').item()
    
    results = {
        'loss_history': loss_history,
        'A_diff_history': A_diff_history,
        'final_loss': final_loss,
        'final_A_diff': final_A_diff,
        'samples': samples,
        'trained_model': trained_model,
        'theoretical_A': A_ref_tensor,
        'data': data
    }
    
    print(f"Training completed!")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Final A difference from theoretical equilibrium: {final_A_diff:.6f}")
    
    return results

def visualize_results(results):
    """
    Visualize the experimental results with English labels
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract results
    loss_history = results['loss_history']
    A_diff_history = results['A_diff_history']
    samples = results['samples'].cpu().numpy()
    data = results['data'].cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Loss History
    epochs = list(range(1, len(loss_history) + 1))
    axes[0, 0].plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss History', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. A Parameter Difference from Theoretical Equilibrium
    if A_diff_history:
        epochs_diff = list(range(1, len(A_diff_history) + 1))
        axes[0, 1].plot(epochs_diff, A_diff_history, 'r-', linewidth=2, 
                       label='||A_trained - A_theory||_F')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Frobenius Norm Difference', fontsize=12)
        axes[0, 1].set_title('Parameter Convergence to Theoretical Equilibrium', 
                           fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # 3. Sample Distribution Scatter Plot
    axes[1, 0].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=20, 
                      color='blue', label='Generated Samples')
    axes[1, 0].scatter(data[:, 0], data[:, 1], color='red', marker='s', 
                      s=150, edgecolors='black', label='Original Data', zorder=5)
    axes[1, 0].set_xlabel('X-coordinate', fontsize=12)
    axes[1, 0].set_ylabel('Y-coordinate', fontsize=12)
    axes[1, 0].set_title('Generated Samples vs Original Data', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal', adjustable='box')
    
    # 4. Sample Density Heatmap
    try:
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="viridis", 
                   fill=True, ax=axes[1, 1], levels=10, alpha=0.7)
        axes[1, 1].scatter(data[:, 0], data[:, 1], color='red', marker='s', 
                          s=150, edgecolors='black', label='Original Data', zorder=5)
        axes[1, 1].set_xlabel('X-coordinate', fontsize=12)
        axes[1, 1].set_ylabel('Y-coordinate', fontsize=12)
        axes[1, 1].set_title('Sample Density Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
    except ValueError as e:
        axes[1, 1].text(0.5, 0.5, f'Density Plot Failed\n{str(e)}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=12, color='red')
        axes[1, 1].set_title('Sample Density Distribution (Failed)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('consistent_implementation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    print(f"Final Training Loss: {results['final_loss']:.6f}")
    print(f"Final A Parameter Difference: {results['final_A_diff']:.6f}")
    print(f"Number of Training Epochs: {len(loss_history)}")
    print(f"Number of Generated Samples: {len(samples)}")
    print(f"Original Data Points: {len(data)}")
    
    if A_diff_history:
        print(f"Initial A Difference: {A_diff_history[0]:.6f}")
        print(f"Final A Difference: {A_diff_history[-1]:.6f}")
        improvement = (A_diff_history[0] - A_diff_history[-1]) / A_diff_history[0] * 100
        print(f"Improvement: {improvement:.2f}%")
    
    print("="*60)

def compare_different_t_grid_sizes():
    """
    Compare results with different t_grid_size values (similar to notebook)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Data setup
    data = torch.tensor([[1.0, 0.0], [6.0, 0.0], [1.0, 5.0]], device=device)
    numpy_array = data.cpu().numpy()
    
    # Model parameters
    input_dim = 2
    feature_dim = 2048
    K_t = 64
    T = 40.0
    
    # Different t_grid_sizes to compare
    t_grid_sizes = [101, 201]
    results_dict = {}
    
    for t_grid_size in t_grid_sizes:
        print(f"\n=== Training with t_grid_size={t_grid_size} ===")
        
        # Create fresh model for each experiment
        torch.manual_seed(114514)
        model = RandomFeatureModel(input_dim=input_dim, feature_dim=feature_dim, 
                                 K_t=K_t, T=T).to(device)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        
        # Extract parameters
        W_x_np = model.W_x.cpu().numpy()
        W_t_np = model.W_t.cpu().numpy()
        b_np = model.b.cpu().numpy()
        
        # Setup for theoretical calculation
        L_t = t_grid_size - 1
        ts = np.linspace(0.0, T, L_t + 1)
        w = np.ones(t_grid_size)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w /= (3.0 * (t_grid_size - 1))
        
        # Calculate theoretical equilibrium
        rng = np.random.default_rng(42)
        U_tilde, V_tilde = compute_UV_unweighted_consistent(
            numpy_array, W_x_np, W_t_np, b_np, ts, w, rng, feature_dim, input_dim, 3, K_t, T
        )
        A_ref, U_truncated = compute_A_infty(U_tilde, V_tilde, feature_dim)
        A_ref_tensor = torch.tensor(A_ref, device=device, dtype=torch.float32)
        
        # Train model
        z_grid = torch.randn(700, *data.shape[1:], device=device)
        loss_history, trained_model, A_diff_history = train_simpson_consistent(
            model, U_tilde_np=U_truncated, V_tilde_np=V_tilde, epochs=10000, lr=0.05, T=T,A_ref=A_ref_tensor  # Reduced for faster comparison
        )
        
        # Generate samples
        samples = sample_exponential(trained_model, num_samples=1000, T=T, device=device)
        
        results_dict[t_grid_size] = {
            'loss_history': loss_history,
            'A_diff_history': A_diff_history,
            'samples': samples.cpu().numpy(),
            'final_loss': loss_history[-1],
            'final_A_diff': A_diff_history[-1] if A_diff_history else None
        }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparison Across Different t_grid_size Values', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red']
    
    # 1. Loss History Comparison
    for i, (t_grid_size, results) in enumerate(results_dict.items()):
        epochs = list(range(1, len(results['loss_history']) + 1))
        axes[0, 0].plot(epochs, results['loss_history'], 
                       color=colors[i], linewidth=2, label=f't_grid_size={t_grid_size}')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. A Difference Comparison
    for i, (t_grid_size, results) in enumerate(results_dict.items()):
        if results['A_diff_history']:
            epochs = list(range(1, len(results['A_diff_history']) + 1))
            axes[0, 1].plot(epochs, results['A_diff_history'], 
                           color=colors[i], linewidth=2, label=f't_grid_size={t_grid_size}')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Frobenius Norm Difference', fontsize=12)
    axes[0, 1].set_title('Parameter Convergence Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sample Distribution Comparison
    for i, (t_grid_size, results) in enumerate(results_dict.items()):
        samples = results['samples']
        axes[1, 0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=15, 
                          color=colors[i], label=f't_grid_size={t_grid_size}')
    
    # Add original data points
    data_np = data.cpu().numpy()
    axes[1, 0].scatter(data_np[:, 0], data_np[:, 1], color='black', marker='s', 
                      s=200, edgecolors='white', label='Original Data', zorder=5)
    axes[1, 0].set_xlabel('X-coordinate', fontsize=12)
    axes[1, 0].set_ylabel('Y-coordinate', fontsize=12)
    axes[1, 0].set_title('Sample Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal', adjustable='box')
    
    # 4. Density Comparison
    for i, (t_grid_size, results) in enumerate(results_dict.items()):
        samples = results['samples']
        try:
            sns.kdeplot(x=samples[:, 0], y=samples[:, 1], 
                       fill=False, levels=5, alpha=0.6, 
                       color=colors[i], ax=axes[1, 1], 
                       label=f't_grid_size={t_grid_size}')
        except ValueError:
            pass
    
    axes[1, 1].scatter(data_np[:, 0], data_np[:, 1], color='black', marker='s', 
                      s=200, edgecolors='white', label='Original Data', zorder=5)
    axes[1, 1].set_xlabel('X-coordinate', fontsize=12)
    axes[1, 1].set_ylabel('Y-coordinate', fontsize=12)
    axes[1, 1].set_title('Density Contour Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('t_grid_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comparison summary
    print("\n" + "="*80)
    print("T_GRID_SIZE COMPARISON SUMMARY")
    print("="*80)
    for t_grid_size, results in results_dict.items():
        print(f"\nt_grid_size={t_grid_size}:")
        print(f"  Final Loss: {results['final_loss']:.6f}")
        print(f"  Final A Difference: {results['final_A_diff']:.6f}" if results['final_A_diff'] else "  Final A Difference: N/A")
    print("="*80)

if __name__ == "__main__":
    # Run the consistent experiment
    print("Running consistent experiment...")
    results = run_consistent_experiment()
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results)


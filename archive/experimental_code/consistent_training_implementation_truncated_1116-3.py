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

import numpy as np

def compute_A_infty(U_tilde_np, V_tilde_np, p):
    U_tilde_f32 = U_tilde_np.astype(np.float64)
    V_tilde_f32 = V_tilde_np.astype(np.float64)

    try:
        U_sym = 0.5 * (U_tilde_f32 + U_tilde_f32.T)
        eigvals, Q = np.linalg.eigh(U_sym)
        tol = np.finfo(np.float32).eps 

        lam = np.clip(eigvals, 0.0, None)
        Qt = Q.T
        invlam = np.zeros_like(lam)
        mask = lam > 1/(2048 * 10000)
        invlam[mask] = 1.0 / lam[mask]

        U_pinv = (Q * invlam) @ Qt
        A_inf = V_tilde_f32 @ U_pinv

    except np.linalg.LinAlgError:
        print("在 float32 下计算伪逆失败")
        return None, None

    lam_truncated_diag = np.zeros_like(lam)
    lam_truncated_diag[mask] = lam[mask]
    U_truncated = (Q * lam_truncated_diag) @ Qt

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
    tau = t / T
    output_dim = 2 * K + 1
    phi = np.zeros(output_dim)
    
    # Constant term
    phi[0] = 1.0
    
    # Interleaved sin/cos terms
    for k in range(1, K + 1):
        ang = 2.0 * math.pi * k * tau
        phi[2 * k - 1] = math.sin(ang)
        phi[2 * k]     = math.cos(ang)
        
    return phi

def compute_UV_unweighted_consistent(X0, W_x, W_tau, b, ts, w, rng, p, d, n, K_t, T, 
                                    z_grid_np, bias_switch=1):
    """
    Consistent implementation of compute_UV_unweighted using Simpson's rule for time integration
    and the same Monte Carlo approach as train_simpson
    """
    n_data = n
    at_values = np.array([a_t(t) for t in ts])  # (L_t+1,)
    ht_values = np.array([h_t(t) for t in ts])  # (L_t+1,)
    phi_vectors = np.array([phi_time(t, K_t, T) for t in ts])  # (L_t+1, 2K+1)
    Wphi_values = phi_vectors @ W_tau.T   # (L_t+1, p) with √K normalization

    U_acc = np.zeros((p, p))
    V_acc = np.zeros((d, p))
    H_sum = np.zeros((p, n_data))
    
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

        z_grid_size = 2000  # Same as train_simpson
        Z_batch = rng.standard_normal((z_grid_size, d)) 
        sqrt_ht_Z_batch = math.sqrt(ht) * Z_batch
        
        # Use the same Monte Carlo approach as train_simpson
        # For each data point, use the same z_grid (repeated pattern like train_simpson)
        WZ_over_sqrt_d = np.einsum('pd,md->pm', W_x, sqrt_ht_Z_batch)   # (p, z_grid_size)
        
        WX_X0T = W_x @ X0.T  # (p, n_data)
        
        # mean term for all centers
        mean_i = WX_X0T * at  # (p, n_data)
        
        # preactivations for this batch - same pattern as train_simpson
        if bias_switch:
            Z_all = mean_i[:, :, None] + WZ_over_sqrt_d[:, None, :] + b[:, None, None] + Wphi[:, None, None]  # (p, n_data, z_grid_size)
        else:
            Z_all = mean_i[:, :, None] + WZ_over_sqrt_d[:, None, :] + Wphi[:, None, None]  # (p, n_data, z_grid_size)
        
        H = np.maximum(0.0, Z_all)  # (p, n_data, z_grid_size)
        
        # Accumulate U_t contribution - same formula as train_simpson
        H_flat = H.reshape(p, -1)  # (p, n_data * z_grid_size)
        U_t_hat = (H_flat @ H_flat.T) / (n_data * z_grid_size * p)

        H_sum = H.sum(axis=2) 
        H_mean = H_sum / z_grid_size  
        V_t_hat = (at / (n_data * p)) * (X0.T @ H_mean.T)
        
        U_acc += wt * U_t_hat
        V_acc += wt * V_t_hat

    return U_acc, V_acc

def train_simpson_consistent(
    model, U_tilde_np, V_tilde_np, C, # <-- 新增：预先计算的 U 和 V
    epochs=10000, lr=0.01, T=10.0, eps=1e-5, device='cuda', A_ref=None): 
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
        
        loss = term1 - term2 + C
        
        loss.backward()
        optimizer.step()

        if A_ref is not None:
            A_current = model.A / math.sqrt(model.feature_dim)
            diff = torch.norm(A_current - A_ref, p='fro').item()
            A_diff_history.append(diff)
        
        avg_epoch_loss = loss.item()
        loss_history.append(avg_epoch_loss)

    return loss_history, model, A_diff_history

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
    
    t_min = 0.000001  # Simulation start time (epsilon)
    t_max = T      # Simulation end time
    L_sde = num_steps # Number of steps

    # Calculate q for exponential scheduling
    if t_min <= 0:
        raise ValueError("t_min must be greater than 0 for exponential scheduling.")
        
    q = (t_max / t_min) ** (1.0 / L_sde)

    time_points_np = np.array([t_max * (q ** (-i)) for i in range(L_sde + 1)])
    
    time_points = torch.tensor(time_points_np, device=device, dtype=dtype)

    with torch.no_grad():
        for i in range(num_steps):
            t_current = time_points[i]
            t_next = time_points[i + 1]
            dt = t_current - t_next

            h_t = 1 - torch.exp(-2 * t_current)
            h_t = torch.clamp(h_t, min=1e-9)

            model_output = model(y, t_current.unsqueeze(0).expand(num_samples))
            diffusion_term = g * math.sqrt(dt) * torch.randn_like(y)
            drift = y - (2 / h_t) * y + (2 / h_t) * model_output
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

def compute_constant_term(data, T):
    if data.dim() == 1:
        # 处理单个数据点的情况 (形状 [d])
        n = 1
        d = data.shape[0]
    else:
        n = data.shape[0]
        d = data.shape[1]

    if n == 0 or d == 0:
        return 0.0

    # 2. 计算数据项: sum_{i=1}^n |x_0^(i)|^2
    # torch.sum(data**2) 计算所有元素的平方和,
    # 这等价于 sum_i (sum_j (x_ij^2)) = sum_i |x_0^(i)|^2
    sum_sq_norms = torch.sum(data**2).item()

    # 3. 计算积分项 (解析解): integral_0^T (exp(-t))^2 dt
    # 即: integral_0^T exp(-2t) dt = 0.5 * (1 - exp(-2T))
    integral_a_sq = 0.5 * (1.0 - math.exp(-2.0 * T))

    # 4. 组合所有项
    C = (1.0 / (n * d)) * integral_a_sq * sum_sq_norms
    
    return C

def compare_different_feature_dims():
    """
    Compare results with different feature_dim values
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Data setup
    data = torch.tensor([[0.0, 0.0], [5.0, 0.0]], device=device)
    data_np = data.cpu().numpy()
    numpy_array = data.cpu().numpy()
    
    # Model parameters
    input_dim = 2
    feature_dims = [2048]  # Different feature dimensions to compare
    K_t = 128
    T = 40.0
    L_t = 1000 # Number of time points for integration

    C = compute_constant_term(data=data, T=T)
    
    results_dict = {}
    
    for feature_dim in feature_dims:
        print(f"\n=== Training with feature_dim={feature_dim} ===")
        
        # Create fresh model for each experiment
        torch.manual_seed(42)
        model = RandomFeatureModel(input_dim=input_dim, feature_dim=feature_dim, 
                                 K_t=K_t, T=T).to(device)
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
        w /= (3.0 * (t_grid_size - 1)) 
        w *= T # Normalize by 3*(n-1)
        
        # Monte Carlo parameters
        d = input_dim
        n = 2  # Number of data points
        p = feature_dim
        
        # Create z_grid for training
        z_grid = torch.randn(2000, *data.shape[1:], device=device)
        z_grid_np = z_grid.detach().cpu().numpy()
        
        # Calculate theoretical equilibrium point using consistent method
        rng = np.random.default_rng(42)
        U_tilde, V_tilde = compute_UV_unweighted_consistent(
            numpy_array, W_x_np, W_t_np, b_np, ts, w, rng, p, d, n, K_t, T, z_grid_np
        )
        
        A_ref, U_truncated = compute_A_infty(U_tilde, V_tilde, feature_dim)
        A_ref_tensor = math.sqrt(feature_dim) * torch.tensor(A_ref, device=device, dtype=torch.float32)

        U_tilde_tensor = torch.tensor(U_tilde, device=device, dtype=torch.float32)
        V_tilde_tensor = torch.tensor(V_tilde, device=device, dtype=torch.float32)

        term1 = (1.0 / d) * torch.trace(A_ref_tensor.T @ A_ref_tensor @ U_tilde_tensor)
        term2 = (2.0 * math.sqrt(p) / d) * torch.trace(V_tilde_tensor @ A_ref_tensor.T)
        
        ground_truth = term1 - term2 
        print(ground_truth)

        loss_history, trained_model, A_diff_history = train_simpson_consistent(model, U_tilde_np=U_tilde, V_tilde_np=V_tilde, C = -ground_truth, epochs=1000, lr=0.0001, T=T,A_ref=A_ref_tensor)

        with torch.no_grad():
            model.A = nn.Parameter(A_ref_tensor.clone().to(device))
    
        final_loss = 0.0

        print("Generating samples...")
        samples = sample_exponential(
            model, num_samples=2000, T=T, device=device
        )

        final_A_diff = 0.0
        # final_A_diff = torch.norm(trained_model.A - A_ref_tensor, p='fro').item()
        
        results_dict[feature_dim] = {
            'loss_history': loss_history,
            'A_diff_history': A_diff_history,
            'samples': samples.cpu().numpy(),
            'final_loss': loss_history[-1] if loss_history else 0.0,
            'final_A_diff': final_A_diff,
            'model_state_dict': trained_model.state_dict(),
            'theoretical_A': A_ref_tensor
        }

    
    try:
        torch.save(results_dict, 'full_results_archive.pth')
        print("Successfully saved all results to 'full_results_archive.pth'")
    except Exception as e:
        print(f"CRITICAL: Failed to save results to disk. Error: {e}")

    
    for feature_dim, results in results_dict.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Results for feature_dim={feature_dim}', fontsize=16, fontweight='bold')
        
        # Loss History
        epochs = list(range(1, len(results['loss_history']) + 1))
        axes[0, 0].plot(epochs, results['loss_history'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Training Loss History', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # A Difference History
        if results['A_diff_history']:
            epochs_diff = list(range(1, len(results['A_diff_history']) + 1))
            axes[0, 1].plot(epochs_diff, results['A_diff_history'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Frobenius Norm Difference', fontsize=12)
            axes[0, 0].set_yscale('log')
            axes[0, 1].set_title('Parameter Convergence to Theoretical Equilibrium', 
                               fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Sample Distribution
        samples = results['samples']
        axes[1, 0].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=20, color='blue')
        axes[1, 0].scatter(data_np[:, 0], data_np[:, 1], color='red', marker='s', 
                          s=150, edgecolors='black', label='Original Data', zorder=5)
        axes[1, 0].set_xlabel('X-coordinate', fontsize=12)
        axes[1, 0].set_ylabel('Y-coordinate', fontsize=12)
        axes[1, 0].set_title('Generated Samples vs Original Data', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect('equal', adjustable='box')
        
        # Sample Density
        try:
            sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="viridis", 
                       fill=True, ax=axes[1, 1], levels=10, alpha=0.7)
            axes[1, 1].scatter(data_np[:, 0], data_np[:, 1], color='red', marker='s', 
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
        plt.savefig(f'feature_dim_{feature_dim}_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        
if __name__ == "__main__":
    # Compare different feature dimensions
    print("\nComparing different feature dimensions...")
    feature_dim_results = compare_different_feature_dims()

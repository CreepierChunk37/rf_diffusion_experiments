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
from scipy.linalg import expm
import matplotlib.pyplot as plt
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
data_sampling_mode = "centers"
m_z = 4000 

def compute_A_infty(U_tilde, V_tilde, p):
    # Symmetrize + tiny ridge for numerics
    U_sym = 0.5 * (U_tilde + U_tilde.T) 
    eigvals, Q = np.linalg.eigh(U_sym)

    # Project to PSD (clip tiny negatives to 0 to prevent overflow)
    lam = np.clip(eigvals, 0.0, None)      # shape (p,)
    Qt = Q.T

    # Build pseudoinverse term: Q diag(1/lam) Q^T with 1/lam = 0 on the nullspace
    invlam = np.zeros_like(lam)
    mask = lam > 1/ (p * 10000)
    invlam[mask] = 1.0 / lam[mask]
    U_pinv = (Q * invlam) @ Qt                      # (p, p)  == Ũ^+

    # Steady-state A(∞)/√p = Ṽ Ũ^+
    A_inf = V_tilde @ U_pinv               # (d, p)
    
    return A_inf

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
def compute_UV_unweighted(X0, W_x, W_tau, b, ts, w, rng, p, d, T, K, bias_switch=1):
    batch_size = 2000
    
    U_acc = np.zeros((p, p))
    V_acc = np.zeros((d, p))

    # Determine the number of data points based on sampling mode
    if data_sampling_mode == "centers":
        n_data = X0.shape[0]  # Number of centers (NUM_MODES)
    else:
        raise ValueError(f"Unknown data_sampling_mode: {data_sampling_mode}")

    # Vectorize over time: compute all time-dependent quantities at once
    at_values = np.array([a_t(t) for t in ts])  # (L_t+1,)
    ht_values = np.array([h_t(t) for t in ts])  # (L_t+1,)
    phi_vectors = np.array([phi_time(t, K, T) for t in ts])  # (L_t+1, 2K+1)
    Wphi_values = phi_vectors @ W_tau.T   # (L_t+1, p) with √K normalization

    # Create progress bar for UV computation
    pbar = create_progress_bar(len(ts), "UV computation", time.time())
    
    for i, (t, wt) in enumerate(zip(ts, w)):
        at, ht = at_values[i], ht_values[i]
        Wphi = Wphi_values[i]  # (p,)

        # Initialize accumulators for this time step
        U_t_hat = np.zeros((p, p))
        V_t_hat = np.zeros((d, p))
        H_sum = np.zeros((p, n_data))  # Accumulator for H sum over all m_z samples
        
        if data_sampling_mode == "centers":
            # Process all centers at once (no batching needed for small number of centers)
            WX_X0T = W_x @ X0.T  # (p, n_data)
            
            # Batch over m_z noise samples
            m_z_batches = (m_z + batch_size - 1) // batch_size
            
            for m_batch_idx in range(m_z_batches):
                start_m = m_batch_idx * batch_size
                end_m = min((m_batch_idx + 1) * batch_size, m_z)
                m_batch_size = end_m - start_m
                
                # Generate noise for this batch of m_z samples
                Z_batch = rng.standard_normal((n_data, m_batch_size, d))  # (n_data, m_batch_size, d)
                WZ_over_sqrt_d_batch = math.sqrt(ht) * np.einsum('pd,nmd->pnm', W_x, Z_batch)  # (p, n_data, m_batch_size)
                
                # mean term for all data points
                mean_i = WX_X0T * (at)  # (p, n_data)
                # preactivations for this batch
                if bias_switch:
                    Z_all_batch = mean_i[:, :, None] + WZ_over_sqrt_d_batch + b[:, None, None] + Wphi[:, None, None]  # (p, n_data, m_batch_size)
                else:
                    Z_all_batch = mean_i[:, :, None] + WZ_over_sqrt_d_batch + Wphi[:, None, None]  # (p, n_data, m_batch_size)
                H_batch = np.maximum(0.0, Z_all_batch)  # (p, n_data, m_batch_size)
                
                # Accumulate U_t contribution from this batch
                H_flat_batch = H_batch.reshape(p, -1)  # (p, n_data * m_batch_size)
                U_t_hat += (H_flat_batch @ H_flat_batch.T) / (n_data * m_z * p)
                
                # Accumulate H sum over all m_z samples
                H_sum += H_batch.sum(axis=2)  # (p, n_data) - sum over m_batch_size samples in this batch
            
            # Compute V_t_hat using proper average over all m_z samples
            H_mean = H_sum / m_z  # (p, n_data) - average over all m_z samples
            V_t_hat = (at / (n_data * p)) * (X0.T @ H_mean.T)  # (d, p)
        
        # Aggregate over time (trapezoid)
        U_acc += wt * U_t_hat
        V_acc += wt * V_t_hat
        
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()

    return U_acc, V_acc
def train_simpson_consistent(
    model, U_tilde_np, V_tilde_np, C, # <-- 新增：预先计算的 U 和 V
    epochs=10000, lr=0.01, T=10.0, eps=0.0, device='cuda', A_ref=None): 
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

def sample_exponential(model, num_samples=1000, dim=2, num_steps=4000, T=40.0, device='cuda',
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
    
    t_min = 1e-4  # Simulation start time (epsilon)
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

def calculate_A_tau(tau: float, TildeU: np.ndarray, TildeV: np.ndarray) -> np.ndarray:
    TildeU_pinv = np.linalg.pinv(TildeU)
    VU_pinv = TildeV @ TildeU_pinv
    matrix_exponential = expm(-2 * tau * TildeU)
    term1 = - VU_pinv 
    term2 = term1 @ matrix_exponential
    result = term2 + VU_pinv

    return result

def visualize_snapshots(snapshots, cols=3, bounds=(-4, 4), save_plot_path=None):
    """
    绘制 snapshot 网格。
    
    Args:
        snapshots: List of tensors, shape (N, 2)
        cols: 列数
        bounds: 绘图边界 (min, max)
    """
    num_snapshots = len(snapshots)
    rows = math.ceil(num_snapshots / cols)
    
    # 设置画布大小，保证每个子图比例合适
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)
    axes_flat = axes.flatten()
    
    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        
        if i < num_snapshots:
            data = snapshots[i].numpy()
            
            # 绘制散点
            # alpha 设置透明度，方便观察高密度区域
            # s 设置点的大小
            ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.6, c='blue', edgecolors='none')
            
            # 设置标题和坐标轴
            ax.set_title(f"Snapshot {i+1} / {num_snapshots}", fontsize=10)
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
            ax.set_aspect('equal') # 保持几何比例
            ax.grid(True, linestyle='--', alpha=0.3)
        else:
            # 隐藏多余的子图 (例如 20 个图，3 列，第 21 个位置空白)
            ax.axis('off')
            
    plt.suptitle(f"Reverse Diffusion Process ({num_snapshots} Snapshots)", fontsize=16)
    if save_plot_path:
        directory = os.path.dirname(save_plot_path)
        # 只有当 directory 不为空字符串时，才去创建文件夹
        if directory: 
            os.makedirs(directory, exist_ok=True)
            
        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
        print(f"[Info] Plot saved to: {save_plot_path}")
        

def compare_different_feature_dims():
    """
    Compare results with different feature_dim values
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Data setup
    data = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], device=device)
    data_np = data.cpu().numpy()
    numpy_array = data.cpu().numpy()
    
    # Model parameters
    input_dim = 2
    feature_dims = [2000]  # Different feature dimensions to compare
    K_t = 128
    T = 40.0
    L_t = 400 # Number of time points for integration

    C = compute_constant_term(data=data, T=T)
    
    results_dict = {}
    
    for feature_dim in feature_dims:
        print(f"\n=== Training with feature_dim={feature_dim} ===")
        
        # Create fresh model for each experiment
        torch.manual_seed(42)
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
        w = np.empty_like(ts)
        if len(ts) >= 2:
            w[1:-1] = (ts[2:] - ts[:-2]) / 2.0
            w[0] = (ts[1] - ts[0]) / 2.0
            w[-1] = (ts[-1] - ts[-2]) / 2.0
        else:
            w[:] = 1.0

        
        # Monte Carlo parameters
        d = input_dim
        n = data.shape[0]  # Number of data points
        p = feature_dim
        rng = np.random.default_rng(42)

        #U_tilde, V_tilde = compute_UV_unweighted(numpy_array, W_x_np, W_t_np, b_np, ts, w, rng, p, d, T, K_t)
        #np.savez_compressed('uv_matrices.npz', U=U_tilde, V=V_tilde)
        #print("矩阵已保存到 uv_matrices.npz")

        data = np.load('uv_matrices.npz')
        U_tilde = data['U']
        V_tilde = data['V']
        print("矩阵加载成功！")
        print(f"U shape: {U_tilde.shape}")
        print(f"V shape: {V_tilde.shape}")

        A_ref = compute_A_infty(U_tilde, V_tilde, feature_dim)
        A_ref_tensor = math.sqrt(feature_dim) * torch.tensor(A_ref, device=device, dtype=torch.float32)

        A_tau = calculate_A_tau(300.0, U_tilde, V_tilde)
        A_true_tau = math.sqrt(feature_dim) * torch.tensor(A_tau, device=device, dtype=torch.float32)

        U_tilde_tensor = torch.tensor(U_tilde, device=device, dtype=torch.float32)
        V_tilde_tensor = torch.tensor(V_tilde, device=device, dtype=torch.float32)

        term1 = (1.0 / d) * torch.trace(A_ref_tensor.T @ A_ref_tensor @ U_tilde_tensor)
        term2 = (2.0 * math.sqrt(p) / d) * torch.trace(V_tilde_tensor @ A_ref_tensor.T)
        
        ground_truth = term1 - term2 
        print(ground_truth + C)

        loss_history, trained_model, A_diff_history = train_simpson_consistent(model, U_tilde_np=U_tilde, V_tilde_np=V_tilde, C = -ground_truth, epochs=600000, lr=0.001, T=T,A_ref=A_ref_tensor)
        
        final_loss = 0.0
        #model_1 = copy.deepcopy(model)
        #with torch.no_grad():
            #model_1.A = nn.Parameter(A_true_tau.clone().to(device))

        print("Generating samples...")
        samples = sample_exponential(
            trained_model, num_samples=2000, T=T, device=device
        )

        with torch.no_grad():
            # 构造一个位于数据中心 x0=[5, 0] 的样本
            # 在 t=0 时，a_t=1，模型应该输出 x0 本身
            x_test = torch.tensor([[5.0, 0.0]], device=device) 
            t_test = torch.tensor([0.0], device=device) # t=0
            
            # 临时赋值权重
            # model.A = nn.Parameter(A_ref_tensor.clone())
            
            # 预测
            out = trained_model(x_test, t_test)
            print(f"Test Input: [5, 0]")
            print(f"Model Output at t=0: {out[0].cpu().numpy()}")

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

        snapshots_data = sample_exponential(model=trained_model, num_samples=2000, device=device, num_snapshots=20)
        visualize_snapshots(snapshots_data, cols=3, bounds=(-2, 7), save_plot_path=f'feature_dim_{feature_dim}_m4000_snapshots_A300_practice.png')

    
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
        axes[1, 0].set_xlim(-2,7)
        axes[1, 0].set_ylim(-2,7)
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
        plt.savefig(f'feature_dim_{feature_dim}_m4000_results_A300_practice.png', dpi=300, bbox_inches='tight')
        plt.show()

        
if __name__ == "__main__":
    # Compare different feature dimensions
    print("\nComparing different feature dimensions...")
    feature_dim_results = compare_different_feature_dims()

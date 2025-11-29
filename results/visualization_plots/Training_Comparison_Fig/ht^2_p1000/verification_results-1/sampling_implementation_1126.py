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
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
data_sampling_mode = "centers"
m_z = 4000  # 用于计算 U, V 的蒙特卡洛样本数
DATA_CENTERS = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])

def compute_A_infty(U_tilde, V_tilde, p, A0):
    d, p_dim = V_tilde.shape
    U_sym = 0.5 * (U_tilde + U_tilde.T) 
    eigvals, Q = np.linalg.eigh(U_sym)
    lam = np.clip(eigvals, 0.0, None)       # shape (p,)
    Qt = Q.T

    invlam = np.zeros_like(lam)
    mask = lam > 1.0 / (p * 10000)
    invlam[mask] = 1.0 / lam[mask]
    
    U_pinv = (Q * invlam) @ Qt              # (p, p)

    term_particular = V_tilde @ U_pinv      # (d, p)
    
    A0_scaled = A0 / np.sqrt(p)
    proj_null = np.eye(p_dim) - U_sym @ U_pinv
    term_null = A0_scaled @ proj_null
    
    A_inf = term_null + term_particular
    
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

def create_progress_bar(total, desc):
    return tqdm(total=total, desc=desc, unit="step", ncols=100, leave=False)
    
def compute_UV_unweighted(X0, W_x, W_tau, b, ts, w, rng, p, d, T, K, bias_switch=1):
    """
    计算理论矩阵 U_tilde 和 V_tilde (GPU 加速版本 - Double Precision).
    为了数值稳定性，全程使用 float64 计算，防止 float32 在处理 1/h_t^2 时溢出或精度不足。
    """
    # 确保使用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64  # 强制使用双精度，与 Numpy 默认行为一致

    # 1. 将静态输入转换为 GPU Tensor (Double)
    X0_torch = torch.from_numpy(X0).to(device, dtype=dtype)
    W_x_torch = torch.from_numpy(W_x).to(device, dtype=dtype)
    W_tau_torch = torch.from_numpy(W_tau).to(device, dtype=dtype)
    b_torch = torch.from_numpy(b).to(device, dtype=dtype)
    
    # 2. 预计算时间相关的量
    at_values = np.array([a_t(t) for t in ts])
    ht_values = np.array([h_t(t) for t in ts])
    phi_vectors = np.array([phi_time(t, K, T) for t in ts])
    
    at_torch = torch.from_numpy(at_values).to(device, dtype=dtype)
    ht_torch = torch.from_numpy(ht_values).to(device, dtype=dtype)
    w_torch = torch.from_numpy(w).to(device, dtype=dtype)
    phi_torch = torch.from_numpy(phi_vectors).to(device, dtype=dtype)
    
    # 预计算 Wphi: (L_t, 2K+1) @ (2K+1, p) -> (L_t, p)
    Wphi_values = phi_torch @ W_tau_torch.T
    
    # 3. 初始化累加器 (GPU Double)
    U_acc = torch.zeros((p, p), device=device, dtype=dtype)
    V_acc = torch.zeros((d, p), device=device, dtype=dtype)
    
    n_data = X0.shape[0]
    batch_size = 2000 
    m_z_batches = (m_z + batch_size - 1) // batch_size
    
    # 预计算数据相关项
    WX_X0T = W_x_torch @ X0_torch.T

    pbar = create_progress_bar(len(ts), "Computing U, V Matrices (GPU-Double)")
    
    # 4. 主时间循环
    for i, (t, _) in enumerate(zip(ts, w)):
        at = at_torch[i]
        ht = ht_torch[i]
        wt_val = w_torch[i]
        
        # 增加微小量保护，虽然 float64 应该没事，但双重保险
        ht_square = ht ** 2 + 1e-16
        
        Wphi = Wphi_values[i]  # (p,)

        U_t_hat = torch.zeros((p, p), device=device, dtype=dtype)
        H_sum = torch.zeros((p, n_data), device=device, dtype=dtype)
        
        for m_batch_idx in range(m_z_batches):
            start_m = m_batch_idx * batch_size
            end_m = min((m_batch_idx + 1) * batch_size, m_z)
            m_batch_size = end_m - start_m
            
            # 这里的 randn 也要生成 double 类型
            Z_batch = torch.randn((n_data, m_batch_size, d), device=device, dtype=dtype)
            
            # Einsum
            WZ_term = torch.sqrt(ht) * torch.einsum('pd,nmd->pnm', W_x_torch, Z_batch)
            
            mean_i = (WX_X0T * at).unsqueeze(2)
            
            if bias_switch:
                Z_all = mean_i + WZ_term + b_torch.view(p, 1, 1) + Wphi.view(p, 1, 1)
            else:
                Z_all = mean_i + WZ_term + Wphi.view(p, 1, 1)
            
            # ReLU
            H_batch = F.relu(Z_all)
            
            # Accumulate U_t
            H_flat = H_batch.reshape(p, -1)
            U_t_hat += (H_flat @ H_flat.T) / (n_data * m_z * p)
            
            # Accumulate H_sum
            H_sum += H_batch.sum(dim=2)
            
        # V_t_hat
        H_mean = H_sum / m_z
        V_t_hat = (at / (n_data * p)) * (X0_torch.T @ H_mean.T)
        
        # Integral Accumulation
        U_acc += wt_val * U_t_hat / ht_square
        V_acc += wt_val * V_t_hat / ht_square
        
        pbar.update(1)
    
    pbar.close()
    
    # 5. 转回 Numpy 返回
    return U_acc.cpu().numpy(), V_acc.cpu().numpy()

def sample_exponential(model, num_samples=1000, dim=2, num_steps=2000, T=40.0, device='cuda'):
    dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    g = math.sqrt(2.0)
    
    y = torch.randn(num_samples, dim, device=device, dtype=dtype)
    t_min = 5e-3
    t_max = T
    
    q = (t_max / t_min) ** (1.0 / num_steps)
    time_points_np = np.array([t_max * (q ** (-i)) for i in range(num_steps + 1)])
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
            
    return y.cpu().numpy()

def compute_constant_term_consistent(data_np, ts, w, T):
    n = data_np.shape[0]
    d = data_np.shape[1]
    at_values = np.exp(-ts)
    ht_values = 1.0 - np.exp(-2.0 * ts)
    ht_square = ht_values ** 2 
    integrand = (at_values ** 2) / ht_square
    integral_val = np.sum(w * integrand)
    sum_sq_norms = np.sum(data_np**2)
    C = (1.0 / (n * d)) * integral_val * sum_sq_norms
    return C

def calculate_A_tau(tau: float, TildeU: np.ndarray, TildeV: np.ndarray, p: int, A0: np.ndarray = None) -> np.ndarray:
    d, p = TildeV.shape

    TildeU_pinv = np.linalg.pinv(TildeU)
    VU_pinv = TildeV @ TildeU_pinv           # V U^+
    A0_scaled = A0 / np.sqrt(p)              # A(0) / sqrt(p)
    matrix_exponential = expm(-2 * tau * TildeU)

    diff_term = A0_scaled - VU_pinv
    result = (diff_term @ matrix_exponential) + VU_pinv

    return result

def count_samples_near_modes(samples, centers, radius=0.5):
    counts = []
    for center in centers:
        # 计算欧氏距离
        dists = np.linalg.norm(samples - center, axis=1)
        count = np.sum(dists < radius)
        counts.append(count)
    return counts

def kde2d_grid(samples, grid_size=150, padding=0.1):
    n, d_ = samples.shape
    assert d_ == 2, "kde2d_grid expects samples with shape (N, 2)."

    # Scott's rule (scalar bandwidth)
    std = samples.std(axis=0, ddof=1)
    h = n ** (-1.0 / (d_ + 4.0)) * float(std.mean())

    mins = samples.min(axis=0); maxs = samples.max(axis=0)
    span = maxs - mins
    mins = mins - padding * span
    maxs = maxs + padding * span

    xs = np.linspace(mins[0], maxs[0], grid_size)
    ys = np.linspace(mins[1], maxs[1], grid_size)
    X, Y = np.meshgrid(xs, ys)

    XY = np.stack([X.ravel(), Y.ravel()], axis=1)   # (G^2, 2)
    diffs = XY[:, None, :] - samples[None, :, :]    # (G^2, N, 2)
    sqnorm = np.sum(diffs * diffs, axis=2)          # (G^2, N)

    coef = 1.0 / (2.0 * np.pi * (h ** 2))
    D = (coef * np.mean(np.exp(-0.5 * sqnorm / (h ** 2)), axis=1)).reshape(grid_size, grid_size)
    return X, Y, D

def visualize_results(samples, centers, counts, feature_dim, tau, loss_val, save_dir="results_newKDE"):
    plt.figure(figsize=(10, 8))
    
    # 标题处理
    if tau == float('inf'):
        tau_str = "Infinity"
        file_suffix = "inf"
    else:
        tau_str = str(tau)
        file_suffix = str(tau)

    # --- 修改部分开始 ---
    # 1. 绘制 KDE 等高线填充图 (Filled Contours)
    try:
        # 计算密度网格
        Xg, Yg, Dg = kde2d_grid(samples, grid_size=150, padding=0.2)
        
        # 使用 contourf 进行填充绘制
        # levels=20: 增加层级让渐变更平滑
        # cmap='Oranges': 使用橙色系渐变 (也可以尝试 'YlOrBr', 'Reds')
        # alpha=0.6: 设置透明度，不要遮挡住蓝色的散点
        cf = plt.contourf(Xg, Yg, Dg, levels=30, cmap='Oranges', alpha=0.6)
        
        # 可选：如果想要边缘更清晰，可以叠加一层很淡的线，不需要的话可以注释掉下面这行
        plt.contour(Xg, Yg, Dg, levels=30, colors=['k'], linewidths=0.3, alpha=0.1)
        
    except Exception as e:
        print(f"Warning: KDE plot failed ({e}), skipping density background.")
    # --- 修改部分结束 ---

    # 2. 绘制生成的样本点 (蓝色小点)
    # zorder=2 保证点在背景之上
    plt.scatter(samples[:, 0], samples[:, 1], s=2, c='blue', alpha=0.4, label='Generated', zorder=2)

    # 3. 绘制真实数据中心 (红色 X)
    # zorder=3 保证中心点在最上层
    plt.scatter(centers[:, 0], centers[:, 1], s=180, c='red', marker='x', linewidth=3, label='True Centers', zorder=3)

    # 4. 在数据中心附近标注数量 (n=xxx)
    for i, center in enumerate(centers):
        count = counts[i]
        # 调整了样式以匹配你图片中的红棕色边框风格
        bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="darkred", lw=1.5, alpha=0.9)
        plt.text(center[0] + 0.4, center[1] + 0.4, f"n={count}", 
                 fontsize=13, fontweight='bold', color='darkred', bbox=bbox_props, zorder=4)

    # 图表装饰
    plt.title(f"Dim={feature_dim}, Tau={tau_str}, Loss={loss_val:.4f}", fontsize=15)
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    plt.gca().set_aspect('equal')
    
    # 图例
    plt.legend(loc='upper right', framealpha=0.9)
    
    # 网格线设置 (虚线，灰色)
    plt.grid(True, linestyle='--', alpha=0.4)

    # 保存逻辑
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"dim_{feature_dim}_tau_{file_suffix}.jpg") # 保存为 jpg 或 png
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")

def calculate_loss(A_tensor, U_tensor, V_tensor, C, p, d):
    term1 = (1.0 / d) * torch.trace(A_tensor.T @ A_tensor @ U_tensor)
    term2 = (2.0 * math.sqrt(p) / d) * torch.trace(V_tensor @ A_tensor.T)
    loss = term1 - term2 + C
    return loss.item()

def run_analytical_experiment(feature_dims, taus):
    # 实验设置
    input_dim = 2
    K_t = 128
    T = 40.0
    L_t = 4000
    
    # 准备时间网格和权重 (Simpson/Trapezoidal weights)
    start_t = 1e-3 
    ts = np.linspace(start_t, T, L_t + 1)
    dt = (T - start_t) / L_t
    w = np.zeros_like(ts)
    w[0] = 1.0; w[-1] = 1.0
    w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    w = w * (dt / 3.0)

    # 外层循环：Feature Dimension
    for p in feature_dims:
        print(f"\n{'='*40}")
        print(f"Processing Feature Dim: {p}")
        print(f"{'='*40}")

        # 1. 初始化模型
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        
        model = RandomFeatureModel(input_dim, p, K_t, T).to(device)

        A_init = rng.standard_normal((input_dim, p))
        
        W_x_np = model.W_x.cpu().numpy()
        W_t_np = model.W_t.cpu().numpy()
        b_np = model.b.cpu().numpy()
        d = input_dim
        
        # 2. 计算 U, V 矩阵
        print(f"Computing U and V matrices for dim={p}...")
        U_tilde, V_tilde = compute_UV_unweighted(
            DATA_CENTERS, W_x_np, W_t_np, b_np, ts, w, rng, p, d, T, K_t
        )
        
        C = compute_constant_term_consistent(DATA_CENTERS, ts, w, T)
        
        U_tensor = torch.tensor(U_tilde, device=device, dtype=torch.float32)
        V_tensor = torch.tensor(V_tilde, device=device, dtype=torch.float32)

        # 内层循环：Tau (Gradient Flow Time)
        for tau in taus:
            if tau == float('inf'):
                print(f"  -> Analyzing Tau = Infinity (Equilibrium)")
                # 计算 A(infinity) / sqrt(p)
                W_tau = compute_A_infty(U_tilde, V_tilde, p, A_init)
            else:
                print(f"  -> Analyzing Tau = {tau}")
                # 计算 A(tau) / sqrt(p)
                W_tau = calculate_A_tau(tau, U_tilde, V_tilde, p, A_init)
            
            # 恢复为模型的参数 A = W(tau) * sqrt(p)
            A_tau = W_tau * math.sqrt(p)
            A_tau_tensor = torch.tensor(A_tau, device=device, dtype=torch.float32)
            
            # 4. 计算 Loss
            current_loss = calculate_loss(A_tau_tensor, U_tensor, V_tensor, C, p, d)
            print(f"     Theoretical Loss: {current_loss:.6f}")
            
            # 5. 更新模型参数并采样
            with torch.no_grad():
                model.A.copy_(A_tau_tensor)
            
            # 采样
            print("     Sampling...")
            samples = sample_exponential(model, num_samples=2000, T=T, device=device)
            
            # 6. 统计与绘图
            counts = count_samples_near_modes(samples, DATA_CENTERS, radius=0.5)
            print(f"     Counts near modes: {counts}")
            
            visualize_results(samples, DATA_CENTERS, counts, p, tau, current_loss)

if __name__ == "__main__":
    DIMS_TO_TEST = [1000]
    TAUS_TO_TEST = [500, 1000, 5000, 10000] 
    
    run_analytical_experiment(DIMS_TO_TEST, TAUS_TO_TEST)


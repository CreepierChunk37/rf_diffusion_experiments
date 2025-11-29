import torch
import torch.optim as optim
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from sampling_implementation_1126 import (
    RandomFeatureModel, compute_UV_unweighted, calculate_A_tau, 
    sample_exponential, visualize_results, count_samples_near_modes,
    compute_constant_term_consistent, DATA_CENTERS,
    a_t, h_t, phi_time 
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def train_simpson_consistent(model, U_tilde_np, V_tilde_np, C, epochs, lr, T, device='cuda', A_ref=None): 
    if A_ref is not None:
        if isinstance(A_ref, np.ndarray):
            A_ref = torch.tensor(A_ref, device=device, dtype=torch.float32)
        else:
            A_ref = A_ref.to(device=device)

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
    
    # 使用 tqdm 显示进度，但如果是分段训练，可能需要外部控制描述
    iterator = range(epochs)
    
    for epoch in tqdm(iterator):
        optimizer.zero_grad()
        A = model.A  # 维度 (d, p)
        
        # Loss 计算: (1/d) Tr(A^T A U) - (2*sqrt(p)/d) Tr(V A^T) + C
        term1 = (1.0 / d) * torch.trace(A.T @ A @ U_tilde_tensor)
        term2 = (2.0 * sqrt_p / d) * torch.trace(V_tilde_tensor @ A.T)
        
        loss = term1 - term2 + C
        
        loss.backward()
        optimizer.step()

        # 记录与 A_ref 的差距 (如果有)
        if A_ref is not None:
            with torch.no_grad():
                diff = torch.norm(model.A - A_ref, p='fro').item()
                A_diff_history.append(diff)
        
        loss_history.append(loss.item())

    return loss_history, model, A_diff_history

def run_verification_experiment():
    # --- 实验配置 ---
    feature_dim = 1000      
    input_dim = 2           
    K_t = 128
    T = 40.0
    lr = 0.004               
    
    checkpoints_tau = [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000]
    
    save_dir = "verification_results"
    os.makedirs(save_dir, exist_ok=True)
    print("Initializing Model and Pre-computing Matrices...")
    
    seed = 42
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model_trained = RandomFeatureModel(input_dim, feature_dim, K_t, T).to(device)
    A0_np = model_trained.A.detach().cpu().numpy().copy()
    
    model_theory = copy.deepcopy(model_trained)

    L_t = 4000
    start_t = 5e-3 
    ts = np.linspace(start_t, T, L_t + 1)
    dt_val = (T - start_t) / L_t
    w = np.zeros_like(ts)
    w[0] = 1.0
    w[-1] = 1.0
    w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    w = w * (dt_val / 3.0)
    
    W_x_np = model_trained.W_x.cpu().numpy()
    W_t_np = model_trained.W_t.cpu().numpy()
    b_np = model_trained.b.cpu().numpy()
    
    print("Computing Integrals (U, V)... this might take a moment.")
    U_tilde, V_tilde = compute_UV_unweighted(
        DATA_CENTERS, W_x_np, W_t_np, b_np, ts, w, rng, feature_dim, input_dim, T, K_t
    )
    C_const = compute_constant_term_consistent(DATA_CENTERS, ts, w, T)
    print("Integrals computed.")
    eigvals = np.linalg.eigvalsh(U_tilde)
    max_lambda = eigvals[-1]
    print(f"Max Eigenvalue: {max_lambda:.6e}")

    # --- 主循环：分段训练与验证 ---
    
    current_tau = 0.0
    global_diff_history = [] # 记录 (tau, diff)
    
    
    for target_tau in checkpoints_tau:
        print(f"\n{'-'*20} Reaching Target Tau = {target_tau} {'-'*20}")
        delta_tau = target_tau - current_tau
        if delta_tau < 0:
            print("Warning: Target tau is behind current tau. Skipping.")
            continue
            
        epochs_needed = int(round((delta_tau * input_dim) / lr))
        print(f"Training for {epochs_needed} epochs (Delta Tau={delta_tau:.2f})...")
        
        # 2. 运行实际训练 (从当前状态继续)
        # 注意：每次调用都会新建 optimizer，对于 SGD (无动量) 来说这相当于继续训练
        _, model_trained, _ = train_simpson_consistent(
            model_trained, U_tilde, V_tilde, C_const, 
            epochs=epochs_needed, lr=lr, T=T, device=device, A_ref=None
        )
        
        # 更新当前时间
        current_tau = target_tau
        
        # 3. 计算理论解 A(tau)
        print(f"Calculating Theoretical Solution for Tau={target_tau}...")
        W_tau_theory = calculate_A_tau(target_tau, U_tilde, V_tilde, feature_dim, A0_np)
        A_tau_theory = W_tau_theory * math.sqrt(feature_dim) # 还原为参数尺度
        
        # 4. 对比权重矩阵 (Frobenius Norm)
        A_trained_np = model_trained.A.detach().cpu().numpy()
        diff_norm = np.linalg.norm(A_trained_np - A_tau_theory, 'fro')
        # 相对误差
        rel_diff = diff_norm / np.linalg.norm(A_tau_theory, 'fro')
        
        global_diff_history.append((target_tau, diff_norm))
        print(f"Checkpoint Tau={target_tau}:")
        print(f"  > Weight Difference (Frobenius): {diff_norm:.6f}")
        print(f"  > Relative Difference: {rel_diff:.6%}")
        
        # 5. 双重采样与可视化
        # 将理论权重加载到影子模型
        with torch.no_grad():
            model_theory.A.copy_(torch.from_numpy(A_tau_theory).to(device))
            
        print("  > Sampling from TRAINED model...")
        samples_trained = sample_exponential(model_trained, num_samples=2000, T=T, device=device)
        counts_trained = count_samples_near_modes(samples_trained, DATA_CENTERS)
        
        print("  > Sampling from THEORY model...")
        samples_theory = sample_exponential(model_theory, num_samples=2000, T=T, device=device)
        counts_theory = count_samples_near_modes(samples_theory, DATA_CENTERS)
        
        # 6. 绘图 (左右对比)
        plot_comparison(
            samples_trained, counts_trained, 
            samples_theory, counts_theory, 
            target_tau, diff_norm, feature_dim, save_dir
        )

    # 最后绘制误差随时间变化的曲线
    plot_diff_history(global_diff_history, save_dir)

def plot_comparison(samp_tr, counts_tr, samp_th, counts_th, tau, diff, dim, save_dir):
    """绘制两张子图：左边是实际训练，右边是理论计算"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    titles = [f"Actual Training (Tau={tau})", f"Theoretical Sol (Tau={tau})"]
    samples_list = [samp_tr, samp_th]
    counts_list = [counts_tr, counts_th]
    
    from sampling_implementation_1126 import kde2d_grid # 确保能调用到
    
    for i, ax in enumerate(axes):
        samples = samples_list[i]
        counts = counts_list[i]
        
        # KDE 背景
        try:
            Xg, Yg, Dg = kde2d_grid(samples, grid_size=100, padding=0.2)
            # 使用带颜色的填充
            ax.contourf(Xg, Yg, Dg, levels=30, cmap='Oranges', alpha=0.6)
        except:
            pass
            
        # 散点
        ax.scatter(samples[:, 0], samples[:, 1], s=2, c='blue', alpha=0.4, zorder=2)
        
        # 中心点
        ax.scatter(DATA_CENTERS[:, 0], DATA_CENTERS[:, 1], s=150, c='red', marker='x', lw=3, zorder=3)
        
        # 计数文字
        for idx, center in enumerate(DATA_CENTERS):
            bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", alpha=0.8)
            ax.text(center[0]+0.3, center[1]+0.3, f"n={counts[idx]}", 
                    fontsize=11, color='darkred', fontweight='bold', bbox=bbox, zorder=4)
            
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlim(-2, 7)
        ax.set_ylim(-2, 7)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle(f"Dim={dim}, Tau={tau} | Matrix Diff (Fro)={diff:.4f}", fontsize=16)
    
    save_path = os.path.join(save_dir, f"compare_tau_{tau}.jpg")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {save_path}")

def plot_diff_history(history, save_dir):
    taus, diffs = zip(*history)
    plt.figure(figsize=(8, 6))
    plt.plot(taus, diffs, 'o-', linewidth=2, color='purple')
    plt.xlabel(r"Gradient Flow Time $\tau$")
    plt.ylabel(r"Frobenius Norm $||A_{train} - A_{theory}||_F$")
    plt.title("Deviation between SGD Training and Theoretical Flow")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.yscale('log') # 通常误差是对数级的，如果还是很大可以看
    
    save_path = os.path.join(save_dir, "diff_history_curve.jpg")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Difference curve saved: {save_path}")

if __name__ == "__main__":
    run_verification_experiment()
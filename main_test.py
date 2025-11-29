"""
Main RF Score experiment script
"""
import numpy as np
import time
from config import *
from utils import show_overall_progress
from data_generation import generate_data_samples, compute_UV_unweighted
from gradient_flow import compute_A_infty
from sde_simulation import simulate_reverse_sde
from visualization import plot_samples_and_kde, plot_snapshots

def main():
    """Main experiment function"""
    print("🚀 Starting RF Score Experiment")
    print("=" * 50)
    script_start_time = time.time()

    # Initialize random number generator
    rng = np.random.default_rng(SEED)
    
    # Print configuration
    config = MODE_CONFIGS[NUM_MODES]
    print(f"📊 Data Distribution: {config['description']}")
    print(f"   Centers: {config['centers']}")
    print(f"   Probabilities: {config['probabilities']}")
    print(f"⏰ Time Schedule: {TIME_SCHEDULE}")
    print(f"🔧 Bias Switch: {bias_switch} ({'Bias b included' if bias_switch else 'Bias b excluded'})")
    print(f"📊 Data Sampling: {data_sampling_mode} ({'Centers only' if data_sampling_mode == 'centers' else 'n iid samples'})")
    print("=" * 50)

    # ---------- 1) Generate random features ----------
    W_x = rng.standard_normal((p, d))        # N(0,1)
    W_tau = rng.standard_normal((p, 2 * K + 1))  # Updated for new time embedding dim
    b = rng.normal(0.0, 1, size=(p,))    # inside bias

    # ---------- 2) Generate data samples ----------
    X0 = generate_data_samples(rng)

    # ---------- 3) Create time grid + trapezoid weights ----------
    ts = np.linspace(0.0, T, L_t + 1)
    w = np.empty_like(ts)
    if len(ts) >= 2:
        w[1:-1] = (ts[2:] - ts[:-2]) / 2.0
        w[0] = (ts[1] - ts[0]) / 2.0
        w[-1] = (ts[-1] - ts[-2]) / 2.0
    else:
        w[:] = 1.0

    # ---------- 4) UV computation ----------
    print("=" * 50)
    show_overall_progress("UV computation")
    start_time = time.time()
    U_tilde, V_tilde = compute_UV_unweighted(X0, W_x, W_tau, b, ts, w, rng)
    uv_time = time.time() - start_time
    print(f"\n✅ UV computation completed in {uv_time:.2f} seconds")
    
    # Debug: Check UV matrices
    print(f"U_tilde shape: {U_tilde.shape}, norm: {np.linalg.norm(U_tilde):.6f}")
    print(f"V_tilde shape: {V_tilde.shape}, norm: {np.linalg.norm(V_tilde):.6f}")
    print(f"U_tilde condition number: {np.linalg.cond(U_tilde):.2e}")
    print("=" * 50)

    # ---------- 5) A(∞) computation ----------
    show_overall_progress("A(∞) computation")
    start_time = time.time()
    A_inf = compute_A_infty(U_tilde, V_tilde)
    a_inf_time = time.time() - start_time
    print(f"\n✅ A(∞) computation completed in {a_inf_time:.2f} seconds")
    
    # Debug: Check A_inf matrix
    print(f"A_inf shape: {A_inf.shape}, norm: {np.linalg.norm(A_inf):.6f}")
    print(f"A_inf range: [{A_inf.min():.3f}, {A_inf.max():.3f}]")
    print(f"A_inf has infs: {np.any(np.isinf(A_inf))}")
    print(f"A_inf has NaNs: {np.any(np.isnan(A_inf))}")
    
    # Check if A_inf is essentially zero
    if np.linalg.norm(A_inf) < 1e-10:
        print("⚠️  WARNING: A_inf is essentially zero! This will cause g=0.")
    elif np.max(np.abs(A_inf)) < 1e-6:
        print("⚠️  WARNING: A_inf has very small values! This might cause g≈0.")

    # ---------- 6) Reverse SDE simulation ----------
    show_overall_progress("Reverse SDE")
    start_time = time.time()
    x, snapshots = simulate_reverse_sde(A_inf, W_x, W_tau, b, rng, capture_snapshots=True, snapshot_interval=snapshot_interval)
    sde_time = time.time() - start_time
    print(f"\n✅ Reverse SDE simulation completed in {sde_time:.2f} seconds")
    print("Generated samples shape:", x.shape)
    
    # Display snapshots if available
    if snapshots:
        print(f"\n📸 Captured {len(snapshots)} snapshots during SDE evolution")
        show_overall_progress("Snapshot Visualization")
        start_time = time.time()
        plot_snapshots(snapshots, "SDE Evolution Snapshots", m_z=m_z, script_source="main_test.py")
        snapshot_time = time.time() - start_time
        print(f"\n✅ Snapshot visualization completed in {snapshot_time:.2f} seconds")

    # ---------- 7) Visualization ----------
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 2:
        show_overall_progress("Visualization")
        start_time = time.time()
        plot_samples_and_kde(x)
        viz_time = time.time() - start_time
        print(f"\n✅ Visualization completed in {viz_time:.2f} seconds")
        
        # Print timing summary
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        total_time = uv_time + a_inf_time + sde_time + viz_time
        if snapshots:
            total_time += snapshot_time
        script_total_time = time.time() - script_start_time
        
        # Calculate percentages
        uv_pct = (uv_time / total_time) * 100
        a_inf_pct = (a_inf_time / total_time) * 100
        sde_pct = (sde_time / total_time) * 100
        viz_pct = (viz_time / total_time) * 100
        
        print(f"UV computation:     {uv_time:.2f}s ({uv_pct:.1f}%)")
        print(f"A(∞) computation:   {a_inf_time:.2f}s ({a_inf_pct:.1f}%)")
        print(f"Reverse SDE:        {sde_time:.2f}s ({sde_pct:.1f}%)")
        if snapshots:
            snapshot_pct = (snapshot_time / total_time) * 100
            print(f"Snapshot viz:        {snapshot_time:.2f}s ({snapshot_pct:.1f}%)")
        print(f"Visualization:      {viz_time:.2f}s ({viz_pct:.1f}%)")
        print("-" * 60)
        print(f"TOTAL TIME:         {total_time:.2f}s (100.0%)")
        print(f"SCRIPT TOTAL:       {script_total_time:.2f}s")
        print("=" * 60)
        print("🎉 Experiment completed successfully!")

    else:
        print("Visualization skipped: `x` not found as an array of shape (N, 2). "
              "Run the main cell first so `x` holds your generated samples.")

if __name__ == "__main__":
    main()

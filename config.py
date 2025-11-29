"""
Configuration settings for RF Score experiment
"""
import numpy as np

# ---------- Basic Config ----------
SEED = 0

# ---------- Data Distribution Config ----------
# Choose number of Dirac modes: 1, 2, 3, 4, 5, 9
NUM_MODES = 4.5  # Change this to 1, 2, 3, 4, 5, 9

# Define mode configurations
MODE_CONFIGS = {
    1: {
        'centers': np.array([[5.0, 0.0]]),
        'probabilities': np.array([1.0]),
        'description': 'Single Dirac at origin'
    },
    2: {
        'centers': np.array([[0.0, 0.0], [5.0, 0.0]]),
        'probabilities': np.array([0.5, 0.5]),
        'description': 'Two Diracs: origin and (5,0)'
    },
    3: {
        'centers': np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]]),
        'probabilities': np.array([1/3, 1/3, 1/3]),
        'description': 'Three Diracs: origin, (5,0), and (0,5)'
    },
    4: {
        'centers': np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]),
        'probabilities': np.array([0.25, 0.25, 0.25, 0.25]),
        'description': 'Four Diracs: corners of square'
    },
    4.5: {
        'centers': np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [2.5, 2.5]]),
        'probabilities': np.array([0.25, 0.25, 0.25, 0.25]),
        'description': 'Four Diracs: triangle with one mass on the side'
    },
    5: {
        'centers': np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0], [2.5, 2.5]]),
        'probabilities': np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
        'description': 'Five Diracs: corners and center of square'
    },
    9: {
        'centers': np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0] ]),
        'probabilities': np.array([1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]),
        'description': '9 Diracs: grid of 3x3 square'
    }
}

# ---------- Time Schedule Config ----------
# Choose time schedule for reverse SDE: 'linear', 'cosine', or 'exponential'
# 
# Schedule Types:
# - 'linear': Uniform steps from T to t_min (constant step size)
# - 'cosine': Cosine annealing from T to t_min (smooth transition, fine near t_min)
# - 'exponential': Exponential decay from T to t_min (very fine near t_min, coarse near T)
#
TIME_SCHEDULE = 'exponential'  # Change this to 'linear', 'cosine', or 'exponential'

# ---------- Dimensions / RF ----------
d = 2                 # ambient dimension
p = 10000              # number of random features (p×d W_x, d×p A)
K = 128                # Fourier harmonics => time embedding dim = 2K+1
# K values: 16 (fast), 32 (balanced), 64 (high quality), 128 (maximum)
# Memory: ~130KB, ~260KB, ~1MB, ~2MB respectively

# ---------- Data / MC sizes ----------
n = 8000             # number of iid samples x0^{(i)} from P used in U_t, V_t
m_z = 4000             # # of z ~ N(0,I) draws per data sample per time
batch_size = 500     # batch size for processing data samples (to avoid memory issues)

# ---------- Data Sampling Control ----------
data_sampling_mode = "centers"  # "samples": n iid samples from mixture, "centers": one sample at each center

# ---------- Memory Management ----------
# For very large n, you can use these settings:
# n = 10000           # Use 10k samples for better quality
# batch_size = 200    # Smaller batches for memory efficiency
# m_z = 32            # Reduce MC samples if needed

# ---------- Time grid ----------
T = 40.0
L_t = 400              # L_t+1 grid points on [0,T]

# ---------- Reverse SDE Time Range ----------
t_max = T            # maximum time (start of reverse process)
t_min = 0.00001           # minimum time (stopping point)

# ---------- Gradient-flow closed-form ----------
ridge = 1e-32          # tiny ridge for numerical stability in eigendecomp

# ---------- Reverse SDE ----------
N_gen = 500          # number of generated samples
L_sde = 4000           # time steps in reverse SDE
drift_scale = 1.0     # scaling factor for drift (try 0.5 or 0.1 if drift is too strong)
snapshot_interval = 200  # capture snapshots every N steps during SDE simulation

# ---------- Bias Parameter Control ----------
bias_switch = 1  # 0: bias b is not added, 1: bias b is added


# ---------- Visualization ----------
KDE_GRID_SIZE = 150
KDE_PADDING = 0.1

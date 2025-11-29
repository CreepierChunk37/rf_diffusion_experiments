"""
Time schedules and embedding functions
"""
import numpy as np
import math

def a_t(t: float) -> float:
    """Forward schedule function"""
    return np.exp(-t)

def h_t(t: float) -> float:
    """Variance schedule function"""
    return 1 - np.exp(- 2 * t)

def phi_time(t: float, K: int, T: float) -> np.ndarray:
    """Fourier time embedding of length 2K+1 at absolute time t in [0,T]."""
    tau = t / T
    ks = np.arange(1, K + 1)  # ks from 1 to K (not 0 to K-1)
    ang = 2.0 * math.pi * ks * tau
    # Include constant term (1) + sin/cos terms
    return np.concatenate([[1.0], np.sin(ang), np.cos(ang)], axis=0)  # (2K+1,)

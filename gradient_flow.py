"""
Gradient flow computation for A(∞)
"""
import numpy as np
import math
from config import d, p, ridge

def compute_A_infty(U_tilde, V_tilde):
    """Compute A(∞) using closed-form steady state solution"""
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
    U_pinv = (Q * invlam) @ Qt                      # (p, p)  == Ũ^+

    # Steady-state A(∞)/√p = Ṽ Ũ^+
    A_inf = V_tilde @ U_pinv               # (d, p)
    
    return A_inf

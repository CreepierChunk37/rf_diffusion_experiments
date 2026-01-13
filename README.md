# Research: Random Feature Diffusion Models

> **Note:** This repository contains the **research code** and experimental logs for the project: Quantifying the generalization of diffusion
models: generation implicit bias made explicit. It targets a submission to **ICML 2026**.

## 1. Project Overview
This project investigates the theoretical and empirical properties of **Random Feature (RF)** methods applied to **Score-based Generative Modeling**. 

Unlike standard neural network approximations, we explore the **Gradient Flow equilibrium** of random feature models in the diffusion setting. We specifically analyze:
- The convergence behavior of RF-based score estimators.
- The comparative performance between **Posterior Mean Parametrization** and **Direct Score Parametrization**.
- The impact of data dimensionality and random feature scaling ($p \to \infty$) on generation quality.
- etc.

## 2. Key Contributions & Features
* **Theoretical Implementation:** Implements the closed-form steady-state solution $A(\infty)$ for gradient flow using eigendecomposition.
* **Dual Approaches:**
    * **Gradient Flow Approach:** Leveraging equilibrium analysis (implemented in `main_test.py`).
    * **Score-based Approach:** Direct estimation of the score function $\nabla \log p_t(x)$ (implemented in `main_score.py`).
* **Flexible Configurations:** Support for various Dirac mode configurations (1-9 modes), time schedules (Linear/Cosine/Exponential), and sampling strategies.

## 3. Project Structure
The repository is organized to separate core logic from experimental logs.

```text
random_feature_diffusion-main/
├── README.md                    # Project documentation
├── config.py                    # Hyperparameter configurations
├── main_test.py                 # Experiment: Gradient Flow Equilibrium
├── main_score.py                # Experiment: Direct Score Matching
├── src/                         # Core functional modules
│   ├── gradient_flow.py         # Eigen-decomposition & A(inf) computation
│   ├── sde_simulation.py        # Reverse-time SDE solvers
│   ├── schedules.py             # Variance schedules (beta_t)
│   └── data_generation.py       # Gaussian mixture / Dirac mode samplers
└── results/                     # Experimental outputs
    ├── generated_samples/       # Final PNGs (300 DPI)
    └── snapshots/               # SDE trajectory snapshots

## 4. Methodological Details
*Gradient Flow Equilibrium*
We compute the limit of the gradient flow $A(\infty)$ explicitly. The core computation involves the eigendecomposition of the kernel matrix $K$:
$$ A(\infty) = \sum_{i} \dots $$
*Score Parametrization*
We verify two types of parameterizations for the score function:
1. Posterior Mean: $\hat{x}_0(x_t) = \mathbb{E}[x_0 | x_t]$
2. Direct Score: $s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)$

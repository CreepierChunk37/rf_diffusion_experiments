# Research: Random Feature Diffusion Models

> **Note:** This repository contains the **research code** and experimental logs for the project: **Quantifying the generalization of diffusion models: generation implicit bias made explicit**.
> It targets a submission to **ICML 2026**.

## 1. Project Overview

This project investigates the theoretical and empirical properties of **Random Feature (RF)** methods applied to **Score-based Generative Modeling**.

Our core objective is to quantify the generalization capability of diffusion models. We move beyond standard loss metrics by defining a theoretical "Ridge" manifold, allowing us to explicitly analyze how generated samples deviate from the true data support in terms of normal and tangent components.

## 2. Key Contributions

### Part I: RF Dynamics & Equilibrium (Training)

* **Theoretical Implementation:** Implements the closed-form steady-state solution  for gradient flow using eigendecomposition.
* **Dual Parameterizations:** Compares **Posterior Mean Parametrization** vs. **Direct Score Parametrization** in the random feature regime.

### Part II: Generalization & Manifold Analysis (Sampling)

* **Ridge Manifold Definition:** We formally define the data manifold  using the properties of the score function and Hessian eigenvalues:


* **Error Decomposition Theory:** Based on the Ridge definition, we derived the theoretical error bounds for generated samples, explicitly decomposing the error into **normal (off-manifold)** and **tangent (on-manifold)** directions relative to the trained model's approximation error.
* **Empirical Verification:**
* **Toy Datasets:** Direct numerical calculation of normal/tangent errors, confirming consistency with theoretical predictions.
* **MNIST:** Utilization of **Level Sets** to characterize and visualize the distance from generated samples to the high-dimensional manifold.



## 3. Project Structure

The repository is organized to separate core logic, manifold analysis, and experimental logs.

```text
random_feature_diffusion-main/
├── README.md                  # Project documentation
├── config.py                  # Hyperparameter configurations
├── main_test.py               # Experiment: Gradient Flow Equilibrium
├── main_score.py              # Experiment: Direct Score Matching
├── src/                       # Core functional modules
│   ├── gradient_flow.py       # Eigen-decomposition & A(inf) computation
│   └── ...                    # SDE solvers, schedules, etc.
├── Sampling_Quantifying/      # [Ongoing] Error decomposition & Manifold analysis
├── MNIST/                     # [Ongoing] MNIST Level Set experiments
└── other                  # Experimental outputs (will not use in the final paper)

```

## 4. Methodological Details

### 4.1 Gradient Flow Equilibrium

We compute the limit of the gradient flow  explicitly via eigendecomposition of the kernel matrix , analyzing the convergence of the RF-based score estimator.

### 4.2 Manifold Analysis & Error Decomposition

To measure generalization, we utilize the **Ridge** definition to define the target manifold .

**Theoretical Insight:**
We derive that for a sample  near , the generation error can be decomposed. The error in the **normal direction** (orthogonal to the manifold) is tightly controlled by the score matching error, while the error in the **tangent direction** represents movement along the data support.

**Experimental Validation:**

1. **Toy Datasets (`Sampling_Quantifying/`):**
* For low-dimensional data, we explicitly compute the manifold .
* We calculate the exact projection of generated samples onto , separating the error vectors into normal and tangent components.
* **Result:** The computed errors align with our theoretical derivation regarding the score estimation error.


2. **MNIST (`MNIST/`):**
* Due to the intractability of the exact manifold in high dimensions, we employ **Level Set** analysis.
* We analyze the distribution of samples across different level sets of the learned score/energy function.
* **Result:** The level sets effectively reflect the distance to the manifold, acting as a proxy to validate the implicit bias of the generation process.



## 5. Configuration

All hyperparameters are centralized in `config.py`.

* **Ridge Parameters:** `MANIFOLD_BETA` (Curvature threshold).
* **Model Parameters:** `p` (Random features), `NUM_MODES`.
* **Experiment Switches:** Flags to toggle between Toy Set error calculation and MNIST level set visualization.

---

**Author:** Yitong Qiu | USTC
**Last Update:** Jan 2026

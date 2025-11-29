## Training
**Formula:**
$$ \mathcal{L} = \frac{1}{nd}\sum_{i=1}^n \int_0^T  \mathbb{E}_{z\sim \mathcal{N}(0,I_d)} \left[\left\lVert  -a_t x_0^{(i)}+ m(t,a_t x_0^{(i)} + \sqrt{h_t}z) \right\rVert^2 \right] \mathrm{d}t. $$
#### 1. Squared Error Term: $\left\lVert  -a_t x_0^{(i)}+ m(t,a_t x_0^{(i)} + \sqrt{h_t}z) \right\rVert^2$
*   **`m(t, a_t x_0^{(i)} + \sqrt{h_t}z)` in the formula**:
    *   Corresponds to `pred = model(xt, t_expanded)` in code.
    *   `xt` is the noisy data constructed according to the forward process:  
        `xt = exp_neg_t.view(*view_shape) * x0_expanded + sigma.view(*view_shape) * z_expanded`.
    *   `exp_neg_t` corresponds to `a_t`, and `sigma` corresponds to `\sqrt{h_t}`. Hence, `xt` is exactly `a_t x_0 + \sqrt{h_t}z`.
    *   The `model` is the function `m(...)`.

*   **`a_t x_0^{(i)}` in the formula**:
    *   Corresponds to `target = a_t.view(*view_shape) * x0_expanded` in code.

*   **The squared error $\left\lVert \cdot \right\rVert^2$**:
    *   Corresponds to `squared_error = (pred - target) ** 2` in code.

**Definitions of `a_t` and `h_t` in code:**
*   `a_t = exp_neg_t = torch.exp(-t_expanded)`, i.e., $a_t = e^{-t}$.
*   `sigma = torch.sqrt(1 - torch.exp(-2 * t_expanded))`, i.e., $\sqrt{h_t} = \sqrt{1 - e^{-2t}}$.

#### 2. Expectation Term: $\mathbb{E}_{z\sim \mathcal{N}(0,I_d)} [...]$
The formula requires taking an expectation over noise `z`. The code approximates this using the **Monte Carlo method**.
*   **Formula**: $\mathbb{E}_{z\sim \mathcal{N}(0,I_d)}[f(z)]$
*   **Code implementation**:
    1.  `z_grid` is a pre-defined, fixed grid of `z` values used to approximate sampling from $\mathcal{N}(0,I_d)$, reducing randomness during training.
    2.  `z_expanded = z_grid.repeat(num_data * current_t_batch_size, *([1] * len(data_dim)))` broadcasts this `z_grid` to match the dimensions of all data points `x0` and time points `t`.
    3.  `squared_error_view = squared_error.view(num_data, current_t_batch_size, z_grid_size, data_flat_size)` reshapes the computed squared errors so that one dimension corresponds to `z_grid_size`.
    4.  In `loss_per_t = squared_error_view.mean(dim=(0, 2, 3))`, `.mean(dim=2)` averages over the `z_grid` dimension to approximate $\mathbb{E}_{z}$, and `.mean(dim=3)` averages over the data dimension, corresponding to $\frac{1}{d}$.

#### 3. Integral Term: $\int_0^T [...] \mathrm{d}t$
The formula requires integration over time $[0, T]$. The code uses **Simpson's rule** for numerical integration.
*   **Formula**: $\int_0^T f(t) \mathrm{d}t$
*   **Code implementation**:
    1.  `t_grid = torch.linspace(eps, T, t_grid_size, ...)` creates a grid of time points from `eps` to `T`.
    2.  `simpson_avg_weights` defines the Simpson’s rule weights:
        *   `simpson_avg_weights[1:-1:2] = 4.0` (odd-indexed interior points have weight 4),
        *   `simpson_avg_weights[2:-1:2] = 2.0` (even-indexed interior points have weight 2),
        *   `simpson_avg_weights /= (3.0 * (t_grid_size - 1))` (normalization factor $1/(3n)$),
        *   `simpson_avg_weights *= T` (multiplied by interval length $T - \varepsilon$, with $\varepsilon = 0$).  
        This matches Simpson’s rule:  
        $\int_a^b f(x)dx \approx \frac{b-a}{3n}[f(x_0) + 4f(x_1) + 2f(x_2) + \dots + 4f(x_{n-1}) + f(x_n)]$.
    3.  `scaled_batch_loss = (loss_per_t * batch_weights).sum()` multiplies the per-time-step loss (already averaged over `z` and `d`) by the corresponding Simpson weights and sums them, thereby approximating the time integral.

#### 4. Summation and Averaging Term: $\frac{1}{nd}\sum_{i=1}^n$
The formula requires averaging over all data samples `i` and data dimensions `d`.
*   **Formula**: $\frac{1}{nd}\sum_{i=1}^n [...]$
*   **Code implementation**:
    *   In `loss_per_t = squared_error_view.mean(dim=(0, 2, 3))`, `.mean(dim=0)` averages over all data points `i` (batch size `num_data`), corresponding to $\frac{1}{n}\sum_{i=1}^n$.
    *   `.mean(dim=3)` averages over the feature dimension `d` (`data_flat_size`), corresponding to $\frac{1}{d}$.

## Computation of Equilibrium

### Target Formulas

The quantities $\tilde{U}$ and $\tilde{V}$ are defined as:
$$ \tilde{U} = \int_0^T \frac{U_t}{p} \,\mathrm{d}t, \quad \tilde{V} = \int_0^T \frac{V_t}{p} \,\mathrm{d}t $$
where $U_t$ and $V_t$ are given by:
$$ U_t = \frac{1}{n}\sum_{i=1}^n \mathbb{E}_z \big[ \sigma_t(x_t^{(i)})\sigma_t(x_t^{(i)})^\intercal \big] $$
$$ V_t = \frac{1}{n}\sum_{i=1}^n \mathbb{E}_z \big[ a_t x_0^{(i)} \sigma_t(x_t^{(i)})^\intercal \big] $$
The final outputs of the code, `U_acc` and `V_acc`, correspond to $\tilde{U}$ and $\tilde{V}$, respectively.
#### 1. $U_t/p$ and $V_t/p$
##### a. Computing $U_t/p$

*   **Formula**:  
    $$
    \frac{U_t}{p} = \frac{1}{n p}\sum_{i=1}^n \mathbb{E}_z \big[ \sigma_t(x_t^{(i)})\sigma_t(x_t^{(i)})^\intercal \big]
    $$

*   **Code implementation**:
    1.  **Activation function $\sigma_t(x_t^{(i)})$**:
        *   `H = np.maximum(0.0, Z_all)` computes the activation values. `H` is a 3 D array of shape `(p, n_data, z_grid_size)`.
        *   Specifically, `H[:, i_data, m]` represents the activation vector $\sigma_t(x_t^{(i)})$ for the `i_data` -th data point using the `m` -th noise sample `z`.
    2.  **Expectation $\mathbb{E}_z$ and summation $\sum_{i=1}^n$**:
        *   `H_flat = H.reshape(p, -1)` flattens all `n_data * z_grid_size` activation vectors into a matrix of shape `(p, n_data * z_grid_size)`.
        *   `H_flat @ H_flat.T` computes the sum of outer products over all activations:  $$
            \sum_{i=1}^n \sum_{m=1}^{z_{\text{grid\_size}}} \sigma_t(x_t^{(i,m)})\sigma_t(x_t^{(i,m)})^\intercal
            $$
    3.  **Averaging and normalization $\frac{1}{n p}$**:
        *   `U_t_hat += (H_flat @ H_flat.T) / (n_data * z_grid_size * p)`
        *   The division by `(n_data * z_grid_size)` yields the empirical average, approximating  
            $\frac{1}{n}\sum_{i=1}^n \mathbb{E}_z[\cdot]$.
        *   The factor `1/p` from the formula is also included here, directly estimating $\frac{U_t}{p}$.

Thus, `U_t_hat` is an unbiased estimator of $\frac{U_t}{p}$.

##### b. Computing $V_t/p$

*   **Formula**:  $$
    \frac{V_t}{p} = \frac{1}{n p}\sum_{i=1}^n \mathbb{E}_z \big[ a_t x_0^{(i)} \sigma_t(x_t^{(i)})^\intercal \big]
    $$
*   **Code implementation**:
    1.  **Components**:
        *   `target_i = at * X0[i_data, :]` corresponds to $a_t x_0^{(i)}$.
        *   `sigma_m = H[:, i_data, m]` corresponds to $\sigma_t(x_t^{(i)})$.
    2.  **Outer product**:
        *   `contribution = np.outer(target_i, sigma_m)` implements  
            $a_t x_0^{(i)} \sigma_t(x_t^{(i)})^\intercal$.
    3.  **Expectation, summation, and normalization**:
        *   Nested loops iterate over all data points `i_data` and noise samples `m`, accumulating `contribution`.
        *   `V_t_hat += contribution / (n_data * z_grid_size * p)`
        *   Again, division by `(n_data * z_grid_size)` approximates  
            $\frac{1}{n}\sum_{i=1}^n \mathbb{E}_z[\cdot]$, and `/ p` accounts for the $1/p$ factor.

Hence, `V_t_hat` is an unbiased estimator of $\frac{V_t}{p}$.

#### 2. Integral Term: $\int_0^T \dots \,\mathrm{d}t$

The formula requires integrating $U_t/p$ and $V_t/p$ over time $[0, T]$. The code uses **Simpson’s rule** for numerical integration.
*   **Formula**:  
    $$
    \tilde{U} = \int_0^T \frac{U_t}{p} \,\mathrm{d}t
    $$

*   **Code implementation**:
    1.  **Simpson weights**:
        *   `simpson_weights` assigns appropriate weights (1, 4, 2, 4, ..., 2, 4, 1) to each time point according to Simpson’s rule and normalizes them to approximate $\int_0^T f(t)\,\mathrm{d}t$.
    2.  **Weighted summation**:
        *   `U_acc += wt * U_t_hat`
        *   `V_acc += wt * V_t_hat`
        *   In each time-step iteration, the estimates `U_t_hat` and `V_t_hat` (i.e., $U_t/p$ and $V_t/p$) are multiplied by the corresponding Simpson weight `wt` and accumulated into `U_acc` and `V_acc`, implementing numerical integration.

Therefore, `U_acc` and `V_acc` are the final computed approximations of $\tilde{U}$ and $\tilde{V}$, respectively.

# Random Feature Diffusion - Research Exploration

This project implements Random Feature (RF) Score-based generative modeling as part of an active research exploration. The codebase contains extensive experimental variations, parameter studies, and comparative analyses developed during the investigation of random feature diffusion methods.

**Note**: This is a research exploration repository with extensive experimental code and results. The project includes multiple approaches, parameter variations, and comparative studies that were developed during the research process.

## Project Structure

```
random_feature_diffusion-main/
├── README.md                    # This file - Main project documentation
├── ORGANIZATION_NOTES.md        # Detailed organization and cleanup notes
├── config.py                    # Configuration parameters
├── main_test.py                 # Gradient flow approach experiment
├── main_score.py                # Score-based approach experiment
├── [functional modules]         # Core implementation files
├── results/                     # All experimental results and visualizations
│   ├── training_experiments/    # Training experiment results
│   ├── visualization_plots/     # Organized visualizations
│   └── comparison_studies/      # Comparative analysis results
└── archive/                     # Historical and deprecated files
    ├── experimental_code/       # Previous experimental versions
    ├── deprecated_results/      # Old experimental results
    └── duplicate_files/         # Duplicate files identified during cleanup
```

### Main Scripts
- **`main_test.py`** - Main experiment script using posterior mean parametrization
- **`main_score.py`** - Alternative experiment script using score-based parametrization
- **`config.py`** - All configuration parameters and settings
- **`utils.py`** - Utility functions (progress bars, time schedules, etc.)

### Functional Modules
- **`data_generation.py`** - Data sampling and UV computation for main_test.py
- **`data_generation_score.py`** - Data sampling and UV computation for main_score.py
- **`gradient_flow.py`** - A(∞) computation using eigendecomposition
- **`sde_simulation.py`** - Reverse SDE simulation for main_test.py
- **`sde_simulation_score.py`** - Reverse SDE simulation for main_score.py
- **`schedules.py`** - Time embedding and schedule functions
- **`visualization.py`** - Plotting and KDE visualization functions

## How to Use

### Run the experiments:

**Gradient Flow Approach:**
```bash
python main_test.py
```

**Score-based Approach:**
```bash
python main_score.py
```

### Modify configurations:
Edit `config.py` to change:
- `NUM_MODES` - Number of Dirac modes (1, 2, 3, 4, 4.5, 5, or 9)
- `TIME_SCHEDULE` - Time schedule type ('linear', 'cosine', 'exponential')
- `data_sampling_mode` - Data sampling strategy ('centers' or 'samples')
- `bias_switch` - Include/exclude bias parameter (0 or 1)
- All other parameters (dimensions, sample sizes, etc.)

### Modify specific functionality:
- **Data generation**: Edit `data_generation.py` (main_test.py) or `data_generation_score.py` (main_score.py)
- **Gradient flow**: Edit `gradient_flow.py`
- **SDE simulation**: Edit `sde_simulation.py` (main_test.py) or `sde_simulation_score.py` (main_score.py)
- **Visualization**: Edit `visualization.py`
- **Time schedules**: Edit `schedules.py`

## Key Features

- **Two experiment approaches**: Gradient flow equilibrium vs. score-based parametrization
- **Modular design**: Each component is in its own file
- **Easy configuration**: All settings in `config.py`
- **Progress tracking**: Built-in tqdm progress bars
- **Flexible time schedules**: Linear, cosine, or exponential
- **Configurable data distributions**: 1-9 Dirac modes with custom configurations
- **Performance monitoring**: Detailed timing breakdown
- **Snapshot visualization**: Intermediate states during SDE evolution
- **Configurable data sampling**: Centers-only or iid samples from mixture
- **Bias parameter control**: Optional bias inclusion in neural network features

## Configuration Options

### Data Distribution
```python
NUM_MODES = 4.5  # 1, 2, 3, 4, 4.5, 5, or 9 Dirac modes
```

### Data Sampling Strategy
```python
data_sampling_mode = "centers"  # "centers": one sample at each center, "samples": n iid samples from mixture
```

### Time Schedule
```python
TIME_SCHEDULE = 'exponential'  # 'linear', 'cosine', or 'exponential'
```

### Dimensions
```python
d = 2          # ambient dimension
p = 10000      # number of random features
K = 128        # Fourier harmonics (time embedding dim = 2K+1)
```

### Sample Sizes
```python
n = 8000       # data samples
m_z = 4000     # noise samples per data sample per time
N_gen = 500    # generated samples
L_t = 400      # time grid points
L_sde = 4000   # SDE time steps
```

### Control Switches
```python
bias_switch = 1  # 0: exclude bias b, 1: include bias b
snapshot_interval = 200  # capture snapshots every N steps
```

## Output Files

The experiments generate several output files with descriptive names:

### Generated Images
- **Final samples**: `final_samples_{data_sampling_mode}_n{n}_p{p}_{TIME_SCHEDULE}_modes{NUM_MODES}.png`
- **SDE snapshots**: `snapshots_{data_sampling_mode}_n{n}_p{p}_{TIME_SCHEDULE}_modes{NUM_MODES}.png`

### Example filenames:
- `final_samples_centers_n8000_p10000_exponential_modes4.5.png`
- `snapshots_centers_n8000_p10000_exponential_modes4.5.png`

## Experiment Differences

### main_test.py (Gradient Flow Approach)
- Uses gradient flow equilibrium A(∞) computation
- Implements closed-form steady-state solution
- Uses `data_generation.py` and `sde_simulation.py`
- Focuses on equilibrium-based score estimation

### main_score.py (Score-based Approach)
- Uses alternative score parametrization
- Implements different V_t_hat computation formula
- Uses `data_generation_score.py` and `sde_simulation_score.py`
- Focuses on direct score function estimation

## Memory Management

For large-scale experiments, adjust these parameters in `config.py`:
```python
# For very large n:
n = 10000           # Use 10k samples for better quality
batch_size = 200    # Smaller batches for memory efficiency
m_z = 32            # Reduce MC samples if needed
```

## Visualization Features

- **KDE contours**: Kernel density estimation overlays on scatter plots
- **True centers**: Red X markers showing actual Dirac delta locations
- **Snapshot progression**: Multiple time steps during SDE evolution
- **High-quality output**: 300 DPI PNG files with tight bounding boxes
- **Descriptive titles**: Include time, parameters, and script source information

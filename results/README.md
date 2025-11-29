# Results Directory

This directory contains all experimental results and visualizations from the Random Feature Diffusion experiments.

## Directory Structure

### `training_experiments/`
Contains training experiment results including:
- Feature dimension comparison plots
- Loss comparison visualizations
- Model learning rate analysis
- Sampling comparison results
- Various parameter ablation studies

### `visualization_plots/`
Organized visualization outputs:
- **`Memorization_Fig/`**: Memory-related experimental visualizations
- **`Sim_Generation_Fig/`**: Simulation generation results with ablation studies
  - `Ablation_Lt/`: Time grid size ablation
  - `Ablation_mz/`: Noise sample count ablation  
  - `Ablation_N_gen/`: Generated sample count ablation
  - `Ablation_p/`: Feature dimension ablation
- **`Structed_Bias_Fig/`**: Structured bias analysis and experiments
- **`Training_Comparison_Fig/`**: Training method comparisons and hyperparameter studies

### `comparison_studies/`
Comparative analysis results:
- Model comparison across different optimizers (Adam, GD, SGD)
- Sampling method comparisons
- Evolution analysis over different modes
- Time grid size comparisons

## File Naming Convention

Most files follow the pattern:
- `final_samples_{mode}_n{samples}_p{features}_{schedule}_modes{num_modes}.png`
- `snapshots_{mode}_n{samples}_p{features}_{schedule}_modes{num_modes}.png`
- `feature_dim_{dimension}_results{variant}.png`

## Usage

These results can be used to:
- Compare different experimental configurations
- Analyze the impact of various hyperparameters
- Understand the behavior of different optimization methods
- Visualize the progression of the diffusion process

For detailed experimental setup and parameters, refer to the main project README and configuration files.

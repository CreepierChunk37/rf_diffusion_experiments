# 文件夹整理说明

## 整理前的状态
原始文件夹结构混乱，包含大量重复文件和未组织的实验结果：
- `RealTraining/` 文件夹包含大量重复的实验代码和结果
- 多个图片文件夹分散在根目录
- 重复的文件名（如 `consistent_training_implementation` 的多个版本）
- 缺乏清晰的分类结构

## 整理后的新结构

### 根目录核心文件
- **主要实验脚本**: `main_test.py`, `main_score.py`
- **配置文件**: `config.py`
- **功能模块**: `data_generation.py`, `gradient_flow.py`, `sde_simulation.py`, `utils.py`, `visualization.py`, `schedules.py`
- **训练相关代码**: `train.py`, `model.py`, `sampling.py`, `consistent_training_implementation.py`, `other_mathods.py`
- **文档**: `README.md`, `debug.md`, `note.md`

### 新的文件夹结构

#### `results/` - 所有结果文件的统一目录
- **`training_experiments/`** - 训练实验结果图片
- **`visualization_plots/`** - 可视化图表
  - `Memorization_Fig/` - 记忆相关图表
  - `Sim_Generation_Fig/` - 模拟生成图表
  - `Structed_Bias_Fig/` - 结构化偏差图表
  - `Training_Comparison_Fig/` - 训练比较图表
- **`comparison_studies/`** - 比较研究结果

#### `archive/` - 归档文件
- **`experimental_code/`** - 实验性代码（重复的 `consistent_training_implementation` 版本）
- **`deprecated_results/`** - 过时的结果文件和文件夹
- **`duplicate_files/`** - 重复的图片文件

## 清理的内容

### 移除的重复文件
- 18个重复的 `consistent_training_implementation` Python文件
- 8个重复的 `all_models_sampling_comparison` 图片文件
- 多个版本的实验结果文件夹

### 重新组织的文件
- 将分散的图片文件统一到 `results/` 目录
- 将核心训练代码移到根目录便于访问
- 将过时的实验代码和结果归档到 `archive/` 目录

## 使用建议

1. **主要开发**: 使用根目录的核心文件进行开发
2. **查看结果**: 在 `results/` 目录中查找相应的实验结果
3. **历史代码**: 如需查看历史实验代码，请检查 `archive/experimental_code/`
4. **新实验**: 新的实验结果应保存到 `results/` 对应的子目录中

## 文件数量统计
- **归档的重复代码**: 18个Python文件
- **归档的重复图片**: 8个PNG文件
- **整理的图片文件**: 100+个图片文件
- **清理的文件夹**: 5个混乱的子文件夹

整理后的结构更加清晰，便于维护和查找文件。

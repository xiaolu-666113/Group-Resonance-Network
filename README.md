# Group Resonance Network (GRN)

[English](#english) | [中文](#中文)

---

## English

### 1. Overview
This repository provides a full, runnable PyTorch implementation of **Group Resonance Network (GRN)** for cross-subject EEG emotion recognition.

GRN addresses strong inter-subject variability by combining:
- individual EEG representation (`F`)
- learnable group prototypes (`R`)
- multi-subject resonance from training-only references (`G`) using PLV + coherence
- resonance-aware fusion for final prediction

Supported experimental protocols:
- Subject-Dependent (SD)
- Subject-Independent LOSO (SI/LOSO)

Supported datasets/adapters:
- SEED (3-class)
- DEAP (binary valence/arousal)

### 2. Method Summary
For each sample:
1. Encode EEG feature tensor into `F in R^d`.
2. Compute attention over learnable prototype bank `P in R^{M x d}` and get `R`.
3. Select `K_r` references from **training-only pool** and build resonance tensor `M_res` with PLV/Coherence.
4. Encode `M_res` into `G`.
5. Fuse `[F, R, G, F-R, F-G, F*R, F*G]` (full model) and classify.

Loss:
- `L = L_cls + lambda * L_proto`

### 3. Project Structure

```text
Group_Resonance_Network/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    seed_sd.yaml
    seed_loso.yaml
    deap_valence.yaml
    deap_arousal.yaml
    ablation_individual_only.yaml
    ablation_prototypes_only.yaml
    ablation_resonance_only.yaml
    ablation_full.yaml
    ablation_full_no_proto_reg.yaml
  data/
    README.md
  src/
    __init__.py
    datasets/
      __init__.py
      base_dataset.py
      seed_dataset.py
      deap_dataset.py
      preprocessing.py
      samplers.py
      reference_selector.py
    models/
      __init__.py
      encoders.py
      prototypes.py
      resonance.py
      fusion.py
      grn.py
    losses/
      __init__.py
      classification.py
      prototype_loss.py
    trainers/
      __init__.py
      trainer.py
      evaluator.py
    utils/
      config.py
      seed.py
      logging.py
      io.py
      metrics.py
      signal_ops.py
      plotting.py
      factory.py
  scripts/
    train.py
    evaluate.py
    run_loso.py
    run_ablation.py
    run_sensitivity.py
    make_confusion_matrix.py
    make_training_curves.py
    preprocess_seed.py
    preprocess_deap.py
    sanity_check.py
  outputs/
    checkpoints/
    logs/
    figures/
    results/
  tests/
    test_shapes.py
    test_reference_leakage.py
    test_signal_ops.py
```

### 4. File-by-File Purpose

#### Root files
- `README.md`: full documentation and reproducibility guide.
- `requirements.txt`: direct runtime dependencies.
- `pyproject.toml`: package/build metadata and pytest config.

#### `configs/`
- `seed_sd.yaml`: SEED subject-dependent training/eval.
- `seed_loso.yaml`: SEED LOSO benchmark.
- `deap_valence.yaml`: DEAP valence LOSO benchmark.
- `deap_arousal.yaml`: DEAP arousal LOSO benchmark.
- `ablation_individual_only.yaml`: disable prototypes and resonance.
- `ablation_prototypes_only.yaml`: keep prototype module only.
- `ablation_resonance_only.yaml`: keep resonance module only.
- `ablation_full.yaml`: full GRN.
- `ablation_full_no_proto_reg.yaml`: full GRN, no prototype regularizer.

#### `data/`
- `data/README.md`: canonical NPZ data format.

#### `src/datasets/`
- `base_dataset.py`: canonical dataset class, shape normalization, and index-aware sample output.
- `seed_dataset.py`: SEED adapter and label normalization.
- `deap_dataset.py`: DEAP adapter, task selection, threshold binarization.
- `preprocessing.py`: raw EEG -> band DE feature extraction and caching.
- `samplers.py`: SD split and LOSO split builders.
- `reference_selector.py`: deterministic reference selection (`random`, `class_balanced`, `nearest`) with leakage checks.

#### `src/models/`
- `encoders.py`: individual EEG encoder (CNN + Transformer).
- `prototypes.py`: learnable prototype bank and attention aggregation.
- `resonance.py`: resonance tensor encoder.
- `fusion.py`: resonance-aware fusion and classifier head.
- `grn.py`: full GRN model assembly + ablation switches.

#### `src/losses/`
- `classification.py`: cross-entropy loss.
- `prototype_loss.py`: prototype alignment regularization.

#### `src/trainers/`
- `trainer.py`: full fold training pipeline (early stop, AMP, checkpointing, metrics export).
- `evaluator.py`: checkpoint evaluation helper.

#### `src/utils/`
- `config.py`: YAML load and override (`key=value`) support.
- `seed.py`: deterministic seed setup.
- `logging.py`: console+file logger creation.
- `io.py`: JSON/CSV/checkpoint read-write utilities.
- `metrics.py`: accuracy/macro-F1/per-class-acc/confusion matrix and fold summary.
- `signal_ops.py`: band filtering, DE, PLV, coherence, resonance tensor construction.
- `plotting.py`: publication-style confusion matrix and training curves.
- `factory.py`: dataset/device/experiment-dir factories.

#### `scripts/`
- `train.py`: single-run training entry (SD or single-fold LOSO).
- `evaluate.py`: evaluate a saved checkpoint without further training.
- `run_loso.py`: full LOSO loop + aggregate summary.
- `run_ablation.py`: predefined ablation suite runner.
- `run_sensitivity.py`: grid search for `K_r` and `M`.
- `make_confusion_matrix.py`: render confusion matrix to PNG/PDF.
- `make_training_curves.py`: render train/val loss+acc curves to PNG/PDF.
- `preprocess_seed.py`: raw SEED preprocessing CLI.
- `preprocess_deap.py`: raw DEAP preprocessing CLI.
- `sanity_check.py`: fast integration checks for model/selection/signal ops.

#### `tests/`
- `test_shapes.py`: model output shape tests.
- `test_reference_leakage.py`: reference leakage safety tests.
- `test_signal_ops.py`: signal processing correctness sanity tests.

### 5. Data Format (Canonical NPZ)

Required keys:
- `x`
  - feature mode: `[N, T, C, Bf]`
  - raw mode: `[N, C, T]` or `[N, T, C]`
- `y`: `[N]` (or DEAP alternatives below)
- `subject_id`: `[N]`

Optional keys:
- `session_id`, `trial_id`
- for DEAP: `valence`, `arousal` (continuous or binary)

DEAP labels are resolved by priority:
1. `valence`/`arousal` keys
2. `y` with shape `[N,2]` (`valence=col0`, `arousal=col1`)
3. `y` with shape `[N]`

### 6. Shape Conventions
- Input `x`: `[B, T, C, Bf]`
- Individual embedding `F`: `[B, d]`
- Prototype bank `P`: `[M, d]`
- Resonance tensor `M_res`: `[B, K_r, C, C, 2]`
- Resonance embedding `G`: `[B, d]`

`M_res[..., 0]` = PLV, `M_res[..., 1]` = coherence.

### 7. Leakage-Safe Reference Selection
Key behavior implemented:
- Reference pool is always built from `train_idx` of current fold.
- LOSO test subject is explicitly forbidden in references.
- Validation/test evaluation uses the same training-only reference pool.
- Deterministic selection via seed-controlled per-sample RNG.

### 8. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional
pip install -e .
```

### 9. Quick Start

#### 9.1 Preprocess raw EEG to DE features
SEED:
```bash
python scripts/preprocess_seed.py \
  --input data/seed_raw.npz \
  --output data/seed_features.npz \
  --fs 200
```

DEAP:
```bash
python scripts/preprocess_deap.py \
  --input data/deap_raw.npz \
  --output data/deap_features.npz \
  --fs 128
```

#### 9.2 Train main experiments
SEED SD:
```bash
python scripts/train.py --config configs/seed_sd.yaml
```

SEED LOSO (full all-fold run):
```bash
python scripts/run_loso.py --config configs/seed_loso.yaml
```

DEAP Valence LOSO:
```bash
python scripts/run_loso.py --config configs/deap_valence.yaml
```

DEAP Arousal LOSO:
```bash
python scripts/run_loso.py --config configs/deap_arousal.yaml
```

#### 9.3 Ablation
```bash
python scripts/run_ablation.py --config configs/seed_loso.yaml
```

Optional subset:
```bash
python scripts/run_ablation.py \
  --config configs/seed_loso.yaml \
  --variants individual_only full_model
```

#### 9.4 Sensitivity (`K_r in {1,3,5}`, `M in {4,8,12}`)
```bash
python scripts/run_sensitivity.py \
  --config configs/seed_loso.yaml \
  --k-refs 1 3 5 \
  --prototypes 4 8 12
```

#### 9.5 Evaluate checkpoint
```bash
python scripts/evaluate.py \
  --config configs/seed_loso.yaml \
  --checkpoint outputs/<exp>/loso_subject_1/checkpoints/best.pt \
  --protocol loso \
  --test-subject 1 \
  --split test
```

#### 9.6 Generate figures
Confusion matrix:
```bash
python scripts/make_confusion_matrix.py \
  --input outputs/<exp>/loso_subject_1/results/test_metrics.json \
  --class-names Negative,Neutral,Positive
```

Training curves:
```bash
python scripts/make_training_curves.py \
  --history-csv outputs/<exp>/loso_subject_1/results/history.csv
```

### 10. Reproducible Pipeline
1. Convert raw EEG to canonical DE NPZ with preprocess scripts.
2. Select config (`SD`, `LOSO`, `DEAP task`).
3. Run training script (`train.py` or `run_loso.py`).
4. Collect fold metrics from `summary.json` / `fold_metrics.csv`.
5. Generate figures from saved metrics/history.
6. Run tests and sanity checks before final report.

### 11. Outputs
Per fold directory includes:
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `logs/train.log`
- `results/history.csv`
- `results/val_metrics.json`
- `results/test_metrics.json`
- `results/test_predictions.npz`

Experiment root includes:
- `resolved_config.json`
- `summary.json`
- optional `fold_metrics.csv` for LOSO aggregation

### 12. CLI Override Examples
Override any YAML field with dotted keys:
```bash
python scripts/train.py \
  --config configs/seed_sd.yaml \
  --override model.prototypes=12 resonance.k_refs=5 training.epochs=40
```

### 13. Tests and Sanity Checks
```bash
pytest
python scripts/sanity_check.py
```

### 14. Engineering Assumptions
- Canonical NPZ input is used to unify private/public dataset layouts.
- If raw preprocessing yields variable numbers of windows, all samples are truncated to the minimum window count for tensor batching.
- Coherence is averaged across frequency bins from `scipy.signal.coherence`.
- Default resonance mode is `precompute` for practical runtime.
- Validation references are also restricted to training pool to avoid any split contamination.

---

## 中文

### 1. 项目简介
本仓库给出 **Group Resonance Network (GRN)** 的完整 PyTorch 实现，用于跨被试 EEG 情绪识别。

核心思想：
- 传统方法只强调被试不变特征，难以充分利用同一情绪刺激下跨被试共享脑活动结构。
- GRN 同时建模：
  - 个体表示 `F`
  - 可学习群体原型 `R`
  - 基于训练集参考样本的多被试共振表示 `G`（PLV + Coherence）
  - 共振感知融合分类

支持协议：
- 被试内（SD）
- 被试间 LOSO（SI）

支持数据集：
- SEED（三分类）
- DEAP（二分类：Valence/Arousal）

### 2. 方法实现概览
对每个样本：
1. 编码得到个体嵌入 `F`。
2. 与原型库计算相似度并注意力加权得到 `R`。
3. 从当前 fold 的训练池选取 `K_r` 个参考样本，构建 `M_res`（PLV+CoH）。
4. `M_res` 经共振编码器得到 `G`。
5. 融合 `[F,R,G,F-R,F-G,F*R,F*G]` 后分类。

目标函数：
- `L = L_cls + lambda * L_proto`

### 3. 目录结构
见英文部分的完整树。

### 4. 每个文件用途（详细）

#### 根目录
- `README.md`：中英文文档、复现实验教程、文件说明。
- `requirements.txt`：运行依赖。
- `pyproject.toml`：打包与 pytest 配置。

#### `configs/`
- `seed_sd.yaml`：SEED 被试内主实验。
- `seed_loso.yaml`：SEED LOSO 主实验。
- `deap_valence.yaml`：DEAP Valence LOSO。
- `deap_arousal.yaml`：DEAP Arousal LOSO。
- `ablation_individual_only.yaml`：仅个体分支。
- `ablation_prototypes_only.yaml`：个体+原型。
- `ablation_resonance_only.yaml`：个体+共振。
- `ablation_full.yaml`：完整模型。
- `ablation_full_no_proto_reg.yaml`：完整模型去掉原型正则。

#### `data/`
- `data/README.md`：规范输入格式说明。

#### `src/datasets/`
- `base_dataset.py`：基础数据类、形状标准化、全局索引返回。
- `seed_dataset.py`：SEED 标签映射与适配。
- `deap_dataset.py`：DEAP 任务标签解析和阈值二值化。
- `preprocessing.py`：原始 EEG 到 DE 特征缓存。
- `samplers.py`：SD/LOSO 划分逻辑。
- `reference_selector.py`：参考样本选择策略与防泄漏校验。

#### `src/models/`
- `encoders.py`：个体编码器（CNN+Transformer）。
- `prototypes.py`：原型库与注意力聚合。
- `resonance.py`：共振张量编码器。
- `fusion.py`：共振感知融合头。
- `grn.py`：整网组装，含消融开关。

#### `src/losses/`
- `classification.py`：交叉熵。
- `prototype_loss.py`：原型对齐正则项。

#### `src/trainers/`
- `trainer.py`：训练主流程（早停、混合精度、断点恢复、checkpoint、指标导出）。
- `evaluator.py`：checkpoint 评估辅助。

#### `src/utils/`
- `config.py`：配置加载与命令行 override。
- `seed.py`：随机种子与确定性配置。
- `logging.py`：终端+文件日志。
- `io.py`：JSON/CSV/checkpoint 读写。
- `metrics.py`：Accuracy/Macro-F1/每类准确率/混淆矩阵。
- `signal_ops.py`：滤波、DE、PLV、Coherence、共振张量构建。
- `plotting.py`：论文风格图（不带图内标题，字体较大）。
- `factory.py`：数据集和设备构造辅助。

#### `scripts/`
- `train.py`：单次训练入口（SD 或单 fold LOSO）。
- `evaluate.py`：加载 checkpoint 仅评估。
- `run_loso.py`：完整 LOSO 循环与汇总。
- `run_ablation.py`：五组 ablation 自动实验。
- `run_sensitivity.py`：`K_r` 与 `M` 灵敏度实验。
- `make_confusion_matrix.py`：生成混淆矩阵 PDF/PNG。
- `make_training_curves.py`：生成训练曲线 PDF/PNG。
- `preprocess_seed.py`：SEED 原始数据预处理。
- `preprocess_deap.py`：DEAP 原始数据预处理。
- `sanity_check.py`：快速联调检查。

#### `tests/`
- `test_shapes.py`：张量维度一致性。
- `test_reference_leakage.py`：LOSO 参考池泄漏检查。
- `test_signal_ops.py`：信号处理模块测试。

### 5. 输入数据格式
推荐使用统一 `.npz`：
- `x`：
  - 特征模式 `[N,T,C,Bf]`
  - 原始模式 `[N,C,T]` 或 `[N,T,C]`
- `y`：标签 `[N]`
- `subject_id`：被试编号 `[N]`
- 可选：`session_id`、`trial_id`
- DEAP 可用 `valence` / `arousal` 或 `y=[N,2]`

### 6. 关键形状约定
- 输入：`[B,T,C,Bf]`
- `F`：`[B,d]`
- 原型库：`[M,d]`
- 共振张量：`[B,K_r,C,C,2]`
- `G`：`[B,d]`

### 7. 参考样本防泄漏机制
已实现并默认开启：
- 参考样本仅来自当前 fold 的 `train_idx`。
- LOSO 下测试被试永不进入参考池。
- 验证/测试阶段也只使用训练池参考样本。
- 参考采样由 seed 控制，保证可复现。

### 8. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 可选
pip install -e .
```

### 9. 详细使用教程

#### 9.1 原始数据预处理（生成 DE 特征）
SEED:
```bash
python scripts/preprocess_seed.py \
  --input data/seed_raw.npz \
  --output data/seed_features.npz \
  --fs 200
```

DEAP:
```bash
python scripts/preprocess_deap.py \
  --input data/deap_raw.npz \
  --output data/deap_features.npz \
  --fs 128
```

#### 9.2 主实验
SEED SD:
```bash
python scripts/train.py --config configs/seed_sd.yaml
```

SEED LOSO:
```bash
python scripts/run_loso.py --config configs/seed_loso.yaml
```

DEAP Valence:
```bash
python scripts/run_loso.py --config configs/deap_valence.yaml
```

DEAP Arousal:
```bash
python scripts/run_loso.py --config configs/deap_arousal.yaml
```

#### 9.3 消融实验
```bash
python scripts/run_ablation.py --config configs/seed_loso.yaml
```

#### 9.4 灵敏度实验
```bash
python scripts/run_sensitivity.py \
  --config configs/seed_loso.yaml \
  --k-refs 1 3 5 \
  --prototypes 4 8 12
```

#### 9.5 单 checkpoint 评估
```bash
python scripts/evaluate.py \
  --config configs/seed_loso.yaml \
  --checkpoint outputs/<exp>/loso_subject_1/checkpoints/best.pt \
  --protocol loso \
  --test-subject 1 \
  --split test
```

#### 9.6 图像生成
混淆矩阵：
```bash
python scripts/make_confusion_matrix.py \
  --input outputs/<exp>/loso_subject_1/results/test_metrics.json \
  --class-names Negative,Neutral,Positive
```

训练曲线：
```bash
python scripts/make_training_curves.py \
  --history-csv outputs/<exp>/loso_subject_1/results/history.csv
```

#### 9.7 自检
```bash
pytest
python scripts/sanity_check.py
```

### 10. 复现实验建议流程
1. 准备并标准化数据为 canonical NPZ。
2. 先跑 `sanity_check.py` 验证环境。
3. 跑主实验（SD/LOSO）。
4. 跑 ablation 与 sensitivity。
5. 导出图表与汇总指标（`summary.json`、`fold_metrics.csv`）。
6. 使用固定 seed 重复一次确认稳定性。

### 11. 默认超参数
- `d=256`
- `M=8`
- `K_r=3`
- Adam, `lr=1e-4`, `weight_decay=1e-4`
- `batch_size=64`
- `epochs=80`
- `patience=10`
- `lambda_proto=0.1`
- `tau=0.07`

### 12. 当前实现中的工程假设
- 原始数据结构不统一时，统一转换到 canonical NPZ。
- 预处理后若窗口数不一致，会截断到最短窗口长度保证批处理。
- coherence 使用 SciPy 频率维平均值。
- 默认 `resonance.mode=precompute`（更实用，训练阶段开销更可控）。


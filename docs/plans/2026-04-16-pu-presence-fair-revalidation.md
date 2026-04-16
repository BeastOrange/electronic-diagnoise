# PU Presence Fair Revalidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在统一 leakage 策略、统一 split、统一训练预算的前提下，重新验证 `PU_Presence` 是否真的能从修复后的物理特征中获益，并据此决定下一阶段是继续做模型优化还是转向数据可学性审计。

**Architecture:** 先建立一套“公平复验”配置，保证 `Bagged LR`、`CNN-1D`、`Qwen QLoRA` 使用同一数据准备口径；再用最小消融矩阵拆开 `PU_Signal_Strength`、`Frequency_Band` 交互项、`top_k` 筛选对结果的影响；最后用统一汇总表和停机阈值决定是否继续堆模型。计划默认先在服务器 `/root/autodl-tmp` 上执行，落盘产物继续写入 `artifacts/prepared_*` 与 `artifacts/runs/*`。

**Tech Stack:** Python 3.12, `uv`, `emc_diag` CLI, YAML configs, server CUDA runtime, markdown/csv artifact reporting.

---

### Task 1: 建立公平复验配置骨架

**Files:**
- Create: `configs/cognitive_radio_presence_fair_baseline.yaml`
- Create: `configs/cognitive_radio_presence_fair_cnn.yaml`
- Create: `configs/cognitive_radio_presence_fair_qwen.yaml`
- Create: `configs/cognitive_radio_presence_signal_strength_ablation.yaml`
- Reference: `configs/cognitive_radio_presence_cnn.yaml`
- Reference: `configs/cognitive_radio_presence_qwen_qlora.yaml`
- Reference: `configs/cognitive_radio_presence_model_benchmark.yaml`

**Step 1: 复制现有 presence 配置作为起点**

基于下面三份已有配置建立公平复验版本：

- `configs/cognitive_radio_presence_cnn.yaml:1-69`
- `configs/cognitive_radio_presence_qwen_qlora.yaml:1-69`
- `configs/cognitive_radio_presence_model_benchmark.yaml:1-84`

新配置必须统一这些字段：

- `dataset.input_path`
- `task.target_column`
- `trainer.train_ratio`
- `trainer.val_ratio`
- `trainer.random_state`
- `runtime.artifacts_dir`

**Step 2: 统一 leakage 策略**

默认公平复验版本先统一为：

```yaml
drop_leakage_columns:
  - time_index
  - PU_bandwidth
  - PU_burst_duration
  - PU_drift_type
```

说明：

- `PU_drift_type` 一律删除
- `Frequency_Band` 默认保留
- `PU_Signal_Strength` 不作为默认 leakage 列，改由消融矩阵单独控制

**Step 3: 统一特征筛选口径**

在三份公平复验主配置中先统一：

```yaml
features:
  method: hybrid
  top_k: full
```

不要一开始继续使用 `top_k: 20` 或 `top_k: 16`，先避免 F-score 筛选把物理特征结构再次压坏。

**Step 4: 建立 `PU_Signal_Strength` 消融 benchmark 配置**

在 `configs/cognitive_radio_presence_signal_strength_ablation.yaml` 中建立 2 个 task 或 2 组 benchmark 变体：

- `presence_keep_strength`
- `presence_drop_strength`

唯一差异是 `drop_leakage_columns` 是否包含 `PU_Signal_Strength`。

**Step 5: Commit**

```bash
git add configs/cognitive_radio_presence_fair_baseline.yaml \
  configs/cognitive_radio_presence_fair_cnn.yaml \
  configs/cognitive_radio_presence_fair_qwen.yaml \
  configs/cognitive_radio_presence_signal_strength_ablation.yaml
git commit -m "feat(configs): 新增 presence 公平复验配置"
```

### Task 2: 验证 prepared/feature bundle 是否真的走到修复后逻辑

**Files:**
- Reference: `src/emc_diag/cli.py`
- Reference: `src/emc_diag/feature_engineering.py`
- Reference: `src/emc_diag/data_pipeline.py`
- Output: `artifacts/prepared_cognitive_presence_fair_*`

**Step 1: 先跑 prepare**

```bash
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_fair_baseline.yaml
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_fair_cnn.yaml
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_fair_qwen.yaml
```

Expected:

- 生成新的 `prepared_dir`
- `metadata.json` 中 `leakage_columns_removed` 不再出现 `Frequency_Band`
- `leakage_columns_removed` 必须出现 `PU_drift_type`

**Step 2: 再跑 extract-features**

```bash
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_fair_baseline.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_fair_cnn.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_fair_qwen.yaml
```

Expected:

- `feature_metadata.json` 中出现 `SU*_cov_flat_mme_*`
- `feature_metadata.json` 中出现 `*_phys_eig_min`
- `feature_metadata.json` 中出现 `cross_su_*`
- `selected_feature_count` 与 `top_k: full` 配置一致，不应被截断到 20

**Step 3: 手工核对关键产物**

检查：

- `artifacts/prepared_cognitive_presence_fair_*/metadata.json`
- `artifacts/prepared_cognitive_presence_fair_*/feature_metadata.json`
- `artifacts/prepared_cognitive_presence_fair_*/tables/selected_features.csv`

若仍看到 `PU_drift_type__*` 进入特征，说明配置或缓存复用仍有问题，停止后转入 `@systematic-debugging`。

**Step 4: Commit**

```bash
git add artifacts/prepared_cognitive_presence_fair_* 2>/dev/null || true
git commit -m "chore(presence): 校验公平复验特征产物"
```

### Task 3: 先跑诚实 baseline 和最小消融矩阵

**Files:**
- Modify: `configs/cognitive_radio_presence_signal_strength_ablation.yaml`
- Output: `artifacts/runs/run-*-pu-presence-fair-*`
- Output: `artifacts/benchmarks/benchmark-*`

**Step 1: 跑 baseline**

先只跑最简单、最便宜、最有诊断价值的模型：

```bash
uv run python -m emc_diag train --config configs/cognitive_radio_presence_fair_baseline.yaml --device cpu
```

若配置使用 benchmark matrix，则改为：

```bash
uv run python -m emc_diag benchmark --configs configs/cognitive_radio_presence_fair_baseline.yaml --device cpu --theme paper-bar
```

Expected:

- 有 `metrics.json`
- 有 `tables/cv_metrics.csv`
- 有 `tables/candidate_scores.csv`

**Step 2: 跑 `PU_Signal_Strength` 消融**

```bash
uv run python -m emc_diag benchmark --configs configs/cognitive_radio_presence_signal_strength_ablation.yaml --device cpu --theme paper-bar
```

必须输出一张 2 行对照：

- keep strength
- drop strength

核心看：

- `accuracy`
- `macro_f1`
- `minority_f1`
- `cv mean/std`

**Step 3: 加一组交互项/筛选强度消融**

如果代码层已支持通过配置关闭交互项或限制特征组，则直接加 benchmark 变体；否则新建一份临时配置，至少比较：

- `top_k: full`
- `top_k: 20`

目的不是调参，而是判断“是物理特征没信号，还是筛选方式把信号切坏了”。

**Step 4: 结果门槛**

用以下阈值决定是否继续：

- 若 baseline `cv mean accuracy >= 0.58` 且 `minority_f1 >= 0.55`，允许进入深度模型复验
- 若 baseline `cv mean accuracy <= 0.53` 且消融差异都很小，优先转 Task 6 的数据可学性审计

**Step 5: Commit**

```bash
git add configs/cognitive_radio_presence_signal_strength_ablation.yaml
git commit -m "feat(presence): 新增公平复验消融矩阵"
```

### Task 4: 跑修复后 CNN-1D 复验

**Files:**
- Modify: `configs/cognitive_radio_presence_fair_cnn.yaml`
- Output: `artifacts/runs/run-*-pu-presence-fair-cnn-*`
- Reference: `docs/runtime.md`

**Step 1: 统一训练预算**

把 CNN 复验配置固定为：

```yaml
model:
  name: cnn_1d
  epochs: 40
  patience: 6
trainer:
  random_state: 42
evaluation:
  cross_validation:
    enabled: false
```

先做单次可比运行，不要一开始就开大矩阵。

**Step 2: 运行训练**

```bash
uv run python -m emc_diag train --config configs/cognitive_radio_presence_fair_cnn.yaml --device cuda
```

**Step 3: 补 evaluate / visualize**

```bash
uv run python -m emc_diag evaluate --run-dir artifacts/runs/<fair-cnn-run-id>
uv run python -m emc_diag visualize --run-dir artifacts/runs/<fair-cnn-run-id> --theme paper-bar
```

Expected:

- `metrics.json`
- `summary.md`
- `figures/confusion_matrix.png`
- `tables/per_class_metrics.csv`

**Step 4: 验收标准**

CNN 只有在满足以下任一条件时才算“值得继续调参”：

- `accuracy >= 0.58`
- `minority_f1 >= 0.55`

如果仍接近 `0.50~0.55`，不要继续换网络结构，先进入 Task 6。

**Step 5: Commit**

```bash
git add configs/cognitive_radio_presence_fair_cnn.yaml
git commit -m "feat(presence): 新增公平复验 cnn 配置"
```

### Task 5: 跑修复后 Qwen QLoRA 复验

**Files:**
- Modify: `configs/cognitive_radio_presence_fair_qwen.yaml`
- Output: `artifacts/runs/run-*-pu-presence-fair-qwen-*`
- Reference: `src/emc_diag/llm_text_adapter.py`

**Step 1: 固定 Qwen 训练预算**

Qwen 公平复验先恢复到不低于这组预算：

```yaml
model:
  epochs: 4
  batch_size: 1
  learning_rate: 0.0001
  patience: 3
  llm:
    gradient_accumulation_steps: 16
    feature_limit: null
```

不要再接受 `epochs: 1` 的结果作为主结论。

**Step 2: 运行 prepare / extract / train**

```bash
uv sync --dev --group llm
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_fair_qwen.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_fair_qwen.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_fair_qwen.yaml --device cuda
```

**Step 3: 验收 Qwen 输出**

必须检查：

- `best_checkpoint.json`
- `metrics.json`
- `tables/per_class_metrics.csv`
- `llm_adapter/adapter_model.safetensors`

若 `best_epoch == 0` 且 `val_loss` 明显高于上一轮最佳 run，则先排查 prompt/feature text，而不是继续延长 epoch。

**Step 4: 验收标准**

只有在这两条之一成立时，Qwen 才值得继续：

- 超过当前最好 clean run
- 至少不低于公平 baseline `+0.02 accuracy`

**Step 5: Commit**

```bash
git add configs/cognitive_radio_presence_fair_qwen.yaml
git commit -m "feat(presence): 新增公平复验 qwen 配置"
```

### Task 6: 汇总结果并做停机决策

**Files:**
- Create: `docs/presence_fair_revalidation_summary.md`
- Output: `artifacts/benchmarks/benchmark-*/benchmark_metrics.csv`
- Output: `artifacts/runs/run-*/metrics.json`

**Step 1: 生成统一汇总表**

至少汇总这些列：

- `config`
- `model`
- `seed`
- `drop_leakage_columns`
- `top_k`
- `accuracy`
- `macro_f1`
- `minority_f1`
- `cv_mean`
- `cv_std`
- `notes`

**Step 2: 写结论**

结论只允许二选一：

1. `公平复验确认数据修复有增益`  
条件：至少一个 clean baseline 和一个 clean 深度模型都显著超过旧的 honest baseline

2. `公平复验未确认增益，问题升级为数据可学性审计`  
条件：所有 clean 方案都停留在 `0.50~0.55` 附近，且消融对结果影响有限

**Step 3: 如果进入数据可学性审计，下一轮只做这些事**

- 标签生成逻辑检查
- `PU_Presence` 与关键字段的单变量 AUC / permutation importance
- 样本去重与近重复检查
- waveform / spectrum 原始路径复验

不要在这个分支继续堆更多 CNN/LSTM/Transformer 变体。

**Step 4: Commit**

```bash
git add docs/presence_fair_revalidation_summary.md
git commit -m "docs(presence): 汇总公平复验结论"
```

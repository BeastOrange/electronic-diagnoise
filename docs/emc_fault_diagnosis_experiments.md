# EMC Fault Diagnosis Experiment Commands

本清单围绕毕设主线“基于 AI 的电磁兼容故障诊断研究”，把可直接执行的主实验命令链整理出来。所有路径都基于当前 `emc_diag` CLI。

当前主结果线说明：

- 主线数据集：`Cognitive Radio Spectrum Sensing Dataset`
- 辅助数据集：`Electrical Fault Detection and Classification`、`VSB Power Line Fault Detection`
- 论文主结果、主要图表、主要方法对比，应优先围绕 `cognitive_radio_*` 配置展开

## 1. presence -> band 监督迁移

先训练认知无线电 presence 任务的源模型：

```bash
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_cnn.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_cnn.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_cnn.yaml --device cuda
```

把生成的 run 目录填入：

- `configs/cognitive_radio_band_transfer_from_presence.yaml`
- `configs/cognitive_radio_paper_benchmark_extended.yaml`

然后执行 band 迁移实验：

```bash
uv run python -m emc_diag prepare --config configs/cognitive_radio_band_transfer_from_presence.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_band_transfer_from_presence.yaml --device cuda
uv run python -m emc_diag evaluate --run-dir artifacts/runs/REPLACE_WITH_BAND_TRANSFER_RUN_DIR
```

这组实验用于说明：共享频谱表征能否从 presence 检测迁移到 band 判别。

## 2. SSL pretrain -> downstream fault diagnosis

### 2.1 认知无线电自监督预训练 -> presence 下游任务

```bash
uv run python -m emc_diag prepare --config configs/cognitive_radio_ssl_pretrain.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_ssl_pretrain.yaml --device cuda
```

把 SSL 预训练 run 目录填入 `configs/cognitive_radio_presence_ssl_transfer.yaml`，然后执行下游分类：

```bash
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_ssl_transfer.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_ssl_transfer.yaml --device cuda
uv run python -m emc_diag evaluate --run-dir artifacts/runs/REPLACE_WITH_PRESENCE_SSL_TRANSFER_RUN_DIR
```

### 2.2 VSB 电力设备故障数据上的 SSL -> fault diagnosis（辅助验证）

先做 VSB 自监督预训练：

```bash
uv run python -m emc_diag prepare --config configs/vsb_fault_ssl_pretrain.yaml
uv run python -m emc_diag train --config configs/vsb_fault_ssl_pretrain.yaml --device cuda
```

把预训练 run 目录填入 `configs/vsb_fault_ssl_transfer.yaml`，再做故障诊断微调：

```bash
uv run python -m emc_diag prepare --config configs/vsb_fault_ssl_transfer.yaml
uv run python -m emc_diag train --config configs/vsb_fault_ssl_transfer.yaml --device cuda
uv run python -m emc_diag evaluate --run-dir artifacts/runs/REPLACE_WITH_VSB_SSL_TRANSFER_RUN_DIR
```

这组实验用于辅助验证“先学信号表征，再迁移到下游故障识别”的思路，但不作为主结果线。

## 3. multitask cognitive radio

```bash
uv run python -m emc_diag prepare --config configs/cognitive_radio_multitask.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_multitask.yaml --device cuda
uv run python -m emc_diag evaluate --run-dir artifacts/runs/REPLACE_WITH_MULTITASK_RUN_DIR
uv run python -m emc_diag visualize --run-dir artifacts/runs/REPLACE_WITH_MULTITASK_RUN_DIR --theme paper-bar
```

这组实验用于说明：presence、band、burst 等任务共享编码器时，是否能提升 EMC/EMI 场景下的多目标感知与诊断表现。

## 4. benchmark.matrix 主实验矩阵

### 4.1 传统 baseline 矩阵

```bash
uv run python -m emc_diag benchmark --configs configs/cognitive_radio_paper_benchmark.yaml --device cpu --theme paper-bar
```

这会自动展开任务 × 模型组合，用于生成跨任务 baseline 对比表。

### 4.2 扩展矩阵：seed / mode / transfer_strategy

把迁移源 run 目录填入 `configs/cognitive_radio_paper_benchmark_extended.yaml` 后运行：

```bash
uv run python -m emc_diag benchmark --configs configs/cognitive_radio_paper_benchmark_extended.yaml --device cuda --theme paper-bar
```

这会展开：

- task
- model
- seed
- mode
- transfer_strategy

适合做论文结果章节中的系统性实验表。

## 5. 建议的论文叙事顺序

1. 单任务 baseline：先说明主线 Cognitive Radio 数据上的 EMC/EMI 感知任务可以被传统 ML 和单任务深度模型建模。
2. 监督迁移：说明跨任务频谱表征具有可迁移性。
3. SSL pretrain：说明在标签受限时，自监督表征学习能提升主线数据集下游任务。
4. multitask cognitive radio：说明多任务共享编码器可以统一感知多个 EMC 相关目标。
5. benchmark.matrix：用主线数据集上的系统实验收尾，展示模型、任务、种子和迁移策略的总体规律。
6. VSB：作为辅助验证章节，补充说明方法在波形式故障诊断数据上的泛化表现。

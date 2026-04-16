# Runtime Notes

## Python

- Project interpreter is pinned to `Python 3.12`.
- Do not rely on the system `Python 3.14` for this project.

## Device Selection

- `--device auto`: prefer `CUDA`, then `MPS`, then `CPU`
- `--device cuda`: force CUDA, fallback to CPU if unavailable
- `--device mps`: force Apple Silicon MPS, fallback to CPU if unavailable
- `--device cpu`: always run on CPU

## Recommended Commands

```bash
uv sync --dev
uv run pytest -q
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_cnn.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_cnn.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_cnn.yaml --device auto
uv run python -m emc_diag benchmark --configs \
  configs/cognitive_radio_presence_cnn_cv.yaml \
  configs/cognitive_radio_burst_duration.yaml \
  configs/cognitive_radio_frequency_band.yaml \
  configs/cognitive_radio_drift_type.yaml \
  --device auto
```

## Open-Source LLM Mainline

```bash
uv sync --dev --group llm
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_qwen_qlora.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_qwen_qlora.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_qwen_qlora.yaml --device cuda
```

如果服务器只能访问国内网络，优先使用上面的镜像环境变量。
这里默认使用 `hf-mirror.com` 作为 Hugging Face 国内镜像。

默认主线现在面向 `24GB` 显存的 `7B + 4bit QLoRA`：

- `configs/cognitive_radio_presence_qwen_qlora.yaml`: 主线配置，默认使用 `Qwen/Qwen2.5-7B-Instruct`
- `configs/cognitive_radio_presence_qwen7b_qlora.yaml`: 显式 Qwen 7B 配置
- `configs/cognitive_radio_presence_deepseek7b_qlora.yaml`: `DeepSeek-R1-Distill-Qwen-7B` 配置

如果你要直接跑 DeepSeek：

```bash
uv sync --dev --group llm
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_deepseek7b_qlora.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_deepseek7b_qlora.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_deepseek7b_qlora.yaml --device cuda
```

训练稳定性建议：

- `4090D 24GB` 优先保持 `batch_size=1`
- `max_length` 先用 `512`
- 通过 `gradient_accumulation_steps` 拉高等效 batch
- 保持 `save_adapter_only=true`，只保存 LoRA adapter

CNN vs Qwen benchmark:

```bash
uv run python -m emc_diag benchmark --configs configs/cognitive_radio_presence_qwen_vs_cnn.yaml --device cuda
```

## Platform Bootstrap

### macOS (Apple Silicon, MPS)

```bash
uv sync --dev
uv run python -m emc_diag train --config configs/cognitive_radio_presence_cnn.yaml --device mps
```

### Linux (CUDA)

```bash
uv sync --dev
uv run python -m emc_diag train --config configs/cognitive_radio_presence_cnn.yaml --device cuda
```

### Windows (PowerShell, CUDA)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1 -ProjectRoot "." -DatasetSource cognitive -RunPipeline
```

Use `-IncludeVSB` when you want to fetch VSB competition data.

## Cross-Platform Notes

- macOS Apple Silicon: use `MPS` for local smoke tests and small experiments
- Linux server: use `CUDA` for full training workloads
- Windows client: prefer native CUDA first, with WSL2 as fallback when drivers or toolchains are unstable

## Cognitive Radio Tasks

- `configs/cognitive_radio_presence_cnn.yaml`: main `PU_Presence` task with CNN-first training
- `configs/cognitive_radio_presence_qwen_qlora.yaml`: main open-source LLM fine-tuning config for `PU_Presence`
- `configs/cognitive_radio_presence_qwen7b_qlora.yaml`: explicit Qwen2.5-7B QLoRA config
- `configs/cognitive_radio_presence_deepseek7b_qlora.yaml`: DeepSeek-R1-Distill-Qwen-7B QLoRA config
- `configs/cognitive_radio_presence_qwen_vs_cnn.yaml`: Qwen vs CNN benchmark matrix
- `configs/cognitive_radio_presence_cnn_cv.yaml`: ablation/CV-friendly `PU_Presence` config with threshold-tuning grid
- `configs/cognitive_radio_burst_duration.yaml`: auxiliary high-score burst-duration task
- `configs/cognitive_radio_frequency_band.yaml`: auxiliary high-score frequency-band task
- `configs/cognitive_radio_drift_type.yaml`: mid-difficulty drift-type task

Each task config records the target column, task name, leakage columns, candidate models, and evaluation settings needed for thesis experiments.

## Teacher-Facing Summary

可以直接这样说明当前方案：

> 这个项目不是只用传统机器学习。  
> 现在我们把电磁/无线信号特征转换成结构化文本输入，使用开源大模型 `Qwen` 或 `DeepSeek` 做 `PU_Presence` 专项分类微调。  
> 训练方式是 `7B + 4bit QLoRA`，输出的是可复现的 LoRA adapter、指标文件、图表和报告。

## Strict Risk-Check Output Contract

Before accepting a run as thesis-ready, verify:

- required: `metrics.json`, `predictions.csv`, `summary.md`
- required: `figures/metrics_overview.png`, `figures/confusion_matrix.png`
- optional but valid when present: `tables/per_class_metrics.csv`, benchmark aggregate CSV/MD outputs

## Runtime Sanity Checks

Use these checks before reporting results:

```bash
uv run python - <<'PY'
import torch
print("cuda:", bool(torch.cuda.is_available()))
print("mps:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
PY
```

Expected interpretation:

- macOS Apple Silicon: usually `mps: True`
- Linux/Windows with NVIDIA: usually `cuda: True`
- CPU-only fallback remains valid for smoke tests

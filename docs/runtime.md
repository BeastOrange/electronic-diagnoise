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

## Qwen QLoRA Mainline

```bash
uv sync --dev --group llm
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_qwen_qlora.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_qwen_qlora.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_qwen_qlora.yaml --device cuda
```

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
- `configs/cognitive_radio_presence_qwen_qlora.yaml`: Qwen2.5-3B QLoRA single-task mainline
- `configs/cognitive_radio_presence_qwen_vs_cnn.yaml`: Qwen vs CNN benchmark matrix
- `configs/cognitive_radio_presence_cnn_cv.yaml`: ablation/CV-friendly `PU_Presence` config with threshold-tuning grid
- `configs/cognitive_radio_burst_duration.yaml`: auxiliary high-score burst-duration task
- `configs/cognitive_radio_frequency_band.yaml`: auxiliary high-score frequency-band task
- `configs/cognitive_radio_drift_type.yaml`: mid-difficulty drift-type task

Each task config records the target column, task name, leakage columns, candidate models, and evaluation settings needed for thesis experiments.

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

# Electronic Diagnosis AI

Research-oriented EMC/EMI fault diagnosis project for graduation delivery. The repository now focuses on one clean outcome: a usable handoff package for the person who will write the paper.

## Project Goal

This project studies AI-based EMC/EMI diagnosis with:

- one primary thesis dataset: `Cognitive Radio Spectrum Sensing Dataset`
- one primary hard task: `PU_Presence`
- several auxiliary tasks for comparison and analysis
- a complete reporting pipeline for metrics, figures, and Markdown reports

The emphasis is not only model training, but also reproducible experiment assets, visual evidence, and a paper-ready research report.

## Final Result Snapshot

Current final conclusions:

- `PU_Presence` is the main difficult task and the weakest core result.
- Best `PU_Presence` model: `bagged_logistic_regression`
- Best `PU_Presence` score: `accuracy=0.5733`, `f1=0.5706`
- Best medium-difficulty task: `PU_drift_type` with `cnn_lstm`
- `PU_burst_duration` and `Frequency_Band` achieve very high scores, but `Frequency_Band` must be treated as a risk-control / proxy-feature task rather than the main innovation result.
- `VSB` is kept only as auxiliary external validation and is not the main thesis line.

## Main Deliverable

The most important output in this repository is the final paper handoff package:

- `artifacts/reports/paper-handoff-package/`
- `artifacts/reports/paper-handoff-package.zip`

That package already contains:

- final research report
- exploration gallery
- final metrics table
- merged benchmark table
- figure manifest
- all figures needed for paper writing

If someone only needs the final usable materials, they should start from that folder, not from the raw experiment directories.

## Repository Structure

- `src/emc_diag/`: core implementation
- `configs/`: experiment configs
- `data/Cognitive Radio Spectrum Sensing Dataset.csv`: main thesis dataset
- `docs/`: supporting documentation
- `tests/`: regression tests
- `artifacts/reports/paper-handoff-package/`: final handoff bundle

## Models Implemented

The project includes both traditional ML and deep learning baselines:

- `bagged_logistic_regression`
- `logistic_regression`
- `random_forest`
- `svc`
- `cnn_1d`
- `cnn_lstm`
- `transformer_1d`
- `cognitive_radio_scalar_hybrid`

The final thesis conclusion currently relies on traditional ML for the hardest main task, while deep models show clearer value on selected auxiliary tasks such as `PU_drift_type`.

## Tech Stack

- Python 3.12
- PyTorch 2.8
- scikit-learn
- NumPy
- Pandas
- SciPy
- PyArrow
- Matplotlib
- Seaborn
- uv

## Quick Start

Install dependencies:

```bash
uv sync --dev
```

Run a single experiment:

```bash
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_cnn.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_cnn.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_cnn.yaml --device auto
uv run python -m emc_diag evaluate --run-dir artifacts/runs/<run-id>
uv run python -m emc_diag visualize --run-dir artifacts/runs/<run-id> --theme paper-bar
uv run python -m emc_diag export-report --run-dir artifacts/runs/<run-id> --format md
```

## Final Handoff Workflow

To rebuild the final handoff package after report updates:

```bash
PYTHONPATH=src python scripts/build_paper_handoff.py
```

This creates:

- `artifacts/reports/paper-handoff-package/`
- `artifacts/reports/paper-handoff-package.zip`

## Thesis Positioning

Recommended writing strategy:

- Main thesis line: `Cognitive Radio Spectrum Sensing Dataset`
- Main difficult task: `PU_Presence`
- Supporting deep-learning value task: `PU_drift_type`
- High-score auxiliary task: `PU_burst_duration`
- Risk-control task: `Frequency_Band`
- External validation only: `VSB`

## Status

This repository has been cleaned for delivery.

- unnecessary old run directories were removed
- only the main dataset is retained locally
- only the final handoff package is preserved under `artifacts/reports`

At this stage, the project is intended for delivery and paper writing, not for open-ended further experimentation.

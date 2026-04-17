# Dataset Guide

## Primary EMC/EMI Dataset (Current Mainline)

- `Cognitive Radio Spectrum Sensing Dataset`
  - Local path: `data/Cognitive Radio Spectrum Sensing Dataset.csv`
  - Kaggle: <https://www.kaggle.com/datasets/ajithdari/cognitive-radio-spectrum-sensing-dataset>
  - Default label column: `PU_Presence`
  - Project config: `configs/cognitive_radio_presence_cnn.yaml`
  - Ablation/CV config: `configs/cognitive_radio_presence_cnn_cv.yaml`
  - Notes: this CSV contains mixed feature types (numeric, categorical, array-like strings). Parsing support is handled in the data pipeline.

## Additional Local Dataset

- `Electrical Fault detection and classification`
  - Raw download may appear under the repository root with irregular whitespace in the folder name.
  - Normalize it first with:
    ```bash
    uv run python scripts/normalize_electrical_fault_dataset.py
    ```
  - Normalized local path: `data/electrical_fault_detection/`
  - Ready-to-run files:
    - `detect_dataset.csv`: binary fault detection
    - `classData_fault_code.csv`: 6-class fault-code classification
  - Recommended role: fast recovery line for customer-facing strong results.

## EMC/EMI Supplement Dataset

- NIST `Radio Frequency Interference Measurements of Industrial Machinery`
  - Catalog page: <https://catalog.data.gov/dataset/radio-frequency-interference-measurements-of-industrial-machinery-34600>
  - This catalog page was still accessible on 2026-03-25 and exposes multiple CSV downloads from `data.nist.gov`.
  - Recommended use: thesis-side EMC/EMI supplementary analysis and visualization, not necessarily the first training dataset.

## Historical/Backup Dataset

- `emi_uci`
  - Kaggle: <https://www.kaggle.com/datasets/ucimachinelearning/electromagnetic-interference-dataset>
  - Backup: <https://archive.ics.uci.edu/dataset/763>
  - CLI hint:
    ```bash
    kaggle datasets download -d ucimachinelearning/electromagnetic-interference-dataset
    ```

## Proxy Datasets

- `vsb_power_line_fault`
  - Kaggle: <https://www.kaggle.com/competitions/vsb-power-line-fault-detection>
  - CLI hint:
    ```bash
    kaggle competitions download -c vsb-power-line-fault-detection
    ```
  - Recommended local folder: `data/vsb-power-line-fault-detection/`
  - Usage: large-scale proxy validation and robustness experiments after Cognitive Radio mainline is stable.

- `electrical_fault`
  - Kaggle: <https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification>
  - CLI hint:
    ```bash
    kaggle datasets download -d esathyaprakash/electrical-fault-detection-and-classification
    ```
  - Recommended run order:
    ```bash
    uv run python scripts/normalize_electrical_fault_dataset.py
    uv run python -m emc_diag prepare --config configs/electrical_fault_detect_rf.yaml
    uv run python -m emc_diag train --config configs/electrical_fault_detect_rf.yaml --device cpu
    uv run python -m emc_diag train --config configs/electrical_fault_fault_code_rf.yaml --device cpu
    ```

## Local Example

The repository includes `data/synthetic_emi.csv` so the full CLI workflow can be demonstrated without downloading a public dataset first.

## Download And Execution Order

1. Download Cognitive Radio dataset first and place CSV under `data/`.
2. Run CNN-first mainline with `configs/cognitive_radio_presence_cnn.yaml`.
3. Run ablation/CV-friendly config `configs/cognitive_radio_presence_cnn_cv.yaml`.
4. Download VSB later and run proxy robustness experiments when storage/time permits.

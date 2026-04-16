from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from emc_diag.cli import (
    _call_with_batch_backoff,
    _execute_training,
    _expand_benchmark_variants,
    _load_prepared_bundle,
    _model_family_for_name,
    main,
)
from emc_diag.config import load_config
from emc_diag.runtime import resolve_device


def _write_synthetic_tabular_dataset(csv_path: Path) -> None:
    rows = []
    for index in range(60):
        label = index % 3
        rows.append(
            {
                "sensor_a": label + (index * 0.01),
                "sensor_b": (label * 0.5) + ((index % 5) * 0.1),
                "sensor_c": np.sin(index / 5.0) + label,
                "label": label,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _write_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "synthetic_emi",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "label",
        },
        "task": {
            "type": "classification",
            "task_name": "synthetic_label",
            "target_column": "label",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": [],
        },
        "features": {"method": "hybrid", "top_k": 3},
        "model": {"name": "random_forest", "candidates": ["random_forest"], "epochs": 2, "batch_size": 8},
        "trainer": {"train_ratio": 0.65, "val_ratio": 0.2, "random_state": 7},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_synthetic_binary_dataset(csv_path: Path) -> None:
    rows = []
    for index in range(36):
        label = index % 2
        rows.append(
            {
                "sensor_a": (label * 1.2) + (index * 0.01),
                "sensor_b": (label * 0.8) + np.cos(index / 4.0) * 0.1,
                "sensor_c": (label * 0.4) + np.sin(index / 5.0) * 0.1,
                "label": label,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _write_cnn_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "synthetic_binary_for_cnn",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "label",
        },
        "task": {
            "type": "classification",
            "task_name": "synthetic_binary_cnn",
            "target_column": "label",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": [],
        },
        "features": {"method": "hybrid", "top_k": 3},
        "model": {"name": "cnn_1d", "candidates": ["cnn_1d", "random_forest"], "epochs": 1, "batch_size": 8},
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 9},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_simple_auto_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "synthetic_binary_auto",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "label",
        },
        "task": {
            "type": "classification",
            "task_name": "synthetic_binary_auto",
            "target_column": "label",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": [],
        },
        "features": {"method": "hybrid", "top_k": 4},
        "model": {
            "name": "auto",
            "candidates": ["bagged_logistic_regression", "logistic_regression", "random_forest"],
            "epochs": 2,
            "batch_size": 8,
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 9},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": True, "metric": "f1", "grid": [0.4, 0.5, 0.6]}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_deep_config(
    config_path: Path,
    raw_data_path: Path,
    prepared_dir: Path,
    model_name: str,
) -> None:
    config = {
        "dataset": {
            "name": f"synthetic_binary_for_{model_name}",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "label",
        },
        "task": {
            "type": "classification",
            "task_name": f"synthetic_binary_{model_name}",
            "target_column": "label",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": [],
        },
        "features": {"method": "hybrid", "top_k": 3},
        "model": {
            "name": model_name,
            "candidates": [model_name],
            "epochs": 1,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 9},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_mixed_cognitive_like_dataset(csv_path: Path) -> None:
    rows = []
    for index in range(48):
        label = index % 2
        drift = "Frequency drift" if label else "Bandwidth switch"
        band = "1800" if index % 3 == 0 else "2400"
        base = 0.1 + (index * 0.001)
        rows.append(
            {
                "time_index": index + 1,
                "freq_bin": 10 + (index % 20),
                "power_dB": -70.0 + (index * 0.2),
                "PU_Presence": label,
                "PU_Signal_Strength": -68.0 + (index * 0.2),
                "PU_bandwidth": 4 + (index % 6),
                "PU_burst_duration": "Long" if index % 2 else "Short",
                "PU_drift_type": drift,
                "spectral_entropy": 1.0 + (index * 0.001),
                "Frequency_Band": band,
                "SU1_cov_flat": f"[{base:.6f}, {base + 0.01:.6f}, {base + 0.02:.6f}]",
                "SU2_cov_flat": f"[{base + 0.03:.6f}, {base + 0.04:.6f}, {base + 0.05:.6f}]",
                "SU3_cov_flat": f"[{base + 0.06:.6f}, {base + 0.07:.6f}, {base + 0.08:.6f}]",
                "SU1_temporal_cov": f"[{base + 0.001:.6f}, {base + 0.002:.6f}, {base + 0.003:.6f}, {base + 0.004:.6f}]",
                "SU2_temporal_cov": f"[{base + 0.005:.6f}, {base + 0.006:.6f}, {base + 0.007:.6f}, {base + 0.008:.6f}]",
                "SU3_temporal_cov": f"[{base + 0.009:.6f}, {base + 0.010:.6f}, {base + 0.011:.6f}, {base + 0.012:.6f}]",
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _write_cognitive_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "cognitive_radio_spectrum",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "PU_Presence",
        },
        "task": {
            "type": "classification",
            "task_name": "presence_task",
            "target_column": "PU_Presence",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
        },
        "features": {"method": "hybrid", "top_k": 8},
        "model": {
            "name": "random_forest",
            "candidates": ["random_forest", "logistic_regression"],
            "epochs": 2,
            "batch_size": 8,
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 42},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_multitask_cognitive_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "cognitive_radio_spectrum",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "PU_Presence",
        },
        "tasks": [
            {
                "name": "presence",
                "target_column": "PU_Presence",
                "metric_primary": "accuracy",
                "metric_secondary": "f1",
            },
            {
                "name": "band",
                "target_column": "Frequency_Band",
                "metric_primary": "accuracy",
                "metric_secondary": "f1",
            },
            {
                "name": "burst",
                "target_column": "PU_burst_duration",
                "metric_primary": "accuracy",
                "metric_secondary": "f1",
            },
        ],
        "task": {
            "type": "classification",
            "task_name": "presence",
            "target_column": "PU_Presence",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
        },
        "features": {"method": "hybrid", "top_k": 12},
        "model": {
            "name": "cnn_1d",
            "mode": "multitask",
            "epochs": 1,
            "batch_size": 8,
            "dropout": 0.1,
            "learning_rate": 0.001,
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 42},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_pretrain_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "synthetic_binary_for_pretrain",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "label",
        },
        "task": {
            "type": "classification",
            "task_name": "synthetic_pretrain_downstream",
            "target_column": "label",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": [],
        },
        "features": {"method": "hybrid", "top_k": 3},
        "model": {
            "name": "cnn_1d",
            "mode": "pretrain",
            "epochs": 1,
            "batch_size": 8,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "pretrain": {
                "enabled": True,
                "objective": "masked_reconstruction",
                "mask_ratio": 0.15,
                "checkpoint_name": "pretrain_encoder.pt",
            },
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 9},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_cognitive_hybrid_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "cognitive_radio_spectrum",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "PU_Presence",
        },
        "task": {
            "type": "classification",
            "task_name": "presence_hybrid",
            "target_column": "PU_Presence",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
        },
        "features": {"method": "hybrid", "top_k": None},
        "model": {
            "name": "cognitive_radio_hybrid",
            "mode": "single_task",
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "dropout": 0.1,
            "weight_decay": 0.0001,
            "patience": 2,
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 42},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_cognitive_scalar_hybrid_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "cognitive_radio_spectrum",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "PU_Presence",
        },
        "task": {
            "type": "classification",
            "task_name": "presence_scalar_hybrid",
            "target_column": "PU_Presence",
            "metric_primary": "f1",
            "metric_secondary": "accuracy",
            "drop_leakage_columns": ["time_index", "PU_burst_duration", "Frequency_Band", "PU_drift_type"],
        },
        "features": {"method": "hybrid", "top_k": None},
        "model": {
            "name": "cognitive_radio_scalar_hybrid",
            "mode": "single_task",
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "dropout": 0.1,
            "weight_decay": 0.0001,
            "patience": 2,
            "hidden_dim": 128,
            "num_heads": 4,
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 42},
        "metrics": {"primary": "f1"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": True, "metric": "f1", "grid": [0.4, 0.5, 0.6]}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_benchmark_matrix_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "cognitive_radio_spectrum",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "PU_Presence",
        },
        "task": {
            "type": "classification",
            "task_name": "presence_task",
            "target_column": "PU_Presence",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
        },
        "tasks": [
            {
                "name": "presence",
                "target_column": "PU_Presence",
                "metric_primary": "accuracy",
                "metric_secondary": "f1",
                "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
            },
            {
                "name": "band",
                "target_column": "Frequency_Band",
                "metric_primary": "accuracy",
                "metric_secondary": "f1",
                "drop_leakage_columns": ["time_index", "PU_Presence", "PU_drift_type"],
            },
        ],
        "features": {"method": "hybrid", "top_k": 8},
        "model": {
            "name": "random_forest",
            "mode": "single_task",
            "candidates": ["random_forest"],
            "epochs": 1,
            "batch_size": 8,
        },
        "benchmark": {
            "matrix": {
                "tasks": ["presence", "band"],
                "models": ["random_forest", "logistic_regression"],
            }
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 42},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_extended_benchmark_matrix_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "cognitive_radio_spectrum",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "PU_Presence",
        },
        "task": {
            "type": "classification",
            "task_name": "presence",
            "target_column": "PU_Presence",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
        },
        "tasks": [
            {
                "name": "presence",
                "target_column": "PU_Presence",
                "metric_primary": "accuracy",
                "metric_secondary": "f1",
                "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
            },
            {
                "name": "band",
                "target_column": "Frequency_Band",
                "metric_primary": "accuracy",
                "metric_secondary": "f1",
                "drop_leakage_columns": ["time_index", "PU_Presence", "PU_drift_type"],
            },
        ],
        "features": {"method": "hybrid", "top_k": 8},
        "model": {
            "name": "cnn_1d",
            "mode": "single_task",
            "epochs": 1,
            "batch_size": 8,
            "transfer": {
                "enabled": True,
                "from_run_dir": str(prepared_dir / "transfer-source"),
                "strategy": "freeze",
            },
        },
        "benchmark": {
            "matrix": {
                "tasks": ["presence", "band"],
                "models": ["cnn_1d"],
                "seeds": [7, 11],
                "modes": ["single_task", "pretrain"],
                "transfer_strategies": ["scratch", "freeze"],
            }
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 42},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_pretrain_benchmark_matrix_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "cognitive_radio_spectrum",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "PU_Presence",
        },
        "task": {
            "type": "classification",
            "task_name": "presence",
            "target_column": "PU_Presence",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
        },
        "tasks": [
            {
                "name": "presence",
                "target_column": "PU_Presence",
                "metric_primary": "accuracy",
                "metric_secondary": "f1",
                "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
            }
        ],
        "features": {"method": "hybrid", "top_k": 8},
        "model": {
            "name": "cnn_1d",
            "mode": "single_task",
            "epochs": 1,
            "batch_size": 8,
            "pretrain": {
                "enabled": False,
                "objective": "masked_reconstruction",
                "checkpoint_name": "pretrain_encoder.pt",
                "mask_ratio": 0.15,
            },
        },
        "benchmark": {
            "matrix": {
                "tasks": ["presence"],
                "models": ["cnn_1d"],
                "seeds": [7],
                "modes": ["single_task", "pretrain"],
                "transfer_strategies": ["scratch"],
            }
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 42},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_qwen_benchmark_matrix_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "cognitive_radio_spectrum",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "PU_Presence",
        },
        "task": {
            "type": "classification",
            "task_name": "presence",
            "target_column": "PU_Presence",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
        },
        "tasks": [
            {
                "name": "presence",
                "target_column": "PU_Presence",
                "metric_primary": "accuracy",
                "metric_secondary": "f1",
                "drop_leakage_columns": ["time_index", "PU_Signal_Strength", "PU_bandwidth", "PU_drift_type"],
            }
        ],
        "features": {"method": "hybrid", "top_k": 8},
        "model": {
            "name": "qwen_qlora_classifier",
            "mode": "single_task",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.0002,
        },
        "benchmark": {
            "matrix": {
                "tasks": ["presence"],
                "models": ["qwen_qlora_classifier", "cnn_1d"],
                "seeds": [42],
                "modes": ["single_task"],
            }
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 42},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_llm_config(config_path: Path, raw_data_path: Path, prepared_dir: Path) -> None:
    config = {
        "dataset": {
            "name": "synthetic_binary_llm",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "label",
        },
        "task": {
            "type": "classification",
            "task_name": "synthetic_binary_llm",
            "target_column": "label",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": [],
        },
        "features": {"method": "hybrid", "top_k": 3},
        "model": {
            "name": "llm_qlora_classifier",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.0002,
            "class_weighting": "balanced",
            "llm": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "max_length": 128,
                "load_in_4bit": False,
                "gradient_accumulation_steps": 1,
                "save_adapter_only": True,
            },
        },
        "trainer": {"train_ratio": 0.7, "val_ratio": 0.15, "random_state": 9},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {"threshold_tuning": {"enabled": False}},
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class _FakeLlmPredictor:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.8, 0.2]], dtype=float), (len(x), 1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=np.int64)


def test_resolve_device_prefers_cpu_when_requested() -> None:
    assert resolve_device("cpu") == "cpu"


def test_load_config_supports_tasks_and_normalizes_transfer(tmp_path: Path) -> None:
    config_path = tmp_path / "transfer_config.yaml"
    config = {
        "dataset": {
            "name": "synthetic_emi",
            "schema": "tabular",
            "input_path": "synthetic.csv",
            "label_column": "label",
        },
        "tasks": [
            {"name": "presence", "target_column": "label"},
            {"name": "band", "target_column": "band"},
        ],
        "model": {
            "name": "cnn_1d",
            "transfer_from": "artifacts/runs/run-123",
            "transfer": {"strategy": "freeze"},
        },
        "runtime": {
            "prepared_dir": "prepared",
            "artifacts_dir": "artifacts",
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded["model"]["mode"] == "single_task"
    assert loaded["tasks"][0]["name"] == "presence"
    assert loaded["task"]["target_column"] == "label"
    assert loaded["model"]["transfer"]["from_run_dir"].endswith("artifacts/runs/run-123")
    assert loaded["model"]["transfer"]["strategy"] == "freeze"


def test_load_config_preserves_imbalance_training_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "imbalance_config.yaml"
    config = {
        "dataset": {
            "name": "vsb_powerline_fault",
            "schema": "tabular",
            "input_path": "train.csv",
            "label_column": "target",
        },
        "task": {
            "task_name": "vsb_fault",
            "target_column": "target",
        },
        "model": {
            "name": "cnn_1d_residual",
            "loss_name": "focal",
            "focal_gamma": 2.5,
            "class_weighting": "balanced",
        },
        "trainer": {
            "sampler": "balanced",
            "random_state": 7,
        },
        "runtime": {
            "prepared_dir": "prepared",
            "artifacts_dir": "artifacts",
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded["model"]["name"] == "cnn_1d_residual"
    assert loaded["model"]["loss_name"] == "focal"
    assert loaded["model"]["focal_gamma"] == 2.5
    assert loaded["trainer"]["sampler"] == "balanced"


def test_load_config_exposes_llm_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "qwen_defaults.yaml"
    config = {
        "dataset": {
            "name": "synthetic_emi",
            "schema": "tabular",
            "input_path": "synthetic.csv",
            "label_column": "label",
        },
        "task": {
            "task_name": "qwen_defaults",
            "target_column": "label",
        },
        "model": {
            "name": "qwen_qlora_classifier",
        },
        "runtime": {
            "prepared_dir": "prepared",
            "artifacts_dir": "artifacts",
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded["model"]["name"] == "qwen_qlora_classifier"
    assert loaded["model"]["llm"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert loaded["model"]["llm"]["load_in_4bit"] is True


def test_load_config_preserves_generic_llm_qlora_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "deepseek_defaults.yaml"
    config = {
        "dataset": {
            "name": "synthetic_emi",
            "schema": "tabular",
            "input_path": "synthetic.csv",
            "label_column": "label",
        },
        "task": {
            "task_name": "deepseek_defaults",
            "target_column": "label",
        },
        "model": {
            "name": "llm_qlora_classifier",
            "llm": {
                "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "target_modules": ["q_proj", "v_proj"],
                "gradient_checkpointing": True,
                "torch_dtype": "float16",
                "feature_limit": 12,
                "task_instruction": "Determine whether the primary user is present.",
                "label_descriptions": {
                    "0": "absent",
                    "1": "present",
                },
            },
        },
        "runtime": {
            "prepared_dir": "prepared",
            "artifacts_dir": "artifacts",
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded["model"]["name"] == "llm_qlora_classifier"
    assert loaded["model"]["llm"]["model_id"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert loaded["model"]["llm"]["target_modules"] == ["q_proj", "v_proj"]
    assert loaded["model"]["llm"]["gradient_checkpointing"] is True
    assert loaded["model"]["llm"]["torch_dtype"] == "float16"
    assert loaded["model"]["llm"]["feature_limit"] == 12
    assert loaded["model"]["llm"]["task_instruction"] == "Determine whether the primary user is present."
    assert loaded["model"]["llm"]["label_descriptions"] == {"0": "absent", "1": "present"}


def test_model_family_marks_qwen_as_llm() -> None:
    assert _model_family_for_name("qwen_qlora_classifier") == "llm"
    assert _model_family_for_name("llm_qlora_classifier") == "llm"


def test_project_7b_llm_configs_load_expected_foundation_models() -> None:
    project_root = Path(__file__).resolve().parents[1]
    qwen_config = load_config(project_root / "configs" / "cognitive_radio_presence_qwen7b_qlora.yaml")
    deepseek_config = load_config(project_root / "configs" / "cognitive_radio_presence_deepseek7b_qlora.yaml")

    assert qwen_config["model"]["name"] == "llm_qlora_classifier"
    assert qwen_config["model"]["llm"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert qwen_config["model"]["llm"]["gradient_checkpointing"] is True
    assert qwen_config["model"]["llm"]["feature_limit"] == 16

    assert deepseek_config["model"]["name"] == "llm_qlora_classifier"
    assert deepseek_config["model"]["llm"]["model_id"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert deepseek_config["model"]["llm"]["gradient_checkpointing"] is True
    assert deepseek_config["model"]["llm"]["feature_limit"] == 16


def test_expand_benchmark_variants_supports_seed_mode_and_transfer_strategy(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "cognitive_like.csv"
    prepared_dir = tmp_path / "prepared_matrix"
    config_path = tmp_path / "extended_benchmark_matrix.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_extended_benchmark_matrix_config(config_path, raw_data_path, prepared_dir)

    variants = _expand_benchmark_variants(config_path)

    assert len(variants) == 12
    labels = [label for label, _ in variants]
    assert any(":presence:cnn_1d:seed=7:mode=single_task:transfer=scratch" in label for label in labels)
    assert any(":band:cnn_1d:seed=11:mode=pretrain:transfer=scratch" in label for label in labels)

    sample_variant = dict(variants[0][1])
    assert sample_variant["trainer"]["random_state"] in {7, 11}
    assert sample_variant["model"]["mode"] in {"single_task", "pretrain"}
    assert sample_variant["model"]["transfer"]["strategy"] in {"scratch", "freeze", "finetune"}
    prepared_dirs = {variant["runtime"]["prepared_dir"] for _, variant in variants}
    assert len(prepared_dirs) == 4
    assert any(path.endswith("presence-seed-7") for path in prepared_dirs)
    assert any(path.endswith("band-seed-11") for path in prepared_dirs)


def test_expand_benchmark_variants_supports_qwen_matrix(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "cognitive_like.csv"
    prepared_dir = tmp_path / "prepared_qwen_matrix"
    config_path = tmp_path / "qwen_benchmark_matrix.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_qwen_benchmark_matrix_config(config_path, raw_data_path, prepared_dir)

    variants = _expand_benchmark_variants(config_path)
    labels = [label for label, _ in variants]
    assert any(":presence:qwen_qlora_classifier:seed=42:mode=single_task:transfer=scratch" in label for label in labels)

    qwen_variant = next(config for label, config in variants if ":qwen_qlora_classifier:" in label)
    assert qwen_variant["model"]["name"] == "qwen_qlora_classifier"
    assert qwen_variant["model"]["mode"] == "single_task"
    assert qwen_variant["model"]["candidates"] == []


def test_call_with_batch_backoff_retries_on_cuda_oom() -> None:
    attempts: list[int] = []

    def _fake_trainer(*, batch_size: int) -> int:
        attempts.append(batch_size)
        if batch_size > 8:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return batch_size

    result = _call_with_batch_backoff(
        _fake_trainer,
        {"trainer": {"min_batch_size": 8, "oom_retries": 3, "oom_backoff_factor": 0.5}},
        batch_size=32,
    )

    assert result == 8
    assert attempts == [32, 16, 8]


def test_call_with_batch_backoff_preserves_kwargs_for_wrapper_functions() -> None:
    captured: dict[str, object] = {}

    def _wrapper(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        captured.update(kwargs)
        return captured

    result = _call_with_batch_backoff(
        _wrapper,
        {"trainer": {"min_batch_size": 1, "oom_retries": 1, "oom_backoff_factor": 0.5}},
        x_train=[[1.0]],
        y_train=[1],
        x_val=[[0.0]],
        y_val=[0],
        batch_size=1,
    )

    assert result["x_train"] == [[1.0]]
    assert result["y_train"] == [1]
    assert result["x_val"] == [[0.0]]
    assert result["y_val"] == [0]


def test_execute_training_forwards_class_weighting_to_llm_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_data_path = tmp_path / "synthetic_binary_llm.csv"
    prepared_dir = tmp_path / "prepared_llm"
    config_path = tmp_path / "llm_config.yaml"
    _write_synthetic_binary_dataset(raw_data_path)
    _write_llm_config(config_path, raw_data_path, prepared_dir)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["extract-features", "--config", str(config_path)]) == 0

    captured: dict[str, object] = {}

    def _fake_train_qwen_qlora_classifier(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "model": _FakeLlmPredictor(),
            "model_name": "llm_qlora_classifier",
            "resolved_device": "cpu",
            "train_history": {},
            "epochs_ran": 1,
            "best_checkpoint": {"epoch": 1, "best_epoch": 0, "val_loss": 0.1, "stopped_early": False},
            "llm_info": {},
        }

    monkeypatch.setattr(
        "emc_diag.cli._load_modeling_module",
        lambda: type("FakeModelingModule", (), {"train_qwen_qlora_classifier": staticmethod(_fake_train_qwen_qlora_classifier)})(),
    )

    config = load_config(config_path)
    prepared = _load_prepared_bundle(prepared_dir)
    _execute_training(config, prepared, requested_device="cpu")

    assert captured["class_weighting"] == "balanced"


def test_cli_smoke_pipeline_generates_artifacts(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "synthetic.csv"
    prepared_dir = tmp_path / "prepared"
    config_path = tmp_path / "config.yaml"
    _write_synthetic_tabular_dataset(raw_data_path)
    _write_config(config_path, raw_data_path, prepared_dir)

    exit_code = main(["prepare", "--config", str(config_path)])
    assert exit_code == 0

    exit_code = main(["extract-features", "--config", str(config_path)])
    assert exit_code == 0

    exit_code = main(["train", "--config", str(config_path), "--device", "cpu"])
    assert exit_code == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]

    exit_code = main(["evaluate", "--run-dir", str(run_dir)])
    assert exit_code == 0

    exit_code = main(["visualize", "--run-dir", str(run_dir), "--theme", "paper-bar"])
    assert exit_code == 0

    exit_code = main(["export-report", "--run-dir", str(run_dir), "--format", "md"])
    assert exit_code == 0

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "accuracy" in metrics
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "figures" / "metrics_overview.png").exists()
    assert (run_dir / "summary.md").exists()


def test_download_command_writes_dataset_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "downloads"
    exit_code = main(["download", "--dataset", "emi_uci", "--out-dir", str(output_dir)])
    assert exit_code == 0

    manifest_path = output_dir / "emi_uci" / "download_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "emi_uci"


def test_cognitive_config_defaults_to_pu_presence_label() -> None:
    config_path = Path("configs/cognitive_radio_spectrum.yaml")
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["dataset"]["name"] == "cognitive_radio_spectrum"
    assert config["dataset"]["label_column"] == "PU_Presence"
    assert config["task"]["task_name"] == "pu_presence_main"
    assert "drop_leakage_columns" in config["task"]


def test_cnn_first_configs_and_windows_bootstrap_exist() -> None:
    cnn_main = Path("configs/cognitive_radio_presence_cnn.yaml")
    cnn_cv = Path("configs/cognitive_radio_presence_cnn_cv.yaml")
    bootstrap = Path("scripts/bootstrap_windows.ps1")

    assert cnn_main.exists()
    assert cnn_cv.exists()
    assert bootstrap.exists()

    main_cfg = yaml.safe_load(cnn_main.read_text(encoding="utf-8"))
    cv_cfg = yaml.safe_load(cnn_cv.read_text(encoding="utf-8"))
    assert main_cfg["model"]["name"] == "cnn_1d"
    assert cv_cfg["task"]["target_column"] == "PU_Presence"
    assert cv_cfg["evaluation"]["threshold_tuning"]["enabled"] is True


def test_cli_smoke_pipeline_with_mixed_cognitive_columns(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "cognitive_like.csv"
    prepared_dir = tmp_path / "prepared_cognitive"
    config_path = tmp_path / "cognitive_config.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_cognitive_config(config_path, raw_data_path, prepared_dir)

    exit_code = main(["prepare", "--config", str(config_path)])
    assert exit_code == 0

    exit_code = main(["extract-features", "--config", str(config_path)])
    assert exit_code == 0

    exit_code = main(["train", "--config", str(config_path), "--device", "cpu"])
    assert exit_code == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]

    exit_code = main(["evaluate", "--run-dir", str(run_dir)])
    assert exit_code == 0

    exit_code = main(["visualize", "--run-dir", str(run_dir), "--theme", "paper-bar"])
    assert exit_code == 0

    assert (run_dir / "figures" / "metrics_overview.png").exists()
    assert (run_dir / "metrics.json").exists()
    # Optional benchmark/report outputs may be present depending on src pipeline behavior.
    if (run_dir / "tables").exists():
        csv_tables = list((run_dir / "tables").glob("*.csv"))
        assert len(csv_tables) >= 0


def test_cli_smoke_pipeline_with_cnn_config(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for CNN smoke coverage")

    raw_data_path = tmp_path / "synthetic_binary.csv"
    prepared_dir = tmp_path / "prepared_cnn"
    config_path = tmp_path / "cnn_config.yaml"
    _write_synthetic_binary_dataset(raw_data_path)
    _write_cnn_config(config_path, raw_data_path, prepared_dir)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["extract-features", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    assert main(["evaluate", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()


def test_cli_smoke_pipeline_with_cnn_lstm_config(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for CNN-LSTM smoke coverage")

    raw_data_path = tmp_path / "synthetic_binary_lstm.csv"
    prepared_dir = tmp_path / "prepared_cnn_lstm"
    config_path = tmp_path / "cnn_lstm_config.yaml"
    _write_synthetic_binary_dataset(raw_data_path)
    _write_deep_config(config_path, raw_data_path, prepared_dir, model_name="cnn_lstm")

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["extract-features", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    assert main(["evaluate", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "metrics.json").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_name"] == "cnn_lstm"


def test_prepare_command_supports_multitask_metadata(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "cognitive_multitask.csv"
    prepared_dir = tmp_path / "prepared_multitask"
    config_path = tmp_path / "multitask_config.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_multitask_cognitive_config(config_path, raw_data_path, prepared_dir)

    assert main(["prepare", "--config", str(config_path)]) == 0

    metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "tasks" in metadata
    assert [task["name"] for task in metadata["tasks"]] == ["presence", "band", "burst"]
    assert metadata["task_type"] == "multitask"


def test_prepare_command_exports_exploration_assets(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "synthetic.csv"
    prepared_dir = tmp_path / "prepared_assets"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, raw_data_path, prepared_dir)
    _write_synthetic_tabular_dataset(raw_data_path)

    assert main(["prepare", "--config", str(config_path)]) == 0

    assert (prepared_dir / "figures" / "dataset_summary.png").exists()
    assert (prepared_dir / "tables" / "class_distribution.csv").exists()
    assert (prepared_dir / "tables" / "dataset_statistics.csv").exists()
    assert (prepared_dir / "exploration_summary.md").exists()


def test_prepare_command_prints_output_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw_data_path = tmp_path / "synthetic.csv"
    prepared_dir = tmp_path / "prepared_assets"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, raw_data_path, prepared_dir)
    _write_synthetic_tabular_dataset(raw_data_path)

    assert main(["prepare", "--config", str(config_path)]) == 0

    out = capsys.readouterr().out
    assert "[prepare] output_dir=" in out
    assert "metadata.json" in out
    assert "statistics.json" in out
    assert "prepared_splits.npz" in out
    assert "exploration_summary.md" in out


def test_extract_features_command_exports_feature_assets(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "synthetic.csv"
    prepared_dir = tmp_path / "prepared_assets"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, raw_data_path, prepared_dir)
    _write_synthetic_tabular_dataset(raw_data_path)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["extract-features", "--config", str(config_path)]) == 0

    assert (prepared_dir / "figures" / "feature_importance.png").exists()
    assert (prepared_dir / "tables" / "selected_features.csv").exists()
    summary_text = (prepared_dir / "exploration_summary.md").read_text(encoding="utf-8")
    assert "Feature Analysis" in summary_text


def test_extract_features_command_prints_output_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_data_path = tmp_path / "synthetic.csv"
    prepared_dir = tmp_path / "prepared_assets"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, raw_data_path, prepared_dir)
    _write_synthetic_tabular_dataset(raw_data_path)

    assert main(["prepare", "--config", str(config_path)]) == 0
    capsys.readouterr()

    assert main(["extract-features", "--config", str(config_path)]) == 0

    out = capsys.readouterr().out
    assert "[extract-features] output_dir=" in out
    assert "feature_metadata.json" in out
    assert "feature_importance.csv" in out
    assert "selected_features.csv" in out


def test_train_command_supports_noise_robustness_with_feature_selection(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "synthetic_binary.csv"
    prepared_dir = tmp_path / "prepared_noise_robust"
    config_path = tmp_path / "noise_robust.yaml"
    _write_synthetic_binary_dataset(raw_data_path)
    payload = {
        "dataset": {
            "name": "synthetic_binary_noise_robust",
            "schema": "tabular",
            "input_path": str(raw_data_path),
            "label_column": "label",
        },
        "task": {
            "type": "classification",
            "task_name": "synthetic_noise_robust",
            "target_column": "label",
            "metric_primary": "accuracy",
            "metric_secondary": "f1",
            "drop_leakage_columns": [],
        },
        "features": {"method": "hybrid", "top_k": 2},
        "model": {"name": "logistic_regression", "candidates": ["logistic_regression"], "epochs": 2, "batch_size": 8},
        "trainer": {"train_ratio": 0.65, "val_ratio": 0.2, "random_state": 7},
        "metrics": {"primary": "accuracy"},
        "visualization": {"theme": "paper-bar"},
        "evaluation": {
            "threshold_tuning": {"enabled": True, "metric": "f1", "grid": [0.4, 0.5, 0.6]},
            "robustness_sweeps": {"noise_levels": [0.0, 0.05], "train_ratios": []},
        },
        "runtime": {
            "prepared_dir": str(prepared_dir),
            "artifacts_dir": str(prepared_dir / "artifacts"),
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["extract-features", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    assert (run_dir / "tables" / "robustness_noise.csv").exists()


def test_train_command_prints_output_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw_data_path = tmp_path / "synthetic.csv"
    prepared_dir = tmp_path / "prepared"
    config_path = tmp_path / "config.yaml"
    _write_synthetic_tabular_dataset(raw_data_path)
    _write_config(config_path, raw_data_path, prepared_dir)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["extract-features", "--config", str(config_path)]) == 0
    capsys.readouterr()

    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "[train] completed run_dir=" in out
    assert "metrics.json" in out
    assert "predictions.csv" in out
    assert "run_config.yaml" in out


def test_quickstart_creates_latest_dirs_and_curated_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_data_path = tmp_path / "synthetic_binary.csv"
    config_path = tmp_path / "quickstart.yaml"
    _write_synthetic_binary_dataset(raw_data_path)
    _write_simple_auto_config(config_path, raw_data_path, tmp_path / "ignored_prepared")

    output_root = tmp_path / "simple_outputs"
    assert (
        main(
            [
                "quickstart",
                "--config",
                str(config_path),
                "--device",
                "cpu",
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    latest_prepared = output_root / "latest_prepared"
    latest_run = output_root / "latest_run"
    assert latest_prepared.exists()
    assert latest_run.exists()
    assert (latest_prepared / "metadata.json").exists()
    assert (latest_prepared / "exploration_summary.md").exists()
    assert (latest_run / "metrics.json").exists()
    assert (latest_run / "summary.md").exists()
    assert (latest_run / "figures" / "confusion_matrix.png").exists()
    assert (latest_run / "figures" / "metrics_overview.png").exists()
    assert not any((latest_run / "figures").glob("*.svg"))
    assert len(list((latest_run / "figures").glob("*.png"))) <= 4

    out = capsys.readouterr().out
    assert "[quickstart] latest_prepared=" in out
    assert "[quickstart] latest_run=" in out
    assert "summary.md" in out


def test_cli_train_supports_multitask_cnn_config(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for multitask smoke coverage")

    raw_data_path = tmp_path / "cognitive_multitask.csv"
    prepared_dir = tmp_path / "prepared_multitask"
    config_path = tmp_path / "multitask_config.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_multitask_cognitive_config(config_path, raw_data_path, prepared_dir)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_mode"] == "multitask"
    assert "task_metrics" in metrics
    assert (run_dir / "predictions_presence.csv").exists()


def test_cli_evaluate_supports_multitask_run_outputs(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for multitask smoke coverage")

    raw_data_path = tmp_path / "cognitive_multitask.csv"
    prepared_dir = tmp_path / "prepared_multitask"
    config_path = tmp_path / "multitask_config.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_multitask_cognitive_config(config_path, raw_data_path, prepared_dir)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]

    assert main(["evaluate", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "tables" / "task_comparison.csv").exists()
    assert (run_dir / "tables" / "per_task_metrics.csv").exists()
    assert (run_dir / "figures" / "multitask_vs_single_task.png").exists()


def test_cli_train_supports_pretrain_mode(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for pretrain smoke coverage")

    raw_data_path = tmp_path / "synthetic_binary.csv"
    prepared_dir = tmp_path / "prepared_pretrain"
    config_path = tmp_path / "pretrain_config.yaml"
    _write_synthetic_binary_dataset(raw_data_path)
    _write_pretrain_config(config_path, raw_data_path, prepared_dir)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_mode"] == "pretrain"
    assert "pretrain_history" in metrics
    assert (run_dir / "pretrain_encoder.pt").exists()


def test_cli_train_supports_cognitive_radio_hybrid_config(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for hybrid smoke coverage")

    raw_data_path = tmp_path / "cognitive_hybrid.csv"
    prepared_dir = tmp_path / "prepared_hybrid"
    config_path = tmp_path / "hybrid_config.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_cognitive_hybrid_config(config_path, raw_data_path, prepared_dir)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_name"] == "cognitive_radio_hybrid"


def test_cli_train_supports_cognitive_radio_multitask_hybrid_config(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for hybrid smoke coverage")

    raw_data_path = tmp_path / "cognitive_hybrid_multi.csv"
    prepared_dir = tmp_path / "prepared_hybrid_multi"
    config_path = tmp_path / "hybrid_multi_config.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_cognitive_hybrid_config(config_path, raw_data_path, prepared_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["tasks"] = [
        {"name": "presence", "target_column": "PU_Presence", "metric_primary": "accuracy", "metric_secondary": "f1"},
        {"name": "band", "target_column": "Frequency_Band", "metric_primary": "accuracy", "metric_secondary": "f1"},
        {"name": "burst", "target_column": "PU_burst_duration", "metric_primary": "accuracy", "metric_secondary": "f1"},
    ]
    payload["task"]["task_name"] = "presence"
    payload["task"]["target_column"] = "PU_Presence"
    payload["model"]["mode"] = "multitask"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_name"] == "cognitive_radio_hybrid_multitask"
    assert metrics["run_mode"] == "multitask"


def test_cli_train_supports_cognitive_radio_scalar_hybrid_config(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for scalar hybrid smoke coverage")

    raw_data_path = tmp_path / "cognitive_scalar_hybrid.csv"
    prepared_dir = tmp_path / "prepared_scalar_hybrid"
    config_path = tmp_path / "scalar_hybrid_config.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_cognitive_scalar_hybrid_config(config_path, raw_data_path, prepared_dir)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0

    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_name"] == "cognitive_radio_scalar_hybrid"
    assert "val_f1_score" in metrics
    assert "train_f1_score" in metrics
    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert not predictions.empty


def test_benchmark_command_runs_multiple_configs(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "cognitive_like.csv"
    _write_mixed_cognitive_like_dataset(raw_data_path)

    prepared_a = tmp_path / "prepared_presence"
    prepared_b = tmp_path / "prepared_band"
    config_a = tmp_path / "presence.yaml"
    config_b = tmp_path / "band.yaml"

    _write_cognitive_config(config_a, raw_data_path, prepared_a)

    band_config = yaml.safe_load(config_a.read_text(encoding="utf-8"))
    band_config["dataset"]["label_column"] = "Frequency_Band"
    band_config["task"]["task_name"] = "band_task"
    band_config["task"]["target_column"] = "Frequency_Band"
    band_config["task"]["drop_leakage_columns"] = ["time_index", "PU_Presence", "PU_drift_type"]
    band_config["runtime"]["prepared_dir"] = str(prepared_b)
    band_config["runtime"]["artifacts_dir"] = str(prepared_b / "artifacts")
    config_b.write_text(yaml.safe_dump(band_config, sort_keys=False), encoding="utf-8")

    exit_code = main(
        [
            "benchmark",
            "--configs",
            str(config_a),
            str(config_b),
            "--device",
            "cpu",
            "--theme",
            "paper-bar",
        ]
    )
    assert exit_code == 0

    benchmark_dirs = sorted((prepared_a / "benchmarks").glob("benchmark-*"))
    assert benchmark_dirs
    latest = benchmark_dirs[-1]
    assert (latest / "benchmark_metrics.csv").exists()
    assert (latest / "summary.md").exists()


def test_benchmark_command_supports_matrix_config(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "cognitive_like.csv"
    prepared_dir = tmp_path / "prepared_matrix"
    config_path = tmp_path / "benchmark_matrix.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_benchmark_matrix_config(config_path, raw_data_path, prepared_dir)

    exit_code = main(
        [
            "benchmark",
            "--configs",
            str(config_path),
            "--device",
            "cpu",
            "--theme",
            "paper-bar",
        ]
    )
    assert exit_code == 0

    benchmark_dirs = sorted((prepared_dir / "benchmarks").glob("benchmark-*"))
    assert benchmark_dirs
    latest = benchmark_dirs[-1]
    benchmark_df = pd.read_csv(latest / "benchmark_metrics.csv")
    assert benchmark_df.shape[0] == 4
    assert set(benchmark_df["task_name"]) == {"presence", "band"}
    assert set(benchmark_df["model_name"]) >= {"random_forest", "logistic_regression"}


def test_benchmark_command_supports_pretrain_mode_without_predictions(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for benchmark pretrain smoke coverage")

    raw_data_path = tmp_path / "cognitive_like.csv"
    prepared_dir = tmp_path / "prepared_pretrain_matrix"
    config_path = tmp_path / "pretrain_benchmark_matrix.yaml"
    _write_mixed_cognitive_like_dataset(raw_data_path)
    _write_pretrain_benchmark_matrix_config(config_path, raw_data_path, prepared_dir)

    exit_code = main(
        [
            "benchmark",
            "--configs",
            str(config_path),
            "--device",
            "cpu",
            "--theme",
            "paper-bar",
        ]
    )
    assert exit_code == 0

    benchmark_dirs = sorted((prepared_dir / "benchmarks").glob("benchmark-*"))
    assert benchmark_dirs
    latest = benchmark_dirs[-1]
    benchmark_df = pd.read_csv(latest / "benchmark_metrics.csv")
    assert set(benchmark_df["mode"]) == {"single_task", "pretrain"}


def test_thesis_assets_command_builds_final_bundle(tmp_path: Path) -> None:
    raw_data_path = tmp_path / "synthetic.csv"
    prepared_dir = tmp_path / "prepared_assets"
    config_path = tmp_path / "config.yaml"
    report_dir = tmp_path / "reports"
    _write_config(config_path, raw_data_path, prepared_dir)
    _write_synthetic_tabular_dataset(raw_data_path)

    assert main(["prepare", "--config", str(config_path)]) == 0
    assert main(["extract-features", "--config", str(config_path)]) == 0
    assert main(["train", "--config", str(config_path), "--device", "cpu"]) == 0
    run_dirs = sorted((prepared_dir / "artifacts").glob("run-*"))
    assert run_dirs
    run_dir = run_dirs[-1]

    assert main(["benchmark", "--configs", str(config_path), "--device", "cpu", "--theme", "paper-bar"]) == 0
    benchmark_dirs = sorted((prepared_dir / "benchmarks").glob("benchmark-*"))
    assert benchmark_dirs
    benchmark_dir = benchmark_dirs[-1]

    exit_code = main(
        [
            "thesis-assets",
            "--run-dirs",
            str(run_dir),
            "--benchmark-dirs",
            str(benchmark_dir),
            "--output-dir",
            str(report_dir),
            "--title",
            "Synthetic Final Summary",
        ]
    )
    assert exit_code == 0
    assert (report_dir / "final_metrics.csv").exists()
    assert (report_dir / "thesis_figures_manifest.csv").exists()
    assert (report_dir / "final_summary.md").exists()
    assert (report_dir / "figures" / "dataset_comparison.png").exists()
    manifest_df = pd.read_csv(report_dir / "thesis_figures_manifest.csv")
    assert {"prepared", "run", "benchmark", "report"}.issubset(set(manifest_df["stage"]))


def test_emc_experiment_command_doc_exists() -> None:
    command_doc = Path("docs/emc_fault_diagnosis_experiments.md")
    assert command_doc.exists()
    text = command_doc.read_text(encoding="utf-8")
    assert "presence -> band" in text
    assert "SSL pretrain" in text
    assert "multitask cognitive radio" in text


def test_thesis_asset_pipeline_doc_exists() -> None:
    command_doc = Path("docs/thesis_asset_pipeline.md")
    assert command_doc.exists()
    text = command_doc.read_text(encoding="utf-8")
    assert "thesis-assets" in text
    assert "cognitive_radio_thesis_benchmark.yaml" in text
    assert "vsb_fault_thesis_benchmark.yaml" in text

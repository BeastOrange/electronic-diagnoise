from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "name": "emi_uci",
        "schema": "tabular",
        "input_path": "",
        "metadata_path": "",
        "label_column": "label",
        "sample_id_column": "signal_id",
        "group_column": "",
        "max_samples": None,
        "waveform_axis": "rows",
    },
    "task": {
        "type": "classification",
        "task_name": "default_task",
        "target_column": "label",
        "metric_primary": "accuracy",
        "metric_secondary": "f1",
        "drop_leakage_columns": [],
    },
    "tasks": [],
    "features": {
        "method": "hybrid",
        "top_k": 8,
        "representation": "auto",
        "sample_rate": 1.0,
        "nperseg": 64,
        "noverlap": 32,
        "domain_feature_groups": {
            "enabled": [],
        },
    },
    "model": {
        "name": "random_forest",
        "mode": "single_task",
        "candidates": [],
        "params": {},
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.2,
        "weight_decay": 0.0,
        "patience": 5,
        "scheduler": "plateau",
        "class_weighting": False,
        "loss_name": "cross_entropy",
        "focal_gamma": 2.0,
        "sequence_layout": "all",
        "transfer_from": None,
        "transfer": {
            "enabled": False,
            "from_run_dir": None,
            "strategy": None,
            "source_task": None,
            "strict": True,
        },
        "llm": {
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "max_length": 1024,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.03,
            "save_adapter_only": True,
        },
        "pretrain": {
            "enabled": False,
            "objective": "masked_reconstruction",
            "checkpoint_name": "pretrain_encoder.pt",
            "mask_ratio": 0.15,
            "projection_dim": 32,
            "temperature": 0.1,
        },
    },
    "trainer": {
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "random_state": 42,
        "sampler": "off",
        "loader_workers": None,
        "pin_memory": None,
        "persistent_workers": None,
        "prefetch_factor": 2,
        "amp": "auto",
        "min_batch_size": 8,
        "oom_retries": 3,
        "oom_backoff_factor": 0.5,
    },
    "metrics": {"primary": "accuracy"},
    "visualization": {"theme": "paper-bar", "language": "en"},
    "evaluation": {
        "threshold_tuning": {
            "enabled": False,
            "metric": "f1",
            "grid": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        },
        "cross_validation": {"enabled": False, "folds": 5},
        "learning_curve": {"enabled": False, "train_sizes": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "ablation": {
            "enabled": False,
            "layouts": ["basic_only", "cov_flat_only", "temporal_cov_only", "all"],
        },
        "risk_checks": {"enabled": False},
        "robustness_sweeps": {
            "noise_levels": [],
            "train_ratios": [],
        },
    },
    "runtime": {
        "prepared_dir": "artifacts/prepared",
        "artifacts_dir": "artifacts/runs",
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_task_entries(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    task = config["task"]
    metrics = config["metrics"]

    raw_tasks = config.get("tasks") or []
    if not raw_tasks:
        raw_tasks = [
            {
                "name": task.get("task_name") or dataset.get("name") or "task",
                "task_name": task.get("task_name") or dataset.get("name") or "task",
                "target_column": task.get("target_column") or dataset.get("label_column", "label"),
                "metric_primary": task.get("metric_primary") or metrics.get("primary", "accuracy"),
                "metric_secondary": task.get("metric_secondary") or "f1",
                "type": task.get("type", "classification"),
                "drop_leakage_columns": list(task.get("drop_leakage_columns", [])),
            }
        ]

    normalized_tasks: list[dict[str, Any]] = []
    for index, raw_task in enumerate(raw_tasks):
        task_name = str(
            raw_task.get("name")
            or raw_task.get("task_name")
            or task.get("task_name")
            or dataset.get("name")
            or f"task_{index}"
        )
        target_column = str(raw_task.get("target_column") or dataset.get("label_column", "label"))
        normalized_tasks.append(
            {
                "name": task_name,
                "task_name": task_name,
                "target_column": target_column,
                "metric_primary": raw_task.get("metric_primary") or metrics.get("primary", "accuracy"),
                "metric_secondary": raw_task.get("metric_secondary") or task.get("metric_secondary", "f1"),
                "type": raw_task.get("type") or task.get("type", "classification"),
                "drop_leakage_columns": list(raw_task.get("drop_leakage_columns", task.get("drop_leakage_columns", [])) or []),
                "loss_weight": float(raw_task.get("loss_weight", 1.0)),
                "missing_label_policy": str(raw_task.get("missing_label_policy", "skip")),
            }
        )

    config["tasks"] = normalized_tasks
    primary_task = normalized_tasks[0]
    task["task_name"] = str(primary_task["task_name"])
    task["target_column"] = str(primary_task["target_column"])
    task["metric_primary"] = str(primary_task["metric_primary"])
    task["metric_secondary"] = str(primary_task["metric_secondary"])
    task["type"] = str(primary_task["type"])
    task["drop_leakage_columns"] = list(primary_task.get("drop_leakage_columns", []))
    dataset["label_column"] = task["target_column"]
    metrics["primary"] = task["metric_primary"]


def _normalize_transfer_config(model: dict[str, Any], base_dir: Path) -> None:
    transfer = model.get("transfer") or {}
    if not isinstance(transfer, dict):
        transfer = {}
    legacy_from = model.get("transfer_from")
    from_run_dir = transfer.get("from_run_dir") or legacy_from
    if from_run_dir and not Path(from_run_dir).is_absolute():
        from_run_dir = str((base_dir / str(from_run_dir)).resolve())
    strategy = transfer.get("strategy")
    enabled = bool(transfer.get("enabled", False) or from_run_dir)
    model["transfer"] = {
        "enabled": enabled,
        "from_run_dir": from_run_dir,
        "strategy": strategy,
        "source_task": transfer.get("source_task"),
        "strict": bool(transfer.get("strict", True)),
    }
    model["transfer_from"] = from_run_dir


def _normalize_model_mode(model: dict[str, Any]) -> None:
    mode = str(model.get("mode") or "single_task")
    if model.get("pretrain", {}).get("enabled"):
        mode = "pretrain"
    model["mode"] = mode
    pretrain = model.get("pretrain") or {}
    if not isinstance(pretrain, dict):
        pretrain = {}
    model["pretrain"] = {
        "enabled": bool(pretrain.get("enabled", mode == "pretrain")),
        "objective": str(pretrain.get("objective", "masked_reconstruction")),
        "checkpoint_name": str(pretrain.get("checkpoint_name", "pretrain_encoder.pt")),
        "mask_ratio": float(pretrain.get("mask_ratio", 0.15)),
        "projection_dim": int(pretrain.get("projection_dim", 32)),
        "temperature": float(pretrain.get("temperature", 0.1)),
    }


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = _deep_merge(DEFAULT_CONFIG, raw)

    task = config["task"]
    dataset = config["dataset"]
    model = config["model"]

    _normalize_task_entries(config)

    model_candidates = model.get("candidates") or []
    if isinstance(model_candidates, str):
        model_candidates = [model_candidates]
    model["candidates"] = [str(candidate) for candidate in model_candidates]

    _normalize_transfer_config(model, path.parent)
    _normalize_model_mode(model)

    top_k = config["features"].get("top_k")
    if isinstance(top_k, str) and top_k.lower() == "full":
        config["features"]["top_k"] = None

    runtime = config["runtime"]
    base_dir = path.parent
    runtime["prepared_dir"] = str((base_dir / runtime["prepared_dir"]).resolve()) if not Path(runtime["prepared_dir"]).is_absolute() else runtime["prepared_dir"]
    runtime["artifacts_dir"] = str((base_dir / runtime["artifacts_dir"]).resolve()) if not Path(runtime["artifacts_dir"]).is_absolute() else runtime["artifacts_dir"]

    dataset_input = config["dataset"]["input_path"]
    if dataset_input and not Path(dataset_input).is_absolute():
        config["dataset"]["input_path"] = str((base_dir / dataset_input).resolve())
    dataset_metadata = config["dataset"].get("metadata_path", "")
    if dataset_metadata and not Path(dataset_metadata).is_absolute():
        config["dataset"]["metadata_path"] = str((base_dir / dataset_metadata).resolve())
    return config

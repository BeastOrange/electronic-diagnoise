from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable

from emc_diag.bootstrap import configure_numeric_runtime_defaults

configure_numeric_runtime_defaults()

import numpy as np
import pandas as pd
import yaml

from emc_diag.artifacts import create_run_dir, load_json, load_npz, save_dataframe, save_json, save_npz
from emc_diag.config import load_config
from emc_diag.data_pipeline import load_local_data, prepare_dataset
from emc_diag.dataset_registry import get_dataset_info
from emc_diag.evaluation import (
    evaluate_predictions,
    per_class_metrics_frame,
    run_noise_robustness_sweep,
    run_train_ratio_sweep,
    score_metric,
)
from emc_diag.feature_engineering import (
    build_cognitive_radio_hybrid_bundle,
    build_sequence_bundle,
    extract_feature_bundle,
    make_feature_group_view,
    make_prepared_layout_view,
)
from emc_diag.runtime import ensure_dir, resolve_device, timestamp_tag

DEEP_MODEL_NAMES = {"cnn_1d", "cnn_1d_residual", "cnn_lstm", "transformer_1d", "cognitive_radio_hybrid", "cognitive_radio_scalar_hybrid"}
QUICKSTART_DEFAULT_CONFIG = "configs/cognitive_radio_spectrum.yaml"
QUICKSTART_CORE_RUN_FILES: tuple[tuple[str, str], ...] = (
    ("metrics.json", "model metrics and key scores"),
    ("predictions.csv", "per-sample predictions"),
    ("summary.md", "human-readable run summary"),
    ("figures/confusion_matrix.png", "confusion matrix figure"),
    ("figures/metrics_overview.png", "overall metrics figure"),
    ("figures/per_class_metrics.png", "per-class performance figure"),
    ("figures/candidate_comparison.png", "candidate model comparison"),
    ("tables/per_class_metrics.csv", "per-class metrics table"),
    ("tables/candidate_scores.csv", "candidate score table"),
)
QUICKSTART_CORE_PREPARED_FILES: tuple[tuple[str, str], ...] = (
    ("metadata.json", "prepared dataset metadata"),
    ("statistics.json", "dataset class counts and statistics"),
    ("exploration_summary.md", "human-readable preparation summary"),
    ("figures/dataset_summary.png", "class distribution figure"),
    ("figures/feature_importance.png", "top feature importance figure"),
    ("tables/class_distribution.csv", "class distribution table"),
    ("tables/dataset_statistics.csv", "dataset statistics table"),
    ("tables/selected_features.csv", "selected feature list"),
)


def _load_modeling_module() -> Any:
    from emc_diag import modeling as modeling_module

    return modeling_module


def _get_model_selection_classes() -> tuple[Any, Any]:
    try:
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    except Exception:  # pragma: no cover
        return None, None
    return StratifiedKFold, StratifiedShuffleSplit


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    ensure_dir(target.parent)
    shutil.copy2(source, target)


def _curate_output_dir(source_dir: Path, target_dir: Path, file_specs: tuple[tuple[str, str], ...]) -> Path:
    _reset_dir(target_dir)
    for relative_path, _ in file_specs:
        _copy_if_exists(source_dir / relative_path, target_dir / relative_path)
    return target_dir


def _existing_file_specs(base_dir: Path, file_specs: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
    return [
        (relative_path, description)
        for relative_path, description in file_specs
        if (base_dir / relative_path).exists()
    ]


def _prediction_collapse_warning(metrics: dict[str, Any]) -> str | None:
    raw_matrix = metrics.get("confusion_matrix")
    if raw_matrix is None:
        return None
    matrix = np.asarray(raw_matrix)
    if matrix.ndim != 2 or matrix.size == 0:
        return None
    empty_columns = np.flatnonzero(matrix.sum(axis=0) == 0)
    if empty_columns.size == 0:
        return None
    labels = [str(item) for item in metrics.get("label_names", metrics.get("labels", []))]
    missing_labels = [
        labels[index] if index < len(labels) else str(index)
        for index in empty_columns.tolist()
    ]
    return "warning: classes never predicted -> " + ", ".join(missing_labels)


def _print_completion_summary(
    command_name: str,
    output_dir: Path,
    file_specs: tuple[tuple[str, str], ...],
    metrics: dict[str, Any] | None = None,
    next_command: str | None = None,
) -> None:
    resolved_output_dir = output_dir.resolve()
    print(f"[{command_name}] output_dir={resolved_output_dir}", flush=True)
    if metrics:
        metric_tokens: list[str] = []
        for key in ("model_name", "accuracy", "f1", "macro_f1"):
            if key not in metrics:
                continue
            value = metrics[key]
            if isinstance(value, float):
                metric_tokens.append(f"{key}={value:.4f}")
            else:
                metric_tokens.append(f"{key}={value}")
        if metric_tokens:
            print(f"[{command_name}] metrics: " + ", ".join(metric_tokens), flush=True)
        warning = _prediction_collapse_warning(metrics)
        if warning:
            print(f"[{command_name}] {warning}", flush=True)

    existing_specs = _existing_file_specs(output_dir, file_specs)
    if existing_specs:
        print(f"[{command_name}] generated:", flush=True)
        for relative_path, description in existing_specs:
            print(f"- {relative_path}: {description}", flush=True)
    if next_command:
        print(f"[{command_name}] next: {next_command}", flush=True)


def _prepare_quickstart_config(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    quickstart_config = deepcopy(config)
    prepared_work_dir = (output_root / "prepared_work").resolve()
    runs_history_dir = (output_root / "runs_history").resolve()
    latest_prepared_dir = (output_root / "latest_prepared").resolve()
    latest_run_dir = (output_root / "latest_run").resolve()

    for path in (prepared_work_dir, latest_prepared_dir, latest_run_dir):
        if path.exists():
            shutil.rmtree(path)

    quickstart_config.setdefault("runtime", {})
    quickstart_config["runtime"]["prepared_dir"] = str(prepared_work_dir)
    quickstart_config["runtime"]["artifacts_dir"] = str(runs_history_dir)

    quickstart_config.setdefault("visualization", {})
    quickstart_config["visualization"]["theme"] = str(
        quickstart_config["visualization"].get("theme", "paper-bar")
    )
    quickstart_config["visualization"]["profile"] = "concise"

    evaluation = quickstart_config.setdefault("evaluation", {})
    for section_name in ("cross_validation", "learning_curve", "ablation"):
        section = evaluation.get(section_name)
        if isinstance(section, dict):
            section["enabled"] = False

    return quickstart_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emc_diag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--dataset", required=True)
    download.add_argument("--out-dir", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True)

    extract_features = subparsers.add_parser("extract-features")
    extract_features.add_argument("--config", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-dir", required=True)

    visualize = subparsers.add_parser("visualize")
    visualize.add_argument("--run-dir", required=True)
    visualize.add_argument("--theme", default="paper-bar")

    export_report = subparsers.add_parser("export-report")
    export_report.add_argument("--run-dir", required=True)
    export_report.add_argument("--format", choices=["md", "csv", "png", "svg"], default="md")

    quickstart = subparsers.add_parser("quickstart")
    quickstart.add_argument("--config", default=QUICKSTART_DEFAULT_CONFIG)
    quickstart.add_argument("--device", default="auto")
    quickstart.add_argument("--output-root", default="artifacts")

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--configs", nargs="+", required=True)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--theme", default="paper-bar")

    thesis_assets = subparsers.add_parser("thesis-assets")
    thesis_assets.add_argument("--run-dirs", nargs="*", default=[])
    thesis_assets.add_argument("--benchmark-dirs", nargs="*", default=[])
    thesis_assets.add_argument("--output-dir", default="")
    thesis_assets.add_argument("--title", default="Final Summary")
    return parser


def _save_prepared_bundle(prepared_dir: Path, prepared: dict[str, Any]) -> None:
    ensure_dir(prepared_dir)
    splits = prepared["splits"]
    arrays: dict[str, Any] = {
        "train_X": splits["train"]["X"],
        "train_y": splits["train"]["y"],
        "val_X": splits["val"]["X"],
        "val_y": splits["val"]["y"],
        "test_X": splits["test"]["X"],
        "test_y": splits["test"]["y"],
    }
    for split_name, split_payload in splits.items():
        for task_name, task_y in split_payload.get("y_tasks", {}).items():
            arrays[f"{split_name}_y_task__{task_name}"] = np.asarray(task_y)
        for task_name, task_mask in split_payload.get("task_masks", {}).items():
            arrays[f"{split_name}_task_mask__{task_name}"] = np.asarray(task_mask)
    save_npz(
        prepared_dir / "prepared_splits.npz",
        **arrays,
    )
    save_json(prepared_dir / "metadata.json", prepared["metadata"])
    save_json(prepared_dir / "statistics.json", prepared["statistics"])
    save_json(prepared_dir / "scaler.json", prepared["scaler"])


def _load_prepared_bundle(prepared_dir: Path) -> dict[str, Any]:
    arrays = load_npz(prepared_dir / "prepared_splits.npz")
    metadata = load_json(prepared_dir / "metadata.json")
    task_names = [str(task.get("name") or task.get("task_name") or task.get("target_column")) for task in metadata.get("tasks", [])]
    splits: dict[str, dict[str, Any]] = {
        "train": {"X": arrays["train_X"], "y": arrays["train_y"]},
        "val": {"X": arrays["val_X"], "y": arrays["val_y"]},
        "test": {"X": arrays["test_X"], "y": arrays["test_y"]},
    }
    for split_name in splits:
        y_tasks: dict[str, np.ndarray] = {}
        task_masks: dict[str, np.ndarray] = {}
        for task_name in task_names:
            y_key = f"{split_name}_y_task__{task_name}"
            mask_key = f"{split_name}_task_mask__{task_name}"
            if y_key in arrays:
                y_tasks[task_name] = arrays[y_key]
            if mask_key in arrays:
                task_masks[task_name] = arrays[mask_key].astype(bool)
        if y_tasks:
            splits[split_name]["y_tasks"] = y_tasks
        if task_masks:
            splits[split_name]["task_masks"] = task_masks

    return {
        "splits": splits,
        "metadata": metadata,
        "statistics": load_json(prepared_dir / "statistics.json"),
        "scaler": load_json(prepared_dir / "scaler.json"),
    }


def _rename_prepared_tasks(prepared: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    task_entries = _task_entries(config)
    if len(task_entries) <= 1:
        return prepared

    rename_map = {
        str(task["target_column"]): str(task.get("name") or task.get("task_name") or task["target_column"])
        for task in task_entries
    }
    renamed_splits: dict[str, dict[str, Any]] = {}
    for split_name, split_payload in prepared["splits"].items():
        payload = dict(split_payload)
        if "y_tasks" in split_payload:
            payload["y_tasks"] = {
                rename_map.get(task_name, task_name): values
                for task_name, values in split_payload["y_tasks"].items()
            }
        if "task_masks" in split_payload:
            payload["task_masks"] = {
                rename_map.get(task_name, task_name): values
                for task_name, values in split_payload["task_masks"].items()
            }
        renamed_splits[split_name] = payload

    metadata = dict(prepared["metadata"])
    metadata["tasks"] = [
        {
            **task,
            "name": str(entry.get("name") or entry.get("task_name") or entry["target_column"]),
            "task_name": str(entry.get("task_name") or entry.get("name") or entry["target_column"]),
            "target_column": str(entry["target_column"]),
            "metric_primary": str(entry.get("metric_primary", "accuracy")),
            "metric_secondary": str(entry.get("metric_secondary", "f1")),
            "loss_weight": float(entry.get("loss_weight", 1.0)),
        }
        for task, entry in zip(prepared["metadata"].get("tasks", []), task_entries, strict=False)
    ]
    primary_task = metadata["tasks"][0]
    metadata["task_name"] = primary_task["task_name"]
    metadata["target_column"] = primary_task["target_column"]
    metadata["labels"] = primary_task.get("labels", metadata.get("labels", []))
    metadata["label_to_index"] = primary_task.get("label_to_index", metadata.get("label_to_index", {}))
    metadata["primary_task_name"] = primary_task["name"]
    return {
        "splits": renamed_splits,
        "metadata": metadata,
        "statistics": prepared.get("statistics", {}),
        "scaler": prepared.get("scaler", {}),
    }


def _prepare_from_config(config: dict[str, Any]) -> dict[str, Any]:
    task_entries = _task_entries(config)
    task_columns = _task_columns(config)
    primary_task = _primary_task_entry(config)
    prepared = prepare_dataset(
        source=config["dataset"]["input_path"],
        schema=config["dataset"]["schema"],
        target_column=primary_task["target_column"],
        task_name=primary_task["task_name"],
        drop_columns=_combined_drop_columns(config),
        train_ratio=config["trainer"]["train_ratio"],
        val_ratio=config["trainer"]["val_ratio"],
        random_state=config["trainer"]["random_state"],
        task_columns=task_columns if len(task_entries) > 1 else None,
    )
    prepared = _rename_prepared_tasks(prepared, config)
    prepared = _apply_feature_group_view_if_enabled(config, prepared)
    return prepared


def _task_entry_by_name(config: dict[str, Any], task_name: str) -> dict[str, Any]:
    for task in _task_entries(config):
        candidate = str(task.get("name") or task.get("task_name") or task.get("target_column"))
        if candidate == task_name:
            return dict(task)
    raise KeyError(f"Unknown task in benchmark matrix: {task_name}")


def _slugify_token(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    compact = "-".join(part for part in normalized.split("-") if part)
    return compact or "item"


def _expand_benchmark_variants(config_path: Path) -> list[tuple[str, dict[str, Any]]]:
    config = load_config(config_path)
    matrix = config.get("benchmark", {}).get("matrix", {})
    if not matrix:
        return [(str(config_path), config)]

    task_names = [str(item) for item in matrix.get("tasks", [])] or [
        str(task.get("name") or task.get("task_name") or task.get("target_column"))
        for task in _task_entries(config)
    ]
    model_names = [str(item) for item in matrix.get("models", [])] or [str(config["model"]["name"])]
    seeds = [int(item) for item in matrix.get("seeds", [])] or [int(config["trainer"]["random_state"])]
    modes = [str(item) for item in matrix.get("modes", [])] or [str(config["model"].get("mode", "single_task"))]
    transfer_strategies = [str(item) for item in matrix.get("transfer_strategies", [])] or [
        str(config["model"].get("transfer", {}).get("strategy") or "scratch")
    ]

    variants: list[tuple[str, dict[str, Any]]] = []
    for task_name in task_names:
        task_entry = _task_entry_by_name(config, task_name)
        for model_name in model_names:
            is_deep = _is_deep_model_name(model_name)
            valid_modes = [
                mode
                for mode in modes
                if mode == "single_task" or (is_deep and _model_supports_pretrain(model_name) and mode == "pretrain")
            ]
            if not valid_modes:
                valid_modes = ["single_task"]
            for seed in seeds:
                for mode in valid_modes:
                    valid_transfer_strategies = transfer_strategies
                    if mode == "pretrain" or not _model_supports_transfer(model_name):
                        valid_transfer_strategies = ["scratch"]
                    for transfer_strategy in valid_transfer_strategies:
                        variant = deepcopy(config)
                        prepared_root = Path(variant["runtime"]["prepared_dir"])
                        variant["tasks"] = [dict(task_entry)]
                        variant["task"] = {
                            **variant["task"],
                            "task_name": str(task_entry.get("task_name", task_name)),
                            "target_column": str(task_entry["target_column"]),
                            "metric_primary": str(task_entry.get("metric_primary", variant["task"].get("metric_primary", "accuracy"))),
                            "metric_secondary": str(task_entry.get("metric_secondary", variant["task"].get("metric_secondary", "f1"))),
                            "drop_leakage_columns": list(task_entry.get("drop_leakage_columns", variant["task"].get("drop_leakage_columns", []))),
                        }
                        variant["dataset"]["label_column"] = variant["task"]["target_column"]
                        variant["trainer"]["random_state"] = int(seed)
                        variant["model"]["name"] = model_name
                        variant["model"]["mode"] = mode
                        if mode == "pretrain":
                            variant["model"].setdefault("pretrain", {})["enabled"] = True
                        else:
                            variant["model"].setdefault("pretrain", {})["enabled"] = False
                        if not is_deep:
                            variant["model"]["candidates"] = [model_name]
                        transfer_config = dict(variant["model"].get("transfer") or {})
                        transfer_enabled = (
                            mode != "pretrain"
                            and _model_supports_transfer(model_name)
                            and transfer_strategy not in {"", "scratch", "none", "off"}
                        )
                        transfer_config["enabled"] = transfer_enabled
                        transfer_config["strategy"] = transfer_strategy
                        if not transfer_enabled:
                            transfer_config["from_run_dir"] = None
                        variant["model"]["transfer"] = transfer_config
                        variant["model"]["transfer_from"] = transfer_config.get("from_run_dir")
                        variant["runtime"]["prepared_dir"] = str(
                            prepared_root / f"{_slugify_token(task_name)}-seed-{seed}"
                        )
                        variant_label = (
                            f"{config_path}:{task_name}:{model_name}:seed={seed}:mode={mode}:transfer={transfer_strategy}"
                        )
                        variants.append((variant_label, variant))
    return variants


def _run_pipeline_config(config: dict[str, Any], requested_device: str, theme_name: str = "paper-bar") -> Path:
    prepared_dir = Path(config["runtime"]["prepared_dir"])
    if (prepared_dir / "prepared_splits.npz").exists():
        prepared = _load_prepared_bundle(prepared_dir)
    else:
        prepared = _prepare_from_config(config)
        _save_prepared_bundle(prepared_dir, prepared)
        _export_prepared_exploration_assets(prepared_dir, prepared)

    if (prepared_dir / "feature_splits.npz").exists() and (prepared_dir / "feature_metadata.json").exists():
        feature_bundle = _load_feature_bundle(prepared_dir)
    else:
        feature_bundle = extract_feature_bundle(
            prepared,
            method=config["features"]["method"],
            top_k=config["features"].get("top_k"),
        )
        _save_feature_bundle(prepared_dir, feature_bundle)
        _export_feature_analysis_assets(prepared_dir, prepared, feature_bundle)
    run_dir = _train_from_config(config, requested_device=requested_device)
    _visualize_run(run_dir, theme_name=theme_name)
    return run_dir


def _save_feature_bundle(prepared_dir: Path, bundle: dict[str, Any]) -> None:
    splits = bundle["splits"]
    save_npz(
        prepared_dir / "feature_splits.npz",
        train_X=splits["train"]["X"],
        train_y=splits["train"]["y"],
        val_X=splits["val"]["X"],
        val_y=splits["val"]["y"],
        test_X=splits["test"]["X"],
        test_y=splits["test"]["y"],
    )
    feature_table = pd.DataFrame(
        {
            "feature": bundle["feature_names"],
            "importance": bundle["feature_scores"],
        }
    ).sort_values("importance", ascending=False)
    save_dataframe(prepared_dir / "feature_importance.csv", feature_table)
    save_json(
        prepared_dir / "feature_metadata.json",
        {
            "method": bundle["method"],
            "selected_indices": np.asarray(bundle["selected_indices"]).tolist(),
            "selected_feature_names": bundle["selected_feature_names"],
            "summary": bundle["summary"],
        },
    )


def _load_feature_bundle(prepared_dir: Path) -> dict[str, Any]:
    arrays = load_npz(prepared_dir / "feature_splits.npz")
    metadata = load_json(prepared_dir / "feature_metadata.json")
    return {
        "splits": {
            "train": {"X": arrays["train_X"], "y": arrays["train_y"]},
            "val": {"X": arrays["val_X"], "y": arrays["val_y"]},
            "test": {"X": arrays["test_X"], "y": arrays["test_y"]},
        },
        "metadata": metadata,
        "feature_importance": pd.read_csv(prepared_dir / "feature_importance.csv"),
    }


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))


def _call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return func(**accepted)


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message or "cudnn" in message)


def _call_with_batch_backoff(
    func: Callable[..., Any],
    config: dict[str, Any],
    **kwargs: Any,
) -> Any:
    signature = inspect.signature(func)
    if "batch_size" not in signature.parameters or "batch_size" not in kwargs:
        return _call_with_supported_kwargs(func, **kwargs)

    current_batch = int(kwargs["batch_size"])
    min_batch_size = int(config.get("trainer", {}).get("min_batch_size", 8))
    retries_remaining = int(config.get("trainer", {}).get("oom_retries", 3))
    backoff_factor = float(config.get("trainer", {}).get("oom_backoff_factor", 0.5))

    while True:
        try:
            return _call_with_supported_kwargs(func, **kwargs)
        except RuntimeError as exc:
            if not _is_cuda_oom_error(exc):
                raise
            next_batch = max(min_batch_size, int(current_batch * backoff_factor))
            if retries_remaining <= 0 or next_batch >= current_batch:
                raise
            print(
                f"[train] cuda_oom batch_size={current_batch} -> retry batch_size={next_batch}",
                flush=True,
            )
            current_batch = next_batch
            kwargs["batch_size"] = current_batch
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            retries_remaining -= 1


def _trainer_performance_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    trainer = config.get("trainer", {})
    return {
        "loader_workers": trainer.get("loader_workers"),
        "pin_memory": trainer.get("pin_memory"),
        "persistent_workers": trainer.get("persistent_workers"),
        "prefetch_factor": trainer.get("prefetch_factor"),
        "amp": trainer.get("amp", "auto"),
    }


def _task_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = list(config.get("tasks") or [])
    if tasks:
        return tasks
    task = config["task"]
    return [
        {
            "name": task.get("task_name", task.get("target_column", "task")),
            "task_name": task.get("task_name", task.get("target_column", "task")),
            "target_column": task.get("target_column", "label"),
            "metric_primary": task.get("metric_primary", "accuracy"),
            "metric_secondary": task.get("metric_secondary", "f1"),
            "drop_leakage_columns": list(task.get("drop_leakage_columns", [])),
            "loss_weight": 1.0,
        }
    ]


def _task_columns(config: dict[str, Any]) -> list[str]:
    return [str(task["target_column"]) for task in _task_entries(config)]


def _task_name_to_column(config: dict[str, Any]) -> dict[str, str]:
    return {
        str(task.get("name") or task.get("task_name") or task.get("target_column")): str(task["target_column"])
        for task in _task_entries(config)
    }


def _primary_task_entry(config: dict[str, Any]) -> dict[str, Any]:
    return _task_entries(config)[0]


def _combined_drop_columns(config: dict[str, Any]) -> list[str]:
    combined: list[str] = []
    seen: set[str] = set()
    for task in _task_entries(config):
        for column in task.get("drop_leakage_columns", []):
            name = str(column)
            if name not in seen:
                seen.add(name)
                combined.append(name)
    for column in config["task"].get("drop_leakage_columns", []):
        name = str(column)
        if name not in seen:
            seen.add(name)
            combined.append(name)
    return combined


def _apply_feature_group_view_if_enabled(config: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    enabled_groups = list(config.get("features", {}).get("domain_feature_groups", {}).get("enabled", []) or [])
    if not enabled_groups:
        return prepared
    return make_feature_group_view(prepared, enabled_groups)


def _load_transfer_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    transfer_config = config.get("model", {}).get("transfer", {}) or {}
    from_run_dir = transfer_config.get("from_run_dir")
    if not from_run_dir:
        return {}
    if "REPLACE_WITH" in str(from_run_dir):
        raise ValueError(
            "transfer.from_run_dir is still a placeholder; replace it with a real run directory before training."
        )

    run_dir = Path(str(from_run_dir))
    if not run_dir.exists():
        raise FileNotFoundError(f"transfer.from_run_dir does not exist: {run_dir}")

    encoder_metadata: dict[str, Any] | None = None
    metadata_path = run_dir / "encoder_metadata.json"
    if metadata_path.exists():
        encoder_metadata = load_json(metadata_path)
    elif (run_dir / "metrics.json").exists():
        previous_metrics = load_json(run_dir / "metrics.json")
        raw_metadata = previous_metrics.get("encoder_metadata")
        if isinstance(raw_metadata, dict):
            encoder_metadata = raw_metadata

    checkpoint_candidates = [
        run_dir / "pretrain_encoder.pt",
        run_dir / "best_checkpoint.pt",
    ]
    checkpoint_path = next((candidate for candidate in checkpoint_candidates if candidate.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No encoder checkpoint found under {run_dir}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for transfer loading") from exc

    encoder_state_dict = torch.load(checkpoint_path, map_location="cpu")
    return {
        "encoder_state_dict": encoder_state_dict,
        "encoder_metadata": encoder_metadata,
        "transfer_strategy": transfer_config.get("strategy"),
        "transfer_from_run_dir": str(run_dir),
    }


def _decode_labels(values: np.ndarray, label_names: list[str]) -> list[str]:
    decoded: list[str] = []
    for value in np.asarray(values).astype(np.int64).tolist():
        decoded.append(label_names[value] if 0 <= value < len(label_names) else str(value))
    return decoded


def _predict_with_model(model: Any, x: np.ndarray, device: str) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(x), dtype=np.int64)

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for neural network prediction") from exc

    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if tensor_x.ndim == 2:
            tensor_x = tensor_x.unsqueeze(1)
        tensor_x = tensor_x.to(device)
        logits = model(tensor_x)
        return torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)


def _predict_scores_with_model(model: Any, x: np.ndarray, device: str) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
        if probabilities.ndim == 1:
            return probabilities
        return None

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=float)
        if decision.ndim == 2 and decision.shape[1] >= 2:
            return decision[:, 1]
        if decision.ndim == 1:
            minimum = float(decision.min())
            maximum = float(decision.max())
            scale = maximum - minimum
            if scale <= 1e-12:
                return np.full(decision.shape, 0.5, dtype=float)
            return (decision - minimum) / scale
        return None

    try:
        import torch
    except Exception:
        return None

    if hasattr(model, "eval"):
        model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(np.asarray(x, dtype=np.float32))
            if tensor_x.ndim == 2:
                tensor_x = tensor_x.unsqueeze(1)
            tensor_x = tensor_x.to(device)
            logits = model(tensor_x)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    return None


def _serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, tuple):
            return [_convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return _convert(dict(metrics))


def _metrics_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    metric_names = ["accuracy", "precision", "recall", "f1", "macro_f1"]
    row = {"model": metrics.get("model_name", "model")}
    for name in metric_names:
        if name in metrics:
            row[name] = metrics[name]
    return pd.DataFrame([row])


def _candidate_scores_frame(candidate_scores: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, payload in candidate_scores.items():
        row = {"model": name}
        if isinstance(payload, dict):
            row.update(payload)
        else:
            row["accuracy"] = float(payload)
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_importance_frame(feature_bundle: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": feature_bundle["feature_names"],
            "importance": feature_bundle["feature_scores"],
        }
    ).sort_values("importance", ascending=False)


def _feature_metadata_payload(feature_bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": feature_bundle["method"],
        "selected_indices": np.asarray(feature_bundle["selected_indices"]).tolist(),
        "selected_feature_names": list(feature_bundle["selected_feature_names"]),
        "summary": feature_bundle["summary"],
    }


def _is_deep_model_name(model_name: str) -> bool:
    return str(model_name) in DEEP_MODEL_NAMES


def _model_family_for_name(model_name: str) -> str:
    return "dl" if _is_deep_model_name(model_name) else "ml"


def _model_supports_transfer(model_name: str) -> bool:
    return str(model_name) == "cnn_1d"


def _model_supports_pretrain(model_name: str) -> bool:
    return str(model_name) == "cnn_1d"


def _history_curve_frame(metrics: dict[str, Any], run_config: dict[str, Any]) -> pd.DataFrame:
    history = metrics.get("train_history", {})
    if history:
        epoch_count = len(history.get("train_loss", history.get("loss", [])))
        epochs = list(range(1, epoch_count + 1))
        return pd.DataFrame(
            {
                "epoch": epochs,
                "train_loss": history.get("train_loss", history.get("loss", [0.0] * epoch_count)),
                "val_loss": history.get("val_loss", history.get("loss", [0.0] * epoch_count)),
                "accuracy": history.get("val_accuracy", [0.0] * epoch_count),
                "f1": history.get("val_f1", history.get("val_accuracy", [0.0] * epoch_count)),
                "train_score": history.get(
                    "train_accuracy",
                    [metrics.get("train_accuracy_score", 0.0)] * epoch_count,
                ),
                "val_score": history.get(
                    "val_accuracy",
                    [metrics.get("val_accuracy_score", metrics.get("accuracy", 0.0))] * epoch_count,
                ),
                "learning_rate": history.get("learning_rate", [0.0] * epoch_count),
            }
        )

    return pd.DataFrame(
        {
            "epoch": [1],
            "train_loss": [1.0 - metrics.get("train_accuracy_score", metrics.get("accuracy", 0.0))],
            "val_loss": [1.0 - metrics.get("val_f1_score", metrics.get("f1", 0.0))],
            "accuracy": [metrics.get("accuracy", 0.0)],
            "f1": [metrics.get("f1", 0.0)],
            "train_score": [metrics.get("train_accuracy_score", metrics.get("accuracy", 0.0))],
            "val_score": [metrics.get("val_accuracy_score", metrics.get("accuracy", 0.0))],
            "learning_rate": [run_config["model"].get("learning_rate", 0.0)],
        }
    )


def _cv_summary_frame(cv_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if cv_df.empty:
        return pd.DataFrame(columns=["model", "cv_mean", "cv_std"])
    if {"model", "cv_mean", "cv_std"}.issubset(cv_df.columns):
        return cv_df[["model", "cv_mean", "cv_std"]].copy()

    metric_column = "accuracy" if "accuracy" in cv_df.columns else "f1"
    numeric = cv_df[pd.to_numeric(cv_df["fold"], errors="coerce").notna()].copy() if "fold" in cv_df.columns else cv_df.copy()
    if numeric.empty:
        numeric = cv_df.copy()
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "cv_mean": float(pd.to_numeric(numeric[metric_column], errors="coerce").mean()),
                "cv_std": float(pd.to_numeric(numeric[metric_column], errors="coerce").std(ddof=0)),
            }
        ]
    )


def _ablation_summary_frame(ablation_df: pd.DataFrame, metric_name: str = "accuracy") -> pd.DataFrame:
    if ablation_df.empty:
        return pd.DataFrame(columns=["variant", "score"])
    if {"variant", "score"}.issubset(ablation_df.columns):
        return ablation_df[["variant", "score"]].copy()
    variant_col = "layout" if "layout" in ablation_df.columns else ablation_df.columns[0]
    score_col = metric_name if metric_name in ablation_df.columns else ("accuracy" if "accuracy" in ablation_df.columns else "f1")
    return pd.DataFrame(
        {
            "variant": ablation_df[variant_col].astype(str),
            "score": pd.to_numeric(ablation_df[score_col], errors="coerce").fillna(0.0),
        }
    )


def _ml_vs_cnn_summary_frame(metrics_rows: pd.DataFrame) -> pd.DataFrame:
    if metrics_rows.empty:
        return pd.DataFrame(columns=["family", "accuracy", "f1"])
    family_col = "family" if "family" in metrics_rows.columns else "model_family"
    summary = (
        metrics_rows.groupby(family_col, dropna=False)[["accuracy", "f1"]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={family_col: "family"})
    )
    return summary


def _load_vsb_waveform_preview(run_config: dict[str, Any], max_points: int = 1024) -> pd.DataFrame:
    source = str(run_config["dataset"]["input_path"])
    source_path = Path(source.split("::", 1)[0])
    if not source_path.is_dir():
        raise ValueError("VSB preview requires a directory source containing train.parquet")
    parquet_path = source_path / "train.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing VSB parquet file: {parquet_path}")

    metadata_path = source_path / "metadata_train.csv"
    preview_column = None
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path, nrows=1)
        if "signal_id" in metadata.columns and not metadata.empty:
            preview_column = str(int(metadata.iloc[0]["signal_id"]))
    if preview_column is None:
        import pyarrow.parquet as pq

        preview_column = pq.ParquetFile(parquet_path).schema.names[0]

    waveform = pd.read_parquet(parquet_path, columns=[preview_column]).iloc[:max_points, 0].to_numpy(dtype=float)
    return pd.DataFrame({"sample_index": np.arange(len(waveform)), "signal": waveform})


def _basic_statistics(x: np.ndarray, y: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    labels, counts = np.unique(y, return_counts=True)
    class_counts: dict[str, int] = {}
    for label, count in zip(labels, counts, strict=False):
        name = label_names[int(label)] if int(label) < len(label_names) else str(int(label))
        class_counts[name] = int(count)
    missing_count = int(np.isnan(x).sum())
    total_count = int(np.prod(x.shape))
    return {
        "class_counts": class_counts,
        "missing_values": missing_count,
        "missing_ratio": float(missing_count / total_count) if total_count else 0.0,
        "feature_mean_head": np.mean(x, axis=0)[: min(5, x.shape[1])].tolist(),
        "feature_std_head": np.std(x, axis=0)[: min(5, x.shape[1])].tolist(),
    }


def _prepared_label_frame(prepared: dict[str, Any]) -> pd.DataFrame:
    labels = list(prepared.get("metadata", {}).get("labels", []))
    rows: list[dict[str, str]] = []
    for split_name, payload in prepared.get("splits", {}).items():
        values = np.asarray(payload.get("y", []), dtype=np.int64)
        for value in values.tolist():
            label = labels[value] if 0 <= value < len(labels) else str(value)
            rows.append({"split": str(split_name), "class": str(label)})
    return pd.DataFrame(rows)


def _prepared_statistics_frame(prepared: dict[str, Any]) -> pd.DataFrame:
    metadata = prepared.get("metadata", {})
    statistics = prepared.get("statistics", {})
    rows: list[dict[str, Any]] = [
        {"metric": "num_samples", "value": int(metadata.get("num_samples", 0))},
        {"metric": "num_features", "value": int(metadata.get("num_features", 0))},
        {"metric": "missing_values", "value": int(statistics.get("missing_values", 0))},
        {"metric": "missing_ratio", "value": float(statistics.get("missing_ratio", 0.0))},
    ]
    class_counts = statistics.get("class_counts", {})
    if isinstance(class_counts, dict):
        for label, count in sorted(class_counts.items()):
            rows.append({"metric": f"class_count[{label}]", "value": int(count)})
    return pd.DataFrame(rows)


def _export_prepared_exploration_assets(prepared_dir: Path, prepared: dict[str, Any]) -> None:
    from emc_diag.reporting import build_prepared_summary
    from emc_diag.visualization import plot_dataset_summary

    figures_dir = ensure_dir(prepared_dir / "figures")
    tables_dir = ensure_dir(prepared_dir / "tables")

    labels_df = _prepared_label_frame(prepared)
    if not labels_df.empty:
        plot_dataset_summary(labels_df[["class"]].copy(), figures_dir)
        split_counts = (
            labels_df.groupby(["split", "class"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["split", "class"])
        )
        save_dataframe(tables_dir / "class_distribution.csv", split_counts)

    statistics_df = _prepared_statistics_frame(prepared)
    save_dataframe(tables_dir / "dataset_statistics.csv", statistics_df)

    dataset_metadata = {
        "dataset": str(prepared.get("metadata", {}).get("task_name", "dataset")),
        "target_column": str(prepared.get("metadata", {}).get("target_column", "label")),
        "num_samples": str(prepared.get("metadata", {}).get("num_samples", 0)),
        "num_features": str(prepared.get("metadata", {}).get("num_features", 0)),
    }
    build_prepared_summary(
        prepared_dir=prepared_dir,
        dataset_metadata=dataset_metadata,
        statistics={
            **prepared.get("statistics", {}),
            "num_samples": int(prepared.get("metadata", {}).get("num_samples", 0)),
            "num_features": int(prepared.get("metadata", {}).get("num_features", 0)),
        },
        figures=sorted(path.name for path in figures_dir.glob("*.png")),
        tables=sorted(path.name for path in tables_dir.glob("*.csv")),
    )


def _export_feature_analysis_assets(prepared_dir: Path, prepared: dict[str, Any], feature_bundle: dict[str, Any]) -> None:
    from emc_diag.reporting import build_prepared_summary
    from emc_diag.visualization import plot_feature_importance

    figures_dir = ensure_dir(prepared_dir / "figures")
    tables_dir = ensure_dir(prepared_dir / "tables")

    feature_df = pd.DataFrame(
        {
            "feature": feature_bundle["feature_names"],
            "importance": feature_bundle["feature_scores"],
        }
    ).sort_values("importance", ascending=False)
    plot_feature_importance(feature_df, figures_dir)

    selected_feature_names = list(feature_bundle.get("selected_feature_names", []))
    selected_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected_feature_names) + 1),
            "feature": selected_feature_names,
        }
    )
    save_dataframe(tables_dir / "selected_features.csv", selected_df)

    feature_summary = {
        "feature_method": str(feature_bundle.get("method", "unknown")),
        "selected_feature_count": str(len(selected_feature_names)),
        "top_feature_names": ", ".join(selected_feature_names[:5]) or "none",
    }
    build_prepared_summary(
        prepared_dir=prepared_dir,
        dataset_metadata={
            "dataset": str(prepared.get("metadata", {}).get("task_name", "dataset")),
            "target_column": str(prepared.get("metadata", {}).get("target_column", "label")),
            "num_samples": str(prepared.get("metadata", {}).get("num_samples", 0)),
            "num_features": str(prepared.get("metadata", {}).get("num_features", 0)),
        },
        statistics={
            **prepared.get("statistics", {}),
            "num_samples": int(prepared.get("metadata", {}).get("num_samples", 0)),
            "num_features": int(prepared.get("metadata", {}).get("num_features", 0)),
        },
        figures=sorted(path.name for path in figures_dir.glob("*.png")),
        tables=sorted(path.name for path in tables_dir.glob("*.csv")),
        feature_summary=feature_summary,
    )


def _tune_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray | None,
    threshold_tuning: dict[str, Any] | None,
    primary_metric: str,
    secondary_metric: str,
) -> tuple[float | None, np.ndarray | None, dict[str, Any] | None]:
    if y_score is None:
        return None, None, None
    tuning = threshold_tuning or {}
    if not tuning.get("enabled", False):
        return None, None, None
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) != 2:
        return None, None, None

    best_threshold: float | None = None
    best_pred: np.ndarray | None = None
    best_metrics: dict[str, Any] | None = None
    best_tuple: tuple[float, float, float] | None = None
    tuning_metric = str(tuning.get("metric", primary_metric))
    grid = [float(item) for item in tuning.get("grid", [0.5])]
    for threshold in grid:
        pred = (y_score >= threshold).astype(np.int64)
        metrics = evaluate_predictions(y_true, pred, y_score=y_score)
        score_tuple = (
            float(metrics.get(tuning_metric, 0.0)),
            float(metrics.get(secondary_metric, 0.0)),
            float(metrics.get("accuracy", 0.0)),
        )
        if best_tuple is None or score_tuple > best_tuple:
            best_tuple = score_tuple
            best_threshold = threshold
            best_pred = pred
            best_metrics = metrics
    return best_threshold, best_pred, best_metrics


def _predict_labels(
    model: Any,
    x: np.ndarray,
    device: str,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    scores = _predict_scores_with_model(model, x, device)
    if threshold is not None and scores is not None:
        return (np.asarray(scores) >= float(threshold)).astype(np.int64), scores
    return _predict_with_model(model, x, device), scores


def _standardize_splits_local(splits: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    x_train = np.asarray(splits["train"]["X"], dtype=float)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)

    normalized: dict[str, dict[str, np.ndarray]] = {}
    for split_name, payload in splits.items():
        normalized[split_name] = {
            "X": (np.asarray(payload["X"], dtype=float) - mean) / std_safe,
            "y": np.asarray(payload["y"]),
        }
    return normalized, {"mean": mean.tolist(), "std": std_safe.tolist()}


def _prepared_from_loaded(
    loaded: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    schema: str,
    target_column: str,
    task_name: str,
    leakage_columns_removed: list[str],
    random_state: int,
) -> dict[str, Any]:
    raw_splits = {
        "train": {"X": loaded["X"][train_idx], "y": loaded["y"][train_idx]},
        "val": {"X": loaded["X"][val_idx], "y": loaded["y"][val_idx]},
        "test": {"X": loaded["X"][test_idx], "y": loaded["y"][test_idx]},
    }
    splits, scaler = _standardize_splits_local(raw_splits)
    label_names = list(loaded.get("label_names", []))
    metadata = {
        "schema": schema,
        "num_samples": int(loaded["X"].shape[0]),
        "num_features": int(loaded["X"].shape[1]),
        "feature_names": list(loaded["feature_names"]),
        "labels": label_names,
        "target_column": target_column,
        "task_name": task_name,
        "random_state": random_state,
        "label_to_index": loaded.get("label_to_index", {}),
        "leakage_columns_removed": leakage_columns_removed,
        "dropped_missing_targets": loaded.get("dropped_missing_targets", 0),
        "sample_id_count": len(loaded.get("sample_ids", [])),
    }
    return {
        "splits": splits,
        "scaler": scaler,
        "metadata": metadata,
        "statistics": _basic_statistics(np.asarray(loaded["X"], dtype=float), np.asarray(loaded["y"]), label_names),
    }


def _save_auxiliary_tables(run_dir: Path, analysis_outputs: dict[str, pd.DataFrame]) -> None:
    for name, frame in analysis_outputs.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        save_dataframe(run_dir / "tables" / f"{name}.csv", frame)


def _torch_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "best_checkpoint.pt"


def _build_training_inputs(config: dict[str, Any], prepared: dict[str, Any], layout_override: str | None = None) -> dict[str, Any]:
    is_deep_model = _is_deep_model_name(config["model"]["name"])
    feature_view = make_prepared_layout_view(prepared, layout_override) if layout_override and not is_deep_model else prepared
    feature_bundle = extract_feature_bundle(
        feature_view,
        method=config["features"]["method"],
        top_k=config["features"].get("top_k"),
    )

    if config["model"]["name"] in {"cognitive_radio_hybrid", "cognitive_radio_scalar_hybrid"}:
        hybrid_bundle = build_cognitive_radio_hybrid_bundle(prepared)
        training_metadata = {
            "sequence_layout": "structured_hybrid",
            "summary": hybrid_bundle["summary"],
        }
        return {
            "feature_bundle": feature_bundle,
            "training_splits": hybrid_bundle["splits"],
            "training_metadata": training_metadata,
        }

    if is_deep_model:
        sequence_layout = layout_override or config["model"].get("sequence_layout", "all")
        if sequence_layout == "feature_selected":
            training_splits = {
                split_name: {
                    "X": np.asarray(split_payload["X"], dtype=np.float32)[:, np.newaxis, :],
                    "y": split_payload["y"],
                }
                for split_name, split_payload in feature_bundle["splits"].items()
            }
            training_metadata = {
                "sequence_layout": sequence_layout,
                "channel_names": ["selected_features"],
                "channel_lengths": [len(feature_bundle["selected_feature_names"])],
                "summary": {
                    "layout": sequence_layout,
                    "channel_count": 1,
                    "max_sequence_length": len(feature_bundle["selected_feature_names"]),
                    "selected_feature_names": feature_bundle["selected_feature_names"],
                },
            }
        else:
            sequence_bundle = build_sequence_bundle(prepared, layout=sequence_layout)
            training_splits = sequence_bundle["splits"]
            training_metadata = {
                "sequence_layout": sequence_layout,
                "channel_names": sequence_bundle["channel_names"],
                "channel_lengths": sequence_bundle["channel_lengths"],
                "summary": sequence_bundle["summary"],
            }
    else:
        training_splits = feature_bundle["splits"]
        training_metadata = {"sequence_layout": None, "summary": feature_bundle["summary"]}

    return {
        "feature_bundle": feature_bundle,
        "training_splits": training_splits,
        "training_metadata": training_metadata,
    }


def _risk_note_from_metrics(metrics: dict[str, Any], config: dict[str, Any], analysis_outputs: dict[str, pd.DataFrame]) -> str:
    notes: list[str] = []
    history = metrics.get("train_history", {})
    if history:
        train_acc = history.get("train_accuracy", [])
        val_acc = history.get("val_accuracy", [])
        if train_acc and val_acc:
            gap = float(train_acc[-1] - val_acc[-1])
            if gap > 0.1:
                notes.append(f"possible_overfit_gap={gap:.3f}")
            else:
                notes.append(f"train_val_gap_stable={gap:.3f}")
    if metrics.get("accuracy", 0.0) >= 0.95 and config.get("evaluation", {}).get("risk_checks", {}).get("enabled", False):
        ablation = analysis_outputs.get("ablation_metrics")
        if isinstance(ablation, pd.DataFrame) and not ablation.empty:
            if float(ablation["accuracy"].min()) >= 0.95:
                notes.append("high_score_persists_across_ablations")
            else:
                notes.append("high_score_sensitive_to_feature_subset")
        else:
            notes.append("high_score_requires_additional_risk_check")
    return "; ".join(notes) if notes else "no_major_overfit_signal_detected"


def _task_metadata_map(prepared: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for task in prepared["metadata"].get("tasks", []):
        task_name = str(task.get("name") or task.get("task_name") or task.get("target_column"))
        mapping[task_name] = dict(task)
    return mapping


def _multitask_predictions(
    model: Any,
    x: np.ndarray,
    task_names: list[str],
    device: str,
) -> dict[str, np.ndarray]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for multitask prediction") from exc

    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if tensor_x.ndim == 2:
            tensor_x = tensor_x.unsqueeze(1)
        tensor_x = tensor_x.to(device)
        logits_map = model(tensor_x)
        return {
            task_name: torch.argmax(logits_map[task_name], dim=1).cpu().numpy().astype(np.int64)
            for task_name in task_names
        }


def _summary_context(
    run_dir: Path,
    metrics: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    transfer_notes: list[str] = []
    if metrics.get("transfer_from_run_dir"):
        transfer_notes.append(
            f"- transfer_from: {metrics['transfer_from_run_dir']} ({metrics.get('transfer_strategy', 'finetune')})"
        )

    pretrain_notes: list[str] = []
    if metrics.get("run_mode") == "pretrain" or run_config.get("model", {}).get("pretrain", {}).get("enabled"):
        pretrain_notes.append(
            f"- pretrain_objective: {metrics.get('pretrain_objective', run_config.get('model', {}).get('pretrain', {}).get('objective', 'masked_reconstruction'))}"
        )

    enabled_feature_groups = list(run_config.get("features", {}).get("domain_feature_groups", {}).get("enabled", []) or [])
    feature_group_summary = ", ".join(enabled_feature_groups) if enabled_feature_groups else None

    robustness_notes: list[str] = []
    noise_path = run_dir / "tables" / "robustness_noise.csv"
    if noise_path.exists():
        robustness_df = pd.read_csv(noise_path)
        if not robustness_df.empty:
            metric_column = next((column for column in robustness_df.columns if column != "noise_sigma"), None)
            if metric_column:
                robustness_notes.append(
                    f"- noise sweep: {metric_column} {float(robustness_df.iloc[0][metric_column]):.4f} -> {float(robustness_df.iloc[-1][metric_column]):.4f}"
                )
    ratio_path = run_dir / "tables" / "robustness_train_ratio.csv"
    if ratio_path.exists():
        ratio_df = pd.read_csv(ratio_path)
        if not ratio_df.empty:
            metric_column = next((column for column in ratio_df.columns if column != "train_ratio"), None)
            if metric_column:
                robustness_notes.append(
                    f"- train-ratio sweep: best {metric_column}={float(pd.to_numeric(ratio_df[metric_column], errors='coerce').max()):.4f}"
                )
    if "minority_f1" in metrics:
        robustness_notes.append(
            (
                f"- minority-class: label={metrics.get('minority_label_name', metrics.get('minority_label'))}, "
                f"precision={float(metrics.get('minority_precision', 0.0)):.4f}, "
                f"recall={float(metrics.get('minority_recall', 0.0)):.4f}, "
                f"f1={float(metrics.get('minority_f1', 0.0)):.4f}, "
                f"support={int(metrics.get('minority_support', 0))}"
            )
        )

    run_metadata = {
        "run_mode": str(metrics.get("run_mode", run_config.get("model", {}).get("mode", "single_task"))),
        "model_name": str(metrics.get("model_name", run_config.get("model", {}).get("name", "model"))),
    }
    if metrics.get("mean_task_f1") is not None:
        run_metadata["mean_task_f1"] = f"{float(metrics['mean_task_f1']):.4f}"
    return {
        "run_metadata": run_metadata,
        "transfer_notes": transfer_notes or None,
        "pretrain_notes": pretrain_notes or None,
        "feature_group_summary": feature_group_summary,
        "robustness_notes": robustness_notes or None,
    }


def _execute_training(
    config: dict[str, Any],
    prepared: dict[str, Any],
    requested_device: str,
    layout_override: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    modeling = _load_modeling_module()
    run_mode = str(config["model"].get("mode", "single_task"))
    prepared_effective = _apply_feature_group_view_if_enabled(config, prepared)
    label_names = list(prepared_effective["metadata"].get("labels", []))
    device = resolve_device(requested_device)
    prepared_inputs = _build_training_inputs(config, prepared_effective, layout_override=layout_override)
    training_splits = prepared_inputs["training_splits"]
    feature_bundle = prepared_inputs["feature_bundle"]
    training_metadata = prepared_inputs["training_metadata"]
    transfer_payload = _load_transfer_artifacts(config) if _is_deep_model_name(config["model"]["name"]) and run_mode != "pretrain" else {}
    trainer_runtime_kwargs = _trainer_performance_kwargs(config)

    if run_mode == "pretrain":
        start = time.perf_counter()
        pretrain_config = config["model"].get("pretrain", {})
        result = _call_with_batch_backoff(
            modeling.pretrain_cnn_encoder,
            config,
            x_train=training_splits["train"]["X"],
            x_val=training_splits["val"]["X"],
            requested_device=device,
            epochs=config["model"]["epochs"],
            batch_size=config["model"]["batch_size"],
            learning_rate=config["model"]["learning_rate"],
            random_seed=config["trainer"]["random_state"],
            dropout=config["model"].get("dropout"),
            weight_decay=config["model"].get("weight_decay"),
            patience=config["model"].get("patience"),
            mask_ratio=pretrain_config.get("mask_ratio"),
            objective=pretrain_config.get("objective"),
            **trainer_runtime_kwargs,
            progress=progress,
        )
        train_seconds = time.perf_counter() - start
        metrics = {
            "run_mode": "pretrain",
            "model_name": result["model_name"],
            "resolved_device": result["resolved_device"],
            "train_seconds": train_seconds,
            "task_name": config["task"]["task_name"],
            "target_column": config["task"]["target_column"],
            "sequence_layout": training_metadata.get("sequence_layout"),
            "model_family": "dl",
            "epochs_ran": result.get("epochs_ran", 0),
            "best_checkpoint": result.get("best_checkpoint", {}),
            "pretrain_history": result.get("train_history", {}),
            "train_history": result.get("train_history", {}),
            "encoder_metadata": result.get("encoder_metadata", {}),
            "pretrain_objective": pretrain_config.get("objective", "masked_reconstruction"),
        }
        return {
            "metrics": metrics,
            "predictions": None,
            "prediction_frames": {},
            "feature_bundle": feature_bundle,
            "dataset_metadata": prepared_effective["metadata"],
            "result": result,
            "training_metadata": training_metadata,
            "tables": {},
            "encoder_checkpoint_name": str(pretrain_config.get("checkpoint_name", "pretrain_encoder.pt")),
        }

    if run_mode == "multitask":
        if config["model"]["name"] == "cognitive_radio_hybrid":
            task_weights = {
                str(task.get("name") or task.get("task_name") or task.get("target_column")): float(task.get("loss_weight", 1.0))
                for task in _task_entries(config)
            }
            start = time.perf_counter()
            result = _call_with_batch_backoff(
                modeling.train_multitask_cognitive_radio_hybrid_model,
                config,
                scalar_train=training_splits["train"]["scalar_X"],
                cov_train=training_splits["train"]["cov_X"],
                temporal_train=training_splits["train"]["temporal_X"],
                y_train_tasks=training_splits["train"]["y_tasks"],
                scalar_val=training_splits["val"]["scalar_X"],
                cov_val=training_splits["val"]["cov_X"],
                temporal_val=training_splits["val"]["temporal_X"],
                y_val_tasks=training_splits["val"]["y_tasks"],
                requested_device=device,
                epochs=config["model"]["epochs"],
                batch_size=config["model"]["batch_size"],
                learning_rate=config["model"]["learning_rate"],
                random_seed=config["trainer"]["random_state"],
                dropout=config["model"].get("dropout"),
                weight_decay=config["model"].get("weight_decay"),
                patience=config["model"].get("patience"),
                scheduler=config["model"].get("scheduler"),
                scheduler_kwargs=config["model"].get("scheduler_kwargs"),
                class_weighting=config["model"].get("class_weighting"),
                task_loss_weights=task_weights,
                fusion_dim=config["model"].get("hidden_dim", 128),
                attention_heads=config["model"].get("num_heads", 4),
                **trainer_runtime_kwargs,
                progress=progress,
            )
            train_seconds = time.perf_counter() - start
            task_meta = _task_metadata_map(prepared_effective)
            task_names = list(result.get("task_names", sorted(training_splits["test"].get("y_tasks", {}).keys())))
            import torch
            with torch.no_grad():
                scalar_test = torch.from_numpy(np.asarray(training_splits["test"]["scalar_X"], dtype=np.float32)).to(result["resolved_device"])
                cov_test = torch.from_numpy(np.asarray(training_splits["test"]["cov_X"], dtype=np.float32)).to(result["resolved_device"])
                temporal_test = torch.from_numpy(np.asarray(training_splits["test"]["temporal_X"], dtype=np.float32)).to(result["resolved_device"])
                logits_map = result["model"](scalar_test, cov_test, temporal_test)
                test_predictions = {
                    task_name: torch.argmax(logits_map[task_name], dim=1).cpu().numpy().astype(np.int64)
                    for task_name in task_names
                }
            prediction_frames: dict[str, pd.DataFrame] = {}
            task_rows: list[dict[str, Any]] = []
            serialized_task_metrics: dict[str, Any] = {}
            primary_task_name = str(_primary_task_entry(config).get("name") or config["task"]["task_name"])
            for task_name in task_names:
                y_true_all = np.asarray(training_splits["test"]["y_tasks"][task_name], dtype=np.int64)
                y_pred_all = np.asarray(test_predictions[task_name], dtype=np.int64)
                valid_mask = y_true_all >= 0
                task_label_names = list(task_meta.get(task_name, {}).get("labels", []))
                if bool(valid_mask.any()):
                    evaluated = evaluate_predictions(y_true_all[valid_mask], y_pred_all[valid_mask], label_names=task_label_names)
                else:
                    evaluated = {
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "macro_f1": 0.0,
                        "confusion_matrix": np.zeros((0, 0), dtype=int),
                        "per_class_metrics": {},
                        "labels": [],
                        "label_names": [],
                    }
                prediction_frames[task_name] = pd.DataFrame(
                    {
                        "y_true": y_true_all,
                        "y_pred": np.where(valid_mask, y_pred_all, -1),
                        "is_labeled": valid_mask.astype(bool),
                        "y_true_label": _decode_labels(y_true_all, task_label_names),
                        "y_pred_label": _decode_labels(np.where(valid_mask, y_pred_all, -1), task_label_names),
                    }
                )
                serialized_task_metrics[task_name] = {
                    "accuracy": float(evaluated.get("accuracy", 0.0)),
                    "f1": float(evaluated.get("f1", 0.0)),
                    "macro_f1": float(evaluated.get("macro_f1", 0.0)),
                    "valid_count": int(valid_mask.sum()),
                    "metrics": evaluated,
                }
                task_rows.append(
                    {
                        "task_name": task_name,
                        "accuracy": float(evaluated.get("accuracy", 0.0)),
                        "f1": float(evaluated.get("f1", 0.0)),
                        "macro_f1": float(evaluated.get("macro_f1", 0.0)),
                        "valid_count": int(valid_mask.sum()),
                    }
                )
            primary_payload = serialized_task_metrics.get(primary_task_name)
            if primary_payload is None:
                primary_payload = serialized_task_metrics[task_names[0]]
            primary_predictions = prediction_frames.get(primary_task_name)
            if primary_predictions is None:
                primary_predictions = next(iter(prediction_frames.values()))
            metrics = dict(primary_payload["metrics"])
            metrics["run_mode"] = "multitask"
            metrics["task_name"] = primary_task_name
            metrics["target_column"] = task_meta.get(primary_task_name, {}).get("target_column", config["task"]["target_column"])
            metrics["model_name"] = result["model_name"]
            metrics["resolved_device"] = result["resolved_device"]
            metrics["train_seconds"] = train_seconds
            metrics["leakage_columns_removed"] = prepared_effective["metadata"].get("leakage_columns_removed", [])
            metrics["sequence_layout"] = training_metadata.get("sequence_layout")
            metrics["model_family"] = "dl"
            metrics["train_history"] = result.get("train_history", {})
            metrics["epochs_ran"] = result.get("epochs_ran", 0)
            metrics["best_checkpoint"] = result.get("best_checkpoint", {})
            metrics["task_metrics"] = serialized_task_metrics
            metrics["mean_task_accuracy"] = float(np.mean([row["accuracy"] for row in task_rows])) if task_rows else 0.0
            metrics["mean_task_f1"] = float(np.mean([row["f1"] for row in task_rows])) if task_rows else 0.0
            return {
                "metrics": metrics,
                "predictions": primary_predictions,
                "prediction_frames": prediction_frames,
                "feature_bundle": feature_bundle,
                "dataset_metadata": prepared_effective["metadata"],
                "result": result,
                "training_metadata": training_metadata,
                "tables": {"task_comparison": pd.DataFrame(task_rows)},
            }
        if not _is_deep_model_name(config["model"]["name"]):
            raise ValueError("multitask mode currently requires a deep model")
        task_weights = {
            str(task.get("name") or task.get("task_name") or task.get("target_column")): float(task.get("loss_weight", 1.0))
            for task in _task_entries(config)
        }
        start = time.perf_counter()
        result = _call_with_batch_backoff(
            modeling.train_multitask_cnn_model,
            config,
            x_train=training_splits["train"]["X"],
            y_train_tasks=training_splits["train"]["y_tasks"],
            x_val=training_splits["val"]["X"],
            y_val_tasks=training_splits["val"]["y_tasks"],
            requested_device=device,
            epochs=config["model"]["epochs"],
            batch_size=config["model"]["batch_size"],
            learning_rate=config["model"]["learning_rate"],
            random_seed=config["trainer"]["random_state"],
            dropout=config["model"].get("dropout"),
            weight_decay=config["model"].get("weight_decay"),
            patience=config["model"].get("patience"),
            scheduler=config["model"].get("scheduler"),
            scheduler_kwargs=config["model"].get("scheduler_kwargs"),
            class_weighting=config["model"].get("class_weighting"),
            task_loss_weights=task_weights,
            encoder_state_dict=transfer_payload.get("encoder_state_dict"),
            encoder_metadata=transfer_payload.get("encoder_metadata"),
            transfer_strategy=transfer_payload.get("transfer_strategy"),
            **trainer_runtime_kwargs,
            progress=progress,
        )
        train_seconds = time.perf_counter() - start
        task_meta = _task_metadata_map(prepared_effective)
        task_names = list(result.get("task_names", sorted(training_splits["test"].get("y_tasks", {}).keys())))
        test_predictions = _multitask_predictions(result["model"], training_splits["test"]["X"], task_names, result["resolved_device"])
        prediction_frames: dict[str, pd.DataFrame] = {}
        task_rows: list[dict[str, Any]] = []
        serialized_task_metrics: dict[str, Any] = {}
        primary_task_name = str(_primary_task_entry(config).get("name") or config["task"]["task_name"])

        for task_name in task_names:
            y_true_all = np.asarray(training_splits["test"]["y_tasks"][task_name], dtype=np.int64)
            y_pred_all = np.asarray(test_predictions[task_name], dtype=np.int64)
            valid_mask = y_true_all >= 0
            task_label_names = list(task_meta.get(task_name, {}).get("labels", []))
            if bool(valid_mask.any()):
                evaluated = evaluate_predictions(y_true_all[valid_mask], y_pred_all[valid_mask], label_names=task_label_names)
                y_true_eval = y_true_all[valid_mask]
                y_pred_eval = y_pred_all[valid_mask]
            else:
                evaluated = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "macro_f1": 0.0,
                    "confusion_matrix": np.zeros((0, 0), dtype=int),
                    "per_class_metrics": {},
                    "labels": [],
                    "label_names": [],
                }
                y_true_eval = np.asarray([], dtype=np.int64)
                y_pred_eval = np.asarray([], dtype=np.int64)

            prediction_frames[task_name] = pd.DataFrame(
                {
                    "y_true": y_true_all,
                    "y_pred": np.where(valid_mask, y_pred_all, -1),
                    "is_labeled": valid_mask.astype(bool),
                    "y_true_label": _decode_labels(y_true_all, task_label_names),
                    "y_pred_label": _decode_labels(np.where(valid_mask, y_pred_all, -1), task_label_names),
                }
            )
            serialized_task_metrics[task_name] = {
                "accuracy": float(evaluated.get("accuracy", 0.0)),
                "f1": float(evaluated.get("f1", 0.0)),
                "macro_f1": float(evaluated.get("macro_f1", 0.0)),
                "valid_count": int(valid_mask.sum()),
                "metrics": evaluated,
            }
            task_rows.append(
                {
                    "task_name": task_name,
                    "accuracy": float(evaluated.get("accuracy", 0.0)),
                    "f1": float(evaluated.get("f1", 0.0)),
                    "macro_f1": float(evaluated.get("macro_f1", 0.0)),
                    "valid_count": int(valid_mask.sum()),
                }
            )

        primary_payload = serialized_task_metrics.get(primary_task_name)
        if primary_payload is None:
            primary_payload = serialized_task_metrics[task_names[0]]
        primary_predictions = prediction_frames.get(primary_task_name)
        if primary_predictions is None:
            primary_predictions = next(iter(prediction_frames.values()))
        metrics = dict(primary_payload["metrics"])
        metrics["run_mode"] = "multitask"
        metrics["task_name"] = primary_task_name
        metrics["target_column"] = task_meta.get(primary_task_name, {}).get("target_column", config["task"]["target_column"])
        metrics["model_name"] = result["model_name"]
        metrics["resolved_device"] = result["resolved_device"]
        metrics["train_seconds"] = train_seconds
        metrics["leakage_columns_removed"] = prepared_effective["metadata"].get("leakage_columns_removed", [])
        metrics["sequence_layout"] = training_metadata.get("sequence_layout")
        metrics["model_family"] = "dl"
        metrics["train_history"] = result.get("train_history", {})
        metrics["epochs_ran"] = result.get("epochs_ran", 0)
        metrics["best_checkpoint"] = result.get("best_checkpoint", {})
        metrics["encoder_metadata"] = result.get("encoder_metadata", {})
        metrics["task_metrics"] = serialized_task_metrics
        metrics["mean_task_accuracy"] = float(np.mean([row["accuracy"] for row in task_rows])) if task_rows else 0.0
        metrics["mean_task_f1"] = float(np.mean([row["f1"] for row in task_rows])) if task_rows else 0.0
        if transfer_payload.get("transfer_from_run_dir"):
            metrics["transfer_from_run_dir"] = transfer_payload["transfer_from_run_dir"]
            metrics["transfer_strategy"] = transfer_payload.get("transfer_strategy")
        return {
            "metrics": metrics,
            "predictions": primary_predictions,
            "prediction_frames": prediction_frames,
            "feature_bundle": feature_bundle,
            "dataset_metadata": prepared_effective["metadata"],
            "result": result,
            "training_metadata": training_metadata,
            "tables": {"task_comparison": pd.DataFrame(task_rows)},
        }

    start = time.perf_counter()
    if config["model"]["name"] == "cognitive_radio_hybrid":
        result = _call_with_batch_backoff(
            modeling.train_cognitive_radio_hybrid_model,
            config,
            scalar_train=training_splits["train"]["scalar_X"],
            cov_train=training_splits["train"]["cov_X"],
            temporal_train=training_splits["train"]["temporal_X"],
            y_train=training_splits["train"]["y"],
            scalar_val=training_splits["val"]["scalar_X"],
            cov_val=training_splits["val"]["cov_X"],
            temporal_val=training_splits["val"]["temporal_X"],
            y_val=training_splits["val"]["y"],
            requested_device=device,
            epochs=config["model"]["epochs"],
            batch_size=config["model"]["batch_size"],
            learning_rate=config["model"]["learning_rate"],
            random_seed=config["trainer"]["random_state"],
            dropout=config["model"].get("dropout"),
            weight_decay=config["model"].get("weight_decay"),
            patience=config["model"].get("patience"),
            scheduler=config["model"].get("scheduler"),
            scheduler_kwargs=config["model"].get("scheduler_kwargs"),
            class_weighting=config["model"].get("class_weighting"),
            fusion_dim=config["model"].get("hidden_dim", 128),
            attention_heads=config["model"].get("num_heads", 4),
            **trainer_runtime_kwargs,
            progress=progress,
        )
        resolved_device = result["resolved_device"]
    elif config["model"]["name"] == "cognitive_radio_scalar_hybrid":
        result = _call_with_batch_backoff(
            modeling.train_cognitive_radio_scalar_hybrid_model,
            config,
            scalar_train=training_splits["train"]["scalar_X"],
            cov_train=training_splits["train"]["cov_X"],
            temporal_train=training_splits["train"]["temporal_X"],
            y_train=training_splits["train"]["y"],
            scalar_val=training_splits["val"]["scalar_X"],
            cov_val=training_splits["val"]["cov_X"],
            temporal_val=training_splits["val"]["temporal_X"],
            y_val=training_splits["val"]["y"],
            requested_device=device,
            epochs=config["model"]["epochs"],
            batch_size=config["model"]["batch_size"],
            learning_rate=config["model"]["learning_rate"],
            random_seed=config["trainer"]["random_state"],
            dropout=config["model"].get("dropout"),
            weight_decay=config["model"].get("weight_decay"),
            patience=config["model"].get("patience"),
            scheduler=config["model"].get("scheduler"),
            scheduler_kwargs=config["model"].get("scheduler_kwargs"),
            class_weighting=config["model"].get("class_weighting"),
            fusion_dim=config["model"].get("hidden_dim", 256),
            attention_heads=config["model"].get("num_heads", 8),
            **trainer_runtime_kwargs,
            progress=progress,
        )
        resolved_device = result["resolved_device"]
    elif _is_deep_model_name(config["model"]["name"]):
        deep_trainers: dict[str, Callable[..., Any]] = {
            "cnn_1d": modeling.train_cnn_model,
            "cnn_1d_residual": modeling.train_cnn_residual_model,
            "cnn_lstm": modeling.train_cnn_lstm_model,
            "transformer_1d": modeling.train_transformer_1d_model,
        }
        trainer = deep_trainers[config["model"]["name"]]
        result = _call_with_batch_backoff(
            trainer,
            config,
            x_train=training_splits["train"]["X"],
            y_train=training_splits["train"]["y"],
            x_val=training_splits["val"]["X"],
            y_val=training_splits["val"]["y"],
            requested_device=device,
            epochs=config["model"]["epochs"],
            batch_size=config["model"]["batch_size"],
            learning_rate=config["model"]["learning_rate"],
            random_seed=config["trainer"]["random_state"],
            dropout=config["model"].get("dropout"),
            weight_decay=config["model"].get("weight_decay"),
            patience=config["model"].get("patience"),
            scheduler=config["model"].get("scheduler"),
            scheduler_kwargs=config["model"].get("scheduler_kwargs"),
            class_weighting=config["model"].get("class_weighting"),
            sampler=config["trainer"].get("sampler", "off"),
            loss_name=config["model"].get("loss_name", "cross_entropy"),
            focal_gamma=config["model"].get("focal_gamma", 2.0),
            lstm_hidden_size=config["model"].get("hidden_dim"),
            lstm_layers=config["model"].get("num_layers"),
            d_model=config["model"].get("hidden_dim"),
            nhead=config["model"].get("num_heads"),
            transformer_layers=config["model"].get("num_layers"),
            dim_feedforward=config["model"].get("dim_feedforward", max(128, int(config["model"].get("hidden_dim", 64)) * 2)),
            encoder_state_dict=transfer_payload.get("encoder_state_dict"),
            encoder_metadata=transfer_payload.get("encoder_metadata"),
            transfer_strategy=transfer_payload.get("transfer_strategy"),
            **trainer_runtime_kwargs,
            progress=progress,
        )
        resolved_device = result["resolved_device"]
    else:
        result = _call_with_supported_kwargs(
            modeling.train_baseline_model,
            x_train=training_splits["train"]["X"],
            y_train=training_splits["train"]["y"],
            x_val=training_splits["val"]["X"],
            y_val=training_splits["val"]["y"],
            random_state=config["trainer"]["random_state"],
            baseline=config["model"]["name"],
            candidates=config["model"].get("candidates") or None,
            baseline_params=config["model"].get("params") or None,
            primary_metric=config["task"].get("metric_primary", "accuracy"),
            secondary_metric=config["task"].get("metric_secondary", "f1"),
            threshold_tuning=config.get("evaluation", {}).get("threshold_tuning", {}),
        )
        resolved_device = device
    train_seconds = time.perf_counter() - start

    threshold = result.get("threshold")
    if config["model"]["name"] in {"cognitive_radio_hybrid", "cognitive_radio_scalar_hybrid"}:
        import torch
        with torch.no_grad():
            scalar_val_threshold = torch.from_numpy(np.asarray(training_splits["val"]["scalar_X"], dtype=np.float32)).to(resolved_device)
            cov_val_threshold = torch.from_numpy(np.asarray(training_splits["val"]["cov_X"], dtype=np.float32)).to(resolved_device)
            temporal_val_threshold = torch.from_numpy(np.asarray(training_splits["val"]["temporal_X"], dtype=np.float32)).to(resolved_device)
            val_logits_threshold = result["model"](scalar_val_threshold, cov_val_threshold, temporal_val_threshold)
            val_probabilities = torch.softmax(val_logits_threshold, dim=1).cpu().numpy()
            val_score_vector = val_probabilities[:, 1] if val_probabilities.ndim == 2 and val_probabilities.shape[1] == 2 else None
    else:
        val_score_vector = _predict_scores_with_model(result["model"], training_splits["val"]["X"], resolved_device)
    val_threshold, _val_threshold_pred, _val_threshold_metrics = _tune_threshold(
        y_true=training_splits["val"]["y"],
        y_score=val_score_vector,
        threshold_tuning=config.get("evaluation", {}).get("threshold_tuning", {}),
        primary_metric=config["task"].get("metric_primary", "accuracy"),
        secondary_metric=config["task"].get("metric_secondary", "f1"),
    )
    if threshold is None:
        threshold = val_threshold

    if config["model"]["name"] in {"cognitive_radio_hybrid", "cognitive_radio_scalar_hybrid"}:
        import torch
        with torch.no_grad():
            scalar_test = torch.from_numpy(np.asarray(training_splits["test"]["scalar_X"], dtype=np.float32)).to(resolved_device)
            cov_test = torch.from_numpy(np.asarray(training_splits["test"]["cov_X"], dtype=np.float32)).to(resolved_device)
            temporal_test = torch.from_numpy(np.asarray(training_splits["test"]["temporal_X"], dtype=np.float32)).to(resolved_device)
            logits_test = result["model"](scalar_test, cov_test, temporal_test)
            probabilities = torch.softmax(logits_test, dim=1).cpu().numpy()
            y_score_test = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] == 2 else None
            if y_score_test is not None and threshold is not None:
                test_pred = (y_score_test >= float(threshold)).astype(np.int64)
            else:
                test_pred = np.argmax(probabilities, axis=1).astype(np.int64)
    else:
        test_pred, y_score_test = _predict_labels(
            result["model"],
            training_splits["test"]["X"],
            resolved_device,
            threshold=threshold if len(label_names) == 2 else None,
        )

    metrics = evaluate_predictions(
        training_splits["test"]["y"],
        test_pred,
        y_score=y_score_test if len(label_names) == 2 else None,
        label_names=label_names,
    )
    metrics["run_mode"] = "single_task"
    metrics["train_seconds"] = train_seconds
    metrics["resolved_device"] = resolved_device
    metrics["model_name"] = result["model_name"]
    metrics["task_name"] = config["task"]["task_name"]
    metrics["target_column"] = config["task"]["target_column"]
    metrics["leakage_columns_removed"] = prepared_effective["metadata"].get("leakage_columns_removed", [])
    metrics["sequence_layout"] = training_metadata.get("sequence_layout")
    metrics["model_family"] = _model_family_for_name(config["model"]["name"])
    if "selected_baseline" in result:
        metrics["selected_baseline"] = result["selected_baseline"]
    if "candidate_scores" in result:
        metrics["candidate_scores"] = result["candidate_scores"]
    if "train_history" in result:
        metrics["train_history"] = result["train_history"]
    if "epochs_ran" in result:
        metrics["epochs_ran"] = result["epochs_ran"]
    if "best_checkpoint" in result:
        metrics["best_checkpoint"] = result["best_checkpoint"]
    if "encoder_metadata" in result:
        metrics["encoder_metadata"] = result["encoder_metadata"]
    if threshold is not None:
        metrics["threshold"] = float(threshold)
    if transfer_payload.get("transfer_from_run_dir"):
        metrics["transfer_from_run_dir"] = transfer_payload["transfer_from_run_dir"]
        metrics["transfer_strategy"] = transfer_payload.get("transfer_strategy")

    if config["model"]["name"] in {"cognitive_radio_hybrid", "cognitive_radio_scalar_hybrid"}:
        import torch
        with torch.no_grad():
            scalar_train_t = torch.from_numpy(np.asarray(training_splits["train"]["scalar_X"], dtype=np.float32)).to(resolved_device)
            cov_train_t = torch.from_numpy(np.asarray(training_splits["train"]["cov_X"], dtype=np.float32)).to(resolved_device)
            temporal_train_t = torch.from_numpy(np.asarray(training_splits["train"]["temporal_X"], dtype=np.float32)).to(resolved_device)
            logits_train = result["model"](scalar_train_t, cov_train_t, temporal_train_t)
            train_prob = torch.softmax(logits_train, dim=1).cpu().numpy()
            if train_prob.ndim == 2 and train_prob.shape[1] == 2 and threshold is not None:
                train_pred = (train_prob[:, 1] >= float(threshold)).astype(np.int64)
            else:
                train_pred = np.argmax(train_prob, axis=1).astype(np.int64)
    else:
        train_pred, _ = _predict_labels(
            result["model"],
            training_splits["train"]["X"],
            resolved_device,
            threshold=threshold if len(label_names) == 2 else None,
        )
    train_metrics = evaluate_predictions(training_splits["train"]["y"], train_pred, label_names=label_names)
    metrics["train_accuracy_score"] = train_metrics["accuracy"]
    metrics["train_f1_score"] = train_metrics["f1"]

    if config["model"]["name"] in {"cognitive_radio_hybrid", "cognitive_radio_scalar_hybrid"}:
        import torch
        with torch.no_grad():
            scalar_val_t = torch.from_numpy(np.asarray(training_splits["val"]["scalar_X"], dtype=np.float32)).to(resolved_device)
            cov_val_t = torch.from_numpy(np.asarray(training_splits["val"]["cov_X"], dtype=np.float32)).to(resolved_device)
            temporal_val_t = torch.from_numpy(np.asarray(training_splits["val"]["temporal_X"], dtype=np.float32)).to(resolved_device)
            logits_val = result["model"](scalar_val_t, cov_val_t, temporal_val_t)
            val_prob = torch.softmax(logits_val, dim=1).cpu().numpy()
            if val_prob.ndim == 2 and val_prob.shape[1] == 2 and threshold is not None:
                val_pred = (val_prob[:, 1] >= float(threshold)).astype(np.int64)
            else:
                val_pred = np.argmax(val_prob, axis=1).astype(np.int64)
    else:
        val_pred, _ = _predict_labels(
            result["model"],
            training_splits["val"]["X"],
            resolved_device,
            threshold=threshold if len(label_names) == 2 else None,
        )
    val_metrics = evaluate_predictions(training_splits["val"]["y"], val_pred, label_names=label_names)
    metrics["val_accuracy_score"] = val_metrics["accuracy"]
    metrics["val_f1_score"] = val_metrics["f1"]
    metrics["overfit_gap"] = float(train_metrics["accuracy"] - val_metrics["accuracy"])

    predictions = pd.DataFrame(
        {
            "y_true": training_splits["test"]["y"],
            "y_pred": test_pred,
            "y_true_label": _decode_labels(training_splits["test"]["y"], label_names),
            "y_pred_label": _decode_labels(test_pred, label_names),
        }
    )
    if y_score_test is not None and len(y_score_test) == len(predictions):
        predictions["score"] = y_score_test

    return {
        "metrics": metrics,
        "predictions": predictions,
        "prediction_frames": {},
        "feature_bundle": feature_bundle,
        "training_splits": training_splits,
        "dataset_metadata": prepared_effective["metadata"],
        "result": result,
        "training_metadata": training_metadata,
        "tables": {},
    }


def _persist_training_result(run_dir: Path, execution: dict[str, Any]) -> None:
    metrics = execution["metrics"]
    feature_bundle = execution["feature_bundle"]
    if isinstance(execution.get("predictions"), pd.DataFrame):
        save_dataframe(run_dir / "predictions.csv", execution["predictions"])
    save_json(run_dir / "metrics.json", _serializable_metrics(metrics))
    if feature_bundle:
        save_json(run_dir / "feature_metadata.json", _feature_metadata_payload(feature_bundle))
    save_json(run_dir / "dataset_metadata.json", execution["dataset_metadata"])
    if feature_bundle:
        save_dataframe(run_dir / "tables" / "feature_importance.csv", _feature_importance_frame(feature_bundle))
    if "per_class_metrics" in metrics:
        save_dataframe(run_dir / "tables" / "per_class_metrics.csv", per_class_metrics_frame(metrics))
    if "minority_f1" in metrics:
        save_dataframe(
            run_dir / "tables" / "minority_class_metrics.csv",
            pd.DataFrame(
                [
                    {
                        "label": metrics.get("minority_label"),
                        "label_name": metrics.get("minority_label_name"),
                        "precision": metrics.get("minority_precision", 0.0),
                        "recall": metrics.get("minority_recall", 0.0),
                        "f1": metrics.get("minority_f1", 0.0),
                        "support": metrics.get("minority_support", 0),
                    }
                ]
            ),
        )
    if metrics.get("candidate_scores"):
        save_dataframe(run_dir / "tables" / "candidate_scores.csv", _candidate_scores_frame(metrics["candidate_scores"]))
    for table_name, frame in execution.get("tables", {}).items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            save_dataframe(run_dir / "tables" / f"{table_name}.csv", frame)
    for task_name, frame in execution.get("prediction_frames", {}).items():
        if isinstance(frame, pd.DataFrame):
            save_dataframe(run_dir / f"predictions_{task_name}.csv", frame)

    result = execution["result"]
    if "best_checkpoint" in result:
        save_json(run_dir / "best_checkpoint.json", result["best_checkpoint"])
    if metrics.get("encoder_metadata"):
        save_json(run_dir / "encoder_metadata.json", metrics["encoder_metadata"])
    if "best_state_dict" in result:
        try:
            import torch

            torch.save(result["best_state_dict"], _torch_checkpoint_path(run_dir))
        except Exception:
            pass
    if "encoder_state_dict" in result:
        try:
            import torch

            checkpoint_name = str(execution.get("encoder_checkpoint_name", "pretrain_encoder.pt"))
            torch.save(result["encoder_state_dict"], run_dir / checkpoint_name)
        except Exception:
            pass


def _ephemeral_run(
    config: dict[str, Any],
    prepared: dict[str, Any],
    requested_device: str,
    layout_override: str | None = None,
) -> dict[str, Any]:
    execution = _execute_training(
        config,
        prepared,
        requested_device=requested_device,
        layout_override=layout_override,
        progress=False,
    )
    return execution


def _cross_validation_analysis(config: dict[str, Any], requested_device: str) -> pd.DataFrame:
    if str(config["model"].get("mode", "single_task")) != "single_task":
        return pd.DataFrame()
    cv_config = config.get("evaluation", {}).get("cross_validation", {})
    if not cv_config.get("enabled", False):
        return pd.DataFrame()
    StratifiedKFold, StratifiedShuffleSplit = _get_model_selection_classes()
    if StratifiedKFold is None or StratifiedShuffleSplit is None:
        return pd.DataFrame()

    loaded = load_local_data(
        source=config["dataset"]["input_path"],
        schema=config["dataset"]["schema"],
        target_column=config["task"]["target_column"],
        drop_columns=config["task"].get("drop_leakage_columns", []),
    )
    x_all = np.asarray(loaded["X"], dtype=np.float32)
    y_all = np.asarray(loaded["y"], dtype=np.int64)
    folds = int(cv_config.get("folds", 5))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config["trainer"]["random_state"])
    rows: list[dict[str, Any]] = []

    for fold_index, (train_val_idx, test_idx) in enumerate(splitter.split(x_all, y_all), start=1):
        val_ratio = float(config["trainer"].get("val_ratio", 0.15))
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=max(val_ratio, 0.1),
            random_state=config["trainer"]["random_state"] + fold_index,
        )
        inner_train_positions, inner_val_positions = next(inner_splitter.split(x_all[train_val_idx], y_all[train_val_idx]))
        train_idx = train_val_idx[inner_train_positions]
        val_idx = train_val_idx[inner_val_positions]
        prepared_fold = _prepared_from_loaded(
            loaded=loaded,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            schema=config["dataset"]["schema"],
            target_column=config["task"]["target_column"],
            task_name=config["task"]["task_name"],
            leakage_columns_removed=config["task"].get("drop_leakage_columns", []),
            random_state=config["trainer"]["random_state"] + fold_index,
        )
        execution = _ephemeral_run(config, prepared_fold, requested_device=requested_device)
        metrics = execution["metrics"]
        rows.append(
            {
                "fold": fold_index,
                "model_name": metrics.get("model_name", "model"),
                "accuracy": metrics.get("accuracy", 0.0),
                "f1": metrics.get("f1", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        summary_row = {
            "fold": "mean",
            "model_name": frame["model_name"].mode().iloc[0],
            "accuracy": frame["accuracy"].mean(),
            "f1": frame["f1"].mean(),
            "macro_f1": frame["macro_f1"].mean(),
            "precision": frame["precision"].mean(),
            "recall": frame["recall"].mean(),
        }
        std_row = {
            "fold": "std",
            "model_name": frame["model_name"].mode().iloc[0],
            "accuracy": frame["accuracy"].std(ddof=0),
            "f1": frame["f1"].std(ddof=0),
            "macro_f1": frame["macro_f1"].std(ddof=0),
            "precision": frame["precision"].std(ddof=0),
            "recall": frame["recall"].std(ddof=0),
        }
        frame = pd.concat([frame, pd.DataFrame([summary_row, std_row])], ignore_index=True)
    return frame


def _learning_curve_analysis(config: dict[str, Any], prepared: dict[str, Any], requested_device: str) -> pd.DataFrame:
    curve_config = config.get("evaluation", {}).get("learning_curve", {})
    if not curve_config.get("enabled", False):
        return pd.DataFrame()
    train_sizes = [float(item) for item in curve_config.get("train_sizes", [1.0])]
    train_x = np.asarray(prepared["splits"]["train"]["X"])
    train_y = np.asarray(prepared["splits"]["train"]["y"])
    rng = np.random.default_rng(config["trainer"]["random_state"])

    rows: list[dict[str, Any]] = []
    for fraction in train_sizes:
        sample_count = max(2, int(round(train_x.shape[0] * fraction)))
        indices = np.arange(train_x.shape[0])
        rng.shuffle(indices)
        selected = indices[:sample_count]
        subset_prepared = {
            "splits": {
                "train": {"X": train_x[selected], "y": train_y[selected]},
                "val": prepared["splits"]["val"],
                "test": prepared["splits"]["test"],
            },
            "metadata": prepared["metadata"],
            "statistics": prepared["statistics"],
            "scaler": prepared["scaler"],
        }
        execution = _ephemeral_run(config, subset_prepared, requested_device=requested_device)
        metrics = execution["metrics"]
        rows.append(
            {
                "train_fraction": fraction,
                "num_train_samples": sample_count,
                "train_accuracy": metrics.get("train_accuracy_score", 0.0),
                "train_f1": metrics.get("train_f1_score", 0.0),
                "val_accuracy": metrics.get("val_accuracy_score", 0.0),
                "val_f1": metrics.get("val_f1_score", 0.0),
                "test_accuracy": metrics.get("accuracy", 0.0),
                "test_f1": metrics.get("f1", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _ablation_analysis(config: dict[str, Any], prepared: dict[str, Any], requested_device: str) -> pd.DataFrame:
    ablation_config = config.get("evaluation", {}).get("ablation", {})
    if not ablation_config.get("enabled", False):
        return pd.DataFrame()
    layouts = [str(item) for item in ablation_config.get("layouts", ["all"])]
    rows: list[dict[str, Any]] = []
    for layout in layouts:
        execution = _ephemeral_run(config, prepared, requested_device=requested_device, layout_override=layout)
        metrics = execution["metrics"]
        rows.append(
            {
                "layout": layout,
                "model_name": metrics.get("model_name", "model"),
                "accuracy": metrics.get("accuracy", 0.0),
                "f1": metrics.get("f1", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _post_training_analyses(
    config: dict[str, Any],
    prepared: dict[str, Any],
    requested_device: str,
) -> dict[str, pd.DataFrame]:
    if str(config["model"].get("mode", "single_task")) != "single_task":
        return {
            "cv_metrics": pd.DataFrame(),
            "learning_curve": pd.DataFrame(),
            "ablation_metrics": pd.DataFrame(),
        }
    return {
        "cv_metrics": _cross_validation_analysis(config, requested_device=requested_device),
        "learning_curve": _learning_curve_analysis(config, prepared, requested_device=requested_device),
        "ablation_metrics": _ablation_analysis(config, prepared, requested_device=requested_device),
    }


def _robustness_analyses(
    config: dict[str, Any],
    prepared: dict[str, Any],
    requested_device: str,
    execution: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    if str(config["model"].get("mode", "single_task")) != "single_task":
        return {
            "robustness_noise": pd.DataFrame(),
            "robustness_train_ratio": pd.DataFrame(),
        }

    sweeps = config.get("evaluation", {}).get("robustness_sweeps", {})
    noise_levels = [float(level) for level in sweeps.get("noise_levels", []) if float(level) >= 0.0]
    train_ratios = [float(level) for level in sweeps.get("train_ratios", []) if 0.0 < float(level) <= 1.0]
    outputs = {
        "robustness_noise": pd.DataFrame(),
        "robustness_train_ratio": pd.DataFrame(),
    }

    if noise_levels:
        if config["model"]["name"] not in {"cognitive_radio_hybrid", "cognitive_radio_scalar_hybrid"}:
            analysis_test_x = np.asarray(execution.get("training_splits", {}).get("test", {}).get("X"))
            threshold = execution["metrics"].get("threshold")
            resolved_device = execution["metrics"].get("resolved_device", resolve_device(requested_device))
            label_names = list(execution["dataset_metadata"].get("labels", []))
            binary_threshold = threshold if len(label_names) == 2 else None

            def _predict_fn(batch: np.ndarray) -> np.ndarray:
                pred, _ = _predict_labels(
                    execution["result"]["model"],
                    np.asarray(batch),
                    resolved_device,
                    threshold=binary_threshold,
                )
                return pred

            outputs["robustness_noise"] = run_noise_robustness_sweep(
                model_predict_fn=_predict_fn,
                x_clean=analysis_test_x,
                y_true=prepared["splits"]["test"]["y"],
                noise_levels=noise_levels,
                metric=config["task"].get("metric_secondary", "macro_f1"),
                random_state=config["trainer"]["random_state"],
            )

    if train_ratios:
        base_train_x = np.asarray(prepared["splits"]["train"]["X"])
        base_train_y = np.asarray(prepared["splits"]["train"]["y"])
        rng = np.random.default_rng(config["trainer"]["random_state"])

        def _train_and_eval(train_ratio: float) -> dict[str, Any]:
            sample_count = max(2, min(base_train_x.shape[0], int(round(base_train_x.shape[0] * train_ratio))))
            indices = np.arange(base_train_x.shape[0])
            rng.shuffle(indices)
            selected = indices[:sample_count]
            subset_prepared = {
                "splits": {
                    "train": {"X": base_train_x[selected], "y": base_train_y[selected]},
                    "val": prepared["splits"]["val"],
                    "test": prepared["splits"]["test"],
                },
                "metadata": prepared["metadata"],
                "statistics": prepared["statistics"],
                "scaler": prepared["scaler"],
            }
            rerun = _ephemeral_run(config, subset_prepared, requested_device=requested_device)
            return {"metrics": rerun["metrics"]}

        outputs["robustness_train_ratio"] = run_train_ratio_sweep(
            train_and_evaluate_fn=_train_and_eval,
            train_ratios=train_ratios,
            metric=config["task"].get("metric_secondary", "macro_f1"),
        )

    return outputs


def _train_from_config(config: dict[str, Any], requested_device: str) -> Path:
    prepared_dir = Path(config["runtime"]["prepared_dir"])
    if (prepared_dir / "prepared_splits.npz").exists():
        prepared_bundle = _load_prepared_bundle(prepared_dir)
    else:
        prepared_bundle = _prepare_from_config(config)
        _save_prepared_bundle(prepared_dir, prepared_bundle)
    execution = _execute_training(config, prepared_bundle, requested_device=requested_device, progress=True)
    analysis_outputs = _post_training_analyses(config, prepared_bundle, requested_device=requested_device)
    analysis_outputs.update(_robustness_analyses(config, prepared_bundle, requested_device=requested_device, execution=execution))
    execution["metrics"]["risk_note"] = _risk_note_from_metrics(execution["metrics"], config, analysis_outputs)

    experiment_name = f"{config['task']['task_name']}-{execution['metrics']['model_name']}"
    run_dir = create_run_dir(config["runtime"]["artifacts_dir"], experiment_name)
    _persist_training_result(run_dir, execution)
    _save_auxiliary_tables(run_dir, analysis_outputs)
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return run_dir


def _visualize_run(run_dir: Path, theme_name: str) -> None:
    from emc_diag.reporting import (
        build_ablation_notes,
        build_overfitting_risk_note,
        build_run_summary,
        summarize_cv_aggregates,
    )
    from emc_diag.visualization import (
        plot_confusion_matrix,
        plot_dataset_summary,
        plot_feature_importance,
        plot_metrics_bar,
        plot_training_curves,
    )

    visualization_module = __import__("emc_diag.visualization", fromlist=["_placeholder"])
    plot_candidate_comparison = getattr(visualization_module, "plot_candidate_comparison", None)
    plot_per_class_metrics = getattr(visualization_module, "plot_per_class_metrics", None)
    plot_binary_curves = getattr(visualization_module, "plot_binary_curves", None)
    plot_task_comparison = getattr(visualization_module, "plot_task_comparison", None)
    plot_cv_comparison = getattr(visualization_module, "plot_cv_comparison", None)
    plot_ablation_comparison = getattr(visualization_module, "plot_ablation_comparison", None)
    plot_ml_vs_cnn_comparison = getattr(visualization_module, "plot_ml_vs_cnn_comparison", None)
    plot_noise_robustness_curve = getattr(visualization_module, "plot_noise_robustness_curve", None)
    plot_feature_group_ablation = getattr(visualization_module, "plot_feature_group_ablation", None)
    plot_transfer_vs_scratch = getattr(visualization_module, "plot_transfer_vs_scratch", None)
    plot_multitask_vs_single_task = getattr(visualization_module, "plot_multitask_vs_single_task", None)
    plot_overfitting_gap = getattr(visualization_module, "plot_overfitting_gap", None)
    plot_waveform_overview = getattr(visualization_module, "plot_waveform_overview", None)
    plot_spectrum_stft_summary = getattr(visualization_module, "plot_spectrum_stft_summary", None)

    run_config = _load_run_config(run_dir)
    prepared_dir = Path(run_config["runtime"]["prepared_dir"])
    statistics = load_json(prepared_dir / "statistics.json")
    dataset_metadata = load_json(run_dir / "dataset_metadata.json")
    metrics = load_json(run_dir / "metrics.json")
    feature_importance = pd.read_csv(run_dir / "tables" / "feature_importance.csv")
    run_mode = str(metrics.get("run_mode", run_config.get("model", {}).get("mode", "single_task")))
    visualization_profile = str(run_config.get("visualization", {}).get("profile", "full")).lower()
    concise_profile = visualization_profile in {"concise", "simple", "latest"}
    predictions = pd.read_csv(run_dir / "predictions.csv") if (run_dir / "predictions.csv").exists() else None

    dataset_rows = []
    for label, count in statistics["class_counts"].items():
        dataset_rows.extend([{"class": str(label)}] * int(count))
    dataset_df = pd.DataFrame(dataset_rows or [{"class": "unknown"}])
    if not concise_profile:
        plot_dataset_summary(dataset_df=dataset_df, output_dir=run_dir / "figures", theme_name=theme_name)
        plot_feature_importance(feature_importance, run_dir / "figures", theme_name)

    curve_df = _history_curve_frame(metrics, run_config)
    if not concise_profile:
        plot_training_curves(curve_df, run_dir / "figures", theme_name)
    if callable(plot_overfitting_gap) and not concise_profile:
        plot_overfitting_gap(curve_df, run_dir / "figures", theme_name)

    if run_mode != "pretrain" and predictions is not None and "confusion_matrix" in metrics:
        plot_confusion_matrix(
            cm=np.asarray(metrics["confusion_matrix"]),
            labels=metrics.get("label_names", dataset_metadata.get("labels", [])),
            output_dir=run_dir / "figures",
            theme_name=theme_name,
        )
        plot_metrics_bar(_metrics_frame(metrics), run_dir / "figures", theme_name)

    if callable(plot_candidate_comparison) and metrics.get("candidate_scores"):
        plot_candidate_comparison(_candidate_scores_frame(metrics["candidate_scores"]), run_dir / "figures", theme_name)
    if callable(plot_per_class_metrics) and (run_dir / "tables" / "per_class_metrics.csv").exists():
        plot_per_class_metrics(pd.read_csv(run_dir / "tables" / "per_class_metrics.csv"), run_dir / "figures", theme_name)
    if (
        callable(plot_binary_curves)
        and not concise_profile
        and predictions is not None
        and "score" in predictions.columns
        and metrics.get("roc_curve")
    ):
        roc_df = pd.DataFrame(metrics.get("roc_curve", []))
        pr_df = pd.DataFrame(metrics.get("pr_curve", []))
        plot_binary_curves(roc_df=roc_df, pr_df=pr_df, output_dir=run_dir / "figures", theme_name=theme_name)
    cv_aggregates: dict[str, float] | None = None
    if callable(plot_cv_comparison) and not concise_profile and (run_dir / "tables" / "cv_metrics.csv").exists():
        cv_metrics_raw = pd.read_csv(run_dir / "tables" / "cv_metrics.csv")
        cv_summary = _cv_summary_frame(cv_metrics_raw, metrics.get("model_name", "model"))
        if not cv_summary.empty:
            plot_cv_comparison(cv_summary, run_dir / "figures", theme_name)
            cv_aggregates = summarize_cv_aggregates(cv_summary.to_dict(orient="records"))

    ablation_notes: list[str] | None = None
    if callable(plot_ablation_comparison) and not concise_profile and (run_dir / "tables" / "ablation_metrics.csv").exists():
        ablation_raw = pd.read_csv(run_dir / "tables" / "ablation_metrics.csv")
        ablation_summary = _ablation_summary_frame(ablation_raw, metric_name=run_config["task"].get("metric_primary", "accuracy"))
        if not ablation_summary.empty:
            plot_ablation_comparison(ablation_summary, run_dir / "figures", theme_name)
            ablation_notes = build_ablation_notes(ablation_summary.to_dict(orient="records"))
    if callable(plot_noise_robustness_curve) and not concise_profile and (run_dir / "tables" / "robustness_noise.csv").exists():
        plot_noise_robustness_curve(
            pd.read_csv(run_dir / "tables" / "robustness_noise.csv"),
            run_dir / "figures",
            theme_name,
        )
    if callable(plot_feature_group_ablation) and not concise_profile and (run_dir / "tables" / "feature_group_ablation.csv").exists():
        plot_feature_group_ablation(
            pd.read_csv(run_dir / "tables" / "feature_group_ablation.csv"),
            run_dir / "figures",
            theme_name,
        )
    if callable(plot_transfer_vs_scratch) and not concise_profile and (run_dir / "tables" / "transfer_vs_scratch.csv").exists():
        plot_transfer_vs_scratch(
            pd.read_csv(run_dir / "tables" / "transfer_vs_scratch.csv"),
            run_dir / "figures",
            theme_name,
        )
    if callable(plot_multitask_vs_single_task) and not concise_profile and (run_dir / "tables" / "multitask_vs_single_task.csv").exists():
        plot_multitask_vs_single_task(
            pd.read_csv(run_dir / "tables" / "multitask_vs_single_task.csv"),
            run_dir / "figures",
            theme_name,
        )
    if callable(plot_task_comparison) and not concise_profile and (run_dir / "tables" / "task_comparison.csv").exists():
        plot_task_comparison(
            pd.read_csv(run_dir / "tables" / "task_comparison.csv"),
            run_dir / "figures",
            theme_name,
        )
    if callable(plot_ml_vs_cnn_comparison) and not concise_profile:
        family_rows = pd.DataFrame(
            [
                {
                    "family": metrics.get("model_family", "ml"),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "f1": metrics.get("f1", 0.0),
                }
            ]
        )
        plot_ml_vs_cnn_comparison(family_rows, run_dir / "figures", theme_name)
    if not concise_profile and (
        dataset_metadata.get("task_name", "").startswith("vsb_")
        or str(run_config["dataset"].get("name", "")).startswith("vsb")
    ):
        try:
            waveform_preview = _load_vsb_waveform_preview(run_config)
            if callable(plot_waveform_overview):
                plot_waveform_overview(waveform_preview, run_dir / "figures", theme_name)
            if callable(plot_spectrum_stft_summary):
                plot_spectrum_stft_summary(
                    waveform_preview,
                    run_dir / "figures",
                    theme_name,
                    nperseg=int(run_config["features"].get("nperseg", 64)),
                    noverlap=int(run_config["features"].get("noverlap", 32)),
                )
        except Exception:
            pass

    figures = sorted(path.name for path in (run_dir / "figures").glob("*.png"))
    tables = sorted(path.name for path in (run_dir / "tables").glob("*.csv"))
    metric_summary = {key: value for key, value in metrics.items() if isinstance(value, (int, float, str))}
    overfitting_risk_note = build_overfitting_risk_note(
        max_gap=float(pd.to_numeric(curve_df["train_score"] - curve_df["val_score"], errors="coerce").max())
        if {"train_score", "val_score"}.issubset(curve_df.columns)
        else float(metrics.get("overfit_gap", 0.0)),
        warning_threshold=float(run_config.get("evaluation", {}).get("risk_checks", {}).get("warning_threshold", 0.05)),
    )
    summary_context = _summary_context(run_dir, metrics, run_config)
    _call_with_supported_kwargs(
        build_run_summary,
        run_name=run_dir.name,
        output_dir=run_dir,
        metrics=metric_summary,
        figures=figures,
        tables=tables,
        task_metadata={
            "task_name": metrics.get("task_name", "task"),
            "target_column": metrics.get("target_column", "label"),
            "selected_baseline": metrics.get("selected_baseline", metrics.get("model_name", "model")),
            "leakage_columns_removed": ", ".join(metrics.get("leakage_columns_removed", [])) or "none",
            "risk_note": metrics.get("risk_note", "none"),
            "sequence_layout": str(metrics.get("sequence_layout", "none")),
        },
        run_metadata=summary_context["run_metadata"],
        transfer_notes=summary_context["transfer_notes"],
        pretrain_notes=summary_context["pretrain_notes"],
        feature_group_summary=summary_context["feature_group_summary"],
        robustness_notes=summary_context["robustness_notes"],
        cv_aggregates=cv_aggregates,
        ablation_notes=ablation_notes,
        overfitting_risk_note=overfitting_risk_note,
    )


def _run_pipeline(config_path: str | Path, requested_device: str, theme_name: str = "paper-bar") -> Path:
    config = load_config(config_path)
    return _run_pipeline_config(config, requested_device=requested_device, theme_name=theme_name)


def _collect_benchmark_frames(benchmark_dirs: list[Path]) -> tuple[list[pd.DataFrame], list[Path]]:
    frames: list[pd.DataFrame] = []
    resolved_dirs: list[Path] = []
    for benchmark_dir in benchmark_dirs:
        metrics_path = benchmark_dir / "benchmark_metrics.csv"
        if not metrics_path.exists():
            continue
        frame = pd.read_csv(metrics_path)
        if frame.empty:
            continue
        frame["benchmark_dir"] = str(benchmark_dir)
        frames.append(frame)
        resolved_dirs.append(benchmark_dir)
    return frames, resolved_dirs


def _collect_run_dirs(explicit_run_dirs: list[Path], benchmark_frames: list[pd.DataFrame]) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for run_dir in explicit_run_dirs:
        resolved = run_dir.resolve()
        if resolved.exists() and resolved not in seen:
            run_dirs.append(resolved)
            seen.add(resolved)
    for frame in benchmark_frames:
        if "run_dir" not in frame.columns:
            continue
        for raw_path in frame["run_dir"].dropna().astype(str).tolist():
            resolved = Path(raw_path).resolve()
            if resolved.exists() and resolved not in seen:
                run_dirs.append(resolved)
                seen.add(resolved)
    return run_dirs


def _run_metric_row(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    run_config = _load_run_config(run_dir)
    task_config = run_config.get("task", {})
    primary_metric = str(task_config.get("metric_primary", "accuracy"))
    secondary_metric = str(task_config.get("metric_secondary", "f1"))
    model_name = str(metrics.get("model_name", run_config.get("model", {}).get("name", "model")))
    return {
        "alias": run_dir.name,
        "run_dir": str(run_dir),
        "prepared_dir": str(run_config.get("runtime", {}).get("prepared_dir", "")),
        "dataset": str(run_config.get("dataset", {}).get("name", "dataset")),
        "task_name": str(metrics.get("task_name", task_config.get("task_name", "task"))),
        "target_column": str(metrics.get("target_column", task_config.get("target_column", "label"))),
        "model_name": model_name,
        "selected_baseline": str(metrics.get("selected_baseline", model_name)),
        "model_family": str(metrics.get("model_family", "ml")),
        "seed": int(run_config.get("trainer", {}).get("random_state", 42)),
        "primary_metric": primary_metric,
        "primary_score": score_metric(metrics, primary_metric),
        "secondary_metric": secondary_metric,
        "secondary_score": score_metric(metrics, secondary_metric),
        "accuracy": score_metric(metrics, "accuracy"),
        "f1": score_metric(metrics, "f1"),
        "macro_f1": score_metric(metrics, "macro_f1"),
        "minority_f1": score_metric(metrics, "minority_f1"),
        "precision": score_metric(metrics, "precision"),
        "recall": score_metric(metrics, "recall"),
        "threshold": metrics.get("threshold", ""),
        "risk_note": str(metrics.get("risk_note", "")),
        "resolved_device": str(metrics.get("resolved_device", "")),
        "overfit_gap": float(metrics.get("overfit_gap", 0.0)),
    }


def _figure_manifest_rows(source_dir: Path, stage: str) -> list[dict[str, str]]:
    figures_dir = source_dir / "figures"
    if not figures_dir.exists():
        return []
    return [
        {
            "alias": source_dir.name,
            "stage": stage,
            "source_dir": str(source_dir),
            "figure": path.name,
        }
        for path in sorted(figures_dir.glob("*.png"))
    ]


def _thesis_output_dir(run_dirs: list[Path], raw_output_dir: str) -> Path:
    if raw_output_dir:
        return ensure_dir(Path(raw_output_dir).resolve())
    base_reports = run_dirs[0].resolve().parents[1] / "reports" if run_dirs else Path("artifacts/reports").resolve()
    return ensure_dir(base_reports / f"thesis-assets-{timestamp_tag()}")


def _handle_download(args: argparse.Namespace) -> int:
    dataset = get_dataset_info(args.dataset)
    dataset_dir = ensure_dir(Path(args.out_dir) / args.dataset)
    manifest = {
        "dataset_id": dataset["id"],
        "name": dataset["name"],
        "schema": dataset["schema"],
        "source_url": dataset["source_url"],
        "download_hint": dataset["download_hint"],
        "status": "manual_download_required",
    }
    save_json(dataset_dir / "download_manifest.json", manifest)
    return 0


def _handle_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    prepared_dir = Path(config["runtime"]["prepared_dir"])
    prepared = _prepare_from_config(config)
    _save_prepared_bundle(prepared_dir, prepared)
    _export_prepared_exploration_assets(prepared_dir, prepared)
    _print_completion_summary(
        "prepare",
        prepared_dir,
        (
            ("metadata.json", "prepared dataset metadata"),
            ("statistics.json", "dataset class counts and statistics"),
            ("scaler.json", "feature scaling parameters"),
            ("prepared_splits.npz", "prepared train/val/test arrays"),
            ("exploration_summary.md", "human-readable preparation summary"),
        ),
        next_command=f"uv run python -m emc_diag extract-features --config {args.config}",
    )
    return 0


def _handle_extract_features(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    prepared_dir = Path(config["runtime"]["prepared_dir"])
    prepared = _load_prepared_bundle(prepared_dir)
    feature_bundle = extract_feature_bundle(
        prepared,
        method=config["features"]["method"],
        top_k=config["features"].get("top_k"),
    )
    _save_feature_bundle(prepared_dir, feature_bundle)
    _export_feature_analysis_assets(prepared_dir, prepared, feature_bundle)
    _print_completion_summary(
        "extract-features",
        prepared_dir,
        (
            ("feature_metadata.json", "feature extraction metadata"),
            ("feature_splits.npz", "feature-engineered train/val/test arrays"),
            ("feature_importance.csv", "full feature importance table"),
            ("tables/selected_features.csv", "selected top feature list"),
            ("exploration_summary.md", "updated preparation summary with feature section"),
        ),
        next_command=f"uv run python -m emc_diag train --config {args.config} --device auto",
    )
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(
        (
            f"[train] mode={config['model'].get('mode', 'single_task')} "
            f"task={config['task']['task_name']} model={config['model']['name']} "
            f"device={args.device}"
        ),
        flush=True,
    )
    run_dir = _train_from_config(config, requested_device=args.device)
    print(f"[train] completed run_dir={run_dir}", flush=True)
    metrics = load_json(run_dir / "metrics.json")
    _print_completion_summary(
        "train",
        run_dir,
        (
            ("metrics.json", "model metrics and evaluation scores"),
            ("predictions.csv", "per-sample predictions"),
            ("run_config.yaml", "resolved run configuration"),
            ("tables/per_class_metrics.csv", "per-class metrics table"),
            ("tables/candidate_scores.csv", "candidate model comparison table"),
        ),
        metrics=metrics,
        next_command=f"uv run python -m emc_diag evaluate --run-dir {run_dir}",
    )
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run_config = _load_run_config(run_dir)
    dataset_metadata = load_json(run_dir / "dataset_metadata.json")
    existing = load_json(run_dir / "metrics.json")
    run_mode = str(existing.get("run_mode", "single_task"))

    predictions = pd.read_csv(run_dir / "predictions.csv") if (run_dir / "predictions.csv").exists() else None
    if predictions is not None:
        score_column = predictions["score"].to_numpy() if "score" in predictions.columns else None
        metrics = evaluate_predictions(
            predictions["y_true"].to_numpy(),
            predictions["y_pred"].to_numpy(),
            y_score=score_column,
            label_names=dataset_metadata.get("labels", []),
        )
    else:
        metrics = {}

    if run_mode == "multitask":
        task_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []
        refreshed_task_metrics: dict[str, Any] = {}
        for task in dataset_metadata.get("tasks", []):
            task_name = str(task.get("name") or task.get("task_name") or task.get("target_column"))
            prediction_path = run_dir / f"predictions_{task_name}.csv"
            if not prediction_path.exists():
                continue
            task_predictions = pd.read_csv(prediction_path)
            labeled_mask = task_predictions["is_labeled"].to_numpy(dtype=bool) if "is_labeled" in task_predictions.columns else np.ones(len(task_predictions), dtype=bool)
            task_label_names = list(task.get("labels", []))
            if bool(labeled_mask.any()):
                task_metrics = evaluate_predictions(
                    task_predictions.loc[labeled_mask, "y_true"].to_numpy(),
                    task_predictions.loc[labeled_mask, "y_pred"].to_numpy(),
                    label_names=task_label_names,
                )
            else:
                task_metrics = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "macro_f1": 0.0,
                    "confusion_matrix": [],
                    "per_class_metrics": {},
                    "labels": [],
                    "label_names": [],
                }
            refreshed_task_metrics[task_name] = {
                "metrics": task_metrics,
                "accuracy": float(task_metrics.get("accuracy", 0.0)),
                "f1": float(task_metrics.get("f1", 0.0)),
                "macro_f1": float(task_metrics.get("macro_f1", 0.0)),
                "valid_count": int(labeled_mask.sum()),
            }
            task_rows.append(
                {
                    "task_name": task_name,
                    "target_column": str(task.get("target_column", task_name)),
                    "accuracy": float(task_metrics.get("accuracy", 0.0)),
                    "f1": float(task_metrics.get("f1", 0.0)),
                    "macro_f1": float(task_metrics.get("macro_f1", 0.0)),
                    "valid_count": int(labeled_mask.sum()),
                }
            )
            comparison_rows.append(
                {
                    "task_name": task_name,
                    "mode": "multitask",
                    "accuracy": float(task_metrics.get("accuracy", 0.0)),
                    "f1": float(task_metrics.get("f1", 0.0)),
                    "macro_f1": float(task_metrics.get("macro_f1", 0.0)),
                }
            )

        if task_rows:
            save_dataframe(run_dir / "tables" / "per_task_metrics.csv", pd.DataFrame(task_rows))
            save_dataframe(run_dir / "tables" / "task_comparison.csv", pd.DataFrame(task_rows))
            save_dataframe(run_dir / "tables" / "multitask_vs_single_task.csv", pd.DataFrame(comparison_rows))
            primary_task_name = str(existing.get("task_name", dataset_metadata.get("primary_task_name", task_rows[0]["task_name"])))
            primary_metrics = refreshed_task_metrics.get(primary_task_name, refreshed_task_metrics[task_rows[0]["task_name"]])["metrics"]
            metrics = {**primary_metrics}
            metrics["task_metrics"] = refreshed_task_metrics
            metrics["mean_task_accuracy"] = float(np.mean([row["accuracy"] for row in task_rows]))
            metrics["mean_task_f1"] = float(np.mean([row["f1"] for row in task_rows]))

    for key in [
        "run_mode",
        "train_seconds",
        "resolved_device",
        "model_name",
        "selected_baseline",
        "candidate_scores",
        "train_history",
        "threshold",
        "task_name",
        "target_column",
        "leakage_columns_removed",
        "epochs_ran",
        "best_checkpoint",
        "risk_note",
        "sequence_layout",
        "model_family",
        "train_accuracy_score",
        "train_f1_score",
        "val_accuracy_score",
        "val_f1_score",
        "overfit_gap",
        "encoder_metadata",
        "transfer_from_run_dir",
        "transfer_strategy",
        "pretrain_history",
        "pretrain_objective",
        "mean_task_accuracy",
        "mean_task_f1",
    ]:
        if key in existing:
            metrics[key] = existing[key]
    save_json(run_dir / "metrics.json", _serializable_metrics(metrics))
    if "per_class_metrics" in metrics:
        save_dataframe(run_dir / "tables" / "per_class_metrics.csv", per_class_metrics_frame(metrics))
    if "minority_f1" in metrics:
        save_dataframe(
            run_dir / "tables" / "minority_class_metrics.csv",
            pd.DataFrame(
                [
                    {
                        "label": metrics.get("minority_label"),
                        "label_name": metrics.get("minority_label_name"),
                        "precision": metrics.get("minority_precision", 0.0),
                        "recall": metrics.get("minority_recall", 0.0),
                        "f1": metrics.get("minority_f1", 0.0),
                        "support": metrics.get("minority_support", 0),
                    }
                ]
            ),
        )
    _visualize_run(run_dir, theme_name=run_config.get("visualization", {}).get("theme", "paper-bar"))
    _print_completion_summary(
        "evaluate",
        run_dir,
        (
            ("metrics.json", "refreshed metrics and evaluation scores"),
            ("tables/per_class_metrics.csv", "per-class metrics table"),
            ("figures/confusion_matrix.png", "confusion matrix figure"),
            ("figures/per_class_metrics.png", "per-class metrics figure"),
            ("summary.md", "updated human-readable run summary"),
        ),
        metrics=metrics,
        next_command=f"uv run python -m emc_diag export-report --run-dir {run_dir} --format md",
    )
    return 0


def _handle_visualize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    _visualize_run(run_dir, theme_name=args.theme)
    metrics = load_json(run_dir / "metrics.json")
    _print_completion_summary(
        "visualize",
        run_dir,
        (
            ("figures/confusion_matrix.png", "confusion matrix figure"),
            ("figures/metrics_overview.png", "overall metrics figure"),
            ("figures/per_class_metrics.png", "per-class metrics figure"),
            ("summary.md", "human-readable run summary"),
        ),
        metrics=metrics,
        next_command=f"uv run python -m emc_diag export-report --run-dir {run_dir} --format md",
    )
    return 0


def _handle_export_report(args: argparse.Namespace) -> int:
    from emc_diag.reporting import (
        build_ablation_notes,
        build_overfitting_risk_note,
        build_run_summary,
        summarize_cv_aggregates,
    )

    run_dir = Path(args.run_dir)
    metrics = load_json(run_dir / "metrics.json")
    run_config = _load_run_config(run_dir)
    figures = sorted(path.name for path in (run_dir / "figures").glob("*.png"))
    tables = sorted(path.name for path in (run_dir / "tables").glob("*.csv"))
    task_metadata = {
        "task_name": metrics.get("task_name", "task"),
        "target_column": metrics.get("target_column", "label"),
        "selected_baseline": metrics.get("selected_baseline", metrics.get("model_name", "model")),
        "leakage_columns_removed": ", ".join(metrics.get("leakage_columns_removed", [])) or "none",
        "risk_note": metrics.get("risk_note", "none"),
        "sequence_layout": str(metrics.get("sequence_layout", "none")),
    }
    cv_aggregates = None
    if (run_dir / "tables" / "cv_metrics.csv").exists():
        cv_raw = pd.read_csv(run_dir / "tables" / "cv_metrics.csv")
        cv_summary = _cv_summary_frame(cv_raw, metrics.get("model_name", "model"))
        if not cv_summary.empty:
            cv_aggregates = summarize_cv_aggregates(cv_summary.to_dict(orient="records"))

    ablation_notes = None
    if (run_dir / "tables" / "ablation_metrics.csv").exists():
        ablation_raw = pd.read_csv(run_dir / "tables" / "ablation_metrics.csv")
        ablation_summary = _ablation_summary_frame(ablation_raw, metric_name=run_config["task"].get("metric_primary", "accuracy"))
        if not ablation_summary.empty:
            ablation_notes = build_ablation_notes(ablation_summary.to_dict(orient="records"))

    curve_df = _history_curve_frame(metrics, run_config)
    overfitting_risk_note = build_overfitting_risk_note(
        max_gap=float(pd.to_numeric(curve_df["train_score"] - curve_df["val_score"], errors="coerce").max())
        if {"train_score", "val_score"}.issubset(curve_df.columns)
        else float(metrics.get("overfit_gap", 0.0)),
        warning_threshold=float(run_config.get("evaluation", {}).get("risk_checks", {}).get("warning_threshold", 0.05)),
    )
    summary_context = _summary_context(run_dir, metrics, run_config)
    _call_with_supported_kwargs(
        build_run_summary,
        run_name=run_dir.name,
        output_dir=run_dir,
        metrics={key: value for key, value in metrics.items() if isinstance(value, (int, float, str))},
        figures=figures,
        tables=tables,
        task_metadata=task_metadata,
        run_metadata=summary_context["run_metadata"],
        transfer_notes=summary_context["transfer_notes"],
        pretrain_notes=summary_context["pretrain_notes"],
        feature_group_summary=summary_context["feature_group_summary"],
        robustness_notes=summary_context["robustness_notes"],
        cv_aggregates=cv_aggregates,
        ablation_notes=ablation_notes,
        overfitting_risk_note=overfitting_risk_note,
    )
    if args.format == "csv":
        save_dataframe(run_dir / "tables" / "metrics_summary.csv", pd.DataFrame([metrics]))
    _print_completion_summary(
        "export-report",
        run_dir,
        (
            ("summary.md", "human-readable run summary"),
            ("tables/metrics_summary.csv", "flat metrics table"),
        ),
        metrics=metrics,
        next_command="Open summary.md and figures/ for the main outputs.",
    )
    return 0


def _handle_quickstart(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    config = load_config(args.config)
    quickstart_config = _prepare_quickstart_config(config, output_root)

    run_dir = _run_pipeline_config(
        quickstart_config,
        requested_device=args.device,
        theme_name=quickstart_config.get("visualization", {}).get("theme", "paper-bar"),
    )
    prepared_work_dir = Path(quickstart_config["runtime"]["prepared_dir"])
    latest_prepared_dir = _curate_output_dir(
        prepared_work_dir,
        output_root / "latest_prepared",
        QUICKSTART_CORE_PREPARED_FILES,
    )
    latest_run_dir = _curate_output_dir(
        run_dir,
        output_root / "latest_run",
        QUICKSTART_CORE_RUN_FILES,
    )

    metrics = load_json(run_dir / "metrics.json")
    print(f"[quickstart] latest_prepared={latest_prepared_dir.resolve()}", flush=True)
    print(f"[quickstart] latest_run={latest_run_dir.resolve()}", flush=True)
    _print_completion_summary(
        "quickstart-prepared",
        latest_prepared_dir,
        QUICKSTART_CORE_PREPARED_FILES,
        next_command="Open latest_prepared/exploration_summary.md for dataset and feature prep outputs.",
    )
    _print_completion_summary(
        "quickstart",
        latest_run_dir,
        QUICKSTART_CORE_RUN_FILES,
        metrics=metrics,
        next_command="Open latest_run/summary.md first, then check latest_run/figures/.",
    )
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    from emc_diag.reporting import (
        build_benchmark_markdown_table,
        build_cross_dataset_benchmark_notes,
        build_run_summary,
        summarize_benchmark_highlights,
        summarize_cross_dataset_benchmark,
    )

    config_paths = [Path(item) for item in args.configs]
    benchmark_variants: list[tuple[str, dict[str, Any]]] = []
    for config_path in config_paths:
        benchmark_variants.extend(_expand_benchmark_variants(config_path))

    run_records: list[tuple[str, dict[str, Any], Path]] = []
    total_variants = len(benchmark_variants)
    for index, (variant_label, variant_config) in enumerate(benchmark_variants, start=1):
        print(f"[benchmark] {index}/{total_variants} {variant_label}", flush=True)
        run_dir = _run_pipeline_config(variant_config, requested_device=args.device, theme_name=args.theme)
        run_records.append((variant_label, variant_config, run_dir))

    artifacts_dir = Path(run_records[0][1]["runtime"]["artifacts_dir"]).resolve()
    benchmark_parent = artifacts_dir.parent / "benchmarks"
    benchmark_root = ensure_dir(benchmark_parent / f"benchmark-{timestamp_tag()}")
    benchmark_rows: list[dict[str, Any]] = []
    for variant_label, config, run_dir in run_records:
        metrics = load_json(run_dir / "metrics.json")
        benchmark_rows.append(
            {
                "config": variant_label,
                "run_dir": str(run_dir),
                "dataset": config["dataset"].get("name", "dataset"),
                "task_name": metrics.get("task_name", "task"),
                "target_column": metrics.get("target_column", "label"),
                "model_name": metrics.get("model_name", "model"),
                "selected_baseline": metrics.get("selected_baseline", metrics.get("model_name", "model")),
                "model_family": metrics.get("model_family", "ml"),
                "seed": config["trainer"].get("random_state", 42),
                "mode": metrics.get("run_mode", config["model"].get("mode", "single_task")),
                "transfer_strategy": metrics.get("transfer_strategy", ""),
                "pretrain_objective": metrics.get("pretrain_objective", ""),
                "accuracy": metrics.get("accuracy", 0.0),
                "f1": metrics.get("f1", 0.0),
                "macro_f1": metrics.get("macro_f1", metrics.get("f1", 0.0)),
                "minority_f1": metrics.get("minority_f1", 0.0),
                "minority_precision": metrics.get("minority_precision", 0.0),
                "minority_recall": metrics.get("minority_recall", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "threshold": metrics.get("threshold", ""),
                "risk_note": metrics.get("risk_note", ""),
            }
        )

    benchmark_df = pd.DataFrame(benchmark_rows)
    save_dataframe(benchmark_root / "benchmark_metrics.csv", benchmark_df)

    visualization_module = __import__("emc_diag.visualization", fromlist=["_placeholder"])
    plot_task_comparison = getattr(visualization_module, "plot_task_comparison", None)
    plot_ml_vs_cnn_comparison = getattr(visualization_module, "plot_ml_vs_cnn_comparison", None)
    plot_transfer_vs_scratch = getattr(visualization_module, "plot_transfer_vs_scratch", None)
    plot_multitask_vs_single_task = getattr(visualization_module, "plot_multitask_vs_single_task", None)
    if callable(plot_task_comparison):
        plot_task_comparison(benchmark_df, benchmark_root / "figures", args.theme)
    if callable(plot_ml_vs_cnn_comparison):
        plot_ml_vs_cnn_comparison(_ml_vs_cnn_summary_frame(benchmark_df), benchmark_root / "figures", args.theme)
    if callable(plot_transfer_vs_scratch):
        transfer_df = benchmark_df.copy()
        transfer_df["strategy"] = transfer_df["transfer_strategy"].fillna("").replace("", "scratch")
        if not transfer_df.empty and transfer_df["strategy"].nunique(dropna=True) > 1:
            save_dataframe(benchmark_root / "transfer_vs_scratch.csv", transfer_df)
            plot_transfer_vs_scratch(transfer_df, benchmark_root / "figures", args.theme)
    if callable(plot_multitask_vs_single_task) and "mode" in benchmark_df.columns:
        multi_df = benchmark_df[["task_name", "mode", "accuracy", "f1", "macro_f1"]].copy()
        if not multi_df.empty:
            save_dataframe(benchmark_root / "multitask_vs_single_task.csv", multi_df)
            plot_multitask_vs_single_task(multi_df, benchmark_root / "figures", args.theme)

    cross_dataset_summary = summarize_cross_dataset_benchmark(benchmark_df.to_dict(orient="records"))
    cross_dataset_notes = build_cross_dataset_benchmark_notes(benchmark_df.to_dict(orient="records"))
    benchmark_highlights = summarize_benchmark_highlights(benchmark_df.to_dict(orient="records"))
    benchmark_table = build_benchmark_markdown_table(benchmark_df.to_dict(orient="records"))
    build_run_summary(
        run_name=benchmark_root.name,
        output_dir=benchmark_root,
        metrics={
            "num_runs": len(benchmark_df),
            "best_accuracy": float(benchmark_df["accuracy"].max()) if not benchmark_df.empty else 0.0,
            "best_macro_f1": float(benchmark_df["macro_f1"].max()) if not benchmark_df.empty else 0.0,
        },
        figures=sorted(path.name for path in (benchmark_root / "figures").glob("*.png")),
        tables=["benchmark_metrics.csv"],
        task_metadata={"scope": "cross-config benchmark"},
        cross_dataset_benchmark_summary=cross_dataset_summary,
        cross_dataset_benchmark_notes=cross_dataset_notes,
        benchmark_highlights=benchmark_highlights,
        benchmark_table=benchmark_table,
    )
    print(f"[benchmark] completed benchmark_root={benchmark_root}", flush=True)
    return 0


def _handle_thesis_assets(args: argparse.Namespace) -> int:
    from emc_diag.reporting import (
        build_thesis_asset_summary,
        summarize_benchmark_highlights,
        summarize_cross_dataset_benchmark,
    )

    visualization_module = __import__("emc_diag.visualization", fromlist=["_placeholder"])
    plot_dataset_comparison = getattr(visualization_module, "plot_dataset_comparison", None)
    plot_task_comparison = getattr(visualization_module, "plot_task_comparison", None)
    plot_ml_vs_cnn_comparison = getattr(visualization_module, "plot_ml_vs_cnn_comparison", None)
    plot_transfer_vs_scratch = getattr(visualization_module, "plot_transfer_vs_scratch", None)
    plot_multitask_vs_single_task = getattr(visualization_module, "plot_multitask_vs_single_task", None)

    benchmark_frames, benchmark_dirs = _collect_benchmark_frames([Path(item) for item in args.benchmark_dirs])
    run_dirs = _collect_run_dirs([Path(item) for item in args.run_dirs], benchmark_frames)
    if not run_dirs:
        raise ValueError("thesis-assets requires at least one run directory or benchmark directory that references run_dir values")

    output_dir = _thesis_output_dir(run_dirs, args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    final_metrics_df = pd.DataFrame([_run_metric_row(run_dir) for run_dir in run_dirs]).sort_values(
        ["dataset", "task_name", "primary_score", "secondary_score"],
        ascending=[True, True, False, False],
    )
    save_dataframe(output_dir / "final_metrics.csv", final_metrics_df)

    benchmark_df = pd.concat(benchmark_frames, ignore_index=True) if benchmark_frames else pd.DataFrame()
    if not benchmark_df.empty:
        save_dataframe(output_dir / "benchmark_metrics_merged.csv", benchmark_df)

    dataset_best_df = (
        final_metrics_df.sort_values(["dataset", "primary_score", "secondary_score"], ascending=[True, False, False])
        .groupby("dataset", as_index=False)
        .first()[["dataset", "primary_score", "primary_metric", "model_name", "task_name"]]
    )
    task_best_df = (
        final_metrics_df.sort_values(["task_name", "primary_score", "secondary_score"], ascending=[True, False, False])
        .groupby("task_name", as_index=False)
        .first()[["task_name", "accuracy", "macro_f1", "f1", "dataset", "model_name"]]
    )
    family_df = _ml_vs_cnn_summary_frame(final_metrics_df)

    if callable(plot_dataset_comparison) and not dataset_best_df.empty:
        plot_dataset_comparison(dataset_best_df, figures_dir, metric="primary_score")
    if callable(plot_task_comparison) and not task_best_df.empty:
        plot_task_comparison(task_best_df, figures_dir)
    if callable(plot_ml_vs_cnn_comparison) and not family_df.empty:
        plot_ml_vs_cnn_comparison(family_df, figures_dir)
    if callable(plot_transfer_vs_scratch) and not benchmark_df.empty and "transfer_strategy" in benchmark_df.columns:
        transfer_df = benchmark_df.copy()
        transfer_df["strategy"] = transfer_df["transfer_strategy"].fillna("").replace("", "scratch")
        if not transfer_df.empty and transfer_df["strategy"].nunique(dropna=True) > 1:
            plot_transfer_vs_scratch(transfer_df, figures_dir)
    if callable(plot_multitask_vs_single_task) and not benchmark_df.empty and "mode" in benchmark_df.columns:
        multitask_df = benchmark_df[["task_name", "mode", "accuracy", "f1", "macro_f1"]].copy()
        if not multitask_df.empty:
            plot_multitask_vs_single_task(multitask_df, figures_dir)

    save_dataframe(tables_dir / "dataset_best_models.csv", dataset_best_df)
    save_dataframe(tables_dir / "task_best_models.csv", task_best_df)
    if not family_df.empty:
        save_dataframe(tables_dir / "model_family_summary.csv", family_df)

    prepared_dirs = sorted(
        {
            Path(item).resolve()
            for item in final_metrics_df["prepared_dir"].dropna().astype(str).tolist()
            if item
        }
    )
    manifest_rows: list[dict[str, str]] = []
    for prepared_dir in prepared_dirs:
        manifest_rows.extend(_figure_manifest_rows(prepared_dir, stage="prepared"))
    for run_dir in run_dirs:
        manifest_rows.extend(_figure_manifest_rows(run_dir, stage="run"))
    for benchmark_dir in benchmark_dirs:
        manifest_rows.extend(_figure_manifest_rows(benchmark_dir, stage="benchmark"))
    manifest_rows.extend(_figure_manifest_rows(output_dir, stage="report"))
    thesis_manifest_df = pd.DataFrame(manifest_rows).drop_duplicates()
    save_dataframe(output_dir / "thesis_figures_manifest.csv", thesis_manifest_df)

    best_overall = final_metrics_df.sort_values(["primary_score", "secondary_score"], ascending=False).iloc[0]
    key_findings = [
        (
            f"Best overall primary score: {best_overall['dataset']} / {best_overall['task_name']} / "
            f"{best_overall['model_name']} = {float(best_overall['primary_score']):.4f}"
        )
    ]
    for _, row in dataset_best_df.iterrows():
        key_findings.append(
            (
                f"{row['dataset']} best: {row['model_name']} on {row['task_name']} "
                f"({row['primary_metric']}={float(row['primary_score']):.4f})"
            )
        )
    if not benchmark_df.empty:
        key_findings.extend(summarize_benchmark_highlights(benchmark_df.to_dict(orient="records")))

    dataset_notes = [
        f"Datasets represented: {', '.join(sorted(final_metrics_df['dataset'].dropna().astype(str).unique().tolist()))}",
        f"Run records collected: {len(final_metrics_df)}",
        f"Prepared exploration directories collected: {len(prepared_dirs)}",
        f"Benchmark directories collected: {len(benchmark_dirs)}",
        f"Figure manifest entries: {len(thesis_manifest_df)}",
    ]
    if not benchmark_df.empty:
        cross_dataset_summary = summarize_cross_dataset_benchmark(benchmark_df.to_dict(orient="records"))
        dataset_notes.append(
            (
                f"Cross-dataset benchmark: best_dataset={cross_dataset_summary['best_dataset']}, "
                f"best_accuracy={float(cross_dataset_summary['best_accuracy']):.4f}, "
                f"mean_accuracy={float(cross_dataset_summary['mean_accuracy']):.4f}"
            )
        )

    build_thesis_asset_summary(
        output_dir=output_dir,
        title=args.title,
        overview_metrics={
            "num_runs": len(final_metrics_df),
            "num_datasets": int(final_metrics_df["dataset"].nunique()),
            "num_benchmarks": len(benchmark_dirs),
            "best_primary_score": float(best_overall["primary_score"]),
            "best_secondary_score": float(best_overall["secondary_score"]),
        },
        key_findings=key_findings,
        dataset_notes=dataset_notes,
        output_files=sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    )
    print(f"[thesis-assets] completed output_dir={output_dir}", flush=True)
    return 0


COMMAND_HANDLERS = {
    "download": _handle_download,
    "prepare": _handle_prepare,
    "extract-features": _handle_extract_features,
    "train": _handle_train,
    "evaluate": _handle_evaluate,
    "visualize": _handle_visualize,
    "export-report": _handle_export_report,
    "quickstart": _handle_quickstart,
    "benchmark": _handle_benchmark,
    "thesis-assets": _handle_thesis_assets,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMAND_HANDLERS[args.command](args)
    except Exception as exc:  # pragma: no cover
        print(f"emc_diag error: {exc}", file=sys.stderr)
        return 1

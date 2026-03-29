from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THESIS_FIGURE_STEMS: tuple[str, ...] = (
    "dataset_summary",
    "waveform_overview",
    "spectrum_stft_summary",
    "feature_importance",
    "training_curves",
    "overfitting_gap",
    "confusion_matrix",
    "metrics_overview",
    "candidate_comparison",
    "cv_comparison",
    "ablation_comparison",
    "ml_vs_cnn_comparison",
    "per_class_metrics",
    "task_comparison",
    "noise_robustness",
    "feature_group_ablation",
    "transfer_vs_scratch",
    "multitask_vs_single_task",
)


def paper_bar_theme() -> dict[str, Any]:
    return {
        "palette": ["#5C78BC", "#E4B295", "#D7B0C0", "#B8C6DD", "#9FB7A8"],
        "font_family": "DejaVu Sans",
        "dpi": 180,
        "title_size": 14,
        "label_size": 11,
        "tick_size": 10,
        "line_width": 1.4,
    }


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_theme(theme_name: str) -> dict[str, Any]:
    if theme_name != "paper-bar":
        raise ValueError(f"Unsupported theme: {theme_name}")
    return paper_bar_theme()


def _save_artifacts(
    fig: plt.Figure, data: pd.DataFrame, output_dir: Path, stem: str
) -> dict[str, Path]:
    _ensure_output_dir(output_dir)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    csv = output_dir / f"{stem}.csv"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    data.to_csv(csv, index=False)
    plt.close(fig)
    return {"png": png, "svg": svg, "csv": csv}


def expected_thesis_figure_stems() -> tuple[str, ...]:
    return THESIS_FIGURE_STEMS


def has_thesis_figure_stems(stems: list[str] | set[str] | tuple[str, ...]) -> bool:
    observed = set(stems)
    return all(stem in observed for stem in THESIS_FIGURE_STEMS)


def plot_dataset_summary(
    dataset_df: pd.DataFrame, output_dir: Path, theme_name: str = "paper-bar"
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    counts = (
        dataset_df["class"].value_counts().rename_axis("class").reset_index(name="count")
    )
    colors = theme["palette"][: len(counts)]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=theme["dpi"])
    bars = ax.bar(counts["class"], counts["count"], color=colors, edgecolor="#4F5562")
    for bar, value in zip(bars, counts["count"].tolist(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"],
        )
    ax.set_title("Dataset Class Summary", fontsize=theme["title_size"])
    ax.set_xlabel("Class", fontsize=theme["label_size"])
    ax.set_ylabel("Count", fontsize=theme["label_size"])
    return _save_artifacts(fig, counts, output_dir, "dataset_summary")


def plot_feature_importance(
    feature_df: pd.DataFrame, output_dir: Path, theme_name: str = "paper-bar"
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    ordered = feature_df.sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=theme["dpi"])
    colors = [theme["palette"][i % len(theme["palette"])] for i in range(len(ordered))]
    bars = ax.bar(ordered["feature"], ordered["importance"], color=colors, edgecolor="#4F5562")
    for bar, value in zip(bars, ordered["importance"].tolist(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"],
        )
    ax.set_title("Feature Importance", fontsize=theme["title_size"])
    ax.set_xlabel("Feature", fontsize=theme["label_size"])
    ax.set_ylabel("Importance", fontsize=theme["label_size"])
    ax.tick_params(axis="x", labelrotation=30)
    return _save_artifacts(fig, ordered, output_dir, "feature_importance")


def plot_training_curves(
    curve_df: pd.DataFrame, output_dir: Path, theme_name: str = "paper-bar"
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=theme["dpi"])

    axes[0].plot(
        curve_df["epoch"],
        curve_df["train_loss"],
        label="train_loss",
        color=theme["palette"][0],
        linewidth=theme["line_width"],
    )
    axes[0].plot(
        curve_df["epoch"],
        curve_df["val_loss"],
        label="val_loss",
        color=theme["palette"][1],
        linewidth=theme["line_width"],
    )
    axes[0].set_title("Loss Curves", fontsize=theme["title_size"])
    axes[0].set_xlabel("Epoch", fontsize=theme["label_size"])
    axes[0].set_ylabel("Loss", fontsize=theme["label_size"])
    axes[0].legend()

    axes[1].plot(
        curve_df["epoch"],
        curve_df["accuracy"],
        label="accuracy",
        color=theme["palette"][2],
        linewidth=theme["line_width"],
    )
    axes[1].plot(
        curve_df["epoch"],
        curve_df["f1"],
        label="f1",
        color=theme["palette"][3],
        linewidth=theme["line_width"],
    )
    axes[1].set_title("Metric Curves", fontsize=theme["title_size"])
    axes[1].set_xlabel("Epoch", fontsize=theme["label_size"])
    axes[1].set_ylabel("Score", fontsize=theme["label_size"])
    axes[1].legend()

    return _save_artifacts(fig, curve_df.copy(), output_dir, "training_curves")


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_dir: Path,
    theme_name: str = "paper-bar",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=theme["dpi"])
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticklabels(labels)
    ax.set_title("Confusion Matrix", fontsize=theme["title_size"])
    ax.set_xlabel("Predicted", fontsize=theme["label_size"])
    ax.set_ylabel("Actual", fontsize=theme["label_size"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels).reset_index(names="actual")
    return _save_artifacts(fig, cm_df, output_dir, "confusion_matrix")


def plot_metrics_bar(
    metrics_df: pd.DataFrame, output_dir: Path, theme_name: str = "paper-bar"
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    data = metrics_df.copy()
    primary_metric = "accuracy" if "accuracy" in data.columns else data.columns[1]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=theme["dpi"])
    bars = ax.bar(
        data["model"],
        data[primary_metric],
        color=theme["palette"][: len(data)],
        edgecolor="#4F5562",
    )
    for bar, value in zip(bars, data[primary_metric].tolist(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1%}" if value <= 1.0 else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"],
        )
    ax.set_title("Model Performance Overview", fontsize=theme["title_size"])
    ax.set_xlabel("Model", fontsize=theme["label_size"])
    ax.set_ylabel(primary_metric.capitalize(), fontsize=theme["label_size"])
    ax.tick_params(axis="x", labelrotation=30)
    return _save_artifacts(fig, data, output_dir, "metrics_overview")


def plot_candidate_comparison(
    candidate_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    data = candidate_df.copy()
    primary_metric = "accuracy" if "accuracy" in data.columns else data.columns[1]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=theme["dpi"])
    colors = [theme["palette"][i % len(theme["palette"])] for i in range(len(data))]
    bars = ax.bar(data["model"], data[primary_metric], color=colors, edgecolor="#4F5562")
    for bar, value in zip(bars, data[primary_metric].tolist(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"],
        )
    ax.set_title("Candidate Model Comparison", fontsize=theme["title_size"])
    ax.set_xlabel("Candidate", fontsize=theme["label_size"])
    ax.set_ylabel(primary_metric.capitalize(), fontsize=theme["label_size"])
    ax.tick_params(axis="x", labelrotation=30)
    return _save_artifacts(fig, data, output_dir, "candidate_comparison")


def plot_per_class_metrics(
    per_class_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    data = per_class_df.copy()
    label_col = "label_name" if "label_name" in data.columns else "label"
    melted = data[[label_col, "precision", "recall", "f1"]].melt(
        id_vars=[label_col],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    pivot = melted.pivot(index=label_col, columns="metric", values="score").reset_index()

    x = np.arange(len(pivot))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=theme["dpi"])
    for offset, metric in enumerate(["precision", "recall", "f1"]):
        values = pivot[metric].to_numpy(dtype=float)
        bars = ax.bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=metric,
            color=theme["palette"][offset],
            edgecolor="#4F5562",
        )
        for bar, value in zip(bars, values.tolist(), strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=theme["tick_size"] - 1,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot[label_col].tolist(), rotation=25)
    ax.set_ylim(0.0, max(1.0, float(melted["score"].max()) + 0.1))
    ax.set_title("Per-Class Metrics", fontsize=theme["title_size"])
    ax.set_xlabel("Class", fontsize=theme["label_size"])
    ax.set_ylabel("Score", fontsize=theme["label_size"])
    ax.legend()
    return _save_artifacts(fig, melted, output_dir, "per_class_metrics")


def plot_binary_curves(
    roc_df: pd.DataFrame,
    pr_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=theme["dpi"])
    axes[0].plot(roc_df["fpr"], roc_df["tpr"], color=theme["palette"][0], linewidth=theme["line_width"])
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.0)
    axes[0].set_title("ROC Curve", fontsize=theme["title_size"])
    axes[0].set_xlabel("False Positive Rate", fontsize=theme["label_size"])
    axes[0].set_ylabel("True Positive Rate", fontsize=theme["label_size"])

    axes[1].plot(pr_df["recall"], pr_df["precision"], color=theme["palette"][1], linewidth=theme["line_width"])
    axes[1].set_title("PR Curve", fontsize=theme["title_size"])
    axes[1].set_xlabel("Recall", fontsize=theme["label_size"])
    axes[1].set_ylabel("Precision", fontsize=theme["label_size"])

    combined = roc_df.copy()
    for column in pr_df.columns:
        combined[f"pr_{column}"] = pr_df[column]
    return _save_artifacts(fig, combined, output_dir, "binary_curves")


def plot_task_comparison(
    benchmark_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    data = benchmark_df.copy()
    primary_metric = "macro_f1" if "macro_f1" in data.columns else "accuracy"
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=theme["dpi"])
    colors = [theme["palette"][i % len(theme["palette"])] for i in range(len(data))]
    bars = ax.bar(data["task_name"], data[primary_metric], color=colors, edgecolor="#4F5562")
    for bar, value in zip(bars, data[primary_metric].tolist(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"],
        )
    ax.set_title("Cross-Task Performance Comparison", fontsize=theme["title_size"])
    ax.set_xlabel("Task", fontsize=theme["label_size"])
    ax.set_ylabel(primary_metric.upper(), fontsize=theme["label_size"])
    ax.tick_params(axis="x", labelrotation=25)
    return _save_artifacts(fig, data, output_dir, "task_comparison")


def plot_dataset_comparison(
    dataset_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
    metric: str = "primary_score",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    data = dataset_df.copy()
    if "dataset" not in data.columns:
        raise ValueError("dataset_df must include 'dataset'")
    metric_col = metric if metric in data.columns else next(
        (column for column in ["primary_score", "macro_f1", "f1", "accuracy"] if column in data.columns),
        None,
    )
    if metric_col is None:
        raise ValueError("dataset_df must include a score column such as 'primary_score', 'macro_f1', 'f1', or 'accuracy'")

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=theme["dpi"])
    colors = [theme["palette"][i % len(theme["palette"])] for i in range(len(data))]
    bars = ax.bar(data["dataset"], data[metric_col], color=colors, edgecolor="#4F5562")
    for bar, value in zip(bars, data[metric_col].tolist(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{float(value):.3f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"],
        )
    ax.set_title("Dataset Comparison", fontsize=theme["title_size"])
    ax.set_xlabel("Dataset", fontsize=theme["label_size"])
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=theme["label_size"])
    ax.tick_params(axis="x", labelrotation=25)
    return _save_artifacts(fig, data, output_dir, "dataset_comparison")


def plot_noise_robustness_curve(
    robustness_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
    metric: str = "macro_f1",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    data = robustness_df.copy()
    if "noise_sigma" not in data.columns:
        raise ValueError("robustness_df must include 'noise_sigma'")
    metric_col = metric if metric in data.columns else next(
        (col for col in data.columns if col != "noise_sigma"), None
    )
    if metric_col is None:
        raise ValueError("robustness_df must include at least one metric column")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=theme["dpi"])
    ax.plot(
        data["noise_sigma"],
        data[metric_col],
        marker="o",
        linewidth=theme["line_width"],
        color=theme["palette"][1],
    )
    ax.set_title("Noise Robustness Curve", fontsize=theme["title_size"])
    ax.set_xlabel("Noise Sigma", fontsize=theme["label_size"])
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=theme["label_size"])
    ax.grid(True, alpha=0.3)

    export_df = data[["noise_sigma", metric_col]].copy()
    return _save_artifacts(fig, export_df, output_dir, "noise_robustness")


def plot_feature_group_ablation(
    ablation_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    data = ablation_df.copy()
    if "variant" not in data.columns or "score" not in data.columns:
        raise ValueError("ablation_df must include 'variant' and 'score'")

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=theme["dpi"])
    colors = [theme["palette"][i % len(theme["palette"])] for i in range(len(data))]
    bars = ax.bar(data["variant"], data["score"], color=colors, edgecolor="#4F5562")
    for bar, value in zip(bars, data["score"].tolist(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"],
        )
    ax.set_title("Feature Group Ablation", fontsize=theme["title_size"])
    ax.set_xlabel("Variant", fontsize=theme["label_size"])
    ax.set_ylabel("Score", fontsize=theme["label_size"])
    ax.tick_params(axis="x", labelrotation=25)
    return _save_artifacts(fig, data[["variant", "score"]].copy(), output_dir, "feature_group_ablation")


def plot_transfer_vs_scratch(
    transfer_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
    metric: str = "macro_f1",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    data = transfer_df.copy()
    if metric not in data.columns:
        raise ValueError(f"transfer_df must include '{metric}' column")
    if "strategy" not in data.columns or "model_name" not in data.columns:
        raise ValueError("transfer_df must include 'strategy' and 'model_name'")
    data["strategy"] = data["strategy"].fillna("").replace("", "scratch")

    grouped = data.groupby(["model_name", "strategy"])[metric].mean().reset_index()
    pivot = grouped.pivot(index="model_name", columns="strategy", values=metric).fillna(0.0)
    pivot_cols = list(pivot.columns)
    x = np.arange(len(pivot))
    width = 0.8 / max(1, len(pivot_cols))

    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=theme["dpi"])
    for idx, strategy in enumerate(pivot_cols):
        values = pivot[strategy].to_numpy(dtype=float)
        bars = ax.bar(
            x + idx * width,
            values,
            width=width,
            label=strategy,
            color=theme["palette"][idx % len(theme["palette"])],
            edgecolor="#4F5562",
        )
        for bar, value in zip(bars, values.tolist(), strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=theme["tick_size"] - 1,
            )
    ax.set_xticks(x + width * (len(pivot_cols) - 1) / 2.0)
    ax.set_xticklabels(pivot.index.tolist(), rotation=25)
    ax.set_title("Transfer vs Scratch Comparison", fontsize=theme["title_size"])
    ax.set_xlabel("Model", fontsize=theme["label_size"])
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=theme["label_size"])
    if pivot_cols:
        ax.legend()

    export_df = grouped.copy()
    return _save_artifacts(fig, export_df, output_dir, "transfer_vs_scratch")


def plot_multitask_vs_single_task(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
    metric: str = "macro_f1",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    data = comparison_df.copy()
    if metric not in data.columns:
        raise ValueError(f"comparison_df must include '{metric}' column")
    if "mode" not in data.columns:
        raise ValueError("comparison_df must include 'mode'")

    task_col = "task_name" if "task_name" in data.columns else "task"
    data[task_col] = data[task_col].fillna("task")
    grouped = data.groupby([task_col, "mode"])[metric].mean().reset_index()
    pivot = grouped.pivot(index=task_col, columns="mode", values=metric).fillna(0.0)
    modes = list(pivot.columns)
    x = np.arange(len(pivot))
    width = 0.8 / max(1, len(modes))

    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=theme["dpi"])
    for idx, mode in enumerate(modes):
        values = pivot[mode].to_numpy(dtype=float)
        bars = ax.bar(
            x + idx * width,
            values,
            width=width,
            label=mode,
            color=theme["palette"][idx % len(theme["palette"])],
            edgecolor="#4F5562",
        )
        for bar, value in zip(bars, values.tolist(), strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=theme["tick_size"] - 1,
            )
    ax.set_xticks(x + width * (len(modes) - 1) / 2.0)
    ax.set_xticklabels(pivot.index.tolist(), rotation=25)
    ax.set_title("Multitask vs Single Task", fontsize=theme["title_size"])
    ax.set_xlabel("Task", fontsize=theme["label_size"])
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=theme["label_size"])
    if modes:
        ax.legend()

    export_df = grouped.copy()
    return _save_artifacts(fig, export_df, output_dir, "multitask_vs_single_task")


def plot_waveform_overview(
    waveform_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
    signal_col: str = "signal",
    index_col: str = "sample_index",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    if signal_col not in waveform_df.columns:
        raise ValueError(f"waveform_df must contain '{signal_col}' column")

    data = waveform_df.copy()
    if index_col not in data.columns:
        data[index_col] = np.arange(len(data))
    data = data[[index_col, signal_col]].copy()
    data["abs_signal"] = data[signal_col].abs()

    signal_values = data[signal_col].to_numpy(dtype=float)
    rms = float(np.sqrt(np.mean(np.square(signal_values)))) if signal_values.size else 0.0
    peak_to_peak = (
        float(np.max(signal_values) - np.min(signal_values)) if signal_values.size else 0.0
    )
    max_abs = float(np.max(np.abs(signal_values))) if signal_values.size else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=theme["dpi"])
    axes[0].plot(
        data[index_col],
        data[signal_col],
        color=theme["palette"][0],
        linewidth=theme["line_width"],
    )
    axes[0].set_title("Waveform Overview", fontsize=theme["title_size"])
    axes[0].set_xlabel("Sample Index", fontsize=theme["label_size"])
    axes[0].set_ylabel("Amplitude", fontsize=theme["label_size"])

    stats_df = pd.DataFrame(
        {"metric": ["rms", "peak_to_peak", "max_abs"], "value": [rms, peak_to_peak, max_abs]}
    )
    bars = axes[1].bar(
        stats_df["metric"],
        stats_df["value"],
        color=[theme["palette"][1], theme["palette"][2], theme["palette"][3]],
        edgecolor="#4F5562",
    )
    for bar, value in zip(bars, stats_df["value"].tolist(), strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"] - 1,
        )
    axes[1].set_title("Waveform Risk Statistics", fontsize=theme["title_size"])
    axes[1].set_ylabel("Value", fontsize=theme["label_size"])

    export_df = data.copy()
    export_df["rms"] = rms
    export_df["peak_to_peak"] = peak_to_peak
    export_df["max_abs"] = max_abs
    return _save_artifacts(fig, export_df, output_dir, "waveform_overview")


def plot_spectrum_stft_summary(
    waveform_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
    signal_col: str = "signal",
    nperseg: int = 64,
    noverlap: int = 32,
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    if signal_col not in waveform_df.columns:
        raise ValueError(f"waveform_df must contain '{signal_col}' column")
    if nperseg <= 0:
        raise ValueError("nperseg must be positive")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")

    signal = waveform_df[signal_col].to_numpy(dtype=float)
    if signal.size == 0:
        raise ValueError("waveform_df must include at least one signal sample")

    window = np.hanning(nperseg)
    step = nperseg - noverlap
    frames: list[np.ndarray] = []
    start = 0
    while start < signal.size:
        segment = signal[start : start + nperseg]
        if segment.size < nperseg:
            padded = np.zeros(nperseg, dtype=float)
            padded[: segment.size] = segment
            segment = padded
        frames.append(segment)
        if start + nperseg >= signal.size:
            break
        start += step

    stft_matrix = np.vstack([np.abs(np.fft.rfft(frame * window)) for frame in frames]).T
    freq_axis = np.fft.rfftfreq(nperseg, d=1.0)
    avg_spectrum = stft_matrix.mean(axis=1)
    dominant_idx = int(np.argmax(avg_spectrum))
    dominant_freq = float(freq_axis[dominant_idx])
    dominant_mag = float(avg_spectrum[dominant_idx])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=theme["dpi"])
    axes[0].plot(
        freq_axis,
        avg_spectrum,
        color=theme["palette"][0],
        linewidth=theme["line_width"],
    )
    axes[0].scatter([dominant_freq], [dominant_mag], color=theme["palette"][1], zorder=3)
    axes[0].text(
        dominant_freq,
        dominant_mag,
        f"peak f={dominant_freq:.3f}",
        fontsize=theme["tick_size"] - 1,
        ha="left",
        va="bottom",
    )
    axes[0].set_title("Average Spectrum", fontsize=theme["title_size"])
    axes[0].set_xlabel("Normalized Frequency", fontsize=theme["label_size"])
    axes[0].set_ylabel("Magnitude", fontsize=theme["label_size"])

    image = axes[1].imshow(
        stft_matrix,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    fig.colorbar(image, ax=axes[1])
    axes[1].set_title("STFT Summary", fontsize=theme["title_size"])
    axes[1].set_xlabel("Frame", fontsize=theme["label_size"])
    axes[1].set_ylabel("Frequency Bin", fontsize=theme["label_size"])

    stft_rows: list[dict[str, float | int]] = []
    for frame_index in range(stft_matrix.shape[1]):
        for freq_index in range(stft_matrix.shape[0]):
            stft_rows.append(
                {
                    "frame_index": frame_index,
                    "freq_bin": float(freq_axis[freq_index]),
                    "magnitude": float(stft_matrix[freq_index, frame_index]),
                }
            )
    export_df = pd.DataFrame(stft_rows)
    export_df["dominant_freq"] = dominant_freq
    export_df["dominant_magnitude"] = dominant_mag
    return _save_artifacts(fig, export_df, output_dir, "spectrum_stft_summary")


def plot_overfitting_gap(
    curve_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
    train_col: str = "train_score",
    val_col: str = "val_score",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    if train_col not in curve_df.columns or val_col not in curve_df.columns:
        raise ValueError(f"curve_df must contain '{train_col}' and '{val_col}' columns")

    data = curve_df.copy()
    if "epoch" not in data.columns:
        data["epoch"] = np.arange(1, len(data) + 1)
    data["overfitting_gap"] = data[train_col] - data[val_col]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=theme["dpi"])
    axes[0].plot(
        data["epoch"],
        data[train_col],
        label=train_col,
        color=theme["palette"][0],
        linewidth=theme["line_width"],
    )
    axes[0].plot(
        data["epoch"],
        data[val_col],
        label=val_col,
        color=theme["palette"][1],
        linewidth=theme["line_width"],
    )
    axes[0].set_title("Learning Curves", fontsize=theme["title_size"])
    axes[0].set_xlabel("Epoch", fontsize=theme["label_size"])
    axes[0].set_ylabel("Score", fontsize=theme["label_size"])
    axes[0].legend()

    bars = axes[1].bar(
        data["epoch"].astype(str),
        data["overfitting_gap"],
        color=theme["palette"][2],
        edgecolor="#4F5562",
    )
    for bar, value in zip(bars, data["overfitting_gap"].tolist(), strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"] - 1,
        )
    axes[1].set_title("Overfitting Gap", fontsize=theme["title_size"])
    axes[1].set_xlabel("Epoch", fontsize=theme["label_size"])
    axes[1].set_ylabel("Train - Val", fontsize=theme["label_size"])
    axes[1].axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    return _save_artifacts(fig, data, output_dir, "overfitting_gap")


def plot_cv_comparison(
    cv_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    required = {"model", "cv_mean", "cv_std"}
    if not required.issubset(cv_df.columns):
        raise ValueError(f"cv_df must contain columns: {sorted(required)}")

    data = cv_df.copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=theme["dpi"])
    colors = [theme["palette"][i % len(theme["palette"])] for i in range(len(data))]
    bars = ax.bar(
        data["model"],
        data["cv_mean"],
        yerr=data["cv_std"],
        capsize=5,
        color=colors,
        edgecolor="#4F5562",
        ecolor="#3A3A3A",
    )
    for bar, mean_value, std_value in zip(
        bars, data["cv_mean"].tolist(), data["cv_std"].tolist(), strict=False
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{mean_value:.3f} +/- {std_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"] - 1,
        )
    ax.set_title("Cross-Validation Comparison", fontsize=theme["title_size"])
    ax.set_xlabel("Model", fontsize=theme["label_size"])
    ax.set_ylabel("CV Mean Score", fontsize=theme["label_size"])
    ax.tick_params(axis="x", labelrotation=25)
    return _save_artifacts(fig, data, output_dir, "cv_comparison")


def plot_ablation_comparison(
    ablation_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    required = {"variant", "score"}
    if not required.issubset(ablation_df.columns):
        raise ValueError(f"ablation_df must contain columns: {sorted(required)}")

    data = ablation_df.copy()
    baseline_score = float(data["score"].iloc[0])
    data["delta_vs_baseline"] = data["score"] - baseline_score

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=theme["dpi"])
    colors = [theme["palette"][i % len(theme["palette"])] for i in range(len(data))]
    bars = ax.bar(data["variant"], data["score"], color=colors, edgecolor="#4F5562")
    for bar, value, delta in zip(
        bars, data["score"].tolist(), data["delta_vs_baseline"].tolist(), strict=False
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f} ({delta:+.3f})",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"] - 1,
        )
    ax.set_title("Ablation Comparison", fontsize=theme["title_size"])
    ax.set_xlabel("Variant", fontsize=theme["label_size"])
    ax.set_ylabel("Score", fontsize=theme["label_size"])
    ax.tick_params(axis="x", labelrotation=25)
    return _save_artifacts(fig, data, output_dir, "ablation_comparison")


def plot_ml_vs_cnn_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    theme_name: str = "paper-bar",
) -> dict[str, Path]:
    theme = _resolve_theme(theme_name)
    required = {"family", "accuracy", "f1"}
    if not required.issubset(comparison_df.columns):
        raise ValueError(f"comparison_df must contain columns: {sorted(required)}")

    data = comparison_df.copy()
    x = np.arange(len(data))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=theme["dpi"])
    bars_accuracy = ax.bar(
        x - width / 2,
        data["accuracy"],
        width=width,
        label="accuracy",
        color=theme["palette"][0],
        edgecolor="#4F5562",
    )
    bars_f1 = ax.bar(
        x + width / 2,
        data["f1"],
        width=width,
        label="f1",
        color=theme["palette"][1],
        edgecolor="#4F5562",
    )
    for bar, value in zip(bars_accuracy, data["accuracy"].tolist(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"] - 1,
        )
    for bar, value in zip(bars_f1, data["f1"].tolist(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=theme["tick_size"] - 1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(data["family"].tolist())
    ax.set_title("ML vs CNN Comparison", fontsize=theme["title_size"])
    ax.set_xlabel("Model Family", fontsize=theme["label_size"])
    ax.set_ylabel("Score", fontsize=theme["label_size"])
    ax.legend()
    return _save_artifacts(fig, data, output_dir, "ml_vs_cnn_comparison")

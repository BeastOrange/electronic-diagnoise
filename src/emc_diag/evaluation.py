from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _auc_from_curve(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if x_values.size < 2 or y_values.size < 2:
        return 0.0
    order = np.argsort(x_values)
    return float(np.trapezoid(y_values[order], x_values[order]))


def _binary_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    thresholds = np.unique(np.concatenate(([0.0], y_score, [1.0])))
    roc_points: list[dict[str, float]] = []
    pr_points: list[dict[str, float]] = []

    positives = float(np.sum(y_true == 1))
    negatives = float(np.sum(y_true == 0))
    for threshold in thresholds[::-1]:
        y_pred = (y_score >= threshold).astype(np.int64)
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        tn = float(np.sum((y_true == 0) & (y_pred == 0)))

        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn) if negatives else 0.0
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn) if positives else 0.0
        roc_points.append({"threshold": float(threshold), "fpr": fpr, "tpr": tpr})
        pr_points.append({"threshold": float(threshold), "precision": precision, "recall": recall})

    roc_df = pd.DataFrame(roc_points).sort_values("fpr")
    pr_df = pd.DataFrame(pr_points).sort_values("recall")
    return {
        "roc_curve": roc_df.to_dict(orient="records"),
        "pr_curve": pr_df.to_dict(orient="records"),
        "roc_auc": _auc_from_curve(roc_df["fpr"].to_numpy(), roc_df["tpr"].to_numpy()),
        "pr_auc": _auc_from_curve(pr_df["recall"].to_numpy(), pr_df["precision"].to_numpy()),
    }


def _build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_index = {int(label): idx for idx, label in enumerate(labels)}
    matrix = np.zeros((labels.size, labels.size), dtype=np.int64)

    for truth, pred in zip(y_true, y_pred, strict=False):
        matrix[label_to_index[int(truth)], label_to_index[int(pred)]] += 1
    return labels, matrix


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays")

    labels, matrix = _build_confusion_matrix(y_true, y_pred)
    total = float(matrix.sum())
    accuracy = _safe_div(float(np.trace(matrix)), total)

    per_class_metrics: dict[int, dict[str, float | int | str]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for idx, label in enumerate(labels):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        support = int(matrix[idx, :].sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) else 0.0

        label_text = label_names[int(label)] if label_names and int(label) < len(label_names) else str(int(label))
        per_class_metrics[int(label)] = {
            "label_name": label_text,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "confusion_matrix": matrix,
        "per_class_metrics": per_class_metrics,
        "labels": [int(label) for label in labels.tolist()],
        "label_names": [
            label_names[int(label)] if label_names and int(label) < len(label_names) else str(int(label))
            for label in labels.tolist()
        ],
    }

    if per_class_metrics:
        minority_label = min(
            per_class_metrics,
            key=lambda label: (
                int(per_class_metrics[label].get("support", 0)),
                int(label),
            ),
        )
        minority_payload = per_class_metrics[minority_label]
        metrics.update(
            {
                "minority_label": int(minority_label),
                "minority_label_name": minority_payload.get("label_name", str(minority_label)),
                "minority_precision": float(minority_payload.get("precision", 0.0)),
                "minority_recall": float(minority_payload.get("recall", 0.0)),
                "minority_f1": float(minority_payload.get("f1", 0.0)),
                "minority_support": int(minority_payload.get("support", 0)),
            }
        )

    if len(labels) == 2 and 1 in per_class_metrics:
        positive_payload = per_class_metrics[1]
        metrics.update(
            {
                "positive_precision": float(positive_payload.get("precision", 0.0)),
                "positive_recall": float(positive_payload.get("recall", 0.0)),
                "positive_f1": float(positive_payload.get("f1", 0.0)),
            }
        )

    if y_score is not None and len(np.unique(y_true)) == 2:
        binary_scores = np.asarray(y_score, dtype=float).reshape(-1)
        metrics.update(_binary_curve_points(y_true, binary_scores))
    return metrics


def score_metric(metrics: dict[str, Any], metric_name: str) -> float:
    aliases = {
        "macro-f1": "macro_f1",
        "macro_f1": "macro_f1",
        "minority-f1": "minority_f1",
        "minority-precision": "minority_precision",
        "minority-recall": "minority_recall",
        "positive-f1": "positive_f1",
        "positive-precision": "positive_precision",
        "positive-recall": "positive_recall",
    }
    key = aliases.get(metric_name, metric_name)
    value = metrics.get(key, 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def per_class_metrics_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, values in metrics.get("per_class_metrics", {}).items():
        rows.append(
            {
                "label": int(label),
                "label_name": values.get("label_name", str(label)),
                "precision": float(values.get("precision", 0.0)),
                "recall": float(values.get("recall", 0.0)),
                "f1": float(values.get("f1", 0.0)),
                "support": int(values.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def run_noise_robustness_sweep(
    model_predict_fn: Callable[[np.ndarray], np.ndarray],
    x_clean: np.ndarray,
    y_true: np.ndarray,
    noise_levels: list[float],
    metric: str = "macro_f1",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Evaluate robustness by adding Gaussian noise with different strengths.

    This function does not modify the input arrays and returns a DataFrame
    with columns ["noise_sigma", metric].
    """

    base_x = np.asarray(x_clean, dtype=float)
    y_arr = np.asarray(y_true)
    rng = np.random.default_rng(random_state)

    records: list[dict[str, Any]] = []
    for sigma in noise_levels:
        if sigma < 0.0:
            continue
        if sigma == 0.0:
            noisy_x = base_x
        else:
            noise = rng.normal(loc=0.0, scale=sigma, size=base_x.shape)
            noisy_x = base_x + noise

        y_pred = np.asarray(model_predict_fn(noisy_x))
        metrics = evaluate_predictions(y_true=y_arr, y_pred=y_pred)
        score = score_metric(metrics, metric)
        records.append({"noise_sigma": float(sigma), metric: float(score)})

    if not records:
        return pd.DataFrame(columns=["noise_sigma", metric])
    df = pd.DataFrame(records)
    df.sort_values("noise_sigma", inplace=True)
    return df.reset_index(drop=True)


def run_train_ratio_sweep(
    train_and_evaluate_fn: Callable[[float], dict[str, Any]],
    train_ratios: list[float],
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """Run a training/evaluation function across multiple train ratios.

    The callable must accept a train_ratio in (0, 1] and return a mapping
    that includes a nested "metrics" dictionary.
    """

    records: list[dict[str, Any]] = []
    for ratio in train_ratios:
        if not (0.0 < ratio <= 1.0):
            continue
        payload = train_and_evaluate_fn(float(ratio))
        metrics = payload.get("metrics", {})
        score = score_metric(metrics, metric)
        records.append({"train_ratio": float(ratio), metric: float(score)})

    if not records:
        return pd.DataFrame(columns=["train_ratio", metric])
    df = pd.DataFrame(records)
    df.sort_values("train_ratio", inplace=True)
    return df.reset_index(drop=True)

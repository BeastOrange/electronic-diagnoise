from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from emc_diag.evaluation import evaluate_predictions, score_metric

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC
except Exception:  # pragma: no cover - optional dependency
    RandomForestClassifier = None
    ExtraTreesClassifier = None
    LogisticRegression = None
    MLPClassifier = None
    SVC = None
    LinearSVC = None
    make_pipeline = None
    StandardScaler = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None


SUPPORTED_BASELINES = {
    "auto",
    "random_forest",
    "extra_trees",
    "logistic_regression",
    "bagged_logistic_regression",
    "svc",
    "linear_svc",
    "mlp_classifier",
}


def _metric_tuple(metrics: dict[str, Any], primary_metric: str, secondary_metric: str) -> tuple[float, float, float]:
    return (
        score_metric(metrics, primary_metric),
        score_metric(metrics, secondary_metric),
        score_metric(metrics, "accuracy"),
    )


def _emit_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


class _MajorityClassifier:
    def __init__(self) -> None:
        self.majority_class: int | None = None

    def fit(self, _x_train: np.ndarray, y_train: np.ndarray) -> "_MajorityClassifier":
        counts = np.bincount(y_train.astype(np.int64))
        self.majority_class = int(np.argmax(counts))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.majority_class is None:
            raise RuntimeError("Model has not been fitted")
        return np.full(shape=(x.shape[0],), fill_value=self.majority_class, dtype=np.int64)


class _BaggedLogisticRegression:
    def __init__(
        self,
        n_estimators: int = 9,
        random_state: int = 42,
        max_iter: int = 2000,
        c: float = 2.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_iter = max_iter
        self.c = c
        self.models: list[Any] = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "_BaggedLogisticRegression":
        if LogisticRegression is None or make_pipeline is None or StandardScaler is None:
            raise RuntimeError("scikit-learn logistic regression dependencies are required")

        rng = np.random.default_rng(self.random_state)
        self.models = []
        sample_count = x_train.shape[0]
        for estimator_index in range(self.n_estimators):
            sample_indices = rng.integers(0, sample_count, sample_count)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=self.max_iter,
                    random_state=self.random_state + estimator_index,
                    class_weight="balanced",
                    C=self.c,
                ),
            )
            model.fit(x_train[sample_indices], y_train[sample_indices])
            self.models.append(model)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model has not been fitted")
        return np.mean([model.predict_proba(x) for model in self.models], axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)
        return np.argmax(probabilities, axis=1).astype(np.int64)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    return float(np.mean(y_true == y_pred))


def _resolve_device(requested_device: str = "auto") -> str:
    normalized = requested_device.lower()
    if normalized not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("requested_device must be one of: auto, cpu, mps, cuda")

    if normalized == "cpu":
        return "cpu"

    if torch is None:
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if normalized == "cuda":
        return "cuda" if cuda_available else "cpu"
    if normalized == "mps":
        return "mps" if mps_available else "cpu"

    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def _score_vector(model: Any, x: np.ndarray) -> np.ndarray | None:
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


def _apply_baseline_params(model_name: str, model: Any, params: dict[str, Any] | None) -> Any:
    if not params:
        return model

    if isinstance(model, _BaggedLogisticRegression):
        merged = {
            "n_estimators": model.n_estimators,
            "random_state": model.random_state,
            "max_iter": model.max_iter,
            "c": model.c,
        }
        merged.update(params)
        return _BaggedLogisticRegression(**merged)

    if hasattr(model, "named_steps") and getattr(model, "named_steps", None):
        final_step_name = next(reversed(model.named_steps.keys()))
        try:
            model.set_params(**{f"{final_step_name}__{key}": value for key, value in params.items()})
            return model
        except ValueError:
            pass

    if hasattr(model, "set_params"):
        model.set_params(**params)
    return model


def _resolve_baseline_param_map(
    baseline_params: dict[str, Any] | None,
    baseline: str,
) -> dict[str, dict[str, Any]]:
    if not baseline_params:
        return {}

    if any(key in SUPPORTED_BASELINES for key in baseline_params):
        resolved: dict[str, dict[str, Any]] = {}
        for key, value in baseline_params.items():
            if key in SUPPORTED_BASELINES and isinstance(value, dict):
                resolved[key] = dict(value)
        return resolved

    if baseline != "auto":
        return {baseline: dict(baseline_params)}
    return {}


def _candidate_models(
    random_state: int,
    n_estimators: int,
    baseline_param_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    if RandomForestClassifier is not None:
        models["random_forest"] = _apply_baseline_params("random_forest", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ), (baseline_param_map or {}).get("random_forest"))
    if ExtraTreesClassifier is not None:
        models["extra_trees"] = _apply_baseline_params("extra_trees", ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ), (baseline_param_map or {}).get("extra_trees"))
    if LogisticRegression is not None and make_pipeline is not None and StandardScaler is not None:
        models["logistic_regression"] = _apply_baseline_params("logistic_regression", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=random_state, class_weight="balanced"),
        ), (baseline_param_map or {}).get("logistic_regression"))
        models["bagged_logistic_regression"] = _apply_baseline_params(
            "bagged_logistic_regression",
            _BaggedLogisticRegression(random_state=random_state),
            (baseline_param_map or {}).get("bagged_logistic_regression"),
        )
    if SVC is not None and make_pipeline is not None and StandardScaler is not None:
        models["svc"] = _apply_baseline_params("svc", make_pipeline(
            StandardScaler(),
            SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state),
        ), (baseline_param_map or {}).get("svc"))
    if LinearSVC is not None and make_pipeline is not None and StandardScaler is not None:
        models["linear_svc"] = _apply_baseline_params("linear_svc", make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, random_state=random_state, class_weight="balanced"),
        ), (baseline_param_map or {}).get("linear_svc"))
    if MLPClassifier is not None and make_pipeline is not None and StandardScaler is not None:
        models["mlp_classifier"] = _apply_baseline_params("mlp_classifier", make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=600,
                early_stopping=True,
                random_state=random_state,
            ),
        ), (baseline_param_map or {}).get("mlp_classifier"))
    return models


def _evaluate_candidate(
    model: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    primary_metric: str,
    secondary_metric: str,
    threshold_tuning: dict[str, Any] | None,
) -> dict[str, Any]:
    score_vector = _score_vector(model, x_val)
    chosen_threshold = 0.5
    native_predictions = np.asarray(model.predict(x_val), dtype=np.int64)
    native_metrics = evaluate_predictions(y_val, native_predictions, y_score=score_vector)
    if score_vector is not None and len(np.unique(y_val)) == 2 and (threshold_tuning or {}).get("enabled", False):
        tuning_metric = str((threshold_tuning or {}).get("metric", primary_metric))
        grid = [float(item) for item in (threshold_tuning or {}).get("grid", [0.5])]
        best_payload: dict[str, Any] | None = None
        for threshold in grid:
            threshold_pred = (score_vector >= threshold).astype(np.int64)
            threshold_metrics = evaluate_predictions(y_val, threshold_pred, y_score=score_vector)
            payload = {
                "threshold": float(threshold),
                "val_predictions": threshold_pred,
                "metrics": threshold_metrics,
            }
            if best_payload is None:
                best_payload = payload
                continue
            current_primary = score_metric(payload["metrics"], tuning_metric)
            current_secondary = score_metric(payload["metrics"], secondary_metric)
            best_primary = score_metric(best_payload["metrics"], tuning_metric)
            best_secondary = score_metric(best_payload["metrics"], secondary_metric)
            if (current_primary, current_secondary) > (best_primary, best_secondary):
                best_payload = payload
        assert best_payload is not None
        threshold_metrics = best_payload["metrics"]
        if _metric_tuple(native_metrics, primary_metric, secondary_metric) >= _metric_tuple(threshold_metrics, primary_metric, secondary_metric):
            chosen_threshold = None
            metrics = native_metrics
            val_predictions = native_predictions
        else:
            chosen_threshold = best_payload["threshold"]
            metrics = threshold_metrics
            val_predictions = best_payload["val_predictions"]
    else:
        chosen_threshold = None
        val_predictions = native_predictions
        metrics = native_metrics

    return {
        "metrics": metrics,
        "val_predictions": val_predictions,
        "threshold": chosen_threshold if score_vector is not None and len(np.unique(y_val)) == 2 else None,
    }


def train_baseline_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 200,
    baseline: str = "auto",
    candidates: list[str] | None = None,
    baseline_params: dict[str, Any] | None = None,
    primary_metric: str = "accuracy",
    secondary_metric: str = "f1",
    threshold_tuning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    if baseline not in SUPPORTED_BASELINES:
        raise ValueError(f"Unsupported baseline '{baseline}'. Supported: {sorted(SUPPORTED_BASELINES)}")

    baseline_param_map = _resolve_baseline_param_map(baseline_params, baseline=baseline)
    all_candidates = _candidate_models(
        random_state=random_state,
        n_estimators=n_estimators,
        baseline_param_map=baseline_param_map,
    )
    candidate_names = candidates or [name for name in all_candidates if name != "auto"]
    available_candidates = {name: all_candidates[name] for name in candidate_names if name in all_candidates}

    chosen_name = baseline
    selected_threshold: float | None = None
    candidate_scores: dict[str, dict[str, float | str | None]] = {}

    if baseline == "auto":
        if not available_candidates:
            chosen_name = "majority_fallback"
            model = _MajorityClassifier().fit(x_train, y_train)
            val_pred = model.predict(x_val)
            val_metrics = evaluate_predictions(y_val, val_pred)
        else:
            best_result: tuple[str, Any, dict[str, Any]] | None = None
            for name, candidate in available_candidates.items():
                candidate.fit(x_train, y_train)
                evaluation = _evaluate_candidate(
                    candidate,
                    x_val=x_val,
                    y_val=y_val,
                    primary_metric=primary_metric,
                    secondary_metric=secondary_metric,
                    threshold_tuning=threshold_tuning,
                )
                metrics = evaluation["metrics"]
                candidate_scores[name] = {
                    "accuracy": score_metric(metrics, "accuracy"),
                    "f1": score_metric(metrics, "f1"),
                    "macro_f1": score_metric(metrics, "macro_f1"),
                    "precision": score_metric(metrics, "precision"),
                    "recall": score_metric(metrics, "recall"),
                    "threshold": evaluation["threshold"],
                }
                score_tuple = _metric_tuple(metrics, primary_metric, secondary_metric)
                if best_result is None:
                    best_result = (name, candidate, evaluation)
                    best_tuple = score_tuple
                    continue
                if score_tuple > best_tuple:
                    best_result = (name, candidate, evaluation)
                    best_tuple = score_tuple
            assert best_result is not None
            chosen_name, model, chosen_evaluation = best_result
            val_pred = np.asarray(chosen_evaluation["val_predictions"], dtype=np.int64)
            val_metrics = chosen_evaluation["metrics"]
            selected_threshold = chosen_evaluation["threshold"]
    else:
        if baseline in all_candidates:
            chosen_name = baseline
            model = all_candidates[baseline]
        else:
            chosen_name = "majority_fallback"
            model = _MajorityClassifier()
        model.fit(x_train, y_train)
        chosen_evaluation = _evaluate_candidate(
            model,
            x_val=x_val,
            y_val=y_val,
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            threshold_tuning=threshold_tuning,
        )
        val_pred = np.asarray(chosen_evaluation["val_predictions"], dtype=np.int64)
        val_metrics = chosen_evaluation["metrics"]
        selected_threshold = chosen_evaluation["threshold"]
        candidate_scores[chosen_name] = {
            "accuracy": score_metric(val_metrics, "accuracy"),
            "f1": score_metric(val_metrics, "f1"),
            "macro_f1": score_metric(val_metrics, "macro_f1"),
            "precision": score_metric(val_metrics, "precision"),
            "recall": score_metric(val_metrics, "recall"),
            "threshold": selected_threshold,
        }

    return {
        "model": model,
        "model_name": chosen_name,
        "selected_baseline": chosen_name,
        "candidate_scores": candidate_scores,
        "val_predictions": val_pred,
        "val_accuracy": _accuracy(y_val, val_pred),
        "val_metrics": val_metrics,
        "threshold": selected_threshold,
    }


def train_qwen_qlora_classifier(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from emc_diag.llm_modeling import train_qwen_qlora_classifier as _train_qwen_qlora_classifier

    return _train_qwen_qlora_classifier(*args, **kwargs)


@dataclass
class _TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    dropout: float
    weight_decay: float
    patience: int | None
    scheduler: str | None
    class_weighting: str
    sampler: str
    loss_name: str
    focal_gamma: float


def _resolve_loader_workers(requested_workers: int | None, resolved_device: str) -> int:
    if isinstance(requested_workers, int) and requested_workers >= 0:
        return requested_workers
    cpu_count = os.cpu_count() or 1
    if resolved_device == "cuda":
        return max(2, min(8, cpu_count // 2 or 1))
    return 0


def _resolve_pin_memory(pin_memory: bool | None, resolved_device: str) -> bool:
    if isinstance(pin_memory, bool):
        return pin_memory
    return resolved_device == "cuda"


def _build_dataloader_kwargs(
    resolved_device: str,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> dict[str, Any]:
    worker_count = _resolve_loader_workers(loader_workers, resolved_device)
    dataloader_kwargs: dict[str, Any] = {
        "num_workers": worker_count,
        "pin_memory": _resolve_pin_memory(pin_memory, resolved_device),
    }
    if worker_count > 0:
        dataloader_kwargs["persistent_workers"] = bool(
            persistent_workers if persistent_workers is not None else True
        )
        dataloader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor or 2))
    return dataloader_kwargs


def _use_amp(amp: bool | str | None, resolved_device: str) -> bool:
    if resolved_device != "cuda" or torch is None:
        return False
    if isinstance(amp, bool):
        return amp
    return str(amp or "auto").lower() in {"auto", "on", "true", "1"}


def _build_grad_scaler(amp_enabled: bool) -> Any:
    if torch is None or not amp_enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(resolved_device: str, amp_enabled: bool) -> Any:
    if torch is None or not amp_enabled:
        return nullcontext()
    return torch.autocast(device_type=resolved_device, dtype=torch.float16)


def _optimizer_step(loss: Any, optimizer: Any, scaler: Any | None) -> None:
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return
    loss.backward()
    optimizer.step()


def _normalize_deep_inputs(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    if x_train.ndim not in {2, 3}:
        raise ValueError("x_train must be 2D [samples, features] or 3D [samples, channels, length]")
    if x_val.ndim not in {2, 3}:
        raise ValueError("x_val must be 2D [samples, features] or 3D [samples, channels, length]")

    if x_train.ndim == 2:
        return x_train[:, None, :], x_val[:, None, :], 1
    return x_train, x_val, int(x_train.shape[1])


def _normalize_class_weighting_name(class_weighting: str | bool) -> str:
    if isinstance(class_weighting, bool):
        return "on" if class_weighting else "off"
    return str(class_weighting).lower()


def _normalize_sampler_name(sampler: str | None) -> str:
    return str(sampler or "off").lower()


def _normalize_loss_name(loss_name: str | None) -> str:
    return str(loss_name or "cross_entropy").lower()


def _validate_deep_training_config(
    epochs: int,
    batch_size: int,
    dropout: float,
    weight_decay: float,
    patience: int | None,
    scheduler_name: str,
    class_weighting_name: str,
    sampler_name: str,
    loss_name: str,
    focal_gamma: float,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not (0.0 <= dropout < 1.0):
        raise ValueError("dropout must be in [0, 1)")
    if weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if patience is not None and patience < 0:
        raise ValueError("patience must be >= 0")
    if scheduler_name not in {"none", "off", "plateau", "cosine", "step"}:
        raise ValueError("scheduler must be one of: none, off, plateau, cosine, step")
    if class_weighting_name not in {"none", "off", "on", "balanced"}:
        raise ValueError("class_weighting must be one of: off, on, balanced")
    if sampler_name not in {"none", "off", "balanced"}:
        raise ValueError("sampler must be one of: off, balanced")
    if loss_name not in {"cross_entropy", "focal"}:
        raise ValueError("loss_name must be one of: cross_entropy, focal")
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be >= 0")


class _FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Any | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: Any, targets: Any) -> Any:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = torch.pow(1.0 - target_probs, self.gamma)
        losses = -focal_term * target_log_probs
        if self.weight is not None:
            class_weights = self.weight.gather(0, targets)
            losses = losses * class_weights
        return losses.mean()


def _build_weighted_loss(
    y_train: np.ndarray,
    num_classes: int,
    class_weighting_name: str,
    device: Any,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
) -> Any:
    criterion_weights = None
    if class_weighting_name in {"on", "balanced"}:
        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        safe_counts = np.where(class_counts > 0, class_counts, 1.0)
        weights = class_counts.sum() / (len(class_counts) * safe_counts)
        criterion_weights = torch.from_numpy(weights).to(device)
    if loss_name == "focal":
        return _FocalLoss(gamma=focal_gamma, weight=criterion_weights)
    return nn.CrossEntropyLoss(weight=criterion_weights)


def _build_balanced_sampler(y_train: np.ndarray, random_seed: int) -> Any | None:
    if WeightedRandomSampler is None:
        return None
    class_counts = np.bincount(np.asarray(y_train, dtype=np.int64))
    safe_counts = np.where(class_counts > 0, class_counts, 1.0)
    sample_weights = 1.0 / safe_counts[np.asarray(y_train, dtype=np.int64)]
    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=min(dropout_rate * 0.5, 0.3))

    def forward(self, inputs: Any) -> Any:
        residual = inputs
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.activation(x)


def _build_scheduler(
    optimizer: Any,
    scheduler_name: str,
    scheduler_kwargs: dict[str, Any],
    epochs: int,
    patience: int | None,
) -> Any | None:
    if scheduler_name == "plateau":
        default_plateau = {
            "mode": "min",
            "factor": 0.5,
            "patience": max(1, (patience or 1) // 2),
        }
        default_plateau.update(scheduler_kwargs)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **default_plateau)
    if scheduler_name == "step":
        default_step = {"step_size": max(1, epochs // 3), "gamma": 0.5}
        default_step.update(scheduler_kwargs)
        return torch.optim.lr_scheduler.StepLR(optimizer, **default_step)
    if scheduler_name == "cosine":
        default_cosine = {"T_max": max(1, epochs)}
        default_cosine.update(scheduler_kwargs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **default_cosine)
    return None


class _CNN1DModel(nn.Module):
    def __init__(self, in_channel_count: int, out_classes: int, dropout_rate: float) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel_count, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.4)),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.4)),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, out_classes),
        )

    def load_encoder_from_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load only the feature_extractor (encoder) weights from a state_dict.

        Keys are filtered by the "feature_extractor." prefix to avoid
        accidentally overwriting classifier layers.
        """

        own_state = self.state_dict()
        prefix = "feature_extractor."
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            if key in own_state:
                own_state[key].copy_(value)

    def forward(self, inputs: Any) -> Any:
        features = self.feature_extractor(inputs)
        return self.classifier(features)


class _ResidualCNN1DModel(nn.Module):
    def __init__(self, in_channel_count: int, out_classes: int, dropout_rate: float) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel_count, 32, kernel_size=5, padding=2),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            _ResidualConvBlock(64, dropout_rate),
            _ResidualConvBlock(64, dropout_rate),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.3)),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, out_classes),
        )

    def load_encoder_from_state_dict(self, state_dict: dict[str, Any]) -> None:
        own_state = self.state_dict()
        prefix = "feature_extractor."
        for key, value in state_dict.items():
            if key.startswith(prefix) and key in own_state:
                own_state[key].copy_(value)

    def forward(self, inputs: Any) -> Any:
        features = self.feature_extractor(inputs)
        return self.classifier(features)


def _build_cnn_encoder_metadata(in_channels: int) -> dict[str, Any]:
    return {
        "encoder_arch": "cnn_1d_encoder",
        "in_channels": int(in_channels),
        "encoder_output_dim": 64,
    }


def _validate_cnn_encoder_transfer_compatibility(
    model: _CNN1DModel,
    encoder_state_dict: dict[str, Any],
    encoder_metadata: dict[str, Any] | None,
) -> None:
    own_state = model.state_dict()
    prefix = "feature_extractor."
    matched = 0
    for key, value in encoder_state_dict.items():
        if not key.startswith(prefix):
            continue
        if key not in own_state:
            continue
        expected_shape = tuple(own_state[key].shape)
        observed_shape = tuple(getattr(value, "shape", ()))
        if observed_shape != expected_shape:
            raise ValueError(
                f"Incompatible encoder weight shape for '{key}': "
                f"expected {expected_shape}, got {observed_shape}"
            )
        matched += 1
    if matched == 0:
        raise ValueError("encoder_state_dict does not contain compatible feature_extractor weights")

    if encoder_metadata is None:
        return

    arch_name = str(encoder_metadata.get("encoder_arch", "")).strip()
    if arch_name and arch_name != "cnn_1d_encoder":
        raise ValueError(f"Unsupported encoder architecture '{arch_name}' for CNN transfer")

    expected_in_channels = encoder_metadata.get("in_channels")
    observed_in_channels = int(model.feature_extractor[0].weight.shape[1])
    if expected_in_channels is not None and int(expected_in_channels) != observed_in_channels:
        raise ValueError(
            f"Incompatible encoder in_channels: expected {observed_in_channels}, got {int(expected_in_channels)}"
        )

    expected_output_dim = encoder_metadata.get("encoder_output_dim")
    observed_output_dim = int(model.classifier[0].in_features)
    if expected_output_dim is not None and int(expected_output_dim) != observed_output_dim:
        raise ValueError(
            f"Incompatible encoder output dim: expected {observed_output_dim}, got {int(expected_output_dim)}"
        )


class _MultiHeadCNN1DModel(nn.Module):
    def __init__(self, in_channel_count: int, task_out_classes: dict[str, int], dropout_rate: float) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel_count, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.4)),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.4)),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.heads = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(64, out_classes),
                )
                for task_name, out_classes in task_out_classes.items()
            }
        )

    def load_encoder_from_state_dict(self, state_dict: dict[str, Any]) -> None:
        own_state = self.state_dict()
        prefix = "feature_extractor."
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            if key in own_state:
                own_state[key].copy_(value)

    def forward(self, inputs: Any, task_name: str | None = None) -> Any:
        features = self.feature_extractor(inputs)
        if task_name is not None:
            return self.heads[task_name](features)
        return {name: head(features) for name, head in self.heads.items()}


class _CNN1DPretrainModel(nn.Module):
    def __init__(self, in_channel_count: int, sequence_length: int, dropout_rate: float) -> None:
        super().__init__()
        self.in_channel_count = int(in_channel_count)
        self.sequence_length = int(sequence_length)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel_count, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.4)),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.4)),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, self.in_channel_count * self.sequence_length),
        )

    def forward(self, inputs: Any) -> Any:
        encoded = self.feature_extractor(inputs)
        reconstructed = self.decoder(encoded)
        return reconstructed.view(-1, self.in_channel_count, self.sequence_length)


class _StructuredBranchEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout_rate: float, out_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.3)),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, inputs: Any) -> Any:
        return self.encoder(inputs)


class _CognitiveRadioHybridBackbone(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        cov_channels: int,
        temporal_channels: int,
        dropout_rate: float,
        fusion_dim: int = 128,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if fusion_dim % attention_heads != 0:
            raise ValueError("fusion_dim must be divisible by attention_heads")
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.cov_encoder = _StructuredBranchEncoder(cov_channels, dropout_rate=dropout_rate, out_dim=fusion_dim)
        self.temporal_encoder = _StructuredBranchEncoder(temporal_channels, dropout_rate=dropout_rate, out_dim=fusion_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=attention_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, scalar_x: Any, cov_x: Any, temporal_x: Any) -> Any:
        scalar_features = self.scalar_encoder(scalar_x)
        cov_features = self.cov_encoder(cov_x)
        temporal_features = self.temporal_encoder(temporal_x)
        tokens = torch.stack([scalar_features, cov_features, temporal_features], dim=1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(cls, tokens, tokens, need_weights=False)
        fused = self.norm(attended.squeeze(1))
        return self.mlp(fused)


class _CognitiveRadioHybridModel(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        cov_channels: int,
        temporal_channels: int,
        out_classes: int,
        dropout_rate: float,
        fusion_dim: int = 128,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = _CognitiveRadioHybridBackbone(
            scalar_dim=scalar_dim,
            cov_channels=cov_channels,
            temporal_channels=temporal_channels,
            dropout_rate=dropout_rate,
            fusion_dim=fusion_dim,
            attention_heads=attention_heads,
        )
        self.scalar_shortcut = nn.Sequential(
            nn.Linear(scalar_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, out_classes),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, out_classes),
        )

    def forward(self, scalar_x: Any, cov_x: Any, temporal_x: Any) -> Any:
        fused = self.backbone(scalar_x, cov_x, temporal_x)
        return self.classifier(fused) + self.scalar_shortcut(scalar_x)


class _CognitiveRadioScalarHybridModel(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        cov_channels: int,
        temporal_channels: int,
        out_classes: int,
        dropout_rate: float,
        fusion_dim: int = 256,
        attention_heads: int = 8,
    ) -> None:
        super().__init__()
        if fusion_dim % attention_heads != 0:
            raise ValueError("fusion_dim must be divisible by attention_heads")
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
        )
        self.cov_encoder = _StructuredBranchEncoder(cov_channels, dropout_rate=dropout_rate, out_dim=fusion_dim // 2)
        self.temporal_encoder = _StructuredBranchEncoder(temporal_channels, dropout_rate=dropout_rate, out_dim=fusion_dim // 2)
        self.scalar_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.struct_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, out_classes),
        )

    def forward(self, scalar_x: Any, cov_x: Any, temporal_x: Any) -> Any:
        scalar_features = self.scalar_encoder(scalar_x)
        structured = torch.cat([self.cov_encoder(cov_x), self.temporal_encoder(temporal_x)], dim=1)
        structured = self.struct_proj(structured)
        gate = self.scalar_gate(scalar_features)
        fused = scalar_features + gate * structured
        return self.classifier(fused)


class _MultiHeadCognitiveRadioHybridModel(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        cov_channels: int,
        temporal_channels: int,
        task_out_classes: dict[str, int],
        dropout_rate: float,
        fusion_dim: int = 128,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = _CognitiveRadioHybridBackbone(
            scalar_dim=scalar_dim,
            cov_channels=cov_channels,
            temporal_channels=temporal_channels,
            dropout_rate=dropout_rate,
            fusion_dim=fusion_dim,
            attention_heads=attention_heads,
        )
        self.heads = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(fusion_dim, out_classes),
                )
                for task_name, out_classes in task_out_classes.items()
            }
        )
        self.scalar_heads = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(scalar_dim, fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(fusion_dim, out_classes),
                )
                for task_name, out_classes in task_out_classes.items()
            }
        )

    def forward(self, scalar_x: Any, cov_x: Any, temporal_x: Any, task_name: str | None = None) -> Any:
        fused = self.backbone(scalar_x, cov_x, temporal_x)
        if task_name is not None:
            return self.heads[task_name](fused) + self.scalar_heads[task_name](scalar_x)
        return {
            name: head(fused) + self.scalar_heads[name](scalar_x)
            for name, head in self.heads.items()
        }


class _CNNLSTM1DModel(nn.Module):
    def __init__(
        self,
        in_channel_count: int,
        out_classes: int,
        dropout_rate: float,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be > 0")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be > 0")

        self.conv = nn.Sequential(
            nn.Conv1d(in_channel_count, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.4)),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(p=min(dropout_rate * 0.5, 0.4)),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = lstm_hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, max(32, out_dim // 2)),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(max(32, out_dim // 2), out_classes),
        )
        self.bidirectional = bidirectional

    def forward(self, inputs: Any) -> Any:
        features = self.conv(inputs)
        seq = features.transpose(1, 2)
        _, (hidden, _) = self.lstm(seq)
        if self.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        return self.head(final_hidden)


class _Transformer1DModel(nn.Module):
    def __init__(
        self,
        in_channel_count: int,
        out_classes: int,
        dropout_rate: float,
        d_model: int = 64,
        nhead: int = 4,
        transformer_layers: int = 2,
        dim_feedforward: int = 128,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if nhead <= 0:
            raise ValueError("nhead must be > 0")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if transformer_layers <= 0:
            raise ValueError("transformer_layers must be > 0")
        if dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be > 0")

        self.input_proj = nn.Linear(in_channel_count, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(d_model, out_classes),
        )

    def _positional_encoding(self, length: int, d_model: int, device: Any) -> Any:
        positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-np.log(10000.0) / d_model)
        )
        encoding = torch.zeros((length, d_model), dtype=torch.float32, device=device)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        return encoding

    def forward(self, inputs: Any) -> Any:
        seq = inputs.transpose(1, 2)
        projected = self.input_proj(seq)
        pos = self._positional_encoding(projected.size(1), projected.size(2), projected.device)
        encoded = self.encoder(projected + pos.unsqueeze(0))
        pooled = self.norm(encoded).mean(dim=1)
        return self.head(pooled)


def _train_deep_model(
    model: Any,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    scheduler: str | None = "plateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    class_weighting: str | bool = "off",
    sampler: str | None = "off",
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required for deep models")

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    x_train_norm, x_val_norm, _ = _normalize_deep_inputs(x_train, x_val)
    resolved_device = _resolve_device(requested_device)
    device = torch.device(resolved_device)
    torch.manual_seed(random_seed)
    amp_enabled = _use_amp(amp, resolved_device)
    scaler = _build_grad_scaler(amp_enabled)

    scheduler_name = str(scheduler).lower() if scheduler is not None else "none"
    class_weighting_name = _normalize_class_weighting_name(class_weighting)
    sampler_name = _normalize_sampler_name(sampler)
    normalized_loss_name = _normalize_loss_name(loss_name)
    _validate_deep_training_config(
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler_name=scheduler_name,
        class_weighting_name=class_weighting_name,
        sampler_name=sampler_name,
        loss_name=normalized_loss_name,
        focal_gamma=focal_gamma,
    )
    scheduler_kwargs = dict(scheduler_kwargs or {})

    num_classes = int(max(y_train.max(), y_val.max()) + 1)
    train_x_tensor = torch.from_numpy(x_train_norm)
    val_x_tensor = torch.from_numpy(x_val_norm)
    train_y_tensor = torch.from_numpy(y_train)
    val_y_tensor = torch.from_numpy(y_val)

    train_loader = DataLoader(
        TensorDataset(train_x_tensor, train_y_tensor),
        batch_size=batch_size,
        shuffle=(sampler_name == "off"),
        sampler=_build_balanced_sampler(y_train, random_seed) if sampler_name == "balanced" else None,
        **_build_dataloader_kwargs(
            resolved_device=resolved_device,
            loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = _build_weighted_loss(
        y_train=y_train,
        num_classes=num_classes,
        class_weighting_name=class_weighting_name,
        device=device,
        loss_name=normalized_loss_name,
        focal_gamma=focal_gamma,
    )
    scheduler_obj = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
        epochs=epochs,
        patience=patience,
    )

    config = _TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler=scheduler_name,
        class_weighting=class_weighting_name,
        sampler=sampler_name,
        loss_name=normalized_loss_name,
        focal_gamma=focal_gamma,
    )

    history: dict[str, list[float]] = {
        "loss": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_accuracy": [],
        "val_f1": [],
        "learning_rate": [],
    }
    best_state_dict: dict[str, Any] | None = None
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    stopped_early = False

    _emit_progress(
        progress,
        f"[{model_name}] start device={resolved_device} epochs={config.epochs} batch_size={batch_size}",
    )

    for epoch_idx in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            with _autocast_context(resolved_device, amp_enabled):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            _optimizer_step(loss, optimizer, scaler)

            batch_size_actual = int(batch_x.shape[0])
            running_loss += float(loss.item()) * batch_size_actual
            seen += batch_size_actual

        train_loss = running_loss / seen if seen else 0.0

        model.eval()
        with torch.no_grad():
            train_logits = model(train_x_tensor.to(device))
            train_pred = torch.argmax(train_logits, dim=1).cpu().numpy().astype(np.int64)
            logits = model(val_x_tensor.to(device))
            val_loss = float(criterion(logits, val_y_tensor.to(device)).item())
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
        train_metrics = evaluate_predictions(y_train, train_pred)
        val_metrics = evaluate_predictions(y_val, pred)
        train_accuracy = score_metric(train_metrics, "accuracy")
        train_f1 = score_metric(train_metrics, "f1")
        val_accuracy = score_metric(val_metrics, "accuracy")
        val_f1 = score_metric(val_metrics, "f1")

        history["loss"].append(train_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["train_f1"].append(train_f1)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)

        if scheduler_obj is not None:
            if scheduler_name == "plateau":
                scheduler_obj.step(val_loss)
            else:
                scheduler_obj.step()
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        _emit_progress(
            progress,
            (
                f"[{model_name}] epoch {epoch_idx + 1}/{config.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} "
                f"train_f1={train_f1:.4f} val_f1={val_f1:.4f} "
                f"lr={float(optimizer.param_groups[0]['lr']):.6f}"
            ),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx
            best_val_accuracy = val_accuracy
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if config.patience is not None and config.patience > 0 and epochs_without_improvement >= config.patience:
            stopped_early = True
            _emit_progress(progress, f"[{model_name}] early_stop epoch={epoch_idx + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        with torch.no_grad():
            best_logits = model(val_x_tensor.to(device))
            pred = torch.argmax(best_logits, dim=1).cpu().numpy().astype(np.int64)
        restored_metrics = evaluate_predictions(y_val, pred)
        restored_accuracy = score_metric(restored_metrics, "accuracy")
        restored_f1 = score_metric(restored_metrics, "f1")
    else:
        restored_accuracy = history["val_accuracy"][-1] if history["val_accuracy"] else 0.0
        restored_f1 = history["val_f1"][-1] if history["val_f1"] else 0.0

    return {
        "model": model,
        "model_name": model_name,
        "resolved_device": resolved_device,
        "train_history": history,
        "val_predictions": pred,
        "val_accuracy": restored_accuracy,
        "val_f1": restored_f1,
        "epochs_ran": len(history["loss"]),
        "best_checkpoint": {
            "epoch": int(best_epoch + 1),
            "val_loss": float(best_val_loss),
            "val_accuracy": float(best_val_accuracy),
            "val_f1": float(best_val_f1),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_accuracy": float(best_val_accuracy),
            "best_val_f1": float(best_val_f1),
            "stopped_early": stopped_early,
        },
        "best_state_dict": best_state_dict,
    }


def train_cnn_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    scheduler: str | None = "plateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    class_weighting: str | bool = "off",
    encoder_state_dict: dict[str, Any] | None = None,
    encoder_metadata: dict[str, Any] | None = None,
    transfer_strategy: str | None = None,
    sampler: str | None = "off",
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for train_cnn_model")

    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_val_np = np.asarray(x_val, dtype=np.float32)
    _, _, in_channels = _normalize_deep_inputs(x_train_np, x_val_np)
    num_classes = int(max(np.asarray(y_train).max(), np.asarray(y_val).max()) + 1)
    model = _CNN1DModel(in_channel_count=in_channels, out_classes=num_classes, dropout_rate=dropout)

    if encoder_state_dict is not None:
        _validate_cnn_encoder_transfer_compatibility(
            model=model,
            encoder_state_dict=encoder_state_dict,
            encoder_metadata=encoder_metadata,
        )
        model.load_encoder_from_state_dict(encoder_state_dict)

    normalized_strategy = (transfer_strategy or "").lower()
    if normalized_strategy in {"freeze", "frozen"}:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    payload = _train_deep_model(
        model=model,
        model_name="cnn_1d",
        x_train=x_train_np,
        y_train=np.asarray(y_train, dtype=np.int64),
        x_val=x_val_np,
        y_val=np.asarray(y_val, dtype=np.int64),
        requested_device=requested_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        class_weighting=class_weighting,
        sampler=sampler,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        loader_workers=loader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        amp=amp,
        progress=progress,
    )
    payload["encoder_metadata"] = _build_cnn_encoder_metadata(in_channels=in_channels)
    return payload


def train_cnn_residual_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    scheduler: str | None = "plateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    class_weighting: str | bool = "off",
    sampler: str | None = "off",
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for train_cnn_residual_model")

    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_val_np = np.asarray(x_val, dtype=np.float32)
    _, _, in_channels = _normalize_deep_inputs(x_train_np, x_val_np)
    num_classes = int(max(np.asarray(y_train).max(), np.asarray(y_val).max()) + 1)
    model = _ResidualCNN1DModel(in_channel_count=in_channels, out_classes=num_classes, dropout_rate=dropout)
    payload = _train_deep_model(
        model=model,
        model_name="cnn_1d_residual",
        x_train=x_train_np,
        y_train=np.asarray(y_train, dtype=np.int64),
        x_val=x_val_np,
        y_val=np.asarray(y_val, dtype=np.int64),
        requested_device=requested_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        class_weighting=class_weighting,
        sampler=sampler,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        loader_workers=loader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        amp=amp,
        progress=progress,
    )
    payload["encoder_metadata"] = {
        "encoder_arch": "cnn_1d_residual_encoder",
        "in_channels": int(in_channels),
        "encoder_output_dim": 64,
    }
    return payload
    payload["encoder_metadata"] = _build_cnn_encoder_metadata(in_channels=in_channels)
    return payload


def train_cnn_lstm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    scheduler: str | None = "plateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    class_weighting: str | bool = "off",
    lstm_hidden_size: int = 64,
    lstm_layers: int = 1,
    bidirectional: bool = False,
    sampler: str | None = "off",
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for train_cnn_lstm_model")

    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_val_np = np.asarray(x_val, dtype=np.float32)
    _, _, in_channels = _normalize_deep_inputs(x_train_np, x_val_np)
    num_classes = int(max(np.asarray(y_train).max(), np.asarray(y_val).max()) + 1)
    model = _CNNLSTM1DModel(
        in_channel_count=in_channels,
        out_classes=num_classes,
        dropout_rate=dropout,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
    )
    return _train_deep_model(
        model=model,
        model_name="cnn_lstm",
        x_train=x_train_np,
        y_train=np.asarray(y_train, dtype=np.int64),
        x_val=x_val_np,
        y_val=np.asarray(y_val, dtype=np.int64),
        requested_device=requested_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        class_weighting=class_weighting,
        sampler=sampler,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        loader_workers=loader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        amp=amp,
        progress=progress,
    )


def train_transformer_1d_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    scheduler: str | None = "plateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    class_weighting: str | bool = "off",
    d_model: int = 64,
    nhead: int = 4,
    transformer_layers: int = 2,
    dim_feedforward: int = 128,
    sampler: str | None = "off",
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for train_transformer_1d_model")

    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_val_np = np.asarray(x_val, dtype=np.float32)
    _, _, in_channels = _normalize_deep_inputs(x_train_np, x_val_np)
    num_classes = int(max(np.asarray(y_train).max(), np.asarray(y_val).max()) + 1)
    model = _Transformer1DModel(
        in_channel_count=in_channels,
        out_classes=num_classes,
        dropout_rate=dropout,
        d_model=d_model,
        nhead=nhead,
        transformer_layers=transformer_layers,
        dim_feedforward=dim_feedforward,
    )
    return _train_deep_model(
        model=model,
        model_name="transformer_1d",
        x_train=x_train_np,
        y_train=np.asarray(y_train, dtype=np.int64),
        x_val=x_val_np,
        y_val=np.asarray(y_val, dtype=np.int64),
        requested_device=requested_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        class_weighting=class_weighting,
        sampler=sampler,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        loader_workers=loader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        amp=amp,
        progress=progress,
    )


def train_cognitive_radio_hybrid_model(
    scalar_train: np.ndarray,
    cov_train: np.ndarray,
    temporal_train: np.ndarray,
    y_train: np.ndarray,
    scalar_val: np.ndarray,
    cov_val: np.ndarray,
    temporal_val: np.ndarray,
    y_val: np.ndarray,
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    scheduler: str | None = "plateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    class_weighting: str | bool = "off",
    fusion_dim: int = 128,
    attention_heads: int = 4,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for train_cognitive_radio_hybrid_model")

    scalar_train_np = np.asarray(scalar_train, dtype=np.float32)
    cov_train_np = np.asarray(cov_train, dtype=np.float32)
    temporal_train_np = np.asarray(temporal_train, dtype=np.float32)
    scalar_val_np = np.asarray(scalar_val, dtype=np.float32)
    cov_val_np = np.asarray(cov_val, dtype=np.float32)
    temporal_val_np = np.asarray(temporal_val, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.int64)
    y_val_np = np.asarray(y_val, dtype=np.int64)

    resolved_device = _resolve_device(requested_device)
    device = torch.device(resolved_device)
    torch.manual_seed(random_seed)
    amp_enabled = _use_amp(amp, resolved_device)
    scaler = _build_grad_scaler(amp_enabled)
    scheduler_name = str(scheduler).lower() if scheduler is not None else "none"
    class_weighting_name = _normalize_class_weighting_name(class_weighting)
    _validate_deep_training_config(
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler_name=scheduler_name,
        class_weighting_name=class_weighting_name,
        sampler_name="off",
        loss_name="cross_entropy",
        focal_gamma=2.0,
    )
    scheduler_kwargs = dict(scheduler_kwargs or {})
    num_classes = int(max(y_train_np.max(), y_val_np.max()) + 1)

    model = _CognitiveRadioHybridModel(
        scalar_dim=int(scalar_train_np.shape[1]),
        cov_channels=int(cov_train_np.shape[1]),
        temporal_channels=int(temporal_train_np.shape[1]),
        out_classes=num_classes,
        dropout_rate=dropout,
        fusion_dim=fusion_dim,
        attention_heads=attention_heads,
    ).to(device)

    train_dataset = TensorDataset(
        torch.from_numpy(scalar_train_np),
        torch.from_numpy(cov_train_np),
        torch.from_numpy(temporal_train_np),
        torch.from_numpy(y_train_np),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **_build_dataloader_kwargs(
            resolved_device=resolved_device,
            loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = _build_weighted_loss(
        y_train=y_train_np,
        num_classes=num_classes,
        class_weighting_name=class_weighting_name,
        device=device,
        loss_name="cross_entropy",
        focal_gamma=2.0,
    )
    scheduler_obj = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
        epochs=epochs,
        patience=patience,
    )

    scalar_val_tensor = torch.from_numpy(scalar_val_np).to(device)
    cov_val_tensor = torch.from_numpy(cov_val_np).to(device)
    temporal_val_tensor = torch.from_numpy(temporal_val_np).to(device)
    scalar_train_tensor = torch.from_numpy(scalar_train_np).to(device)
    cov_train_tensor = torch.from_numpy(cov_train_np).to(device)
    temporal_train_tensor = torch.from_numpy(temporal_train_np).to(device)
    y_val_tensor = torch.from_numpy(y_val_np).to(device)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_accuracy": [],
        "val_f1": [],
        "learning_rate": [],
    }
    best_state_dict: dict[str, Any] | None = None
    best_epoch = -1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    _emit_progress(
        progress,
        f"[cognitive_radio_hybrid] start device={resolved_device} epochs={int(epochs)} batch_size={batch_size}",
    )

    for epoch_idx in range(int(epochs)):
        model.train()
        running_loss = 0.0
        seen = 0
        for scalar_batch, cov_batch, temporal_batch, y_batch in train_loader:
            scalar_batch = scalar_batch.to(device)
            cov_batch = cov_batch.to(device)
            temporal_batch = temporal_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            with _autocast_context(resolved_device, amp_enabled):
                logits = model(scalar_batch, cov_batch, temporal_batch)
                loss = criterion(logits, y_batch)
            _optimizer_step(loss, optimizer, scaler)

            batch_size_actual = int(y_batch.shape[0])
            running_loss += float(loss.item()) * batch_size_actual
            seen += batch_size_actual

        train_loss = running_loss / seen if seen else 0.0
        model.eval()
        with torch.no_grad():
            train_logits = model(scalar_train_tensor, cov_train_tensor, temporal_train_tensor)
            train_pred = torch.argmax(train_logits, dim=1).cpu().numpy().astype(np.int64)
            val_logits = model(scalar_val_tensor, cov_val_tensor, temporal_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy().astype(np.int64)
        train_metrics = evaluate_predictions(y_train_np, train_pred)
        val_metrics = evaluate_predictions(y_val_np, val_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(score_metric(train_metrics, "accuracy"))
        history["train_f1"].append(score_metric(train_metrics, "f1"))
        history["val_accuracy"].append(score_metric(val_metrics, "accuracy"))
        history["val_f1"].append(score_metric(val_metrics, "f1"))

        if scheduler_obj is not None:
            if scheduler_name == "plateau":
                scheduler_obj.step(val_loss)
            else:
                scheduler_obj.step()
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        _emit_progress(
            progress,
            (
                f"[cognitive_radio_hybrid] epoch {epoch_idx + 1}/{int(epochs)} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={history['train_accuracy'][-1]:.4f} val_acc={history['val_accuracy'][-1]:.4f} "
                f"train_f1={history['train_f1'][-1]:.4f} val_f1={history['val_f1'][-1]:.4f} "
                f"lr={float(optimizer.param_groups[0]['lr']):.6f}"
            ),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if patience is not None and patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            _emit_progress(progress, f"[cognitive_radio_hybrid] early_stop epoch={epoch_idx + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    with torch.no_grad():
        val_logits = model(scalar_val_tensor, cov_val_tensor, temporal_val_tensor)
        val_pred = torch.argmax(val_logits, dim=1).cpu().numpy().astype(np.int64)
    val_metrics = evaluate_predictions(y_val_np, val_pred)
    return {
        "model": model,
        "model_name": "cognitive_radio_hybrid",
        "resolved_device": resolved_device,
        "train_history": history,
        "val_predictions": val_pred,
        "val_accuracy": score_metric(val_metrics, "accuracy"),
        "val_f1": score_metric(val_metrics, "f1"),
        "epochs_ran": len(history["train_loss"]),
        "best_checkpoint": {
            "epoch": int(best_epoch + 1),
            "best_epoch": int(best_epoch),
            "val_loss": float(best_val_loss),
            "stopped_early": stopped_early,
        },
        "best_state_dict": best_state_dict,
    }


def train_cognitive_radio_scalar_hybrid_model(
    scalar_train: np.ndarray,
    cov_train: np.ndarray,
    temporal_train: np.ndarray,
    y_train: np.ndarray,
    scalar_val: np.ndarray,
    cov_val: np.ndarray,
    temporal_val: np.ndarray,
    y_val: np.ndarray,
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    scheduler: str | None = "plateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    class_weighting: str | bool = "off",
    fusion_dim: int = 256,
    attention_heads: int = 8,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for train_cognitive_radio_scalar_hybrid_model")

    scalar_train_np = np.asarray(scalar_train, dtype=np.float32)
    cov_train_np = np.asarray(cov_train, dtype=np.float32)
    temporal_train_np = np.asarray(temporal_train, dtype=np.float32)
    scalar_val_np = np.asarray(scalar_val, dtype=np.float32)
    cov_val_np = np.asarray(cov_val, dtype=np.float32)
    temporal_val_np = np.asarray(temporal_val, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.int64)
    y_val_np = np.asarray(y_val, dtype=np.int64)

    resolved_device = _resolve_device(requested_device)
    device = torch.device(resolved_device)
    torch.manual_seed(random_seed)
    amp_enabled = _use_amp(amp, resolved_device)
    scaler = _build_grad_scaler(amp_enabled)
    scheduler_name = str(scheduler).lower() if scheduler is not None else "none"
    class_weighting_name = _normalize_class_weighting_name(class_weighting)
    _validate_deep_training_config(
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler_name=scheduler_name,
        class_weighting_name=class_weighting_name,
        sampler_name="off",
        loss_name="cross_entropy",
        focal_gamma=2.0,
    )
    scheduler_kwargs = dict(scheduler_kwargs or {})
    num_classes = int(max(y_train_np.max(), y_val_np.max()) + 1)

    model = _CognitiveRadioScalarHybridModel(
        scalar_dim=int(scalar_train_np.shape[1]),
        cov_channels=int(cov_train_np.shape[1]),
        temporal_channels=int(temporal_train_np.shape[1]),
        out_classes=num_classes,
        dropout_rate=dropout,
        fusion_dim=fusion_dim,
        attention_heads=attention_heads,
    ).to(device)

    train_dataset = TensorDataset(
        torch.from_numpy(scalar_train_np),
        torch.from_numpy(cov_train_np),
        torch.from_numpy(temporal_train_np),
        torch.from_numpy(y_train_np),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **_build_dataloader_kwargs(
            resolved_device=resolved_device,
            loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = _build_weighted_loss(
        y_train=y_train_np,
        num_classes=num_classes,
        class_weighting_name=class_weighting_name,
        device=device,
        loss_name="cross_entropy",
        focal_gamma=2.0,
    )
    scheduler_obj = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
        epochs=epochs,
        patience=patience,
    )

    scalar_val_tensor = torch.from_numpy(scalar_val_np).to(device)
    cov_val_tensor = torch.from_numpy(cov_val_np).to(device)
    temporal_val_tensor = torch.from_numpy(temporal_val_np).to(device)
    scalar_train_tensor = torch.from_numpy(scalar_train_np).to(device)
    cov_train_tensor = torch.from_numpy(cov_train_np).to(device)
    temporal_train_tensor = torch.from_numpy(temporal_train_np).to(device)
    y_val_tensor = torch.from_numpy(y_val_np).to(device)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_accuracy": [],
        "val_f1": [],
        "learning_rate": [],
    }
    best_state_dict: dict[str, Any] | None = None
    best_epoch = -1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    _emit_progress(
        progress,
        f"[cognitive_radio_scalar_hybrid] start device={resolved_device} epochs={int(epochs)} batch_size={batch_size}",
    )

    for epoch_idx in range(int(epochs)):
        model.train()
        running_loss = 0.0
        seen = 0
        for scalar_batch, cov_batch, temporal_batch, y_batch in train_loader:
            scalar_batch = scalar_batch.to(device)
            cov_batch = cov_batch.to(device)
            temporal_batch = temporal_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            with _autocast_context(resolved_device, amp_enabled):
                logits = model(scalar_batch, cov_batch, temporal_batch)
                loss = criterion(logits, y_batch)
            _optimizer_step(loss, optimizer, scaler)
            batch_size_actual = int(y_batch.shape[0])
            running_loss += float(loss.item()) * batch_size_actual
            seen += batch_size_actual

        train_loss = running_loss / seen if seen else 0.0
        model.eval()
        with torch.no_grad():
            train_logits = model(scalar_train_tensor, cov_train_tensor, temporal_train_tensor)
            train_pred = torch.argmax(train_logits, dim=1).cpu().numpy().astype(np.int64)
            val_logits = model(scalar_val_tensor, cov_val_tensor, temporal_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy().astype(np.int64)
        train_metrics = evaluate_predictions(y_train_np, train_pred)
        val_metrics = evaluate_predictions(y_val_np, val_pred)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(score_metric(train_metrics, "accuracy"))
        history["train_f1"].append(score_metric(train_metrics, "f1"))
        history["val_accuracy"].append(score_metric(val_metrics, "accuracy"))
        history["val_f1"].append(score_metric(val_metrics, "f1"))
        if scheduler_obj is not None:
            if scheduler_name == "plateau":
                scheduler_obj.step(val_loss)
            else:
                scheduler_obj.step()
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        _emit_progress(
            progress,
            (
                f"[cognitive_radio_scalar_hybrid] epoch {epoch_idx + 1}/{int(epochs)} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={history['train_accuracy'][-1]:.4f} val_acc={history['val_accuracy'][-1]:.4f} "
                f"train_f1={history['train_f1'][-1]:.4f} val_f1={history['val_f1'][-1]:.4f} "
                f"lr={float(optimizer.param_groups[0]['lr']):.6f}"
            ),
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if patience is not None and patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            _emit_progress(progress, f"[cognitive_radio_scalar_hybrid] early_stop epoch={epoch_idx + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    with torch.no_grad():
        val_logits = model(scalar_val_tensor, cov_val_tensor, temporal_val_tensor)
        val_pred = torch.argmax(val_logits, dim=1).cpu().numpy().astype(np.int64)
    val_metrics = evaluate_predictions(y_val_np, val_pred)
    return {
        "model": model,
        "model_name": "cognitive_radio_scalar_hybrid",
        "resolved_device": resolved_device,
        "train_history": history,
        "val_predictions": val_pred,
        "val_accuracy": score_metric(val_metrics, "accuracy"),
        "val_f1": score_metric(val_metrics, "f1"),
        "epochs_ran": len(history["train_loss"]),
        "best_checkpoint": {
            "epoch": int(best_epoch + 1),
            "best_epoch": int(best_epoch),
            "val_loss": float(best_val_loss),
            "stopped_early": stopped_early,
        },
        "best_state_dict": best_state_dict,
    }


def train_multitask_cognitive_radio_hybrid_model(
    scalar_train: np.ndarray,
    cov_train: np.ndarray,
    temporal_train: np.ndarray,
    y_train_tasks: dict[str, np.ndarray],
    scalar_val: np.ndarray,
    cov_val: np.ndarray,
    temporal_val: np.ndarray,
    y_val_tasks: dict[str, np.ndarray],
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    scheduler: str | None = "plateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    class_weighting: str | bool = "off",
    task_loss_weights: dict[str, float] | None = None,
    fusion_dim: int = 128,
    attention_heads: int = 4,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for train_multitask_cognitive_radio_hybrid_model")

    scalar_train_np = np.asarray(scalar_train, dtype=np.float32)
    cov_train_np = np.asarray(cov_train, dtype=np.float32)
    temporal_train_np = np.asarray(temporal_train, dtype=np.float32)
    scalar_val_np = np.asarray(scalar_val, dtype=np.float32)
    cov_val_np = np.asarray(cov_val, dtype=np.float32)
    temporal_val_np = np.asarray(temporal_val, dtype=np.float32)

    task_names = sorted(set(y_train_tasks).intersection(set(y_val_tasks)))
    if not task_names:
        raise ValueError("y_train_tasks and y_val_tasks must share at least one task name")

    train_tasks = _normalize_task_labels({name: y_train_tasks[name] for name in task_names}, scalar_train_np.shape[0])
    val_tasks = _normalize_task_labels({name: y_val_tasks[name] for name in task_names}, scalar_val_np.shape[0])
    task_out_classes = {name: _task_num_classes(train_tasks[name], val_tasks[name], name) for name in task_names}

    resolved_device = _resolve_device(requested_device)
    device = torch.device(resolved_device)
    torch.manual_seed(random_seed)
    amp_enabled = _use_amp(amp, resolved_device)
    scaler = _build_grad_scaler(amp_enabled)
    scheduler_name = str(scheduler).lower() if scheduler is not None else "none"
    class_weighting_name = _normalize_class_weighting_name(class_weighting)
    _validate_deep_training_config(
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler_name=scheduler_name,
        class_weighting_name=class_weighting_name,
        sampler_name="off",
        loss_name="cross_entropy",
        focal_gamma=2.0,
    )
    scheduler_kwargs = dict(scheduler_kwargs or {})
    task_loss_weights = {name: float(weight) for name, weight in (task_loss_weights or {}).items()}

    model = _MultiHeadCognitiveRadioHybridModel(
        scalar_dim=int(scalar_train_np.shape[1]),
        cov_channels=int(cov_train_np.shape[1]),
        temporal_channels=int(temporal_train_np.shape[1]),
        task_out_classes=task_out_classes,
        dropout_rate=dropout,
        fusion_dim=fusion_dim,
        attention_heads=attention_heads,
    ).to(device)

    scalar_train_tensor = torch.from_numpy(scalar_train_np)
    cov_train_tensor = torch.from_numpy(cov_train_np)
    temporal_train_tensor = torch.from_numpy(temporal_train_np)
    scalar_val_tensor = torch.from_numpy(scalar_val_np)
    cov_val_tensor = torch.from_numpy(cov_val_np)
    temporal_val_tensor = torch.from_numpy(temporal_val_np)
    train_task_tensors = {name: torch.from_numpy(values) for name, values in train_tasks.items()}
    val_task_tensors = {name: torch.from_numpy(values) for name, values in val_tasks.items()}
    index_loader = DataLoader(
        torch.arange(scalar_train_tensor.shape[0]),
        batch_size=batch_size,
        shuffle=True,
        **_build_dataloader_kwargs(
            resolved_device=resolved_device,
            loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterions: dict[str, Any] = {}
    for task_name in task_names:
        if class_weighting_name in {"on", "balanced"}:
            valid = train_tasks[task_name][train_tasks[task_name] >= 0]
            class_counts = np.bincount(valid, minlength=task_out_classes[task_name]).astype(np.float32)
            safe_counts = np.where(class_counts > 0, class_counts, 1.0)
            weights = class_counts.sum() / (len(class_counts) * safe_counts)
            criterions[task_name] = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
        else:
            criterions[task_name] = nn.CrossEntropyLoss()
    scheduler_obj = _build_scheduler(optimizer, scheduler_name, scheduler_kwargs, epochs, patience)

    history: dict[str, Any] = {
        "loss": [],
        "val_loss": [],
        "learning_rate": [],
        "per_task": {
            name: {
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "train_f1": [],
                "val_accuracy": [],
                "val_f1": [],
            }
            for name in task_names
        },
    }
    best_state_dict: dict[str, Any] | None = None
    best_epoch = -1
    best_val_score = float("-inf")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    _emit_progress(
        progress,
        f"[cognitive_radio_hybrid_multitask] start device={resolved_device} epochs={int(epochs)} tasks={','.join(task_names)} batch_size={batch_size}",
    )

    def _compute_split_metrics(
        scalar_tensor: Any,
        cov_tensor: Any,
        temporal_tensor: Any,
        task_tensors: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        split_metrics: dict[str, Any] = {}
        split_losses: list[float] = []
        with torch.no_grad():
            logits_map = model(
                scalar_tensor.to(device),
                cov_tensor.to(device),
                temporal_tensor.to(device),
            )
            for task_name in task_names:
                targets_all = task_tensors[task_name].to(device)
                valid_mask = targets_all >= 0
                predictions_full = np.full(targets_all.shape[0], -1, dtype=np.int64)
                if not bool(valid_mask.any()):
                    split_metrics[task_name] = {
                        "metrics": {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0},
                        "predictions": predictions_full,
                        "valid_count": 0,
                        "loss": 0.0,
                    }
                    continue
                logits = logits_map[task_name][valid_mask]
                labels = targets_all[valid_mask]
                loss = float(criterions[task_name](logits, labels).item())
                preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
                valid_indices = valid_mask.cpu().numpy().astype(bool)
                predictions_full[valid_indices] = preds
                metrics = evaluate_predictions(labels.cpu().numpy().astype(np.int64), preds)
                split_metrics[task_name] = {
                    "metrics": metrics,
                    "predictions": predictions_full,
                    "valid_count": int(valid_mask.sum().item()),
                    "loss": loss,
                }
                split_losses.append(loss)
        return split_metrics, (sum(split_losses) / len(split_losses) if split_losses else 0.0)

    for epoch_idx in range(int(epochs)):
        model.train()
        running_loss = 0.0
        seen_batches = 0
        for batch_indices in index_loader:
            scalar_batch = scalar_train_tensor[batch_indices].to(device)
            cov_batch = cov_train_tensor[batch_indices].to(device)
            temporal_batch = temporal_train_tensor[batch_indices].to(device)
            with _autocast_context(resolved_device, amp_enabled):
                logits_map = model(scalar_batch, cov_batch, temporal_batch)
            weighted_losses: list[Any] = []
            weights: list[float] = []
            for task_name in task_names:
                labels = train_task_tensors[task_name][batch_indices].to(device)
                valid_mask = labels >= 0
                if not bool(valid_mask.any()):
                    continue
                logits = logits_map[task_name][valid_mask]
                with _autocast_context(resolved_device, amp_enabled):
                    loss = criterions[task_name](logits, labels[valid_mask])
                weight = task_loss_weights.get(task_name, 1.0)
                weighted_losses.append(loss * weight)
                weights.append(float(weight))
            if not weighted_losses:
                continue
            optimizer.zero_grad()
            total_loss = torch.stack(weighted_losses).sum() / max(sum(weights), 1e-12)
            _optimizer_step(total_loss, optimizer, scaler)
            running_loss += float(total_loss.item())
            seen_batches += 1

        train_epoch_loss = running_loss / seen_batches if seen_batches else 0.0
        model.eval()
        train_split_metrics, _ = _compute_split_metrics(
            scalar_train_tensor,
            cov_train_tensor,
            temporal_train_tensor,
            train_task_tensors,
        )
        val_split_metrics, val_epoch_loss = _compute_split_metrics(
            scalar_val_tensor,
            cov_val_tensor,
            temporal_val_tensor,
            val_task_tensors,
        )
        val_scores = [
            score_metric(payload["metrics"], "f1")
            for payload in val_split_metrics.values()
            if payload["valid_count"] > 0
        ]
        mean_val_score = float(sum(val_scores) / len(val_scores)) if val_scores else 0.0

        history["loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        for task_name in task_names:
            train_payload = train_split_metrics[task_name]
            val_payload = val_split_metrics[task_name]
            history["per_task"][task_name]["train_loss"].append(float(train_payload["loss"]))
            history["per_task"][task_name]["val_loss"].append(float(val_payload["loss"]))
            history["per_task"][task_name]["train_accuracy"].append(score_metric(train_payload["metrics"], "accuracy"))
            history["per_task"][task_name]["train_f1"].append(score_metric(train_payload["metrics"], "f1"))
            history["per_task"][task_name]["val_accuracy"].append(score_metric(val_payload["metrics"], "accuracy"))
            history["per_task"][task_name]["val_f1"].append(score_metric(val_payload["metrics"], "f1"))

        if scheduler_obj is not None:
            if scheduler_name == "plateau":
                scheduler_obj.step(val_epoch_loss)
            else:
                scheduler_obj.step()
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        _emit_progress(
            progress,
            (
                f"[cognitive_radio_hybrid_multitask] epoch {epoch_idx + 1}/{int(epochs)} "
                f"train_loss={train_epoch_loss:.4f} val_loss={val_epoch_loss:.4f} "
                f"mean_val_f1={mean_val_score:.4f} lr={float(optimizer.param_groups[0]['lr']):.6f}"
            ),
        )

        if (mean_val_score > best_val_score) or (np.isclose(mean_val_score, best_val_score) and val_epoch_loss < best_val_loss):
            best_val_score = mean_val_score
            best_val_loss = val_epoch_loss
            best_epoch = epoch_idx
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if patience is not None and patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            _emit_progress(progress, f"[cognitive_radio_hybrid_multitask] early_stop epoch={epoch_idx + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    final_train_metrics, _ = _compute_split_metrics(
        scalar_train_tensor,
        cov_train_tensor,
        temporal_train_tensor,
        train_task_tensors,
    )
    final_val_metrics, final_val_loss = _compute_split_metrics(
        scalar_val_tensor,
        cov_val_tensor,
        temporal_val_tensor,
        val_task_tensors,
    )

    per_task_metrics: dict[str, Any] = {}
    val_predictions_tasks: dict[str, np.ndarray] = {}
    for task_name in task_names:
        train_payload = final_train_metrics[task_name]
        val_payload = final_val_metrics[task_name]
        val_predictions_tasks[task_name] = val_payload["predictions"]
        per_task_metrics[task_name] = {
            "train_accuracy": score_metric(train_payload["metrics"], "accuracy"),
            "train_f1": score_metric(train_payload["metrics"], "f1"),
            "val_accuracy": score_metric(val_payload["metrics"], "accuracy"),
            "val_f1": score_metric(val_payload["metrics"], "f1"),
            "train_valid_count": int(train_payload["valid_count"]),
            "val_valid_count": int(val_payload["valid_count"]),
            "train_metrics": train_payload["metrics"],
            "val_metrics": val_payload["metrics"],
        }

    mean_val_accuracy = float(np.mean([metrics["val_accuracy"] for metrics in per_task_metrics.values()])) if per_task_metrics else 0.0
    mean_val_f1 = float(np.mean([metrics["val_f1"] for metrics in per_task_metrics.values()])) if per_task_metrics else 0.0
    return {
        "model": model,
        "model_name": "cognitive_radio_hybrid_multitask",
        "resolved_device": resolved_device,
        "task_names": task_names,
        "train_history": history,
        "per_task_metrics": per_task_metrics,
        "val_predictions_tasks": val_predictions_tasks,
        "val_accuracy": mean_val_accuracy,
        "val_f1": mean_val_f1,
        "val_loss": float(final_val_loss),
        "epochs_ran": len(history["loss"]),
        "best_checkpoint": {
            "epoch": int(best_epoch + 1),
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_score),
            "best_val_loss": float(best_val_loss),
            "stopped_early": stopped_early,
        },
        "best_state_dict": best_state_dict,
    }


def _normalize_task_labels(task_labels: dict[str, np.ndarray], expected_length: int) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for task_name, raw_values in task_labels.items():
        values = np.asarray(raw_values)
        if values.ndim != 1:
            raise ValueError(f"Task '{task_name}' labels must be one-dimensional")
        if values.shape[0] != expected_length:
            raise ValueError(
                f"Task '{task_name}' labels length mismatch: expected {expected_length}, got {values.shape[0]}"
            )

        if values.dtype.kind == "f":
            missing_mask = np.isnan(values) | (values < 0)
            encoded = np.where(missing_mask, -1, values).astype(np.int64)
        else:
            encoded = values.astype(np.int64, copy=True)
            encoded[encoded < 0] = -1
        normalized[task_name] = encoded
    return normalized


def _task_num_classes(train_labels: np.ndarray, val_labels: np.ndarray, task_name: str) -> int:
    valid_train = train_labels[train_labels >= 0]
    valid_val = val_labels[val_labels >= 0]
    combined = np.concatenate([valid_train, valid_val]) if valid_val.size else valid_train
    if combined.size == 0:
        raise ValueError(f"Task '{task_name}' has no valid labels")
    return int(combined.max()) + 1


def _extract_encoder_state_from_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
        if key.startswith("feature_extractor.")
    }


def train_multitask_cnn_model(
    x_train: np.ndarray,
    y_train_tasks: dict[str, np.ndarray],
    x_val: np.ndarray,
    y_val_tasks: dict[str, np.ndarray],
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    scheduler: str | None = "plateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    class_weighting: str | bool = "off",
    task_loss_weights: dict[str, float] | None = None,
    encoder_state_dict: dict[str, Any] | None = None,
    encoder_metadata: dict[str, Any] | None = None,
    transfer_strategy: str | None = None,
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for train_multitask_cnn_model")

    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_val_np = np.asarray(x_val, dtype=np.float32)
    x_train_norm, x_val_norm, in_channels = _normalize_deep_inputs(x_train_np, x_val_np)

    task_names = sorted(set(y_train_tasks).intersection(set(y_val_tasks)))
    if not task_names:
        raise ValueError("y_train_tasks and y_val_tasks must share at least one task name")

    train_tasks = _normalize_task_labels(
        task_labels={name: y_train_tasks[name] for name in task_names},
        expected_length=x_train_norm.shape[0],
    )
    val_tasks = _normalize_task_labels(
        task_labels={name: y_val_tasks[name] for name in task_names},
        expected_length=x_val_norm.shape[0],
    )
    task_out_classes = {
        name: _task_num_classes(train_tasks[name], val_tasks[name], name)
        for name in task_names
    }

    model = _MultiHeadCNN1DModel(
        in_channel_count=in_channels,
        task_out_classes=task_out_classes,
        dropout_rate=dropout,
    )
    if encoder_state_dict is not None:
        proxy = _CNN1DModel(
            in_channel_count=in_channels,
            out_classes=max(task_out_classes.values()),
            dropout_rate=dropout,
        )
        _validate_cnn_encoder_transfer_compatibility(
            model=proxy,
            encoder_state_dict=encoder_state_dict,
            encoder_metadata=encoder_metadata,
        )
        model.load_encoder_from_state_dict(encoder_state_dict)

    normalized_strategy = (transfer_strategy or "").lower()
    if normalized_strategy in {"freeze", "frozen"}:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    resolved_device = _resolve_device(requested_device)
    device = torch.device(resolved_device)
    torch.manual_seed(random_seed)
    amp_enabled = _use_amp(amp, resolved_device)
    scaler = _build_grad_scaler(amp_enabled)
    scheduler_name = str(scheduler).lower() if scheduler is not None else "none"
    class_weighting_name = _normalize_class_weighting_name(class_weighting)
    _validate_deep_training_config(
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler_name=scheduler_name,
        class_weighting_name=class_weighting_name,
        sampler_name="off",
        loss_name="cross_entropy",
        focal_gamma=2.0,
    )
    scheduler_kwargs = dict(scheduler_kwargs or {})
    task_loss_weights = {name: float(weight) for name, weight in (task_loss_weights or {}).items()}

    model = model.to(device)
    train_x_tensor = torch.from_numpy(x_train_norm)
    val_x_tensor = torch.from_numpy(x_val_norm)
    train_task_tensors = {name: torch.from_numpy(values) for name, values in train_tasks.items()}
    val_task_tensors = {name: torch.from_numpy(values) for name, values in val_tasks.items()}
    index_loader = DataLoader(
        torch.arange(train_x_tensor.shape[0]),
        batch_size=batch_size,
        shuffle=True,
        **_build_dataloader_kwargs(
            resolved_device=resolved_device,
            loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterions: dict[str, Any] = {}
    for task_name in task_names:
        if class_weighting_name in {"on", "balanced"}:
            valid = train_tasks[task_name][train_tasks[task_name] >= 0]
            class_counts = np.bincount(valid, minlength=task_out_classes[task_name]).astype(np.float32)
            safe_counts = np.where(class_counts > 0, class_counts, 1.0)
            weights = class_counts.sum() / (len(class_counts) * safe_counts)
            criteria_weights = torch.from_numpy(weights).to(device)
            criterions[task_name] = nn.CrossEntropyLoss(weight=criteria_weights)
        else:
            criterions[task_name] = nn.CrossEntropyLoss()
    scheduler_obj = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
        epochs=epochs,
        patience=patience,
    )

    history: dict[str, Any] = {
        "loss": [],
        "val_loss": [],
        "learning_rate": [],
        "per_task": {
            name: {
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "train_f1": [],
                "val_accuracy": [],
                "val_f1": [],
            }
            for name in task_names
        },
    }
    best_state_dict: dict[str, Any] | None = None
    best_epoch = -1
    best_val_score = float("-inf")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    _emit_progress(
        progress,
        f"[cnn_1d_multitask] start device={resolved_device} epochs={int(epochs)} tasks={','.join(task_names)} batch_size={batch_size}",
    )

    def _compute_split_metrics(split_x_tensor: Any, task_tensors: dict[str, Any]) -> tuple[dict[str, Any], float]:
        split_metrics: dict[str, Any] = {}
        split_losses: list[float] = []
        with torch.no_grad():
            logits_map = model(split_x_tensor.to(device))
            for task_name in task_names:
                targets_all = task_tensors[task_name].to(device)
                valid_mask = targets_all >= 0
                predictions_full = np.full(targets_all.shape[0], -1, dtype=np.int64)
                if not bool(valid_mask.any()):
                    split_metrics[task_name] = {
                        "metrics": {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0},
                        "predictions": predictions_full,
                        "valid_count": 0,
                        "loss": 0.0,
                    }
                    continue

                logits = logits_map[task_name][valid_mask]
                labels = targets_all[valid_mask]
                loss = float(criterions[task_name](logits, labels).item())
                preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
                valid_indices = valid_mask.cpu().numpy().astype(bool)
                predictions_full[valid_indices] = preds

                labels_np = labels.cpu().numpy().astype(np.int64)
                metrics = evaluate_predictions(labels_np, preds)
                split_metrics[task_name] = {
                    "metrics": metrics,
                    "predictions": predictions_full,
                    "valid_count": int(valid_mask.sum().item()),
                    "loss": loss,
                }
                split_losses.append(loss)
        return split_metrics, (sum(split_losses) / len(split_losses) if split_losses else 0.0)

    for epoch_idx in range(int(epochs)):
        model.train()
        running_loss = 0.0
        seen_batches = 0
        for batch_indices in index_loader:
            batch_x = train_x_tensor[batch_indices].to(device)
            with _autocast_context(resolved_device, amp_enabled):
                logits_map = model(batch_x)
            weighted_losses: list[Any] = []
            weights: list[float] = []
            for task_name in task_names:
                labels = train_task_tensors[task_name][batch_indices].to(device)
                valid_mask = labels >= 0
                if not bool(valid_mask.any()):
                    continue
                logits = logits_map[task_name][valid_mask]
                with _autocast_context(resolved_device, amp_enabled):
                    loss = criterions[task_name](logits, labels[valid_mask])
                weight = task_loss_weights.get(task_name, 1.0)
                weighted_losses.append(loss * weight)
                weights.append(float(weight))

            if not weighted_losses:
                continue

            optimizer.zero_grad()
            total_loss = torch.stack(weighted_losses).sum() / max(sum(weights), 1e-12)
            _optimizer_step(total_loss, optimizer, scaler)
            running_loss += float(total_loss.item())
            seen_batches += 1

        train_epoch_loss = running_loss / seen_batches if seen_batches else 0.0
        model.eval()
        train_split_metrics, _ = _compute_split_metrics(train_x_tensor, train_task_tensors)
        val_split_metrics, val_epoch_loss = _compute_split_metrics(val_x_tensor, val_task_tensors)
        val_scores = [
            score_metric(payload["metrics"], "f1")
            for payload in val_split_metrics.values()
            if payload["valid_count"] > 0
        ]
        mean_val_score = float(sum(val_scores) / len(val_scores)) if val_scores else 0.0

        history["loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        for task_name in task_names:
            train_payload = train_split_metrics[task_name]
            val_payload = val_split_metrics[task_name]
            history["per_task"][task_name]["train_loss"].append(float(train_payload["loss"]))
            history["per_task"][task_name]["val_loss"].append(float(val_payload["loss"]))
            history["per_task"][task_name]["train_accuracy"].append(score_metric(train_payload["metrics"], "accuracy"))
            history["per_task"][task_name]["train_f1"].append(score_metric(train_payload["metrics"], "f1"))
            history["per_task"][task_name]["val_accuracy"].append(score_metric(val_payload["metrics"], "accuracy"))
            history["per_task"][task_name]["val_f1"].append(score_metric(val_payload["metrics"], "f1"))

        if scheduler_obj is not None:
            if scheduler_name == "plateau":
                scheduler_obj.step(val_epoch_loss)
            else:
                scheduler_obj.step()
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        _emit_progress(
            progress,
            (
                f"[cnn_1d_multitask] epoch {epoch_idx + 1}/{int(epochs)} "
                f"train_loss={train_epoch_loss:.4f} val_loss={val_epoch_loss:.4f} "
                f"mean_val_f1={mean_val_score:.4f} lr={float(optimizer.param_groups[0]['lr']):.6f}"
            ),
        )

        if (mean_val_score > best_val_score) or (
            np.isclose(mean_val_score, best_val_score) and val_epoch_loss < best_val_loss
        ):
            best_val_score = mean_val_score
            best_val_loss = val_epoch_loss
            best_epoch = epoch_idx
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if patience is not None and patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            _emit_progress(progress, f"[cnn_1d_multitask] early_stop epoch={epoch_idx + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()
    final_train_metrics, _ = _compute_split_metrics(train_x_tensor, train_task_tensors)
    final_val_metrics, final_val_loss = _compute_split_metrics(val_x_tensor, val_task_tensors)

    per_task_metrics: dict[str, Any] = {}
    val_predictions_tasks: dict[str, np.ndarray] = {}
    for task_name in task_names:
        train_payload = final_train_metrics[task_name]
        val_payload = final_val_metrics[task_name]
        val_predictions_tasks[task_name] = val_payload["predictions"]
        per_task_metrics[task_name] = {
            "train_accuracy": score_metric(train_payload["metrics"], "accuracy"),
            "train_f1": score_metric(train_payload["metrics"], "f1"),
            "val_accuracy": score_metric(val_payload["metrics"], "accuracy"),
            "val_f1": score_metric(val_payload["metrics"], "f1"),
            "train_valid_count": int(train_payload["valid_count"]),
            "val_valid_count": int(val_payload["valid_count"]),
            "train_metrics": train_payload["metrics"],
            "val_metrics": val_payload["metrics"],
        }

    mean_val_accuracy = float(
        np.mean([metrics["val_accuracy"] for metrics in per_task_metrics.values()])
    ) if per_task_metrics else 0.0
    mean_val_f1 = float(
        np.mean([metrics["val_f1"] for metrics in per_task_metrics.values()])
    ) if per_task_metrics else 0.0

    return {
        "model": model,
        "model_name": "cnn_1d_multitask",
        "resolved_device": resolved_device,
        "task_names": task_names,
        "train_history": history,
        "per_task_metrics": per_task_metrics,
        "val_predictions_tasks": val_predictions_tasks,
        "val_accuracy": mean_val_accuracy,
        "val_f1": mean_val_f1,
        "val_loss": float(final_val_loss),
        "epochs_ran": len(history["loss"]),
        "best_checkpoint": {
            "epoch": int(best_epoch + 1),
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_score),
            "best_val_loss": float(best_val_loss),
            "stopped_early": stopped_early,
        },
        "best_state_dict": best_state_dict,
        "shared_encoder_state_dict": _extract_encoder_state_from_state_dict(
            best_state_dict if best_state_dict is not None else model.state_dict()
        ),
        "encoder_metadata": _build_cnn_encoder_metadata(in_channels=in_channels),
    }


def pretrain_cnn_encoder(
    x_train: np.ndarray,
    x_val: np.ndarray | None = None,
    requested_device: str = "auto",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int | None = 5,
    mask_ratio: float = 0.15,
    objective: str = "masked_reconstruction",
    loader_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool | str | None = "auto",
    progress: bool = False,
) -> dict[str, Any]:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for pretrain_cnn_encoder")
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in [0, 1)")
    if objective not in {"masked_reconstruction", "denoising_reconstruction"}:
        raise ValueError("objective must be one of: masked_reconstruction, denoising_reconstruction")

    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_val_np = np.asarray(x_val, dtype=np.float32) if x_val is not None else np.asarray(x_train_np, dtype=np.float32)
    x_train_norm, x_val_norm, in_channels = _normalize_deep_inputs(x_train_np, x_val_np)
    sequence_length = int(x_train_norm.shape[-1])

    resolved_device = _resolve_device(requested_device)
    device = torch.device(resolved_device)
    torch.manual_seed(random_seed)
    amp_enabled = _use_amp(amp, resolved_device)
    scaler = _build_grad_scaler(amp_enabled)
    _validate_deep_training_config(
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        scheduler_name="none",
        class_weighting_name="off",
        sampler_name="off",
        loss_name="cross_entropy",
        focal_gamma=2.0,
    )

    model = _CNN1DPretrainModel(
        in_channel_count=in_channels,
        sequence_length=sequence_length,
        dropout_rate=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_x_tensor = torch.from_numpy(x_train_norm)
    val_x_tensor = torch.from_numpy(x_val_norm)
    train_loader = DataLoader(
        train_x_tensor,
        batch_size=batch_size,
        shuffle=True,
        **_build_dataloader_kwargs(
            resolved_device=resolved_device,
            loader_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state_dict: dict[str, Any] | None = None
    best_epoch = -1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    _emit_progress(
        progress,
        (
            f"[cnn_1d_encoder_pretrain] start device={resolved_device} epochs={int(epochs)} "
            f"objective={objective} batch_size={batch_size}"
        ),
    )

    for epoch_idx in range(int(epochs)):
        model.train()
        running_loss = 0.0
        seen_batches = 0
        for clean_batch in train_loader:
            clean_batch = clean_batch.to(device)
            if objective == "masked_reconstruction":
                mask = torch.rand_like(clean_batch) < mask_ratio
                corrupted = clean_batch.clone()
                corrupted[mask] = 0.0
            else:
                noise = torch.randn_like(clean_batch) * mask_ratio
                corrupted = clean_batch + noise

            optimizer.zero_grad()
            with _autocast_context(resolved_device, amp_enabled):
                recon = model(corrupted)
                loss = criterion(recon, clean_batch)
            _optimizer_step(loss, optimizer, scaler)

            running_loss += float(loss.item())
            seen_batches += 1

        train_loss = running_loss / seen_batches if seen_batches else 0.0
        model.eval()
        with torch.no_grad():
            clean_val = val_x_tensor.to(device)
            if objective == "masked_reconstruction":
                mask_val = torch.rand_like(clean_val) < mask_ratio
                corrupted_val = clean_val.clone()
                corrupted_val[mask_val] = 0.0
            else:
                noise_val = torch.randn_like(clean_val) * mask_ratio
                corrupted_val = clean_val + noise_val
            val_recon = model(corrupted_val)
            val_loss = float(criterion(val_recon, clean_val).item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        _emit_progress(
            progress,
            (
                f"[cnn_1d_encoder_pretrain] epoch {epoch_idx + 1}/{int(epochs)} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            ),
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if patience is not None and patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            _emit_progress(progress, f"[cnn_1d_encoder_pretrain] early_stop epoch={epoch_idx + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_state = best_state_dict if best_state_dict is not None else model.state_dict()
    encoder_state_dict = _extract_encoder_state_from_state_dict(final_state)
    encoder_metadata = _build_cnn_encoder_metadata(in_channels=in_channels)
    encoder_metadata.update(
        {
            "objective": objective,
            "mask_ratio": float(mask_ratio),
            "sequence_length": sequence_length,
        }
    )

    return {
        "model": model,
        "model_name": "cnn_1d_encoder_pretrain",
        "resolved_device": resolved_device,
        "train_history": history,
        "epochs_ran": len(history["train_loss"]),
        "best_checkpoint": {
            "epoch": int(best_epoch + 1),
            "best_epoch": int(best_epoch),
            "val_loss": float(best_val_loss),
            "stopped_early": stopped_early,
        },
        "best_state_dict": best_state_dict,
        "encoder_state_dict": encoder_state_dict,
        "encoder_metadata": encoder_metadata,
    }

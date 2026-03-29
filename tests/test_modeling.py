from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emc_diag.evaluation import evaluate_predictions, score_metric
from emc_diag.modeling import (
    _evaluate_candidate,
    pretrain_cnn_encoder,
    train_baseline_model,
    train_cognitive_radio_hybrid_model,
    train_cognitive_radio_scalar_hybrid_model,
    train_multitask_cognitive_radio_hybrid_model,
    train_cnn_lstm_model,
    train_cnn_model,
    train_cnn_residual_model,
    train_multitask_cnn_model,
    train_transformer_1d_model,
)


def _make_classification_data(
    n_samples: int = 120,
    n_features: int = 8,
    n_classes: int = 3,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, n_features)) * 2.0
    labels = rng.integers(0, n_classes, size=n_samples)
    features = centers[labels] + rng.normal(scale=0.5, size=(n_samples, n_features))
    return features.astype(np.float32), labels.astype(np.int64)


def test_train_baseline_model_smoke() -> None:
    x, y = _make_classification_data()
    model_info = train_baseline_model(
        x_train=x[:90],
        y_train=y[:90],
        x_val=x[90:],
        y_val=y[90:],
        random_state=11,
    )
    assert "model" in model_info
    assert model_info["model_name"] in {
        "random_forest",
        "extra_trees",
        "logistic_regression",
        "bagged_logistic_regression",
        "svc",
        "linear_svc",
        "mlp_classifier",
        "majority_fallback",
    }
    assert 0.0 <= model_info["val_accuracy"] <= 1.0


@pytest.mark.parametrize(
    "baseline",
    ["random_forest", "extra_trees", "logistic_regression", "bagged_logistic_regression", "svc", "linear_svc", "mlp_classifier"],
)
def test_train_baseline_model_supports_explicit_baseline_selection(baseline: str) -> None:
    x, y = _make_classification_data(n_samples=140, n_features=12, n_classes=3, seed=13)
    model_info = train_baseline_model(
        x_train=x[:100],
        y_train=y[:100],
        x_val=x[100:],
        y_val=y[100:],
        random_state=13,
        baseline=baseline,
    )
    assert model_info["selected_baseline"] in {
        "random_forest",
        "extra_trees",
        "logistic_regression",
        "bagged_logistic_regression",
        "svc",
        "linear_svc",
        "mlp_classifier",
        "majority_fallback",
    }
    assert 0.0 <= model_info["val_accuracy"] <= 1.0


def test_train_baseline_model_auto_records_candidate_scores() -> None:
    x, y = _make_classification_data(n_samples=180, n_features=16, n_classes=2, seed=23)
    model_info = train_baseline_model(
        x_train=x[:130],
        y_train=y[:130],
        x_val=x[130:],
        y_val=y[130:],
        random_state=23,
        baseline="auto",
    )
    assert model_info["selected_baseline"] == model_info["model_name"]
    assert isinstance(model_info["candidate_scores"], dict)
    assert len(model_info["candidate_scores"]) >= 1
    first_payload = next(iter(model_info["candidate_scores"].values()))
    assert "accuracy" in first_payload
    assert "f1" in first_payload


def test_train_baseline_model_supports_candidate_filter_and_threshold_tuning() -> None:
    x, y = _make_classification_data(n_samples=180, n_features=10, n_classes=2, seed=31)
    model_info = train_baseline_model(
        x_train=x[:130],
        y_train=y[:130],
        x_val=x[130:],
        y_val=y[130:],
        random_state=31,
        baseline="auto",
        candidates=["logistic_regression", "svc"],
        threshold_tuning={"enabled": True, "metric": "f1", "grid": [0.4, 0.5, 0.6]},
    )

    assert set(model_info["candidate_scores"]).issubset({"logistic_regression", "svc"})
    assert model_info["threshold"] is None or model_info["threshold"] in {0.4, 0.5, 0.6}


def test_train_baseline_model_supports_baseline_params_for_svc() -> None:
    x, y = _make_classification_data(n_samples=160, n_features=10, n_classes=2, seed=37)
    model_info = train_baseline_model(
        x_train=x[:120],
        y_train=y[:120],
        x_val=x[120:],
        y_val=y[120:],
        random_state=37,
        baseline="svc",
        baseline_params={"C": 7.0, "gamma": 0.04, "class_weight": "balanced"},
    )

    svc_model = model_info["model"].named_steps["svc"]
    assert svc_model.C == pytest.approx(7.0)
    assert svc_model.gamma == pytest.approx(0.04)
    assert svc_model.class_weight == "balanced"


def test_evaluate_candidate_prefers_native_predict_when_threshold_tuning_is_worse() -> None:
    class _DummyProbModel:
        def predict(self, _x: np.ndarray) -> np.ndarray:
            return np.array([0, 1, 0, 1], dtype=np.int64)

        def predict_proba(self, _x: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    [0.4, 0.6],
                    [0.4, 0.6],
                    [0.6, 0.4],
                    [0.6, 0.4],
                ],
                dtype=float,
            )

    x_val = np.zeros((4, 2), dtype=np.float32)
    y_val = np.array([0, 1, 0, 1], dtype=np.int64)

    evaluation = _evaluate_candidate(
        _DummyProbModel(),
        x_val=x_val,
        y_val=y_val,
        primary_metric="f1",
        secondary_metric="accuracy",
        threshold_tuning={"enabled": True, "metric": "f1", "grid": [0.4, 0.5, 0.6]},
    )

    np.testing.assert_array_equal(evaluation["val_predictions"], y_val)
    assert evaluation["threshold"] is None
    assert evaluation["metrics"]["f1"] == pytest.approx(1.0)


def test_evaluate_predictions_exposes_minority_class_metrics() -> None:
    y_true = np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 1, 0, 0], dtype=np.int64)

    metrics = evaluate_predictions(y_true, y_pred)

    assert "minority_label" in metrics
    assert metrics["minority_label"] == 1
    assert "minority_f1" in metrics
    assert "minority_precision" in metrics
    assert "minority_recall" in metrics
    assert score_metric(metrics, "minority_f1") == metrics["minority_f1"]
    assert score_metric(metrics, "minority_recall") == metrics["minority_recall"]


def test_train_baseline_model_rejects_unknown_baseline() -> None:
    x, y = _make_classification_data()
    with pytest.raises(ValueError, match="Unsupported baseline"):
        train_baseline_model(
            x_train=x[:90],
            y_train=y[:90],
            x_val=x[90:],
            y_val=y[90:],
            baseline="unknown_baseline",
        )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_train_cnn_model_smoke_and_device_support(device: str) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")
    x, y = _make_classification_data(n_samples=96, n_features=16, n_classes=2, seed=9)
    model_info = train_cnn_model(
        x_train=x[:72],
        y_train=y[:72],
        x_val=x[72:],
        y_val=y[72:],
        requested_device=device,
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        random_seed=42,
        loader_workers=1 if device == "cpu" else 0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        amp=False,
    )
    assert "model" in model_info
    assert "train_history" in model_info
    assert len(model_info["train_history"]["train_loss"]) >= 1
    assert len(model_info["train_history"]["val_loss"]) >= 1
    assert len(model_info["train_history"]["val_accuracy"]) >= 1
    assert len(model_info["train_history"]["val_f1"]) >= 1
    assert len(model_info["train_history"]["learning_rate"]) >= 1
    assert model_info["resolved_device"] in {"cpu", "mps", "cuda"}
    assert 0.0 <= model_info["val_accuracy"] <= 1.0
    assert "best_checkpoint" in model_info
    assert "best_state_dict" in model_info
    assert "epochs_ran" in model_info


def test_train_cnn_model_supports_regularization_scheduler_and_early_stopping() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")
    x, y = _make_classification_data(n_samples=160, n_features=20, n_classes=2, seed=19)
    model_info = train_cnn_model(
        x_train=x[:120],
        y_train=y[:120],
        x_val=x[120:],
        y_val=y[120:],
        requested_device="cpu",
        epochs=12,
        batch_size=16,
        learning_rate=1e-2,
        random_seed=19,
        dropout=0.35,
        weight_decay=1e-3,
        patience=2,
        scheduler="step",
        scheduler_kwargs={"step_size": 2, "gamma": 0.7},
        class_weighting=True,
    )
    assert 1 <= model_info["epochs_ran"] <= 12
    assert model_info["best_checkpoint"]["epoch"] <= model_info["epochs_ran"]
    assert "val_loss" in model_info["best_checkpoint"]
    assert "val_accuracy" in model_info["best_checkpoint"]
    assert "val_f1" in model_info["best_checkpoint"]
    assert isinstance(model_info["best_state_dict"], dict)
    assert len(model_info["best_state_dict"]) > 0
    assert len(model_info["train_history"]["learning_rate"]) == model_info["epochs_ran"]
    assert len(model_info["train_history"]["train_loss"]) == model_info["epochs_ran"]
    assert model_info["val_predictions"].shape[0] == y[120:].shape[0]
    assert model_info["val_accuracy"] == pytest.approx(model_info["best_checkpoint"]["val_accuracy"], rel=1e-6)


def test_train_cnn_model_supports_focal_loss_and_balanced_sampler() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")

    x, y = _make_classification_data(n_samples=120, n_features=12, n_classes=2, seed=29)
    y[:90] = 0
    y[90:] = 1
    model_info = train_cnn_model(
        x_train=x[:96],
        y_train=y[:96],
        x_val=x[96:],
        y_val=y[96:],
        requested_device="cpu",
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        random_seed=29,
        class_weighting="balanced",
        sampler="balanced",
        loss_name="focal",
        focal_gamma=2.0,
    )

    assert model_info["model_name"] == "cnn_1d"
    assert model_info["epochs_ran"] >= 1
    assert model_info["resolved_device"] == "cpu"


@pytest.mark.parametrize(
    ("trainer_fn", "expected_model_name"),
    [
        (train_cnn_lstm_model, "cnn_lstm"),
        (train_transformer_1d_model, "transformer_1d"),
    ],
)
def test_additional_deep_models_smoke(trainer_fn, expected_model_name: str) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")
    x, y = _make_classification_data(n_samples=120, n_features=24, n_classes=2, seed=123)
    x_train = x[:90].reshape(90, 3, 8)
    x_val = x[90:].reshape(30, 3, 8)
    model_info = trainer_fn(
        x_train=x_train,
        y_train=y[:90],
        x_val=x_val,
        y_val=y[90:],
        requested_device="cpu",
        epochs=4,
        batch_size=16,
        learning_rate=1e-2,
        random_seed=123,
        dropout=0.2,
        weight_decay=1e-4,
        patience=2,
        scheduler="step",
        scheduler_kwargs={"step_size": 2, "gamma": 0.7},
        class_weighting="balanced",
    )

    assert model_info["model_name"] == expected_model_name
    assert model_info["resolved_device"] == "cpu"
    assert model_info["epochs_ran"] >= 1
    assert len(model_info["train_history"]["train_loss"]) == model_info["epochs_ran"]
    assert len(model_info["train_history"]["val_loss"]) == model_info["epochs_ran"]
    assert len(model_info["train_history"]["learning_rate"]) == model_info["epochs_ran"]
    assert model_info["val_predictions"].shape[0] == y[90:].shape[0]
    assert "best_checkpoint" in model_info
    assert "best_state_dict" in model_info


def test_train_cnn_residual_model_smoke() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")

    x, y = _make_classification_data(n_samples=96, n_features=18, n_classes=2, seed=144)
    model_info = train_cnn_residual_model(
        x_train=x[:72],
        y_train=y[:72],
        x_val=x[72:],
        y_val=y[72:],
        requested_device="cpu",
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        random_seed=144,
        dropout=0.2,
    )

    assert model_info["model_name"] == "cnn_1d_residual"
    assert model_info["resolved_device"] == "cpu"
    assert model_info["epochs_ran"] >= 1


def test_train_cognitive_radio_hybrid_model_smoke() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")

    rng = np.random.default_rng(2026)
    scalar_train = rng.normal(size=(24, 3)).astype(np.float32)
    scalar_val = rng.normal(size=(8, 3)).astype(np.float32)
    cov_train = rng.normal(size=(24, 3, 6)).astype(np.float32)
    cov_val = rng.normal(size=(8, 3, 6)).astype(np.float32)
    temporal_train = rng.normal(size=(24, 3, 8)).astype(np.float32)
    temporal_val = rng.normal(size=(8, 3, 8)).astype(np.float32)
    y_train = rng.integers(0, 2, size=24).astype(np.int64)
    y_val = rng.integers(0, 2, size=8).astype(np.int64)

    model_info = train_cognitive_radio_hybrid_model(
        scalar_train=scalar_train,
        cov_train=cov_train,
        temporal_train=temporal_train,
        y_train=y_train,
        scalar_val=scalar_val,
        cov_val=cov_val,
        temporal_val=temporal_val,
        y_val=y_val,
        requested_device="cpu",
        epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        random_seed=17,
        dropout=0.2,
    )

    assert model_info["model_name"] == "cognitive_radio_hybrid"
    assert model_info["resolved_device"] == "cpu"
    assert model_info["epochs_ran"] >= 1
    assert len(model_info["train_history"]["train_loss"]) == model_info["epochs_ran"]
    assert model_info["val_predictions"].shape[0] == y_val.shape[0]


def test_train_cognitive_radio_scalar_hybrid_model_smoke() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")

    rng = np.random.default_rng(2028)
    scalar_train = rng.normal(size=(24, 5)).astype(np.float32)
    scalar_val = rng.normal(size=(8, 5)).astype(np.float32)
    cov_train = rng.normal(size=(24, 3, 6)).astype(np.float32)
    cov_val = rng.normal(size=(8, 3, 6)).astype(np.float32)
    temporal_train = rng.normal(size=(24, 3, 8)).astype(np.float32)
    temporal_val = rng.normal(size=(8, 3, 8)).astype(np.float32)
    y_train = rng.integers(0, 2, size=24).astype(np.int64)
    y_val = rng.integers(0, 2, size=8).astype(np.int64)

    model_info = train_cognitive_radio_scalar_hybrid_model(
        scalar_train=scalar_train,
        cov_train=cov_train,
        temporal_train=temporal_train,
        y_train=y_train,
        scalar_val=scalar_val,
        cov_val=cov_val,
        temporal_val=temporal_val,
        y_val=y_val,
        requested_device="cpu",
        epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        random_seed=28,
        dropout=0.2,
    )

    assert model_info["model_name"] == "cognitive_radio_scalar_hybrid"
    assert model_info["resolved_device"] == "cpu"
    assert model_info["epochs_ran"] >= 1


def test_train_multitask_cognitive_radio_hybrid_model_smoke() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")

    rng = np.random.default_rng(2027)
    scalar_train = rng.normal(size=(24, 3)).astype(np.float32)
    scalar_val = rng.normal(size=(8, 3)).astype(np.float32)
    cov_train = rng.normal(size=(24, 3, 6)).astype(np.float32)
    cov_val = rng.normal(size=(8, 3, 6)).astype(np.float32)
    temporal_train = rng.normal(size=(24, 3, 8)).astype(np.float32)
    temporal_val = rng.normal(size=(8, 3, 8)).astype(np.float32)
    y_presence_train = rng.integers(0, 2, size=24).astype(np.int64)
    y_presence_val = rng.integers(0, 2, size=8).astype(np.int64)
    y_band_train = rng.integers(0, 3, size=24).astype(np.int64)
    y_band_val = rng.integers(0, 3, size=8).astype(np.int64)

    model_info = train_multitask_cognitive_radio_hybrid_model(
        scalar_train=scalar_train,
        cov_train=cov_train,
        temporal_train=temporal_train,
        y_train_tasks={"presence": y_presence_train, "band": y_band_train},
        scalar_val=scalar_val,
        cov_val=cov_val,
        temporal_val=temporal_val,
        y_val_tasks={"presence": y_presence_val, "band": y_band_val},
        requested_device="cpu",
        epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        random_seed=27,
        dropout=0.2,
    )

    assert model_info["model_name"] == "cognitive_radio_hybrid_multitask"
    assert set(model_info["task_names"]) == {"presence", "band"}
    assert model_info["epochs_ran"] >= 1
    assert "presence" in model_info["per_task_metrics"]


def test_train_transformer_1d_model_validates_attention_dimensions() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")
    x, y = _make_classification_data(n_samples=80, n_features=12, n_classes=2, seed=202)
    with pytest.raises(ValueError, match="d_model must be divisible by nhead"):
        train_transformer_1d_model(
            x_train=x[:60],
            y_train=y[:60],
            x_val=x[60:],
            y_val=y[60:],
            requested_device="cpu",
            epochs=2,
            batch_size=8,
            learning_rate=1e-2,
            random_seed=11,
            d_model=30,
            nhead=8,
        )


@pytest.mark.parametrize(
    ("scheduler", "class_weighting"),
    [("unknown", "off"), ("none", "bad_weighting")],
)
def test_train_cnn_model_rejects_invalid_config_options(scheduler: str, class_weighting: str) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment")
    x, y = _make_classification_data(n_samples=80, n_features=12, n_classes=2, seed=101)
    with pytest.raises(ValueError):
        train_cnn_model(
            x_train=x[:60],
            y_train=y[:60],
            x_val=x[60:],
            y_val=y[60:],
            requested_device="cpu",
            epochs=2,
            batch_size=8,
            learning_rate=1e-2,
            random_seed=11,
            scheduler=scheduler,
            class_weighting=class_weighting,
        )


def test_run_noise_robustness_sweep_shapes_and_order() -> None:
    x = np.zeros((5, 3), dtype=float)
    y = np.array([0, 1, 0, 1, 0], dtype=np.int64)

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        # simple deterministic classifier: always predicts 0
        return np.zeros(batch.shape[0], dtype=np.int64)

    from emc_diag.evaluation import run_noise_robustness_sweep

    df = run_noise_robustness_sweep(
        model_predict_fn=predict_fn,
        x_clean=x,
        y_true=y,
        noise_levels=[0.0, 0.1, 0.05],
        metric="macro_f1",
        random_state=7,
    )

    assert list(df.columns) == ["noise_sigma", "macro_f1"]
    assert df.shape[0] == 3
    assert df["noise_sigma"].tolist() == sorted(df["noise_sigma"].tolist())


@pytest.mark.parametrize("device", ["cpu"])  # keep transfer test simple and fast
def test_train_cnn_model_supports_encoder_transfer_freeze(device: str) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for CNN transfer test")

    x, y = _make_classification_data(n_samples=40, n_features=8, n_classes=2, seed=321)
    x_train, x_val = x[:30], x[30:]
    y_train, y_val = y[:30], y[30:]

    # First run: baseline training to get a reference encoder state_dict.
    first = train_cnn_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        requested_device=device,
        epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        random_seed=1,
        dropout=0.1,
    )

    best_state = first["best_state_dict"]
    assert isinstance(best_state, dict)
    encoder_state = {k: v for k, v in best_state.items() if k.startswith("feature_extractor.")}
    assert encoder_state, "encoder_state_dict should not be empty"

    # Second run: use encoder_state_dict with freeze strategy. This is a smoke
    # test to ensure the transfer arguments are wired correctly.
    second = train_cnn_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        requested_device=device,
        epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        random_seed=2,
        dropout=0.1,
        encoder_state_dict=encoder_state,
        transfer_strategy="freeze",
    )

    assert second["model_name"] == "cnn_1d"
    assert second["resolved_device"] in {"cpu", "mps", "cuda"}
    assert second["epochs_ran"] >= 1


def test_train_cnn_model_rejects_incompatible_encoder_metadata() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for CNN transfer metadata test")

    x, y = _make_classification_data(n_samples=36, n_features=12, n_classes=2, seed=222)
    x_small = x[:30].reshape(30, 2, 6)
    x_small_val = x[30:].reshape(6, 2, 6)
    first = train_cnn_model(
        x_train=x_small,
        y_train=y[:30],
        x_val=x_small_val,
        y_val=y[30:],
        requested_device="cpu",
        epochs=1,
        batch_size=6,
        learning_rate=1e-2,
        random_seed=3,
        dropout=0.1,
    )

    x_other = x[:30].reshape(30, 3, 4)
    x_other_val = x[30:].reshape(6, 3, 4)
    with pytest.raises(ValueError, match="Incompatible encoder (weight shape|in_channels)"):
        train_cnn_model(
            x_train=x_other,
            y_train=y[:30],
            x_val=x_other_val,
            y_val=y[30:],
            requested_device="cpu",
            epochs=1,
            batch_size=6,
            learning_rate=1e-2,
            random_seed=4,
            dropout=0.1,
            encoder_state_dict=first["best_state_dict"],
            encoder_metadata=first["encoder_metadata"],
            transfer_strategy="freeze",
        )


def test_train_multitask_cnn_model_smoke_with_missing_labels() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for multitask CNN test")

    x, y = _make_classification_data(n_samples=72, n_features=10, n_classes=2, seed=444)
    rng = np.random.default_rng(444)
    y_aux = rng.integers(0, 3, size=72).astype(np.int64)

    y_presence = y.copy()
    y_aux_missing = y_aux.copy()
    y_presence[[5, 9, 40]] = -1
    y_aux_missing[[2, 12, 41, 50]] = -1

    model_info = train_multitask_cnn_model(
        x_train=x[:54],
        y_train_tasks={
            "presence": y_presence[:54],
            "band": y_aux_missing[:54],
        },
        x_val=x[54:],
        y_val_tasks={
            "presence": y_presence[54:],
            "band": y_aux_missing[54:],
        },
        requested_device="cpu",
        epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        random_seed=12,
        dropout=0.1,
    )

    assert model_info["model_name"] == "cnn_1d_multitask"
    assert model_info["resolved_device"] == "cpu"
    assert model_info["epochs_ran"] >= 1
    assert set(model_info["task_names"]) == {"presence", "band"}
    assert "presence" in model_info["per_task_metrics"]
    assert "band" in model_info["per_task_metrics"]
    assert "presence" in model_info["val_predictions_tasks"]
    assert model_info["val_predictions_tasks"]["presence"].shape[0] == 18
    assert isinstance(model_info["shared_encoder_state_dict"], dict)
    assert model_info["shared_encoder_state_dict"]
    assert 0.0 <= model_info["per_task_metrics"]["presence"]["val_accuracy"] <= 1.0
    assert 0.0 <= model_info["per_task_metrics"]["band"]["val_accuracy"] <= 1.0


def test_pretrain_cnn_encoder_smoke_and_transfer() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not available for encoder pretraining test")

    x, y = _make_classification_data(n_samples=64, n_features=12, n_classes=2, seed=555)
    pretrain_info = pretrain_cnn_encoder(
        x_train=x[:48],
        x_val=x[48:],
        requested_device="cpu",
        epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        random_seed=10,
        dropout=0.1,
        mask_ratio=0.2,
    )

    assert pretrain_info["model_name"] == "cnn_1d_encoder_pretrain"
    assert pretrain_info["resolved_device"] == "cpu"
    assert pretrain_info["epochs_ran"] >= 1
    assert len(pretrain_info["train_history"]["train_loss"]) == pretrain_info["epochs_ran"]
    assert len(pretrain_info["train_history"]["val_loss"]) == pretrain_info["epochs_ran"]
    assert isinstance(pretrain_info["encoder_state_dict"], dict)
    assert pretrain_info["encoder_state_dict"]
    assert pretrain_info["encoder_metadata"]["encoder_arch"] == "cnn_1d_encoder"

    transfer_info = train_cnn_model(
        x_train=x[:48],
        y_train=y[:48],
        x_val=x[48:],
        y_val=y[48:],
        requested_device="cpu",
        epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        random_seed=11,
        dropout=0.1,
        encoder_state_dict=pretrain_info["encoder_state_dict"],
        encoder_metadata=pretrain_info["encoder_metadata"],
        transfer_strategy="freeze",
    )
    assert transfer_info["model_name"] == "cnn_1d"
    assert transfer_info["epochs_ran"] >= 1

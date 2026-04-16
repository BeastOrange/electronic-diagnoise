from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import emc_diag.llm_modeling as llm_modeling


class _FakeModel:
    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.best_marker = "initial"
        self.saved_marker = ""
        self.loaded_markers: list[str] = []
        self.config = type("Config", (), {"pad_token_id": 0, "use_cache": False})()

    def parameters(self):
        yield self.param

    def train(self) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"marker": torch.tensor([len(self.best_marker)], dtype=torch.float32)}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        _ = strict
        marker_length = int(state_dict["marker"].item())
        self.best_marker = "best" if marker_length == 4 else "last"
        self.loaded_markers.append(self.best_marker)

    def save_pretrained(self, output_dir: Path, safe_serialization: bool = True) -> None:
        _ = output_dir
        _ = safe_serialization
        self.saved_marker = self.best_marker

    def gradient_checkpointing_enable(self) -> None:
        return None


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 0

    def save_pretrained(self, output_dir: Path) -> None:
        _ = output_dir


class _FakeClassifier:
    def __init__(self, *, model: _FakeModel, **_: object) -> None:
        self.model = model

    def predict_proba(self, x: np.ndarray | list[str]) -> np.ndarray:
        size = len(x)
        return np.tile(np.asarray([[0.2, 0.8]], dtype=float), (size, 1))


class _FakeQuantizedModel(_FakeModel):
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "marker": torch.tensor([len(self.best_marker)], dtype=torch.float32),
            "quant_state.bitsandbytes__nf4": torch.tensor([1.0], dtype=torch.float32),
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        if strict and "quant_state.bitsandbytes__nf4" in state_dict:
            raise RuntimeError("Unexpected key(s) in state_dict: quant_state.bitsandbytes__nf4")
        super().load_state_dict(state_dict, strict=strict)


def test_progress_iter_uses_tqdm_when_enabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_tqdm(iterable, **kwargs):
        captured["desc"] = kwargs.get("desc")
        captured["total"] = kwargs.get("total")
        return list(iterable)

    monkeypatch.setattr(llm_modeling, "tqdm", _fake_tqdm)

    items = llm_modeling._progress_iter([1, 2, 3], enabled=True, description="train")

    assert list(items) == [1, 2, 3]
    assert captured["desc"] == "train"
    assert captured["total"] == 3


def test_build_weighted_loss_for_balanced_qwen_training() -> None:
    criterion = llm_modeling._build_weighted_loss(
        y_train=np.asarray([0, 0, 0, 1], dtype=np.int64),
        num_classes=2,
        class_weighting_name="balanced",
        device=torch.device("cpu"),
    )

    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.weight is not None
    assert torch.allclose(criterion.weight.cpu(), torch.tensor([2.0 / 3.0, 2.0], dtype=torch.float32))


def test_train_qwen_qlora_classifier_restores_best_adapter_before_save(monkeypatch, tmp_path: Path) -> None:
    fake_model = _FakeModel()
    fake_tokenizer = _FakeTokenizer()
    epoch_states = iter(["best", "last"])
    eval_losses = iter([0.2, 0.6])

    monkeypatch.setattr(
        llm_modeling,
        "_import_llm_stack",
        lambda: {"get_linear_schedule_with_warmup": lambda **_: object()},
    )
    monkeypatch.setattr(
        llm_modeling,
        "_build_model_and_tokenizer",
        lambda **_: (fake_model, fake_tokenizer),
    )
    monkeypatch.setattr(
        llm_modeling,
        "_build_dataloaders",
        lambda *args, **kwargs: ([{"labels": torch.tensor([1])}], [{"labels": torch.tensor([1])}]),
    )

    def _fake_train_epoch(model, *args, **kwargs) -> float:
        _ = args
        _ = kwargs
        model.best_marker = next(epoch_states)
        return 0.1

    monkeypatch.setattr(llm_modeling, "_run_training_epoch", _fake_train_epoch)
    monkeypatch.setattr(
        llm_modeling,
        "_run_eval_epoch",
        lambda *args, **kwargs: (np.asarray([[0.1, 0.9]], dtype=float), next(eval_losses)),
    )
    monkeypatch.setattr(
        llm_modeling,
        "batched_forward_logits",
        lambda *args, **kwargs: np.asarray([[0.1, 0.9]], dtype=float),
    )
    monkeypatch.setattr(
        llm_modeling,
        "QLoRAFeatureClassifier",
        _FakeClassifier,
    )

    result = llm_modeling.train_qwen_qlora_classifier(
        x_train=np.asarray([[0.1, 0.2]], dtype=float),
        y_train=np.asarray([1], dtype=np.int64),
        x_val=np.asarray([[0.3, 0.4]], dtype=float),
        y_val=np.asarray([1], dtype=np.int64),
        epochs=2,
        batch_size=1,
        patience=1,
        output_dir=str(tmp_path / "adapter"),
    )

    assert fake_model.loaded_markers == ["best"]
    assert fake_model.saved_marker == "best"
    assert result["best_checkpoint"]["epoch"] == 1


def test_train_qwen_qlora_classifier_restores_best_adapter_with_quantized_state(monkeypatch, tmp_path: Path) -> None:
    fake_model = _FakeQuantizedModel()
    fake_tokenizer = _FakeTokenizer()
    epoch_states = iter(["best", "last"])
    eval_losses = iter([0.2, 0.6])

    monkeypatch.setattr(
        llm_modeling,
        "_import_llm_stack",
        lambda: {"get_linear_schedule_with_warmup": lambda **_: object()},
    )
    monkeypatch.setattr(
        llm_modeling,
        "_build_model_and_tokenizer",
        lambda **_: (fake_model, fake_tokenizer),
    )
    monkeypatch.setattr(
        llm_modeling,
        "_build_dataloaders",
        lambda *args, **kwargs: ([{"labels": torch.tensor([1])}], [{"labels": torch.tensor([1])}]),
    )

    def _fake_train_epoch(model, *args, **kwargs) -> float:
        _ = args
        _ = kwargs
        model.best_marker = next(epoch_states)
        return 0.1

    monkeypatch.setattr(llm_modeling, "_run_training_epoch", _fake_train_epoch)
    monkeypatch.setattr(
        llm_modeling,
        "_run_eval_epoch",
        lambda *args, **kwargs: (np.asarray([[0.1, 0.9]], dtype=float), next(eval_losses)),
    )
    monkeypatch.setattr(
        llm_modeling,
        "batched_forward_logits",
        lambda *args, **kwargs: np.asarray([[0.1, 0.9]], dtype=float),
    )
    monkeypatch.setattr(
        llm_modeling,
        "QLoRAFeatureClassifier",
        _FakeClassifier,
    )

    result = llm_modeling.train_qwen_qlora_classifier(
        x_train=np.asarray([[0.1, 0.2]], dtype=float),
        y_train=np.asarray([1], dtype=np.int64),
        x_val=np.asarray([[0.3, 0.4]], dtype=float),
        y_val=np.asarray([1], dtype=np.int64),
        epochs=2,
        batch_size=1,
        patience=1,
        output_dir=str(tmp_path / "adapter"),
    )

    assert fake_model.loaded_markers == ["best"]
    assert fake_model.saved_marker == "best"
    assert result["best_checkpoint"]["epoch"] == 1

from __future__ import annotations
import copy
import math
from pathlib import Path
from tempfile import mkdtemp
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from emc_diag.evaluation import evaluate_predictions, score_metric
from emc_diag.llm_text_adapter import (
    EncodedTextDataset,
    QLoRAFeatureClassifier,
    batched_forward_logits,
    tabular_matrix_to_texts,
)
from emc_diag.runtime import resolve_device
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _clone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if not dtype_name:
        return None
    resolved = getattr(torch, str(dtype_name), None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return resolved


def _normalize_class_weighting_name(class_weighting: str | bool) -> str:
    if isinstance(class_weighting, bool):
        return "on" if class_weighting else "off"
    return str(class_weighting).lower()


def _build_weighted_loss(
    y_train: np.ndarray,
    num_classes: int,
    class_weighting_name: str,
    device: torch.device,
) -> torch.nn.CrossEntropyLoss:
    if class_weighting_name not in {"none", "off", "on", "balanced"}:
        raise ValueError("class_weighting must be one of: off, on, balanced")

    criterion_weights = None
    if class_weighting_name in {"on", "balanced"}:
        class_counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=num_classes).astype(np.float32)
        safe_counts = np.where(class_counts > 0, class_counts, 1.0)
        weights = class_counts.sum() / (len(class_counts) * safe_counts)
        criterion_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return torch.nn.CrossEntropyLoss(weight=criterion_weights)


def _progress_iter(iterable: Any, enabled: bool, description: str) -> Any:
    if not enabled:
        return iterable
    total = len(iterable) if hasattr(iterable, "__len__") else None
    # Keep the bar simple so remote terminals can still show speed and ETA.
    return tqdm(iterable, total=total, desc=description, leave=False)

def _import_llm_stack() -> dict[str, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
            get_linear_schedule_with_warmup,
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Qwen QLoRA requires optional dependencies. Install with: `uv sync --group llm`."
        ) from exc

    return {
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }

def _normalize_training_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str] | None,
) -> dict[str, Any]:
    x_train_np = np.asarray(x_train, dtype=float)
    x_val_np = np.asarray(x_val, dtype=float)
    y_train_np = np.asarray(y_train, dtype=np.int64)
    y_val_np = np.asarray(y_val, dtype=np.int64)
    if x_train_np.ndim != 2 or x_val_np.ndim != 2:
        raise ValueError("Qwen classifier expects tabular 2D features [samples, features].")

    names = list(feature_names or [f"f{idx}" for idx in range(x_train_np.shape[1])])
    return {
        "x_train": x_train_np,
        "y_train": y_train_np,
        "x_val": x_val_np,
        "y_val": y_val_np,
        "feature_names": names,
    }

def _build_model_and_tokenizer(
    llm_stack: dict[str, Any],
    model_id: str,
    resolved_device: str,
    num_classes: int,
    load_in_4bit: bool,
    bnb_4bit_quant_type: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str] | None,
    torch_dtype_name: str | None,
    gradient_checkpointing: bool,
) -> tuple[Any, Any]:
    AutoTokenizer = llm_stack["AutoTokenizer"]
    AutoModelForSequenceClassification = llm_stack["AutoModelForSequenceClassification"]
    BitsAndBytesConfig = llm_stack["BitsAndBytesConfig"]
    LoraConfig = llm_stack["LoraConfig"]
    TaskType = llm_stack["TaskType"]
    get_peft_model = llm_stack["get_peft_model"]
    prepare_model_for_kbit_training = llm_stack["prepare_model_for_kbit_training"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    resolved_torch_dtype = _resolve_torch_dtype(torch_dtype_name) or torch.float16
    model_kwargs: dict[str, Any] = {
        "num_labels": int(num_classes),
        "ignore_mismatched_sizes": True,
        "trust_remote_code": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(bnb_4bit_quant_type),
            bnb_4bit_compute_dtype=resolved_torch_dtype,
        )
        model_kwargs["device_map"] = "auto"
    elif resolved_device == "cuda":
        model_kwargs["torch_dtype"] = resolved_torch_dtype

    model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if gradient_checkpointing:
        # Large 7B checkpoints fit more reliably in 24GB when activation
        # memory is traded for extra compute during backpropagation.
        model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(target_modules or DEFAULT_LORA_TARGET_MODULES),
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    return get_peft_model(model, lora_cfg), tokenizer

def _build_dataloaders(
    tokenizer: Any,
    train_texts: list[str],
    val_texts: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    max_length: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    train_ds = EncodedTextDataset(train_enc, y_train)
    val_ds = EncodedTextDataset(val_enc, y_val)
    return (
        DataLoader(train_ds, batch_size=max(1, int(batch_size)), shuffle=True),
        DataLoader(val_ds, batch_size=max(1, int(batch_size)), shuffle=False),
    )

def _run_training_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: Any,
    scheduler: Any,
    model_device: torch.device,
    gradient_accumulation_steps: int,
    criterion: torch.nn.Module | None = None,
    progress: bool = False,
    progress_description: str = "train",
) -> float:
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    seen = 0
    accum = max(1, int(gradient_accumulation_steps))
    for batch_idx, batch in enumerate(
        _progress_iter(loader, enabled=progress, description=progress_description),
        start=1,
    ):
        labels = batch["labels"].to(model_device)
        inputs = {key: value.to(model_device) for key, value in batch.items() if key != "labels"}
        if criterion is None:
            outputs = model(**inputs, labels=labels)
            loss_value = outputs.loss
        else:
            outputs = model(**inputs)
            loss_value = criterion(outputs.logits, labels)
        loss = loss_value / accum
        loss.backward()
        if batch_idx % accum == 0 or batch_idx == len(loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size_actual = int(labels.shape[0])
        running_loss += float(loss.item()) * batch_size_actual * accum
        seen += batch_size_actual
    return running_loss / seen if seen else 0.0

def _run_eval_epoch(
    model: Any,
    loader: DataLoader,
    model_device: torch.device,
    num_classes: int,
    criterion: torch.nn.Module | None = None,
    progress: bool = False,
    progress_description: str = "eval",
) -> tuple[np.ndarray, float]:
    logits_rows: list[np.ndarray] = []
    loss_sum = 0.0
    seen = 0
    with torch.no_grad():
        for batch in _progress_iter(loader, enabled=progress, description=progress_description):
            labels = batch["labels"].to(model_device)
            inputs = {key: value.to(model_device) for key, value in batch.items() if key != "labels"}
            if criterion is None:
                outputs = model(**inputs, labels=labels)
                loss_value = outputs.loss
            else:
                outputs = model(**inputs)
                loss_value = criterion(outputs.logits, labels)
            logits_rows.append(outputs.logits.detach().cpu().numpy())
            loss_sum += float(loss_value.item()) * int(labels.shape[0])
            seen += int(labels.shape[0])
    val_loss = loss_sum / seen if seen else 0.0
    logits = np.vstack(logits_rows) if logits_rows else np.zeros((0, num_classes), dtype=float)
    return logits, val_loss

def _adapter_output_dir(output_dir: str | None) -> Path:
    target = Path(output_dir or mkdtemp(prefix="emc_diag_qwen_adapter_"))
    target.mkdir(parents=True, exist_ok=True)
    return target

def train_qwen_qlora_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    requested_device: str = "auto",
    epochs: int = 2,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    random_seed: int = 42,
    weight_decay: float = 0.0,
    patience: int | None = 2,
    feature_names: list[str] | None = None,
    model_id: str = DEFAULT_QWEN_MODEL_ID,
    max_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.03,
    save_adapter_only: bool = True,
    gradient_checkpointing: bool = False,
    torch_dtype: str | None = None,
    task_instruction: str | None = None,
    label_descriptions: dict[str, str] | None = None,
    feature_limit: int | None = None,
    class_weighting: str | bool = "off",
    classifier_name: str = "qwen_qlora_classifier",
    output_dir: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    llm_stack = _import_llm_stack()
    arrays = _normalize_training_arrays(x_train, y_train, x_val, y_val, feature_names)
    resolved_device = resolve_device(requested_device)
    quantized = bool(load_in_4bit and resolved_device == "cuda")
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))

    train_texts = tabular_matrix_to_texts(
        arrays["x_train"],
        arrays["feature_names"],
        task_instruction=task_instruction,
        label_descriptions=label_descriptions,
        feature_limit=feature_limit,
    )
    val_texts = tabular_matrix_to_texts(
        arrays["x_val"],
        arrays["feature_names"],
        task_instruction=task_instruction,
        label_descriptions=label_descriptions,
        feature_limit=feature_limit,
    )
    num_classes = int(max(arrays["y_train"].max(), arrays["y_val"].max()) + 1)
    model, tokenizer = _build_model_and_tokenizer(
        llm_stack=llm_stack,
        model_id=model_id,
        resolved_device=resolved_device,
        num_classes=num_classes,
        load_in_4bit=quantized,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        torch_dtype_name=torch_dtype,
        gradient_checkpointing=gradient_checkpointing,
    )
    train_loader, val_loader = _build_dataloaders(
        tokenizer, train_texts, val_texts, arrays["y_train"], arrays["y_val"], int(max_length), int(batch_size)
    )
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=float(learning_rate), weight_decay=float(weight_decay))
    total_steps = max(1, int(epochs) * max(1, math.ceil(len(train_loader) / max(1, int(gradient_accumulation_steps)))))
    scheduler = llm_stack["get_linear_schedule_with_warmup"](
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * max(0.0, float(warmup_ratio))),
        num_training_steps=total_steps,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [], "val_f1": []}
    best_val_loss, best_epoch, epochs_without_improvement, stopped_early = float("inf"), -1, 0, False
    best_state_dict: dict[str, Any] | None = None
    model_device = next(model.parameters()).device
    class_weighting_name = _normalize_class_weighting_name(class_weighting)
    # Use the same weighted CE policy as the existing deep models so minority
    # classes are not silently under-trained in the LLM path.
    criterion = _build_weighted_loss(arrays["y_train"], num_classes, class_weighting_name, model_device)
    for epoch_idx in range(int(epochs)):
        model.train()
        train_loss = _run_training_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            model_device,
            int(gradient_accumulation_steps),
            criterion=criterion,
            progress=progress,
            progress_description=f"train epoch {epoch_idx + 1}/{int(epochs)}",
        )
        model.eval()
        val_logits, val_loss = _run_eval_epoch(
            model,
            val_loader,
            model_device,
            num_classes,
            criterion=criterion,
            progress=progress,
            progress_description=f"eval epoch {epoch_idx + 1}/{int(epochs)}",
        )
        val_pred = np.argmax(val_logits, axis=1).astype(np.int64) if len(val_logits) else np.asarray([], dtype=np.int64)
        val_metrics = evaluate_predictions(arrays["y_val"], val_pred)
        train_pred = np.argmax(batched_forward_logits(model, tokenizer, train_texts, int(max_length), int(batch_size)), axis=1).astype(np.int64)
        train_metrics = evaluate_predictions(arrays["y_train"], train_pred)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(score_metric(train_metrics, "accuracy"))
        history["val_accuracy"].append(score_metric(val_metrics, "accuracy"))
        history["val_f1"].append(score_metric(val_metrics, "f1"))
        if progress:
            print(f"[qwen_qlora_classifier] epoch {epoch_idx + 1}/{int(epochs)} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={history['val_accuracy'][-1]:.4f} val_f1={history['val_f1'][-1]:.4f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_without_improvement = val_loss, epoch_idx, 0
            # Save the best adapter weights instead of the last epoch so the
            # exported LoRA artifact matches the recorded best checkpoint.
            best_state_dict = _clone_state_dict(model.state_dict())
        else:
            epochs_without_improvement += 1
        if patience is not None and int(patience) > 0 and epochs_without_improvement >= int(patience):
            stopped_early = True
            break

    if best_state_dict is not None:
        # QLoRA 4bit models expose extra bitsandbytes state entries that are
        # harmless when reloading into the live training model. We only need
        # the adapter weights restored here, so a non-strict load avoids
        # failing on quantization bookkeeping keys.
        model.load_state_dict(best_state_dict, strict=False)
    adapter_dir = _adapter_output_dir(output_dir)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    if not bool(save_adapter_only):
        tokenizer.save_pretrained(adapter_dir)
    classifier = QLoRAFeatureClassifier(
        model=model.eval(),
        tokenizer=tokenizer,
        feature_names=arrays["feature_names"],
        max_length=int(max_length),
        infer_batch_size=max(1, int(batch_size)),
        task_instruction=task_instruction,
        label_descriptions=label_descriptions,
        feature_limit=feature_limit,
    )
    val_predictions = np.argmax(classifier.predict_proba(arrays["x_val"]), axis=1).astype(np.int64)
    val_metrics = evaluate_predictions(arrays["y_val"], val_predictions)
    return {
        "model": classifier,
        "model_name": str(classifier_name),
        "resolved_device": resolved_device,
        "train_history": history,
        "val_predictions": val_predictions,
        "val_accuracy": score_metric(val_metrics, "accuracy"),
        "val_f1": score_metric(val_metrics, "f1"),
        "epochs_ran": len(history["train_loss"]),
        "best_checkpoint": {"epoch": int(best_epoch + 1), "best_epoch": int(best_epoch), "val_loss": float(best_val_loss), "stopped_early": stopped_early},
        "adapter_artifact_dir": str(adapter_dir),
        "llm_info": {
            "foundation_model_id": str(model_id),
            "lora_config": {"r": int(lora_r), "alpha": int(lora_alpha), "dropout": float(lora_dropout), "target_modules": list(target_modules or DEFAULT_LORA_TARGET_MODULES)},
            "quantization_config": {"load_in_4bit": quantized, "bnb_4bit_quant_type": str(bnb_4bit_quant_type)},
            "max_length": int(max_length),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "torch_dtype": str(torch_dtype or "float16"),
            "task_instruction": str(task_instruction or ""),
            "feature_limit": feature_limit,
            "class_weighting": class_weighting_name,
        },
    }

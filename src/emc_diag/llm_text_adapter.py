from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.6g}"


def tabular_matrix_to_texts(
    matrix: np.ndarray,
    feature_names: list[str] | None = None,
) -> list[str]:
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected 2D matrix [samples, features] for LLM classifier.")
    names = list(feature_names or [f"f{idx}" for idx in range(x.shape[1])])
    if len(names) != x.shape[1]:
        names = [f"f{idx}" for idx in range(x.shape[1])]

    texts: list[str] = []
    for row in x:
        feature_pairs = [f"{name}={_format_value(value)}" for name, value in zip(names, row, strict=False)]
        texts.append(
            "Classify the EMC sample.\n"
            "Features: "
            + "; ".join(feature_pairs)
            + "\nReturn only the class id."
        )
    return texts


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = torch.as_tensor(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def batched_forward_logits(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = EncodedTextDataset(encoded, np.zeros(len(texts), dtype=np.int64))
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=False)
    first_param = next(model.parameters())
    device = first_param.device
    logits_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            _ = labels
            prepared = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**prepared)
            logits_rows.append(outputs.logits.detach().cpu().numpy())
    if not logits_rows:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(logits_rows)


@dataclass
class QLoRAFeatureClassifier:
    model: Any
    tokenizer: Any
    feature_names: list[str]
    max_length: int = 1024
    infer_batch_size: int = 16

    def _coerce_texts(self, x: np.ndarray | list[str]) -> list[str]:
        if isinstance(x, list) and (not x or isinstance(x[0], str)):
            return list(x)
        return tabular_matrix_to_texts(np.asarray(x, dtype=float), self.feature_names)

    def predict_proba(self, x: np.ndarray | list[str]) -> np.ndarray:
        texts = self._coerce_texts(x)
        logits = batched_forward_logits(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=texts,
            max_length=self.max_length,
            batch_size=self.infer_batch_size,
        )
        probs = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()
        return np.asarray(probs, dtype=float)

    def predict(self, x: np.ndarray | list[str]) -> np.ndarray:
        probabilities = self.predict_proba(x)
        return np.argmax(probabilities, axis=1).astype(np.int64)

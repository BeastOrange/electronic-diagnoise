from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re

try:
    import torch
except Exception:  # pragma: no cover - torch is optional at import time
    torch = None


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "run"


def timestamp_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def resolve_device(requested_device: str = "auto") -> str:
    normalized = requested_device.lower()
    if normalized not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("requested_device must be one of: auto, cpu, mps, cuda")

    if normalized == "cpu":
        return "cpu"

    if torch is None:
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and torch.backends.mps.is_available())

    if normalized == "cuda":
        return "cuda" if cuda_available else "cpu"
    if normalized == "mps":
        return "mps" if mps_available else "cpu"
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"

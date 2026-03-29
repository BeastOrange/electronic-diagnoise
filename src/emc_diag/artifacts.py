from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from emc_diag.runtime import ensure_dir, slugify, timestamp_tag


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def save_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return target


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_npz(path: str | Path, **arrays: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    np.savez(target, **arrays)
    return target


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_dataframe(path: str | Path, frame: pd.DataFrame) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    frame.to_csv(target, index=False)
    return target


def create_run_dir(artifacts_dir: str | Path, experiment_name: str) -> Path:
    root = ensure_dir(artifacts_dir)
    run_dir = root / f"run-{timestamp_tag()}-{slugify(experiment_name)}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "tables")
    return run_dir

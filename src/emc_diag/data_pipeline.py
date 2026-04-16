from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from emc_diag.vsb_features import build_vsb_feature_matrix


SUPPORTED_SCHEMAS = {"tabular", "waveform", "spectrum"}
_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_DEFAULT_VSB_MAX_SIGNALS = 256
_DEFAULT_VSB_MAX_TIMESTEPS = 200_000
_DEFAULT_VSB_SIGNAL_CHUNK_SIZE = 256


def _validate_schema(schema: str) -> None:
    if schema not in SUPPORTED_SCHEMAS:
        allowed = ", ".join(sorted(SUPPORTED_SCHEMAS))
        raise ValueError(f"Unsupported schema '{schema}'. Allowed: {allowed}")


def _read_dataframe(source: Path) -> pd.DataFrame:
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported file extension '{suffix}' for {source}")


def _parse_source_options(source: str | Path) -> tuple[Path, dict[str, str]]:
    if isinstance(source, Path):
        return source, {}

    raw = str(source)
    if "::" not in raw:
        return Path(raw), {}

    head, *tail = raw.split("::")
    options: dict[str, str] = {}
    for item in tail:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key:
            options[key] = value
    return Path(head), options


def _sanitize_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    cleaned = df.copy()
    drop_columns = [
        column
        for column in cleaned.columns
        if column != target_column and (cleaned[column].isna().all() or str(column).startswith("Unnamed:"))
    ]
    if drop_columns:
        cleaned = cleaned.drop(columns=drop_columns)
    return cleaned


def _normalize_target_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.astype("string").fillna("missing")


def _encode_target_series(series: pd.Series) -> tuple[np.ndarray, list[str], dict[str, int]]:
    normalized = _normalize_target_series(series)
    if pd.api.types.is_numeric_dtype(normalized):
        normalized = normalized.dropna()
        unique_labels = sorted(pd.unique(normalized).tolist())
        label_names = [str(label) for label in unique_labels]
        label_to_index = {str(label): index for index, label in enumerate(unique_labels)}
        encoded = normalized.map(lambda value: label_to_index[str(value)]).to_numpy(dtype=np.int64)
        return encoded, label_names, {str(label): index for index, label in enumerate(unique_labels)}

    normalized = normalized.astype("string")
    unique_labels = sorted(normalized.unique().tolist())
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded = normalized.map(label_to_index).to_numpy(dtype=np.int64)
    return encoded, unique_labels, label_to_index


def _encode_task_column(
    series: pd.Series,
) -> tuple[np.ndarray, list[str], dict[str, int], np.ndarray]:
    normalized = _normalize_target_series(series)
    missing_mask = normalized.isna().to_numpy(dtype=bool)
    non_missing = normalized[~missing_mask]
    label_names: list[str] = []
    label_to_index: dict[str, int] = {}
    if not non_missing.empty:
        if pd.api.types.is_numeric_dtype(non_missing):
            unique_values = sorted(pd.unique(non_missing).tolist())
        else:
            non_missing = non_missing.astype("string")
            unique_values = sorted(non_missing.unique().tolist())
        label_names = [str(value) for value in unique_values]
        label_to_index = {str(value): idx for idx, value in enumerate(unique_values)}
    encoded = np.full(len(series), -1, dtype=np.int64)
    for position, value in non_missing.items():
        encoded[int(position)] = label_to_index.get(str(value), -1)
    return encoded, label_names, label_to_index, missing_mask


def _safe_to_int(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _safe_to_int_or_default(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_array_like_value(value: Any) -> np.ndarray | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    if text.startswith("array(") and text.endswith(")"):
        text = text[6:-1].strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None

    inner = text[1:-1].replace("\n", " ").replace(",", " ").strip()
    if not inner:
        return np.asarray([], dtype=float)

    try:
        parsed = np.fromstring(inner, sep=" ", dtype=float)
    except ValueError:
        parsed = np.asarray([], dtype=float)
    if parsed.size > 0:
        return parsed

    matches = _FLOAT_PATTERN.findall(inner)
    if not matches:
        return None
    return np.asarray([float(match) for match in matches], dtype=float)


def _expand_array_column(column_name: str, series: pd.Series) -> pd.DataFrame:
    parsed_values = [None if pd.isna(item) else _parse_array_like_value(item) for item in series]
    lengths = [value.size for value in parsed_values if value is not None]
    if not lengths:
        return pd.DataFrame(index=series.index)

    max_len = max(lengths)
    matrix = np.zeros((len(series), max_len), dtype=float)
    for row_index, parsed in enumerate(parsed_values):
        if parsed is None or parsed.size == 0:
            continue
        limit = min(parsed.size, max_len)
        matrix[row_index, :limit] = parsed[:limit]

    columns = [f"{column_name}_{index}" for index in range(max_len)]
    return pd.DataFrame(matrix, columns=columns, index=series.index)


def _is_probably_array_column(series: pd.Series) -> bool:
    values = [item for item in series.dropna().tolist() if isinstance(item, str)]
    if not values:
        return False
    parsed_count = sum(1 for item in values if _parse_array_like_value(item) is not None)
    ratio = parsed_count / len(values)
    return ratio >= 0.8


def _expand_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    expanded_parts: list[pd.DataFrame] = []
    feature_order: list[str] = []

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").astype(float)
            fill_value = float(numeric.mean()) if not np.isnan(numeric.mean()) else 0.0
            numeric = numeric.fillna(fill_value)
            numeric_df = pd.DataFrame({column: numeric}, index=df.index)
            expanded_parts.append(numeric_df)
            feature_order.append(column)
            continue

        if _is_probably_array_column(series):
            array_df = _expand_array_column(column, series)
            expanded_parts.append(array_df)
            feature_order.extend(array_df.columns.tolist())
            continue

        categorical = series.astype("string").fillna("missing")
        dummy_df = pd.get_dummies(categorical, prefix=column, prefix_sep="__", dtype=float)
        expanded_parts.append(dummy_df)
        feature_order.extend(dummy_df.columns.tolist())

    if not expanded_parts:
        return pd.DataFrame(index=df.index)

    expanded = pd.concat(expanded_parts, axis=1)
    return expanded[feature_order]


def _load_array_directory(source: Path) -> dict[str, Any]:
    npy_files = sorted(source.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found under directory: {source}")

    labels_csv = source / "labels.csv"
    label_map: dict[str, Any] = {}
    if labels_csv.exists():
        df_labels = pd.read_csv(labels_csv)
        if {"file", "label"}.issubset(df_labels.columns):
            label_map = {str(row["file"]): row["label"] for _, row in df_labels.iterrows()}

    rows: list[np.ndarray] = []
    labels: list[Any] = []
    sample_ids: list[str] = []
    for file_path in npy_files:
        arr = np.asarray(np.load(file_path), dtype=float).reshape(-1)
        rows.append(arr)
        sample_ids.append(file_path.name)
        if file_path.name in label_map:
            label = label_map[file_path.name]
        else:
            label = _safe_to_int(file_path.stem.split("_")[0])
        labels.append(label)

    lengths = {item.shape[0] for item in rows}
    if len(lengths) != 1:
        raise ValueError("All .npy files in directory mode must have same flattened length.")

    x = np.vstack(rows)
    y = np.asarray(labels)
    feature_names = [f"f{i}" for i in range(x.shape[1])]
    return {"X": x, "y": y, "feature_names": feature_names, "sample_ids": sample_ids}


def _is_vsb_fault_directory(source: Path) -> bool:
    return source.is_dir() and (source / "train.parquet").exists() and (source / "metadata_train.csv").exists()


def _sample_metadata_with_stratification(
    metadata: pd.DataFrame,
    target_column: str,
    max_signals: int,
    seed: int,
) -> pd.DataFrame:
    if max_signals >= len(metadata):
        return metadata

    grouped = [group for _, group in metadata.groupby(target_column, dropna=False)]
    total = len(metadata)
    desired = []
    for group in grouped:
        quota = max(1, int(round(max_signals * len(group) / total)))
        desired.append(min(len(group), quota))

    current = sum(desired)
    while current > max_signals:
        candidates = [index for index, count in enumerate(desired) if count > 1]
        if not candidates:
            break
        largest = max(candidates, key=lambda index: desired[index])
        desired[largest] -= 1
        current -= 1

    while current < max_signals:
        capacities = [
            index
            for index, group in enumerate(grouped)
            if desired[index] < len(group)
        ]
        if not capacities:
            break
        largest = max(capacities, key=lambda index: len(grouped[index]) - desired[index])
        desired[largest] += 1
        current += 1

    sampled_frames = []
    for group, quota in zip(grouped, desired, strict=False):
        sampled_frames.append(group.sample(n=quota, random_state=seed))
    sampled = pd.concat(sampled_frames, ignore_index=True)

    if len(sampled) < max_signals:
        remaining = metadata.loc[~metadata.index.isin(sampled.index)]
        if not remaining.empty:
            extra = remaining.sample(n=min(max_signals - len(sampled), len(remaining)), random_state=seed)
            sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled


def _extract_vsb_signal_features(
    parquet_path: Path,
    signal_ids: list[int],
    max_timesteps: int,
    signal_chunk_size: int,
) -> pd.DataFrame:
    chunk_size = max(1, int(signal_chunk_size))
    feature_frames: list[pd.DataFrame] = []

    for start_index in range(0, len(signal_ids), chunk_size):
        chunk_signal_ids = signal_ids[start_index : start_index + chunk_size]
        column_names = [str(signal_id) for signal_id in chunk_signal_ids]
        signal_frame = pd.read_parquet(parquet_path, columns=column_names)
        if max_timesteps > 0:
            signal_frame = signal_frame.iloc[:max_timesteps]
        signal_matrix = signal_frame.to_numpy(dtype=float)
        if signal_matrix.shape[0] < 2:
            raise ValueError("VSB train.parquet needs at least 2 timesteps to compute stable signal features.")

        waveform_matrix = signal_matrix.T
        feature_matrix, feature_names = build_vsb_feature_matrix(
            waveform_matrix,
            sample_rate=1.0,
            nperseg=min(256, signal_matrix.shape[0]),
            noverlap=min(128, max(1, signal_matrix.shape[0] // 4)),
        )
        compatibility_names = [
            name.replace("stat_", "signal_") if name.startswith("stat_") else name
            for name in feature_names
        ]
        features = pd.DataFrame(feature_matrix, columns=compatibility_names)
        zero_cross = np.mean(np.signbit(signal_matrix[1:, :]) != np.signbit(signal_matrix[:-1, :]), axis=0)
        features["signal_zero_cross_rate"] = zero_cross
        features.insert(0, "signal_id", chunk_signal_ids)
        feature_frames.append(features)

    if not feature_frames:
        return pd.DataFrame(columns=["signal_id"])
    return pd.concat(feature_frames, ignore_index=True)


def _vsb_feature_cache_path(
    source: Path,
    target_column: str,
    max_signals: int,
    max_timesteps: int,
    seed: int,
) -> Path:
    cache_dir = source / ".emc_diag_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_payload = {
        "target_column": target_column,
        "max_signals": int(max_signals),
        "max_timesteps": int(max_timesteps),
        "seed": int(seed),
    }
    digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"vsb_features_{digest}.parquet"


def _load_vsb_fault_directory(
    source: Path,
    target_column: str,
    drop_columns: list[str] | None,
    source_options: dict[str, str],
) -> dict[str, Any]:
    metadata_path = source / "metadata_train.csv"
    train_parquet_path = source / "train.parquet"
    metadata = pd.read_csv(metadata_path)

    required_columns = {"signal_id", target_column}
    missing_required = [column for column in required_columns if column not in metadata.columns]
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise ValueError(f"VSB metadata_train.csv is missing required column(s): {missing_text}")

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - pyarrow is dependency-managed
        raise RuntimeError("pyarrow is required for VSB parquet loading") from exc

    parquet_schema = pq.ParquetFile(train_parquet_path).schema.names
    available_signal_columns = set(parquet_schema)
    metadata = metadata.loc[metadata["signal_id"].astype(str).isin(available_signal_columns)].copy()
    if metadata.empty:
        raise ValueError("No matching signal_id entries between metadata_train.csv and train.parquet columns.")

    normalized_target = _normalize_target_series(metadata[target_column])
    if pd.api.types.is_numeric_dtype(normalized_target):
        valid_rows = ~normalized_target.isna()
    else:
        valid_rows = pd.Series(True, index=metadata.index)
    dropped_missing_targets = int((~valid_rows).sum())
    metadata = metadata.loc[valid_rows].copy()
    normalized_target = normalized_target.loc[valid_rows]
    metadata[target_column] = normalized_target

    max_signals = _safe_to_int_or_default(source_options.get("max_signals"), _DEFAULT_VSB_MAX_SIGNALS)
    max_timesteps = _safe_to_int_or_default(source_options.get("max_timesteps"), _DEFAULT_VSB_MAX_TIMESTEPS)
    seed = _safe_to_int_or_default(source_options.get("seed"), 42)
    signal_chunk_size = _safe_to_int_or_default(source_options.get("signal_chunk_size"), _DEFAULT_VSB_SIGNAL_CHUNK_SIZE)

    cache_path = _vsb_feature_cache_path(
        source=source,
        target_column=target_column,
        max_signals=max_signals,
        max_timesteps=max_timesteps,
        seed=seed,
    )
    if cache_path.exists():
        merged = pd.read_parquet(cache_path)
    else:
        sampled_metadata = _sample_metadata_with_stratification(
            metadata=metadata,
            target_column=target_column,
            max_signals=min(max_signals, len(metadata)),
            seed=seed,
        ).copy()
        sampled_metadata = sampled_metadata.sort_values("signal_id").reset_index(drop=True)
        selected_signal_ids = sampled_metadata["signal_id"].astype(int).tolist()
        signal_features = _extract_vsb_signal_features(
            parquet_path=train_parquet_path,
            signal_ids=selected_signal_ids,
            max_timesteps=max_timesteps,
            signal_chunk_size=signal_chunk_size,
        )

        merged = sampled_metadata.merge(signal_features, on="signal_id", how="inner")
        if merged.empty:
            raise ValueError("VSB loading produced zero rows after feature merge.")
        merged.to_parquet(cache_path, index=False)

    y, label_names, label_to_index = _encode_target_series(merged[target_column])
    feature_frame = merged.drop(columns=[target_column])
    drop_columns = [column for column in (drop_columns or []) if column in feature_frame.columns]
    if drop_columns:
        feature_frame = feature_frame.drop(columns=drop_columns)

    expanded_features = _expand_tabular_features(feature_frame)
    x = expanded_features.to_numpy(dtype=float)
    return {
        "X": x,
        "y": y,
        "feature_names": list(expanded_features.columns),
        "sample_ids": merged["signal_id"].astype(str).tolist(),
        "label_names": label_names,
        "label_to_index": label_to_index,
        "dropped_feature_columns": drop_columns,
        "dropped_missing_targets": dropped_missing_targets,
    }


def _load_from_tabular_file(
    source: Path,
    target_column: str,
    drop_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
) -> dict[str, Any]:
    df = _sanitize_dataframe(_read_dataframe(source), target_column=target_column)
    df = df.reset_index(drop=True)
    requested_columns = target_columns or [target_column]
    missing_columns = [column for column in requested_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"target_column(s) {missing_columns} are not in columns: {list(df.columns)}")

    dropped_missing_targets = 0
    if len(requested_columns) == 1:
        primary_mask = df[requested_columns[0]].notna()
        dropped_missing_targets = int((~primary_mask).sum())
        if dropped_missing_targets:
            df = df.loc[primary_mask].reset_index(drop=True)

    task_label_info: dict[str, tuple[np.ndarray, list[str], dict[str, int], np.ndarray]] = {}
    for column in requested_columns:
        encoded, labels, mapping, mask = _encode_task_column(df[column])
        task_label_info[column] = (encoded, labels, mapping, mask)

    primary_column = requested_columns[0]
    primary_encoded, primary_labels, primary_mapping, _ = task_label_info[primary_column]

    drop_columns_filtered = [column for column in (drop_columns or []) if column in df.columns and column not in requested_columns]
    feature_frame = df.drop(columns=requested_columns, errors="ignore")
    if drop_columns_filtered:
        feature_frame = feature_frame.drop(columns=drop_columns_filtered)

    expanded_features = _expand_tabular_features(feature_frame)
    x = expanded_features.to_numpy(dtype=float)

    return {
        "X": x,
        "y": primary_encoded,
        "feature_names": list(expanded_features.columns),
        "sample_ids": list(df.index.astype(str)),
        "label_names": primary_labels,
        "label_to_index": primary_mapping,
        "dropped_feature_columns": drop_columns_filtered,
        "dropped_missing_targets": dropped_missing_targets,
        "task_columns": requested_columns,
        "task_labels": {column: info[0] for column, info in task_label_info.items()},
        "task_label_names": {column: info[1] for column, info in task_label_info.items()},
        "task_label_to_index": {column: info[2] for column, info in task_label_info.items()},
        "task_label_masks": {column: info[3] for column, info in task_label_info.items()},
    }


def load_local_data(
    source: str | Path,
    schema: str = "tabular",
    target_column: str = "label",
    drop_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
) -> dict[str, Any]:
    _validate_schema(schema)
    source_path, source_options = _parse_source_options(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Data source does not exist: {source_path}")

    multitask_requested = bool(target_columns and len(target_columns) > 1)
    if multitask_requested and schema != "tabular":
        raise ValueError("Multitask loading is only supported for tabular schema currently.")

    if schema == "tabular" and _is_vsb_fault_directory(source_path):
        if multitask_requested:
            raise ValueError("Multitask loading is not supported for VSB directories.")
        return _load_vsb_fault_directory(
            source=source_path,
            target_column=target_column,
            drop_columns=drop_columns,
            source_options=source_options,
        )

    if source_path.is_file():
        return _load_from_tabular_file(
            source_path,
            target_column=target_column,
            drop_columns=drop_columns,
            target_columns=target_columns,
        )

    # Directory input is supported for all schemas as a minimal implementation.
    # For tabular directories we look for an obvious file first, then fallback to .npy arrays.
    if schema == "tabular":
        candidates = sorted(source_path.glob("*.csv")) + sorted(source_path.glob("*.parquet"))
        if candidates:
            return _load_from_tabular_file(
                candidates[0],
                target_column=target_column,
                drop_columns=drop_columns,
                target_columns=target_columns,
            )
    return _load_array_directory(source_path)


def _split_indices(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
    y: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Split sample indices into train / val / test.

    When *y* is provided, uses stratified splitting so that each split
    preserves the original class distribution.  Falls back to simple
    random shuffle when *y* is ``None`` or has too few samples per class.
    """
    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1) and val_ratio in [0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    n_total = n_samples
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = max(1, min(n_train, n_total - 2))
    n_val = max(1, min(n_val, n_total - n_train - 1))
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        raise ValueError("Computed test split has zero size. Increase dataset size or change ratios.")

    # Attempt stratified split when labels are available
    if y is not None:
        labels = np.asarray(y)
        classes, counts = np.unique(labels, return_counts=True)
        # Need at least 3 samples per class (one per split) to stratify
        min_per_class = 3
        if len(classes) >= 2 and counts.min() >= min_per_class:
            return _stratified_split_indices(
                labels, classes, n_train, n_val, random_state,
            )

    # Fallback: plain random shuffle
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def _stratified_split_indices(
    labels: np.ndarray,
    classes: np.ndarray,
    n_train: int,
    n_val: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    """Stratified train/val/test split preserving class proportions."""
    rng = np.random.default_rng(random_state)
    n_total = len(labels)
    n_test = n_total - n_train - n_val

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for cls in classes:
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        cls_n = len(cls_idx)
        # Proportional allocation for this class
        cls_train = max(1, int(round(cls_n * n_train / n_total)))
        cls_val = max(1, int(round(cls_n * n_val / n_total)))
        cls_test = cls_n - cls_train - cls_val
        if cls_test <= 0:
            cls_test = 1
            cls_train = cls_n - cls_val - cls_test

        train_indices.extend(cls_idx[:cls_train])
        val_indices.extend(cls_idx[cls_train:cls_train + cls_val])
        test_indices.extend(cls_idx[cls_train + cls_val:])

    # Shuffle within each split to avoid class-ordered batches
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return {
        "train": np.array(train_indices, dtype=np.intp),
        "val": np.array(val_indices, dtype=np.intp),
        "test": np.array(test_indices, dtype=np.intp),
    }


def _build_splits(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
    y_tasks: dict[str, np.ndarray] | None = None,
    task_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    index_splits = _split_indices(x.shape[0], train_ratio, val_ratio, random_state, y=y)
    if task_masks:
        missing_any = np.zeros(x.shape[0], dtype=bool)
        for mask in task_masks.values():
            missing_any |= np.asarray(mask, dtype=bool)
        if int(missing_any.sum()) >= len(index_splits):
            rng = np.random.default_rng(random_state)
            capacities = {name: len(indices) for name, indices in index_splits.items()}
            missing_indices = np.flatnonzero(missing_any)
            observed_indices = np.flatnonzero(~missing_any)
            rng.shuffle(missing_indices)
            rng.shuffle(observed_indices)
            assigned: dict[str, list[int]] = {name: [] for name in index_splits}

            for split_name in ("train", "val", "test"):
                if capacities.get(split_name, 0) <= 0 or missing_indices.size == 0:
                    continue
                assigned[split_name].append(int(missing_indices[0]))
                missing_indices = missing_indices[1:]

            remaining = np.concatenate([missing_indices, observed_indices])
            rng.shuffle(remaining)
            cursor = 0
            for split_name in ("train", "val", "test"):
                need = capacities[split_name] - len(assigned[split_name])
                if need <= 0:
                    continue
                assigned[split_name].extend(int(item) for item in remaining[cursor : cursor + need])
                cursor += need
            index_splits = {
                name: np.asarray(values, dtype=np.int64)
                for name, values in assigned.items()
            }
    splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name, indices in index_splits.items():
        payload: dict[str, np.ndarray] = {"X": x[indices], "y": y[indices]}
        if y_tasks:
            payload["y_tasks"] = {task: labels[indices] for task, labels in y_tasks.items()}
        if task_masks:
            payload["task_masks"] = {task: mask[indices] for task, mask in task_masks.items()}
        splits[split_name] = payload
    return splits


def _standardize_splits(splits: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    x_train = splits["train"]["X"].astype(float)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)

    normalized: dict[str, dict[str, np.ndarray]] = {}
    for split_name, payload in splits.items():
        normalized_payload: dict[str, np.ndarray] = {
            "X": (payload["X"].astype(float) - mean) / std_safe,
            "y": payload["y"],
        }
        if "y_tasks" in payload:
            normalized_payload["y_tasks"] = payload["y_tasks"]
        if "task_masks" in payload:
            normalized_payload["task_masks"] = payload["task_masks"]
        normalized[split_name] = normalized_payload

    scaler = {"mean": mean.tolist(), "std": std_safe.tolist()}
    return normalized, scaler


def _basic_statistics(x: np.ndarray, y: np.ndarray, label_names: list[str] | None = None) -> dict[str, Any]:
    labels, counts = np.unique(y, return_counts=True)
    missing_count = int(np.isnan(x).sum())
    total_count = int(np.prod(x.shape))
    readable_counts: dict[str, int] = {}
    for label, count in zip(labels, counts, strict=False):
        if label_names and int(label) < len(label_names):
            readable_counts[label_names[int(label)]] = int(count)
        else:
            readable_counts[str(label)] = int(count)
    return {
        "class_counts": readable_counts,
        "missing_values": missing_count,
        "missing_ratio": float(missing_count / total_count) if total_count else 0.0,
        "feature_mean_head": np.mean(x, axis=0)[: min(5, x.shape[1])].tolist(),
        "feature_std_head": np.std(x, axis=0)[: min(5, x.shape[1])].tolist(),
    }


def prepare_dataset(
    source: str | Path,
    schema: str = "tabular",
    target_column: str = "label",
    task_name: str | None = None,
    drop_columns: list[str] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
    task_columns: list[str] | None = None,
) -> dict[str, Any]:
    requested_tasks = task_columns or [target_column]
    loaded = load_local_data(
        source=source,
        schema=schema,
        target_column=target_column,
        drop_columns=drop_columns,
        target_columns=requested_tasks,
    )
    x = np.asarray(loaded["X"], dtype=float)
    y = np.asarray(loaded["y"])

    if x.ndim != 2:
        raise ValueError(f"Expected 2D features after loading; got shape {x.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Mismatched number of samples between X and y.")

    y_tasks = {task: np.asarray(labels, dtype=np.int64) for task, labels in loaded.get("task_labels", {}).items()}
    task_masks = {
        task: np.asarray(mask, dtype=bool) for task, mask in loaded.get("task_label_masks", {}).items()
    }
    raw_splits = _build_splits(
        x=x,
        y=y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_state=random_state,
        y_tasks=y_tasks if y_tasks else None,
        task_masks=task_masks if task_masks else None,
    )
    splits, scaler = _standardize_splits(raw_splits)
    label_names = loaded.get("label_names")
    stats = _basic_statistics(x, y, label_names=label_names)

    task_metadata_columns = loaded.get("task_columns", requested_tasks)
    task_definitions: list[dict[str, Any]] = []
    task_label_names = loaded.get("task_label_names", {})
    task_label_to_index = loaded.get("task_label_to_index", {})
    for column in task_metadata_columns:
        labels_for_task = task_label_names.get(column, label_names if column == target_column else [])
        mapping_for_task = task_label_to_index.get(column, loaded.get("label_to_index", {}) if column == target_column else {})
        task_definitions.append(
            {
                "target_column": column,
                "task_name": column,
                "labels": labels_for_task,
                "label_to_index": mapping_for_task,
            }
        )

    primary_task = task_definitions[0] if task_definitions else {}
    metadata = {
        "schema": schema,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "feature_names": loaded["feature_names"],
        "labels": primary_task.get("labels", label_names or sorted({str(item) for item in y.tolist()})),
        "target_column": target_column,
        "task_name": task_name or primary_task.get("task_name", target_column),
        "random_state": random_state,
        "label_to_index": primary_task.get("label_to_index", loaded.get("label_to_index", {})),
        "leakage_columns_removed": loaded.get("dropped_feature_columns", []),
        "dropped_missing_targets": loaded.get("dropped_missing_targets", 0),
        "task_columns": task_metadata_columns,
        "task_type": "multitask" if len(task_metadata_columns) > 1 else "single_task",
        "tasks": task_definitions,
    }
    if "sample_ids" in loaded:
        metadata["sample_id_count"] = len(loaded["sample_ids"])

    return {
        "splits": splits,
        "scaler": scaler,
        "metadata": metadata,
        "statistics": stats,
    }

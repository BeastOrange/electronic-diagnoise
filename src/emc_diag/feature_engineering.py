from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np


SEQUENCE_LAYOUTS = {"all", "basic_only", "cov_flat_only", "temporal_cov_only"}
_PREFIX_INDEX_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)$")
_COGNITIVE_STRUCTURED_PATTERN = re.compile(r"^(?P<prefix>SU\d+_(?:cov_flat|temporal_cov))_(?P<index>\d+)$")


def _compute_feature_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    overall_mean = x.mean(axis=0)
    labels = np.unique(y)
    between = np.zeros(x.shape[1], dtype=float)
    within = np.zeros(x.shape[1], dtype=float)

    for label in labels:
        class_x = x[y == label]
        if class_x.size == 0:
            continue
        class_mean = class_x.mean(axis=0)
        between += class_x.shape[0] * np.square(class_mean - overall_mean)
        within += np.square(class_x - class_mean).sum(axis=0)

    return between / np.maximum(within, 1e-8)


def _waveform_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    stats = np.column_stack(
        [
            x.mean(axis=1),
            x.std(axis=1),
            x.max(axis=1),
            x.min(axis=1),
            np.sqrt(np.mean(np.square(x), axis=1)),
        ]
    )
    spectrum = np.abs(np.fft.rfft(x, axis=1))
    spectrum_head = spectrum[:, : min(8, spectrum.shape[1])]
    feature_names = [
        "mean",
        "std",
        "max",
        "min",
        "rms",
        *[f"fft_{index}" for index in range(spectrum_head.shape[1])],
    ]
    return np.hstack([stats, spectrum_head]), feature_names


def assign_feature_group(name: str) -> str:
    """Assign a coarse domain feature group based on a feature name.

    Groups are intentionally simple and string-pattern based so that
    configuration can refer to them in a stable way.
    """

    lower = name.lower()
    if "_cov_flat_" in lower:
        return "covariance_based"
    if "_temporal_cov_" in lower:
        return "temporal_covariance"
    if "entropy" in lower:
        return "entropy_related"
    if lower.startswith("fft_") or "spectrum" in lower:
        return "spectral_energy"
    return "basic_statistical"


def group_features_by_domain(feature_names: list[str]) -> Dict[str, List[int]]:
    """Group feature indices by coarse domain groups.

    Returns a mapping from group name to a list of column indices.
    """

    groups: Dict[str, List[int]] = {}
    for index, name in enumerate(feature_names):
        group = assign_feature_group(name)
        groups.setdefault(group, []).append(index)
    return groups


def _build_prefix_groups(feature_names: list[str]) -> list[tuple[str, list[int]]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for column_index, name in enumerate(feature_names):
        match = _PREFIX_INDEX_PATTERN.match(name)
        if not match:
            continue
        prefix = match.group("prefix")
        vector_index = int(match.group("index"))
        grouped.setdefault(prefix, []).append((vector_index, column_index))

    ordered_groups: list[tuple[str, list[int]]] = []
    for prefix, entries in grouped.items():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda item: item[0])
        ordered_groups.append((prefix, [column_index for _, column_index in entries]))
    ordered_groups.sort(key=lambda item: item[0])
    return ordered_groups


def _compute_prefix_aggregations(matrix: np.ndarray, groups: list[tuple[str, list[int]]]) -> tuple[np.ndarray, list[str]]:
    if not groups:
        return np.empty((matrix.shape[0], 0), dtype=float), []

    aggregate_blocks: list[np.ndarray] = []
    aggregate_names: list[str] = []
    for prefix, indices in groups:
        sub = matrix[:, indices]
        abs_sub = np.abs(sub)
        diff_sub = np.diff(sub, axis=1) if sub.shape[1] > 1 else np.zeros((sub.shape[0], 1), dtype=float)
        q25 = np.quantile(sub, 0.25, axis=1, keepdims=True)
        q50 = np.quantile(sub, 0.5, axis=1, keepdims=True)
        q75 = np.quantile(sub, 0.75, axis=1, keepdims=True)
        zero_cross = (
            np.mean(np.signbit(sub[:, 1:]) != np.signbit(sub[:, :-1]), axis=1, keepdims=True)
            if sub.shape[1] > 1
            else np.zeros((sub.shape[0], 1), dtype=float)
        )
        aggregate_blocks.extend(
            [
                np.mean(sub, axis=1, keepdims=True),
                np.std(sub, axis=1, keepdims=True),
                np.max(sub, axis=1, keepdims=True),
                np.min(sub, axis=1, keepdims=True),
                q25,
                q50,
                q75,
                (q75 - q25),
                (np.max(sub, axis=1, keepdims=True) - np.min(sub, axis=1, keepdims=True)),
                np.mean(abs_sub, axis=1, keepdims=True),
                np.sum(abs_sub, axis=1, keepdims=True),
                np.mean(np.square(sub), axis=1, keepdims=True),
                np.sqrt(np.sum(np.square(sub), axis=1, keepdims=True)),
                np.mean(np.abs(diff_sub), axis=1, keepdims=True),
                zero_cross,
            ]
        )
        aggregate_names.extend(
            [
                f"{prefix}__mean",
                f"{prefix}__std",
                f"{prefix}__max",
                f"{prefix}__min",
                f"{prefix}__q25",
                f"{prefix}__median",
                f"{prefix}__q75",
                f"{prefix}__iqr",
                f"{prefix}__range",
                f"{prefix}__abs_mean",
                f"{prefix}__l1_energy",
                f"{prefix}__energy",
                f"{prefix}__l2_energy",
                f"{prefix}__variation",
                f"{prefix}__zero_cross_rate",
            ]
        )

    return np.hstack(aggregate_blocks), aggregate_names


def _augment_tabular_with_prefix_aggregates(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    groups = _build_prefix_groups(feature_names)
    if not groups:
        return train_x, val_x, test_x, feature_names

    train_agg, aggregate_names = _compute_prefix_aggregations(train_x, groups)
    val_agg, _ = _compute_prefix_aggregations(val_x, groups)
    test_agg, _ = _compute_prefix_aggregations(test_x, groups)

    augmented_names = feature_names + aggregate_names
    return (
        np.hstack([train_x, train_agg]),
        np.hstack([val_x, val_agg]),
        np.hstack([test_x, test_agg]),
        augmented_names,
    )


def _layout_indices(feature_names: list[str], layout: str) -> tuple[np.ndarray, list[str]]:
    normalized = layout.lower()
    if normalized not in SEQUENCE_LAYOUTS:
        raise ValueError(f"Unsupported sequence layout '{layout}'. Allowed: {sorted(SEQUENCE_LAYOUTS)}")

    indices: list[int] = []
    selected_names: list[str] = []
    for idx, name in enumerate(feature_names):
        is_cov_flat = "_cov_flat_" in name
        is_temporal_cov = "_temporal_cov_" in name
        is_basic = not (is_cov_flat or is_temporal_cov)

        include = (
            normalized == "all"
            or (normalized == "basic_only" and is_basic)
            or (normalized == "cov_flat_only" and is_cov_flat)
            or (normalized == "temporal_cov_only" and is_temporal_cov)
        )
        if include:
            indices.append(idx)
            selected_names.append(name)

    if not indices:
        raise ValueError(f"Sequence layout '{layout}' selected zero features.")
    return np.asarray(indices, dtype=np.int64), selected_names


def filter_feature_layout(
    matrix: np.ndarray,
    feature_names: list[str],
    layout: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    indices, selected_names = _layout_indices(feature_names, layout)
    return matrix[:, indices], selected_names, indices


def _sequence_channels_from_matrix(
    matrix: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str], list[int]]:
    groups = _build_prefix_groups(feature_names)
    grouped_indices = {index for _, indices in groups for index in indices}
    scalar_indices = [idx for idx in range(len(feature_names)) if idx not in grouped_indices]

    channel_specs: list[tuple[str, list[int]]] = [(prefix, indices) for prefix, indices in groups]
    channel_specs.extend((feature_names[idx], [idx]) for idx in scalar_indices)
    if not channel_specs:
        raise ValueError("No channels available for sequence conversion.")

    max_length = max(len(indices) for _, indices in channel_specs)
    channels = np.zeros((matrix.shape[0], len(channel_specs), max_length), dtype=np.float32)
    channel_names: list[str] = []
    channel_lengths: list[int] = []
    for channel_index, (name, indices) in enumerate(channel_specs):
        sub = matrix[:, indices]
        width = sub.shape[1]
        channels[:, channel_index, :width] = sub
        channel_names.append(name)
        channel_lengths.append(width)
    return channels, channel_names, channel_lengths


def _structured_branch_tensor(
    matrix: np.ndarray,
    feature_names: list[str],
    prefixes: list[str],
) -> np.ndarray:
    if not prefixes:
        return np.zeros((matrix.shape[0], 1, 1), dtype=np.float32)

    prefix_to_indices: dict[str, list[int]] = {}
    for prefix in prefixes:
        matched = []
        for idx, name in enumerate(feature_names):
            if name.startswith(prefix + "_"):
                matched.append(idx)
        prefix_to_indices[prefix] = matched

    max_length = max((len(indices) for indices in prefix_to_indices.values()), default=1)
    tensor = np.zeros((matrix.shape[0], len(prefixes), max_length), dtype=np.float32)
    for prefix_index, prefix in enumerate(prefixes):
        indices = prefix_to_indices[prefix]
        if not indices:
            continue
        sub = matrix[:, indices]
        tensor[:, prefix_index, : sub.shape[1]] = sub
    return tensor


def build_cognitive_radio_hybrid_bundle(prepared: dict[str, Any]) -> dict[str, Any]:
    feature_names = list(prepared["metadata"]["feature_names"])
    scalar_indices: list[int] = []
    cov_prefixes: list[str] = []
    temporal_prefixes: list[str] = []
    seen_cov: set[str] = set()
    seen_temporal: set[str] = set()

    for idx, name in enumerate(feature_names):
        match = _COGNITIVE_STRUCTURED_PATTERN.match(name)
        if not match:
            scalar_indices.append(idx)
            continue
        prefix = match.group("prefix")
        if prefix.endswith("cov_flat") and prefix not in seen_cov:
            seen_cov.add(prefix)
            cov_prefixes.append(prefix)
        elif prefix.endswith("temporal_cov") and prefix not in seen_temporal:
            seen_temporal.add(prefix)
            temporal_prefixes.append(prefix)

    if not cov_prefixes or not temporal_prefixes:
        raise ValueError("Cognitive radio hybrid bundle requires cov_flat and temporal_cov structured features.")

    scalar_feature_names = [feature_names[idx] for idx in scalar_indices]
    hybrid_splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split_payload in prepared["splits"].items():
        matrix = np.asarray(split_payload["X"], dtype=np.float32)
        scalar_x = matrix[:, scalar_indices] if scalar_indices else np.zeros((matrix.shape[0], 1), dtype=np.float32)
        cov_x = _structured_branch_tensor(matrix, feature_names, cov_prefixes)
        temporal_x = _structured_branch_tensor(matrix, feature_names, temporal_prefixes)
        payload: dict[str, np.ndarray] = {
            "scalar_X": scalar_x.astype(np.float32),
            "cov_X": cov_x.astype(np.float32),
            "temporal_X": temporal_x.astype(np.float32),
            "y": np.asarray(split_payload["y"]),
        }
        if "y_tasks" in split_payload:
            payload["y_tasks"] = split_payload["y_tasks"]
        if "task_masks" in split_payload:
            payload["task_masks"] = split_payload["task_masks"]
        hybrid_splits[split_name] = payload

    return {
        "splits": hybrid_splits,
        "summary": {
            "scalar_feature_names": scalar_feature_names,
            "cov_sensor_names": cov_prefixes,
            "temporal_sensor_names": temporal_prefixes,
            "scalar_dim": len(scalar_feature_names) if scalar_feature_names else 1,
            "cov_channels": len(cov_prefixes),
            "temporal_channels": len(temporal_prefixes),
            "cov_length": int(hybrid_splits["train"]["cov_X"].shape[-1]),
            "temporal_length": int(hybrid_splits["train"]["temporal_X"].shape[-1]),
        },
    }


def build_sequence_bundle(prepared: dict[str, Any], layout: str = "all") -> dict[str, Any]:
    raw_feature_names = list(prepared["metadata"]["feature_names"])
    selected_indices, selected_names = _layout_indices(raw_feature_names, layout)

    sequence_splits: dict[str, dict[str, np.ndarray]] = {}
    channel_names: list[str] = []
    channel_lengths: list[int] = []
    for split_name, split_payload in prepared["splits"].items():
        split_matrix = np.asarray(split_payload["X"], dtype=np.float32)[:, selected_indices]
        sequence_matrix, current_channel_names, current_channel_lengths = _sequence_channels_from_matrix(
            split_matrix,
            selected_names,
        )
        payload: dict[str, np.ndarray] = {"X": sequence_matrix, "y": split_payload["y"]}
        if "y_tasks" in split_payload:
            payload["y_tasks"] = split_payload["y_tasks"]
        if "task_masks" in split_payload:
            payload["task_masks"] = split_payload["task_masks"]
        sequence_splits[split_name] = payload
        channel_names = current_channel_names
        channel_lengths = current_channel_lengths

    return {
        "layout": layout,
        "selected_feature_names": selected_names,
        "selected_indices": selected_indices,
        "channel_names": channel_names,
        "channel_lengths": channel_lengths,
        "splits": sequence_splits,
        "summary": {
            "layout": layout,
            "channel_count": len(channel_names),
            "max_sequence_length": int(max(channel_lengths) if channel_lengths else 0),
            "channel_names": channel_names,
        },
    }


def make_prepared_layout_view(prepared: dict[str, Any], layout: str) -> dict[str, Any]:
    feature_names = list(prepared["metadata"]["feature_names"])
    filtered_splits: dict[str, dict[str, np.ndarray]] = {}
    selected_names: list[str] = []
    for split_name, split_payload in prepared["splits"].items():
        filtered_matrix, selected_names, _ = filter_feature_layout(
            np.asarray(split_payload["X"], dtype=np.float32),
            feature_names,
            layout=layout,
        )
        payload: dict[str, np.ndarray] = {"X": filtered_matrix, "y": split_payload["y"]}
        if "y_tasks" in split_payload:
            payload["y_tasks"] = split_payload["y_tasks"]
        if "task_masks" in split_payload:
            payload["task_masks"] = split_payload["task_masks"]
        filtered_splits[split_name] = payload

    metadata = dict(prepared["metadata"])
    metadata["feature_names"] = selected_names
    metadata["num_features"] = len(selected_names)
    metadata["layout"] = layout
    return {
        "splits": filtered_splits,
        "metadata": metadata,
        "statistics": prepared.get("statistics", {}),
        "scaler": prepared.get("scaler", {}),
    }


def make_feature_group_view(prepared: dict[str, Any], enabled_groups: list[str]) -> dict[str, Any]:
    """Return a prepared-style view filtered by domain feature groups.

    This operates on the already-expanded tabular feature space. It does not
    mutate the input and returns a new prepared-like dictionary.
    """

    feature_names = list(prepared["metadata"]["feature_names"])
    groups = group_features_by_domain(feature_names)
    enabled_set = {group for group in enabled_groups}

    selected_indices: list[int] = []
    selected_names: list[str] = []
    for group_name, indices in groups.items():
        if group_name not in enabled_set:
            continue
        for index in indices:
            selected_indices.append(index)
            selected_names.append(feature_names[index])

    if not selected_indices:
        # Fallback: if nothing selected, keep original features to avoid empty design.
        selected_indices = list(range(len(feature_names)))
        selected_names = feature_names

    index_array = np.asarray(selected_indices, dtype=np.int64)
    filtered_splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split_payload in prepared["splits"].items():
        matrix = np.asarray(split_payload["X"], dtype=np.float32)
        filtered_matrix = matrix[:, index_array]
        payload: dict[str, np.ndarray] = {"X": filtered_matrix, "y": split_payload["y"]}
        if "y_tasks" in split_payload:
            payload["y_tasks"] = split_payload["y_tasks"]
        if "task_masks" in split_payload:
            payload["task_masks"] = split_payload["task_masks"]
        filtered_splits[split_name] = payload

    metadata = dict(prepared["metadata"])
    metadata["feature_names"] = selected_names
    metadata["num_features"] = len(selected_names)
    metadata["domain_feature_groups"] = {
        group: [feature_names[index] for index in indices]
        for group, indices in groups.items()
    }
    return {
        "splits": filtered_splits,
        "metadata": metadata,
        "statistics": prepared.get("statistics", {}),
        "scaler": prepared.get("scaler", {}),
    }


def extract_feature_bundle(prepared: dict[str, Any], method: str = "hybrid", top_k: int | None = None) -> dict[str, Any]:
    schema = prepared["metadata"]["schema"]
    splits = prepared["splits"]
    feature_names = list(prepared["metadata"]["feature_names"])
    method_name = str(method or "hybrid").lower()

    if schema in {"waveform", "spectrum"}:
        train_x, generated_names = _waveform_features(splits["train"]["X"])
        val_x, _ = _waveform_features(splits["val"]["X"])
        test_x, _ = _waveform_features(splits["test"]["X"])
        feature_names = generated_names
    else:
        train_x = splits["train"]["X"]
        val_x = splits["val"]["X"]
        test_x = splits["test"]["X"]
        if method_name == "hybrid":
            train_x, val_x, test_x, feature_names = _augment_tabular_with_prefix_aggregates(
                train_x,
                val_x,
                test_x,
                feature_names,
            )
        elif method_name in {"basic", "raw", "plain"}:
            pass
        else:
            raise ValueError(f"Unsupported feature extraction method: {method}")

    scores = _compute_feature_scores(train_x, splits["train"]["y"])
    order = np.argsort(scores)[::-1]
    if isinstance(top_k, int) and top_k > 0:
        order = order[:top_k]

    selected_names = [feature_names[index] for index in order]
    reduced_splits: dict[str, dict[str, np.ndarray]] = {
        "train": {"X": train_x[:, order], "y": splits["train"]["y"]},
        "val": {"X": val_x[:, order], "y": splits["val"]["y"]},
        "test": {"X": test_x[:, order], "y": splits["test"]["y"]},
    }
    for split_name, split_payload in splits.items():
        if "y_tasks" in split_payload:
            reduced_splits[split_name]["y_tasks"] = split_payload["y_tasks"]
        if "task_masks" in split_payload:
            reduced_splits[split_name]["task_masks"] = split_payload["task_masks"]

    return {
        "method": method,
        "feature_names": feature_names,
        "feature_scores": scores,
        "selected_indices": order,
        "selected_feature_names": selected_names,
        "splits": reduced_splits,
        "summary": {
            "schema": schema,
            "selected_feature_count": len(selected_names),
            "top_feature_names": selected_names[: min(5, len(selected_names))],
        },
    }

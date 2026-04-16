from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np


SEQUENCE_LAYOUTS = {"all", "basic_only", "cov_flat_only", "temporal_cov_only"}
_PREFIX_INDEX_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)$")
_COGNITIVE_STRUCTURED_PATTERN = re.compile(r"^(?P<prefix>SU\d+_(?:cov_flat|temporal_cov))_(?P<index>\d+)$")

# Number of elements per covariance sub-block (2x2 complex Hermitian, flattened)
_COV_BLOCK_SIZE = 8


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


def _reconstruct_hermitian_2x2(block: np.ndarray) -> np.ndarray:
    """Reconstruct a 2x2 complex Hermitian matrix from an 8-element flat vector."""

    n = block.shape[0]
    real_part = block[:, :4].reshape(n, 2, 2)
    imag_part = block[:, 4:8].reshape(n, 2, 2)
    return real_part + 1j * imag_part


def _hermitian_2x2_features(matrices: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    """Extract classical cognitive radio detection statistics from 2x2 Hermitian matrices."""

    herm = (matrices + matrices.conj().transpose(0, 2, 1)) / 2.0
    eigvals = np.linalg.eigvalsh(herm.real if np.isrealobj(herm) else herm)
    lmin = eigvals[:, 0]
    lmax = eigvals[:, 1]

    tr = np.real(np.trace(herm, axis1=1, axis2=2))
    det = np.real(np.linalg.det(herm))
    total_abs = np.abs(herm).sum(axis=(1, 2))
    diag_abs = np.abs(herm[:, 0, 0]) + np.abs(herm[:, 1, 1])
    off_diag_mag = np.abs(herm[:, 0, 1])
    diag_product = np.abs(herm[:, 0, 0]) * np.abs(herm[:, 1, 1])
    coherence = off_diag_mag / np.maximum(np.sqrt(np.maximum(diag_product, 0)), 1e-15)

    safe_lmin = np.where(np.abs(lmin) < 1e-15, 1e-15, lmin)
    safe_tr = np.where(np.abs(tr) < 1e-15, 1e-15, tr)

    features = np.column_stack(
        [
            lmax,
            lmin,
            lmax - lmin,
            tr,
            lmax / np.abs(safe_lmin),
            lmax / np.abs(safe_tr),
            diag_abs / np.maximum(total_abs, 1e-15),
            coherence,
            det,
            np.log(np.maximum(np.abs(det), 1e-30)),
        ]
    )
    names = [
        f"{prefix}_eig_max",
        f"{prefix}_eig_min",
        f"{prefix}_eig_spread",
        f"{prefix}_trace_ed",
        f"{prefix}_mme",
        f"{prefix}_rle",
        f"{prefix}_cav",
        f"{prefix}_coherence",
        f"{prefix}_det",
        f"{prefix}_log_det",
    ]
    return features, names


def _collect_su_columns(
    feature_names: list[str],
    suffix: str,
) -> dict[str, list[int]]:
    """Find columns matching SU<n>_<suffix>_<index> and group by SU prefix."""

    pattern = re.compile(rf"^(SU\d+_{re.escape(suffix)})_(\d+)$")
    groups: dict[str, list[tuple[int, int]]] = {}
    for col_idx, name in enumerate(feature_names):
        match = pattern.match(name)
        if not match:
            continue
        prefix = match.group(1)
        elem_idx = int(match.group(2))
        groups.setdefault(prefix, []).append((elem_idx, col_idx))

    result: dict[str, list[int]] = {}
    for prefix, entries in groups.items():
        entries.sort(key=lambda item: item[0])
        result[prefix] = [col_idx for _, col_idx in entries]
    return result


def _extract_cognitive_radio_physics_features(
    matrix: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Extract physics-informed features from cognitive radio covariance data."""

    n_samples = matrix.shape[0]
    all_features: list[np.ndarray] = []
    all_names: list[str] = []

    cov_groups = _collect_su_columns(feature_names, "cov_flat")
    su_cov_stats: dict[str, np.ndarray] = {}

    for prefix, col_indices in cov_groups.items():
        block = matrix[:, col_indices]
        if block.shape[1] == _COV_BLOCK_SIZE:
            cov_mat = _reconstruct_hermitian_2x2(block)
            feats, names = _hermitian_2x2_features(cov_mat, prefix + "_phys")
        else:
            feats = np.column_stack(
                [
                    np.linalg.norm(block, axis=1),
                    block.mean(axis=1),
                    block.std(axis=1),
                ]
            )
            names = [f"{prefix}_phys_norm", f"{prefix}_phys_mean", f"{prefix}_phys_std"]
        all_features.append(feats)
        all_names.extend(names)
        su_cov_stats[prefix] = feats

    temp_groups = _collect_su_columns(feature_names, "temporal_cov")
    for prefix, col_indices in temp_groups.items():
        block = matrix[:, col_indices]
        n_blocks = block.shape[1] // _COV_BLOCK_SIZE
        if n_blocks >= 1 and block.shape[1] == n_blocks * _COV_BLOCK_SIZE:
            sub_features_list = []
            for block_index in range(n_blocks):
                start = block_index * _COV_BLOCK_SIZE
                sub_block = block[:, start : start + _COV_BLOCK_SIZE]
                cov_mat = _reconstruct_hermitian_2x2(sub_block)
                sub_feats, _ = _hermitian_2x2_features(cov_mat, "")
                sub_features_list.append(sub_feats)

            sub_stack = np.stack(sub_features_list, axis=1)
            time_mean = sub_stack.mean(axis=1)
            time_std = sub_stack.std(axis=1)
            time_diff = (
                np.abs(sub_stack[:, -1, :] - sub_stack[:, 0, :])
                if n_blocks > 1
                else np.zeros_like(time_mean)
            )

            stat_suffixes = [
                "eig_max",
                "eig_min",
                "eig_spread",
                "trace_ed",
                "mme",
                "rle",
                "cav",
                "coherence",
                "det",
                "log_det",
            ]
            for agg_name, agg_values in [("tmean", time_mean), ("tstd", time_std), ("tdelta", time_diff)]:
                for stat_index, stat_suffix in enumerate(stat_suffixes):
                    all_features.append(agg_values[:, stat_index : stat_index + 1])
                    all_names.append(f"{prefix}_phys_{agg_name}_{stat_suffix}")
        else:
            feats = np.column_stack(
                [
                    np.linalg.norm(block, axis=1),
                    block.mean(axis=1),
                    block.std(axis=1),
                ]
            )
            names = [f"{prefix}_phys_norm", f"{prefix}_phys_mean", f"{prefix}_phys_std"]
            all_features.append(feats)
            all_names.extend(names)

    cov_prefixes = sorted(cov_groups.keys())
    if len(cov_prefixes) >= 2:
        for left_index in range(len(cov_prefixes)):
            for right_index in range(left_index + 1, len(cov_prefixes)):
                left = matrix[:, cov_groups[cov_prefixes[left_index]]]
                right = matrix[:, cov_groups[cov_prefixes[right_index]]]
                dot = (left * right).sum(axis=1)
                left_norm = np.linalg.norm(left, axis=1)
                right_norm = np.linalg.norm(right, axis=1)
                cosine = dot / np.maximum(left_norm * right_norm, 1e-15)
                l2_distance = np.linalg.norm(left - right, axis=1)
                all_features.append(cosine.reshape(-1, 1))
                all_features.append(l2_distance.reshape(-1, 1))
                pair = f"{cov_prefixes[left_index]}_vs_{cov_prefixes[right_index]}"
                all_names.extend([f"{pair}_cosine", f"{pair}_l2dist"])

        if len(su_cov_stats) >= 2:
            stat_matrices = np.stack([su_cov_stats[prefix] for prefix in cov_prefixes], axis=1)
            cross_mean = stat_matrices.mean(axis=1)
            cross_std = stat_matrices.std(axis=1)
            stat_suffixes = [
                "eig_max",
                "eig_min",
                "eig_spread",
                "trace_ed",
                "mme",
                "rle",
                "cav",
                "coherence",
                "det",
                "log_det",
            ]
            for stat_index in range(min(cross_mean.shape[1], len(stat_suffixes))):
                suffix = stat_suffixes[stat_index]
                all_features.append(cross_mean[:, stat_index : stat_index + 1])
                all_features.append(cross_std[:, stat_index : stat_index + 1])
                all_names.extend([f"cross_su_cov_{suffix}_mean", f"cross_su_cov_{suffix}_std"])

    scalar_candidates = {"power_dB", "spectral_entropy", "freq_bin", "Frequency_Band"}
    scalar_map: dict[str, int] = {}
    for idx, name in enumerate(feature_names):
        if name in scalar_candidates:
            scalar_map[name] = idx

    key_stat_indices = [3, 4, 7]
    key_stat_names = ["trace_ed", "mme", "coherence"]
    for scalar_name, scalar_idx in scalar_map.items():
        scalar_vals = matrix[:, scalar_idx]
        for prefix in cov_prefixes:
            if prefix not in su_cov_stats:
                continue
            n_stats = su_cov_stats[prefix].shape[1]
            for stat_idx, stat_name in zip(key_stat_indices, key_stat_names, strict=False):
                if stat_idx >= n_stats:
                    continue
                interaction = scalar_vals * su_cov_stats[prefix][:, stat_idx]
                all_features.append(interaction.reshape(-1, 1))
                all_names.append(f"{prefix}_{stat_name}_x_{scalar_name}")

    if not all_features:
        return np.empty((n_samples, 0), dtype=float), []

    return np.hstack(all_features), all_names


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

        has_su_cov = any("_cov_flat_" in name for name in feature_names)
        if has_su_cov and method_name == "hybrid":
            raw_names = list(prepared["metadata"]["feature_names"])
            for split_label, split_x in [("train", train_x), ("val", val_x), ("test", test_x)]:
                raw_x = splits[split_label]["X"]
                phys_feats, phys_names = _extract_cognitive_radio_physics_features(raw_x, raw_names)
                if split_label == "train":
                    train_x = np.hstack([split_x, phys_feats])
                    feature_names = feature_names + phys_names
                elif split_label == "val":
                    val_x = np.hstack([split_x, phys_feats])
                else:
                    test_x = np.hstack([split_x, phys_feats])

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

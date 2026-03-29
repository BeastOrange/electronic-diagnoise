from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emc_diag.data_pipeline import load_local_data, prepare_dataset
from emc_diag.dataset_registry import DATASET_REGISTRY, get_dataset_info, list_datasets
from emc_diag.feature_engineering import (
    build_cognitive_radio_hybrid_bundle,
    build_sequence_bundle,
    extract_feature_bundle,
    make_prepared_layout_view,
    make_feature_group_view,
)


def test_dataset_registry_contains_required_datasets() -> None:
    ids = {item["id"] for item in list_datasets()}
    assert {"emi_uci", "vsb_power_line_fault", "electrical_fault"}.issubset(ids)
    assert get_dataset_info("emi_uci")["download_hint"]
    assert len(DATASET_REGISTRY) >= 3


def test_load_local_data_from_csv_and_parquet(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "f1": np.linspace(1.0, 2.0, 10),
            "f2": np.linspace(2.0, 3.0, 10),
            "label": [0, 1] * 5,
        }
    )

    csv_path = tmp_path / "sample.csv"
    pq_path = tmp_path / "sample.parquet"
    data.to_csv(csv_path, index=False)
    data.to_parquet(pq_path, index=False)

    loaded_csv = load_local_data(csv_path, schema="tabular", target_column="label")
    loaded_pq = load_local_data(pq_path, schema="tabular", target_column="label")

    assert loaded_csv["X"].shape == (10, 2)
    assert loaded_csv["y"].shape == (10,)
    assert loaded_pq["X"].shape == (10, 2)
    assert loaded_pq["feature_names"] == ["f1", "f2"]


def test_load_waveform_and_spectrum_from_directory(tmp_path: Path) -> None:
    for idx in range(6):
        label = idx % 2
        np.save(tmp_path / f"{label}_wave_{idx}.npy", np.random.randn(64))

    waveform = load_local_data(tmp_path, schema="waveform", target_column="label")
    spectrum = load_local_data(tmp_path, schema="spectrum", target_column="label")

    assert waveform["X"].shape[0] == 6
    assert waveform["X"].shape[1] == 64
    assert sorted(np.unique(waveform["y"]).tolist()) == [0, 1]
    assert spectrum["X"].shape == waveform["X"].shape


def test_prepare_dataset_smoke_tabular(tmp_path: Path) -> None:
    size = 40
    data = pd.DataFrame(
        {
            "f1": np.random.randn(size),
            "f2": np.random.randn(size),
            "f3": np.random.randn(size),
            "label": [0, 1] * (size // 2),
        }
    )
    csv_path = tmp_path / "tab.csv"
    data.to_csv(csv_path, index=False)

    prepared = prepare_dataset(
        source=csv_path,
        schema="tabular",
        target_column="label",
        train_ratio=0.7,
        val_ratio=0.15,
        random_state=7,
    )

    total = (
        prepared["splits"]["train"]["X"].shape[0]
        + prepared["splits"]["val"]["X"].shape[0]
        + prepared["splits"]["test"]["X"].shape[0]
    )
    assert total == size
    assert prepared["metadata"]["num_features"] == 3
    assert "mean" in prepared["scaler"]
    assert "std" in prepared["scaler"]
    assert "class_counts" in prepared["statistics"]


def test_prepare_dataset_smoke_waveform_and_spectrum(tmp_path: Path) -> None:
    for idx in range(20):
        label = idx % 2
        np.save(tmp_path / f"{label}_sig_{idx}.npy", np.random.randn(32))

    waveform = prepare_dataset(tmp_path, schema="waveform", random_state=21)
    spectrum = prepare_dataset(tmp_path, schema="spectrum", random_state=21)

    assert waveform["metadata"]["schema"] == "waveform"
    assert spectrum["metadata"]["schema"] == "spectrum"
    assert waveform["splits"]["train"]["X"].ndim == 2
    assert spectrum["splits"]["test"]["X"].shape[1] == 32


def test_load_local_data_drops_empty_unnamed_columns(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "Output (S)": [0, 1, 0, 1],
            "Ia": [1.0, 2.0, 1.5, 2.5],
            "Ib": [0.5, 0.7, 0.6, 0.8],
            "Unnamed: 7": [np.nan, np.nan, np.nan, np.nan],
            "Unnamed: 8": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    csv_path = tmp_path / "detect_dataset.csv"
    data.to_csv(csv_path, index=False)

    loaded = load_local_data(csv_path, schema="tabular", target_column="Output (S)")

    assert loaded["X"].shape == (4, 2)
    assert loaded["feature_names"] == ["Ia", "Ib"]


def test_load_local_data_supports_cognitive_radio_mixed_columns(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "PU_Presence": [0, 1, 0, 1],
            "power_dB": [-55.1, -62.0, -74.5, -49.3],
            "PU_drift_type": ["Frequency drift", "Bandwidth switch", "Frequency drift", "No drift"],
            "SU1_cov_flat": [
                "[1.0, 2.0, 3.0]",
                "[4.0, 5.0, 6.0]",
                "[7.0, 8.0, 9.0]",
                "[0.1, 0.2, 0.3]",
            ],
            "SU2_temporal_cov": [
                "array([1.00e-06, 2.00e-07,\n 3.00e-08])",
                "array([2.00e-06, 3.00e-07,\n 4.00e-08])",
                "array([3.00e-06, 4.00e-07,\n 5.00e-08])",
                "array([4.00e-06, 5.00e-07,\n 6.00e-08])",
            ],
        }
    )
    csv_path = tmp_path / "cognitive_radio_sample.csv"
    data.to_csv(csv_path, index=False)

    loaded = load_local_data(csv_path, schema="tabular", target_column="PU_Presence")

    assert loaded["X"].shape[0] == 4
    assert loaded["X"].shape[1] >= 8
    assert np.issubdtype(loaded["X"].dtype, np.floating)
    assert any(name.startswith("SU1_cov_flat_") for name in loaded["feature_names"])
    assert any(name.startswith("SU2_temporal_cov_") for name in loaded["feature_names"])
    assert any(name.startswith("PU_drift_type__") for name in loaded["feature_names"])


def test_prepare_dataset_supports_cognitive_radio_style_csv(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "PU_Presence": [0, 1, 0, 1, 0, 1],
            "power_dB": [-55.1, -62.0, -74.5, -49.3, -58.2, -60.4],
            "Frequency_Band": ["1800", "5000", "5000", "1800", "2400", "2400"],
            "SU1_cov_flat": [
                "[1.0, 2.0, 3.0]",
                "[4.0, 5.0, 6.0]",
                "[7.0, 8.0, 9.0]",
                "[0.1, 0.2, 0.3]",
                "[0.3, 0.4, 0.5]",
                "[0.8, 0.9, 1.0]",
            ],
        }
    )
    csv_path = tmp_path / "cognitive_radio_prepare.csv"
    data.to_csv(csv_path, index=False)

    prepared = prepare_dataset(
        source=csv_path,
        schema="tabular",
        target_column="PU_Presence",
        train_ratio=0.5,
        val_ratio=0.2,
        random_state=11,
    )

    assert prepared["metadata"]["num_samples"] == 6
    assert prepared["metadata"]["num_features"] >= 5
    assert set(prepared["metadata"]["labels"]) == {"0", "1"}
    assert prepared["metadata"]["task_name"] == "PU_Presence"
    assert prepared["metadata"]["task_type"] == "single_task"
    assert prepared["metadata"]["tasks"][0]["target_column"] == "PU_Presence"


def test_prepare_dataset_supports_multitask_contract(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "power_dB": np.random.randn(12),
            "PU_Presence": [0, 1, 0, 1] * 3,
            "Frequency_Band": ["1800", "5000", "1800", np.nan] * 3,
            "payload": np.linspace(0, 1, 12),
        }
    )
    csv_path = tmp_path / "multitask.csv"
    data.to_csv(csv_path, index=False)

    prepared = prepare_dataset(
        source=csv_path,
        schema="tabular",
        target_column="PU_Presence",
        task_columns=["PU_Presence", "Frequency_Band"],
        train_ratio=0.5,
        val_ratio=0.2,
        random_state=13,
    )

    metadata = prepared["metadata"]
    assert metadata["task_type"] == "multitask"
    assert metadata["task_columns"] == ["PU_Presence", "Frequency_Band"]
    assert len(metadata["tasks"]) == 2
    assert metadata["tasks"][1]["target_column"] == "Frequency_Band"

    combined_masks = []
    combined_labels = []
    for split in prepared["splits"].values():
        assert "y_tasks" in split
        assert {"PU_Presence", "Frequency_Band"} <= set(split["y_tasks"].keys())
        assert "task_masks" in split
        mask = split["task_masks"]["Frequency_Band"]
        assert mask.dtype == bool
        combined_masks.append(mask)
        combined_labels.append(split["y_tasks"]["Frequency_Band"])

    overall_mask = np.concatenate(combined_masks)
    overall_labels = np.concatenate(combined_labels)
    assert np.any(overall_mask == False)
    assert np.any(overall_labels == -1)


def test_load_local_data_supports_nested_temporal_cov_arrays(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "PU_Presence": [0, 1],
            "power_dB": [-55.0, -60.0],
            "SU1_temporal_cov": [
                "[array([1.00e-06, 2.00e-07, 3.00e-08]), array([4.00e-06, 5.00e-07, 6.00e-08])]",
                "[array([2.00e-06, 3.00e-07, 4.00e-08]), array([5.00e-06, 6.00e-07, 7.00e-08])]",
            ],
        }
    )
    csv_path = tmp_path / "cognitive_temporal_cov.csv"
    data.to_csv(csv_path, index=False)

    loaded = load_local_data(csv_path, schema="tabular", target_column="PU_Presence")

    assert loaded["X"].shape[0] == 2
    assert any(name.startswith("SU1_temporal_cov_") for name in loaded["feature_names"])


def test_extract_feature_bundle_adds_prefix_aggregations_for_tabular() -> None:
    train_x = np.array(
        [
            [1.0, 2.0, 3.0, 0.5],
            [2.0, 3.0, 4.0, 0.3],
            [4.0, 5.0, 6.0, 0.7],
        ],
        dtype=float,
    )
    val_x = np.array([[1.2, 2.2, 3.2, 0.4]], dtype=float)
    test_x = np.array([[2.2, 3.2, 4.2, 0.6]], dtype=float)
    prepared = {
        "metadata": {
            "schema": "tabular",
            "feature_names": [
                "SU1_cov_flat_0",
                "SU1_cov_flat_1",
                "SU1_cov_flat_2",
                "power_dB",
            ],
        },
        "splits": {
            "train": {"X": train_x, "y": np.array([0, 1, 1], dtype=int)},
            "val": {"X": val_x, "y": np.array([1], dtype=int)},
            "test": {"X": test_x, "y": np.array([0], dtype=int)},
        },
    }

    bundle = extract_feature_bundle(prepared, top_k=None)
    names = bundle["feature_names"]

    assert "SU1_cov_flat__mean" in names
    assert "SU1_cov_flat__std" in names
    assert "SU1_cov_flat__max" in names
    assert "SU1_cov_flat__min" in names
    assert "SU1_cov_flat__energy" in names
    assert "SU1_cov_flat__iqr" in names
    assert "SU1_cov_flat__variation" in names
    assert bundle["splits"]["train"]["X"].shape[1] == len(names)


def test_extract_feature_bundle_basic_method_keeps_raw_tabular_features() -> None:
    prepared = {
        "metadata": {
            "schema": "tabular",
            "feature_names": [
                "SU1_cov_flat_0",
                "SU1_cov_flat_1",
                "power_dB",
            ],
        },
        "splits": {
            "train": {"X": np.array([[1.0, 2.0, 0.5], [2.0, 3.0, 0.6]], dtype=float), "y": np.array([0, 1], dtype=int)},
            "val": {"X": np.array([[1.5, 2.5, 0.4]], dtype=float), "y": np.array([1], dtype=int)},
            "test": {"X": np.array([[2.5, 3.5, 0.7]], dtype=float), "y": np.array([0], dtype=int)},
        },
    }

    bundle = extract_feature_bundle(prepared, method="basic", top_k=None)

    assert bundle["feature_names"] == ["SU1_cov_flat_0", "SU1_cov_flat_1", "power_dB"]
    assert set(bundle["selected_feature_names"]) == {"SU1_cov_flat_0", "SU1_cov_flat_1", "power_dB"}
    assert bundle["splits"]["train"]["X"].shape[1] == 3


def test_prepare_dataset_encodes_string_targets_and_tracks_labels(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "power_dB": [-55.1, -62.0, -74.5, -49.3, -58.2, -60.4],
            "PU_burst_duration": ["Long", "Medium", "Short", "Long", "Short", "Medium"],
            "SU1_cov_flat": [
                "[1.0, 2.0, 3.0]",
                "[4.0, 5.0, 6.0]",
                "[7.0, 8.0, 9.0]",
                "[0.1, 0.2, 0.3]",
                "[0.3, 0.4, 0.5]",
                "[0.8, 0.9, 1.0]",
            ],
        }
    )
    csv_path = tmp_path / "cognitive_burst.csv"
    data.to_csv(csv_path, index=False)

    prepared = prepare_dataset(
        source=csv_path,
        schema="tabular",
        target_column="PU_burst_duration",
        task_name="burst_task",
        train_ratio=0.5,
        val_ratio=0.2,
        random_state=5,
    )

    assert prepared["metadata"]["task_name"] == "burst_task"
    assert prepared["metadata"]["labels"] == ["Long", "Medium", "Short"]
    assert prepared["splits"]["train"]["y"].dtype.kind in {"i", "u"}


def test_make_feature_group_view_filters_by_domain_groups() -> None:
    prepared = {
        "metadata": {
            "schema": "tabular",
            "feature_names": [
                "power_dB",  # basic_statistical
                "SU1_cov_flat_0",  # covariance_based
                "SU1_temporal_cov_0",  # temporal_covariance
                "spectral_entropy",  # entropy_related
                "fft_0",  # spectral_energy
            ],
        },
        "splits": {
            "train": {"X": np.ones((4, 5), dtype=float), "y": np.array([0, 1, 0, 1])},
            "val": {"X": np.ones((2, 5), dtype=float), "y": np.array([0, 1])},
            "test": {"X": np.ones((2, 5), dtype=float), "y": np.array([0, 1])},
        },
        "statistics": {},
        "scaler": {},
    }

    view = make_feature_group_view(prepared, ["covariance_based", "entropy_related"])
    names = view["metadata"]["feature_names"]

    assert "SU1_cov_flat_0" in names
    assert "spectral_entropy" in names
    assert "power_dB" not in names
    assert "fft_0" not in names
    assert view["metadata"]["num_features"] == len(names)
    assert "domain_feature_groups" in view["metadata"]


def test_make_prepared_layout_view_filters_cov_and_temporal_features() -> None:
    prepared = {
        "metadata": {
            "schema": "tabular",
            "feature_names": [
                "power_dB",
                "SU1_cov_flat_0",
                "SU1_cov_flat_1",
                "SU1_temporal_cov_0",
                "SU1_temporal_cov_1",
            ],
        },
        "splits": {
            "train": {"X": np.ones((3, 5), dtype=float), "y": np.array([0, 1, 0])},
            "val": {"X": np.ones((2, 5), dtype=float), "y": np.array([1, 0])},
            "test": {"X": np.ones((2, 5), dtype=float), "y": np.array([0, 1])},
        },
        "statistics": {},
        "scaler": {},
    }

    basic_view = make_prepared_layout_view(prepared, "basic_only")
    cov_view = make_prepared_layout_view(prepared, "cov_flat_only")
    temporal_view = make_prepared_layout_view(prepared, "temporal_cov_only")

    assert basic_view["metadata"]["feature_names"] == ["power_dB"]
    assert cov_view["metadata"]["feature_names"] == ["SU1_cov_flat_0", "SU1_cov_flat_1"]
    assert temporal_view["metadata"]["feature_names"] == ["SU1_temporal_cov_0", "SU1_temporal_cov_1"]


def test_build_sequence_bundle_creates_multichannel_input() -> None:
    prepared = {
        "metadata": {
            "schema": "tabular",
            "feature_names": [
                "power_dB",
                "SU1_cov_flat_0",
                "SU1_cov_flat_1",
                "SU1_temporal_cov_0",
                "SU1_temporal_cov_1",
                "SU1_temporal_cov_2",
            ],
        },
        "splits": {
            "train": {"X": np.arange(18, dtype=float).reshape(3, 6), "y": np.array([0, 1, 0])},
            "val": {"X": np.arange(12, dtype=float).reshape(2, 6), "y": np.array([1, 0])},
            "test": {"X": np.arange(12, dtype=float).reshape(2, 6), "y": np.array([0, 1])},
        },
    }

    sequence_bundle = build_sequence_bundle(prepared, layout="all")

    assert sequence_bundle["splits"]["train"]["X"].ndim == 3
    assert sequence_bundle["splits"]["train"]["X"].shape[1] == 3
    assert sequence_bundle["channel_names"] == ["SU1_cov_flat", "SU1_temporal_cov", "power_dB"]
    assert sequence_bundle["summary"]["max_sequence_length"] == 3


def test_build_cognitive_radio_hybrid_bundle_reconstructs_structured_inputs() -> None:
    prepared = {
        "metadata": {
            "schema": "tabular",
            "feature_names": [
                "power_dB",
                "spectral_entropy",
                "SU1_cov_flat_0",
                "SU1_cov_flat_1",
                "SU2_cov_flat_0",
                "SU2_cov_flat_1",
                "SU1_temporal_cov_0",
                "SU1_temporal_cov_1",
                "SU2_temporal_cov_0",
                "SU2_temporal_cov_1",
            ],
        },
        "splits": {
            "train": {"X": np.arange(40, dtype=float).reshape(4, 10), "y": np.array([0, 1, 0, 1])},
            "val": {"X": np.arange(20, dtype=float).reshape(2, 10), "y": np.array([1, 0])},
            "test": {"X": np.arange(20, dtype=float).reshape(2, 10), "y": np.array([0, 1])},
        },
    }

    bundle = build_cognitive_radio_hybrid_bundle(prepared)

    train_split = bundle["splits"]["train"]
    assert train_split["scalar_X"].shape == (4, 2)
    assert train_split["cov_X"].shape == (4, 2, 2)
    assert train_split["temporal_X"].shape == (4, 2, 2)
    assert bundle["summary"]["scalar_feature_names"] == ["power_dB", "spectral_entropy"]
    assert bundle["summary"]["cov_sensor_names"] == ["SU1_cov_flat", "SU2_cov_flat"]
    assert bundle["summary"]["temporal_sensor_names"] == ["SU1_temporal_cov", "SU2_temporal_cov"]


def test_load_local_data_supports_vsb_train_parquet_with_smoke_option(tmp_path: Path) -> None:
    vsb_dir = tmp_path / "vsb-power-line-fault-detection"
    vsb_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame(
        {
            "signal_id": [0, 1, 2, 3, 4, 5],
            "id_measurement": [0, 0, 1, 1, 2, 2],
            "phase": [0, 1, 2, 0, 1, 2],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    metadata.to_csv(vsb_dir / "metadata_train.csv", index=False)

    rows = 120
    train_df = pd.DataFrame(
        {
            str(signal_id): np.sin(np.linspace(0.0, 3.14, rows) + signal_id)
            for signal_id in metadata["signal_id"]
        }
    )
    train_df.to_parquet(vsb_dir / "train.parquet", index=False)

    source = f"{vsb_dir}::max_signals=4::max_timesteps=80::seed=13"
    loaded = load_local_data(source=source, schema="tabular", target_column="target")

    assert loaded["X"].shape[0] == 4
    assert loaded["y"].shape == (4,)
    assert np.issubdtype(loaded["X"].dtype, np.floating)
    assert "signal_mean" in loaded["feature_names"]
    assert "signal_zero_cross_rate" in loaded["feature_names"]
    assert loaded["label_names"] == ["0", "1"]


def test_prepare_dataset_supports_vsb_train_parquet_with_smoke_option(tmp_path: Path) -> None:
    vsb_dir = tmp_path / "vsb-power-line-fault-detection"
    vsb_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame(
        {
            "signal_id": list(range(12)),
            "id_measurement": [index // 3 for index in range(12)],
            "phase": [index % 3 for index in range(12)],
            "target": [0, 1] * 6,
        }
    )
    metadata.to_csv(vsb_dir / "metadata_train.csv", index=False)

    rows = 200
    train_df = pd.DataFrame(
        {
            str(signal_id): np.cos(np.linspace(0.0, 6.28, rows) + signal_id * 0.1)
            for signal_id in metadata["signal_id"]
        }
    )
    train_df.to_parquet(vsb_dir / "train.parquet", index=False)

    source = f"{vsb_dir}::max_signals=10::max_timesteps=120::seed=7"
    prepared = prepare_dataset(
        source=source,
        schema="tabular",
        target_column="target",
        task_name="vsb_smoke",
        train_ratio=0.6,
        val_ratio=0.2,
        random_state=7,
    )

    assert prepared["metadata"]["task_name"] == "vsb_smoke"
    assert prepared["metadata"]["num_samples"] == 10
    assert prepared["metadata"]["num_features"] >= 6
    assert set(prepared["metadata"]["labels"]) == {"0", "1"}


def test_load_local_data_reuses_vsb_feature_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from emc_diag import data_pipeline as data_pipeline_module

    vsb_dir = tmp_path / "vsb-power-line-fault-detection"
    vsb_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame(
        {
            "signal_id": list(range(8)),
            "id_measurement": [index // 2 for index in range(8)],
            "phase": [index % 3 for index in range(8)],
            "target": [0, 1] * 4,
        }
    )
    metadata.to_csv(vsb_dir / "metadata_train.csv", index=False)

    rows = 128
    train_df = pd.DataFrame(
        {
            str(signal_id): np.sin(np.linspace(0.0, 6.28, rows) + signal_id * 0.2)
            for signal_id in metadata["signal_id"]
        }
    )
    train_df.to_parquet(vsb_dir / "train.parquet", index=False)

    source = f"{vsb_dir}::max_signals=6::max_timesteps=64::seed=11::signal_chunk_size=2"
    first_loaded = load_local_data(source=source, schema="tabular", target_column="target")
    cache_files = sorted((vsb_dir / ".emc_diag_cache").glob("vsb_features_*.parquet"))
    assert cache_files

    def _raise_if_recomputed(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("expected cached VSB features to be reused")

    monkeypatch.setattr(data_pipeline_module, "_extract_vsb_signal_features", _raise_if_recomputed)
    second_loaded = load_local_data(source=source, schema="tabular", target_column="target")

    assert first_loaded["X"].shape == second_loaded["X"].shape
    np.testing.assert_allclose(first_loaded["X"], second_loaded["X"])


def test_load_local_data_reuses_vsb_feature_cache(tmp_path: Path, monkeypatch: Any) -> None:
    vsb_dir = tmp_path / "vsb-power-line-fault-detection"
    vsb_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame(
        {
            "signal_id": [0, 1, 2, 3],
            "id_measurement": [0, 0, 1, 1],
            "phase": [0, 1, 2, 0],
            "target": [0, 1, 0, 1],
        }
    )
    metadata.to_csv(vsb_dir / "metadata_train.csv", index=False)

    train_df = pd.DataFrame(
        {
            "0": np.sin(np.linspace(0.0, 3.14, 64)),
            "1": np.cos(np.linspace(0.0, 3.14, 64)),
            "2": np.sin(np.linspace(0.0, 3.14, 64) + 0.2),
            "3": np.cos(np.linspace(0.0, 3.14, 64) + 0.2),
        }
    )
    train_df.to_parquet(vsb_dir / "train.parquet", index=False)

    from emc_diag import data_pipeline as data_pipeline_module

    call_count = {"value": 0}
    original = data_pipeline_module._extract_vsb_signal_features

    def _wrapped(*args: Any, **kwargs: Any) -> pd.DataFrame:
        call_count["value"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(data_pipeline_module, "_extract_vsb_signal_features", _wrapped)

    source = f"{vsb_dir}::max_signals=4::max_timesteps=64::seed=13::signal_chunk_size=2"
    first = load_local_data(source=source, schema="tabular", target_column="target")
    second = load_local_data(source=source, schema="tabular", target_column="target")

    assert first["X"].shape == second["X"].shape
    assert call_count["value"] == 1

from __future__ import annotations

import numpy as np

from emc_diag.vsb_features import (
    build_vsb_feature_matrix,
    fft_summary_features,
    statistics_features,
    stft_summary_features,
)


def test_statistics_features_returns_expected_shape_and_names() -> None:
    waveforms = np.array(
        [
            [1.0, -1.0, 1.0, -1.0],
            [0.0, 1.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    features, names = statistics_features(waveforms)

    assert features.shape == (2, len(names))
    assert "stat_mean" in names
    assert "stat_std" in names
    assert "stat_rms" in names

    idx_mean = names.index("stat_mean")
    idx_peak = names.index("stat_peak_to_peak")
    assert np.isclose(features[0, idx_mean], 0.0)
    assert np.isclose(features[1, idx_peak], 3.0)


def test_fft_summary_features_captures_dominant_frequency() -> None:
    sample_rate = 100.0
    time_points = np.arange(0, 2.0, 1.0 / sample_rate)
    sine = np.sin(2.0 * np.pi * 5.0 * time_points)
    waveforms = sine.reshape(1, -1)

    features, names = fft_summary_features(waveforms, sample_rate=sample_rate)
    idx_dom = names.index("fft_dominant_frequency")

    assert features.shape == (1, len(names))
    assert np.isclose(features[0, idx_dom], 5.0, atol=0.6)


def test_stft_summary_features_runs_on_short_waveforms() -> None:
    waveforms = np.array(
        [
            np.linspace(-1.0, 1.0, 24),
            np.sin(np.linspace(0.0, 4.0 * np.pi, 24)),
        ],
        dtype=float,
    )
    features, names = stft_summary_features(waveforms, sample_rate=48.0, nperseg=32, noverlap=16)

    assert features.shape == (2, len(names))
    assert "stft_energy" in names
    assert np.all(np.isfinite(features))


def test_build_vsb_feature_matrix_concatenates_all_groups() -> None:
    rng = np.random.default_rng(7)
    waveforms = rng.normal(size=(3, 64))

    matrix, names = build_vsb_feature_matrix(
        waveforms,
        sample_rate=64.0,
        nperseg=32,
        noverlap=16,
    )

    assert matrix.shape == (3, len(names))
    assert any(name.startswith("stat_") for name in names)
    assert any(name.startswith("fft_") for name in names)
    assert any(name.startswith("stft_") for name in names)

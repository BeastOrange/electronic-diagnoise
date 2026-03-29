from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import stft


def _as_2d_waveforms(waveforms: np.ndarray) -> np.ndarray:
    array = np.asarray(waveforms, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"waveforms must be 1D or 2D, got shape {array.shape}")
    return array


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return numerator / np.maximum(denominator, eps)


def statistics_features(waveforms: np.ndarray) -> Tuple[np.ndarray, list[str]]:
    x = _as_2d_waveforms(waveforms)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    abs_mean = np.mean(np.abs(x), axis=1)
    maximum = x.max(axis=1)
    minimum = x.min(axis=1)
    peak_to_peak = maximum - minimum
    crest_factor = _safe_divide(np.abs(maximum), rms)

    features = np.column_stack(
        [
            mean,
            std,
            rms,
            abs_mean,
            maximum,
            minimum,
            peak_to_peak,
            crest_factor,
        ]
    )
    names = [
        "stat_mean",
        "stat_std",
        "stat_rms",
        "stat_abs_mean",
        "stat_max",
        "stat_min",
        "stat_peak_to_peak",
        "stat_crest_factor",
    ]
    return features, names


def fft_summary_features(waveforms: np.ndarray, sample_rate: float = 1.0) -> Tuple[np.ndarray, list[str]]:
    x = _as_2d_waveforms(waveforms)
    n_samples, n_points = x.shape
    if n_points < 2:
        raise ValueError("waveforms must contain at least 2 points for FFT features")

    fft_complex = np.fft.rfft(x, axis=1)
    magnitudes = np.abs(fft_complex)
    frequencies = np.fft.rfftfreq(n_points, d=1.0 / sample_rate)

    power = np.square(magnitudes)
    total_power = power.sum(axis=1)
    dominant_idx = np.argmax(magnitudes, axis=1)
    dominant_frequency = frequencies[dominant_idx]
    dominant_amplitude = magnitudes[np.arange(n_samples), dominant_idx]
    spectral_centroid = _safe_divide((power * frequencies).sum(axis=1), total_power)
    spectral_bandwidth = np.sqrt(
        _safe_divide((power * np.square(frequencies.reshape(1, -1) - spectral_centroid.reshape(-1, 1))).sum(axis=1), total_power)
    )
    flatness = _safe_divide(
        np.exp(np.mean(np.log(np.maximum(magnitudes, 1e-12)), axis=1)),
        np.mean(magnitudes, axis=1),
    )

    features = np.column_stack(
        [
            total_power,
            dominant_frequency,
            dominant_amplitude,
            spectral_centroid,
            spectral_bandwidth,
            flatness,
        ]
    )
    names = [
        "fft_total_power",
        "fft_dominant_frequency",
        "fft_dominant_amplitude",
        "fft_spectral_centroid",
        "fft_spectral_bandwidth",
        "fft_spectral_flatness",
    ]
    return features, names


def stft_summary_features(
    waveforms: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: int = 64,
    noverlap: int = 32,
) -> Tuple[np.ndarray, list[str]]:
    x = _as_2d_waveforms(waveforms)
    n_points = x.shape[1]
    effective_nperseg = min(max(2, int(nperseg)), n_points)
    effective_noverlap = min(max(0, int(noverlap)), effective_nperseg - 1)

    rows = []
    for row in x:
        _, _, zxx = stft(
            row,
            fs=sample_rate,
            nperseg=effective_nperseg,
            noverlap=effective_noverlap,
            boundary=None,
            padded=False,
        )
        magnitude = np.abs(zxx)
        energy = np.mean(np.square(magnitude))
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        max_mag = np.max(magnitude)
        # Temporal flux: frame-to-frame average magnitude variation.
        if magnitude.shape[1] > 1:
            flux = np.mean(np.abs(np.diff(magnitude, axis=1)))
        else:
            flux = 0.0
        rows.append([energy, mean_mag, std_mag, max_mag, flux])

    features = np.asarray(rows, dtype=float)
    names = [
        "stft_energy",
        "stft_mean_magnitude",
        "stft_std_magnitude",
        "stft_max_magnitude",
        "stft_temporal_flux",
    ]
    return features, names


def build_vsb_feature_matrix(
    waveforms: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: int = 64,
    noverlap: int = 32,
) -> Tuple[np.ndarray, list[str]]:
    stats_matrix, stats_names = statistics_features(waveforms)
    fft_matrix, fft_names = fft_summary_features(waveforms, sample_rate=sample_rate)
    stft_matrix, stft_names = stft_summary_features(
        waveforms,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    matrix = np.hstack([stats_matrix, fft_matrix, stft_matrix])
    names = stats_names + fft_names + stft_names
    return matrix, names

from __future__ import annotations

import numpy as np
import pytest

from emc_diag.llm_text_adapter import tabular_matrix_to_texts


def test_tabular_matrix_to_texts_formats_named_features() -> None:
    matrix = np.asarray([[1.234567, -2.0], [np.nan, 0.5]], dtype=float)
    names = ["power_dB", "spectral_entropy"]

    texts = tabular_matrix_to_texts(matrix, names)

    assert len(texts) == 2
    assert "power_dB=1.23457" in texts[0]
    assert "spectral_entropy=-2" in texts[0]
    assert "power_dB=nan" in texts[1]


def test_tabular_matrix_to_texts_includes_emc_task_instruction_and_labels() -> None:
    matrix = np.asarray([[0.12, 0.34]], dtype=float)
    names = ["spectral_entropy", "energy_ratio"]

    texts = tabular_matrix_to_texts(
        matrix,
        names,
        task_instruction="Determine whether the primary user is present in this EMC sensing sample.",
        label_descriptions={"0": "absent", "1": "present"},
        feature_limit=1,
    )

    assert len(texts) == 1
    assert "EMC sensing sample" in texts[0]
    assert "Label meanings: 0=absent; 1=present." in texts[0]
    assert "Observed features (spectrum-sensing style statistics where applicable): spectral_entropy=0.12" in texts[0]
    assert "energy_ratio=0.34" not in texts[0]
    assert "Return only the class id." not in texts[0]


def test_tabular_matrix_to_texts_humanizes_physics_feature_names() -> None:
    """Physics-engineered names should expose ED/MME/RLE semantics to the LLM."""
    matrix = np.asarray([[0.1, 0.2, 0.3]], dtype=float)
    names = [
        "SU1_cov_flat_phys_mme",
        "cross_su_cov_eig_min_mean",
        "SU2_cov_flat_mme_x_freq_bin",
    ]
    texts = tabular_matrix_to_texts(matrix, names)
    assert len(texts) == 1
    body = texts[0]
    assert "SU1_cov_flat_max_min_eigenvalue_ratio=0.1" in body
    assert "cross_sensor_eig_min_mean=0.2" in body
    assert "SU2_cov_flat_max_min_eigenvalue_ratio_times_freq_bin=0.3" in body


def test_tabular_matrix_to_texts_requires_2d_matrix() -> None:
    with pytest.raises(ValueError, match="Expected 2D matrix"):
        tabular_matrix_to_texts(np.asarray([1.0, 2.0], dtype=float))

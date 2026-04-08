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


def test_tabular_matrix_to_texts_requires_2d_matrix() -> None:
    with pytest.raises(ValueError, match="Expected 2D matrix"):
        tabular_matrix_to_texts(np.asarray([1.0, 2.0], dtype=float))

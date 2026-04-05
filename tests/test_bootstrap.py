from __future__ import annotations

import importlib
import os
import sys


def test_configure_numeric_runtime_defaults_sets_thread_limits_when_unset(monkeypatch) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    bootstrap = importlib.import_module("emc_diag.bootstrap")
    bootstrap.configure_numeric_runtime_defaults()

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"
    assert os.environ["BLIS_NUM_THREADS"] == "1"


def test_cli_import_keeps_training_stack_lazy() -> None:
    for module_name in (
        "emc_diag.cli",
        "emc_diag.modeling",
        "sklearn.model_selection",
    ):
        sys.modules.pop(module_name, None)

    importlib.import_module("emc_diag.cli")

    assert "emc_diag.modeling" not in sys.modules
    assert "sklearn.model_selection" not in sys.modules

from __future__ import annotations

import os


_THREAD_LIMIT_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def configure_numeric_runtime_defaults() -> None:
    for key, value in _THREAD_LIMIT_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

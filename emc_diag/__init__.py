from __future__ import annotations

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "emc_diag"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))  # type: ignore[attr-defined]

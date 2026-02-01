from __future__ import annotations

import pkgutil
from pathlib import Path

__path__ = pkgutil.extend_path(__path__, __name__)

_SRC_PKG = Path(__file__).resolve().parent.parent / "src" / "c3rnt2"
if _SRC_PKG.exists():
    __path__.append(str(_SRC_PKG))

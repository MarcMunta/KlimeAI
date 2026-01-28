from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from rich.logging import RichHandler
except Exception:  # pragma: no cover - optional
    RichHandler = None


_DEF_FMT = "%(message)s" if RichHandler else "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def setup_logging(level: str | None = None) -> None:
    log_level = (level or os.getenv("C3RNT2_LOG_LEVEL") or "INFO").upper()
    if RichHandler:
        logging.basicConfig(
            level=log_level,
            format=_DEF_FMT,
            datefmt="%H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(level=log_level, format=_DEF_FMT)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "c3rnt2")

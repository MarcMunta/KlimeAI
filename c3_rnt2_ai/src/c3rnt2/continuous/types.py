from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    prompt: str
    response: str
    source_kind: str = "unknown"

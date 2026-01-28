from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Profiler:
    tokens: int = 0
    start_time: float = field(default_factory=time.time)
    page_faults: int = 0
    vram_mb: float = 0.0

    def tick(self, tokens: int = 1, page_faults: int = 0):
        self.tokens += tokens
        self.page_faults += page_faults

    def tokens_per_second(self) -> float:
        elapsed = max(1e-6, time.time() - self.start_time)
        return self.tokens / elapsed

    def report(self) -> dict:
        return {
            "tokens": self.tokens,
            "tokens_per_second": round(self.tokens_per_second(), 3),
            "page_faults": self.page_faults,
            "vram_mb": self.vram_mb,
        }

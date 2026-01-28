from __future__ import annotations

from dataclasses import dataclass
from time import time
from urllib.parse import urlparse


@dataclass
class WebPolicy:
    allowlist: list[str]
    rate_limit_per_min: int = 30
    _last_reset: float = 0.0
    _count: int = 0

    def allow_url(self, url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        return any(domain.endswith(a) for a in self.allowlist)

    def check_rate(self) -> bool:
        now = time()
        if now - self._last_reset > 60:
            self._last_reset = now
            self._count = 0
        if self._count >= self.rate_limit_per_min:
            return False
        self._count += 1
        return True

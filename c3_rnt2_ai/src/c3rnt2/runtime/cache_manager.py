from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Hashable, Optional


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_in: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class CacheEntry:
    value: object
    size_bytes: int
    stability: float = 0.0


class CacheManager:
    """LRU cache with stability bias for C3 tiles."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self._entries: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self.metrics = CacheMetrics()

    def get(self, key: Hashable) -> Optional[object]:
        if key in self._entries:
            entry = self._entries.pop(key)
            entry.stability += 0.1
            self._entries[key] = entry
            self.metrics.hits += 1
            return entry.value
        self.metrics.misses += 1
        return None

    def put(self, key: Hashable, value: object, size_bytes: int, stability: float = 0.0) -> None:
        if key in self._entries:
            self.current_bytes -= self._entries[key].size_bytes
            self._entries.pop(key, None)
        entry = CacheEntry(value=value, size_bytes=size_bytes, stability=stability)
        self._entries[key] = entry
        self.current_bytes += size_bytes
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while self.current_bytes > self.capacity_bytes and self._entries:
            # evict lowest stability among oldest
            lowest_key = None
            lowest_score = None
            for k, entry in self._entries.items():
                score = entry.stability
                if lowest_score is None or score < lowest_score:
                    lowest_score = score
                    lowest_key = k
            if lowest_key is None:
                break
            entry = self._entries.pop(lowest_key)
            self.current_bytes -= entry.size_bytes
            self.metrics.evictions += 1

    def stats(self) -> Dict[str, float]:
        return {
            "capacity_bytes": float(self.capacity_bytes),
            "current_bytes": float(self.current_bytes),
            "hit_rate": self.metrics.hit_rate,
            "hits": float(self.metrics.hits),
            "misses": float(self.metrics.misses),
            "evictions": float(self.metrics.evictions),
        }

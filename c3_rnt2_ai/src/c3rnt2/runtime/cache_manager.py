from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Hashable, Optional


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_in: int = 0
    bytes_h2d: int = 0
    bytes_compressed: int = 0

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
    """Segmented-LRU cache (probation/protected) for C3 tiles."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self._probation: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self._protected: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self.metrics = CacheMetrics()

    def _promote(self, key: Hashable) -> Optional[CacheEntry]:
        entry = self._probation.pop(key, None)
        if entry is not None:
            self._protected[key] = entry
        return entry

    def get(self, key: Hashable) -> Optional[object]:
        entry = self._protected.get(key)
        if entry is not None:
            entry.stability += 0.1
            self._protected.move_to_end(key)
            self.metrics.hits += 1
            return entry.value
        entry = self._probation.get(key)
        if entry is not None:
            entry.stability += 0.1
            self._promote(key)
            self.metrics.hits += 1
            return entry.value
        self.metrics.misses += 1
        return None

    def put(self, key: Hashable, value: object, size_bytes: int, stability: float = 0.0) -> None:
        if key in self._protected:
            self.current_bytes -= self._protected[key].size_bytes
            self._protected.pop(key, None)
        if key in self._probation:
            self.current_bytes -= self._probation[key].size_bytes
            self._probation.pop(key, None)
        entry = CacheEntry(value=value, size_bytes=int(size_bytes), stability=float(stability))
        self._probation[key] = entry
        self.current_bytes += int(size_bytes)
        self.metrics.bytes_in += int(size_bytes)
        self._evict_if_needed()

    def record_transfer(self, bytes_compressed: int, bytes_h2d: int) -> None:
        self.metrics.bytes_compressed += int(bytes_compressed)
        self.metrics.bytes_h2d += int(bytes_h2d)

    def _evict_from_probation(self) -> bool:
        if not self._probation:
            return False
        victim_key = min(self._probation, key=lambda k: (self._probation[k].stability))
        entry = self._probation.pop(victim_key)
        self.current_bytes -= entry.size_bytes
        self.metrics.evictions += 1
        return True

    def _evict_from_protected(self) -> bool:
        if not self._protected:
            return False
        key, entry = self._protected.popitem(last=False)
        self.current_bytes -= entry.size_bytes
        self.metrics.evictions += 1
        return True

    def _evict_if_needed(self) -> None:
        while self.current_bytes > self.capacity_bytes:
            if self._probation:
                if self._evict_from_probation():
                    continue
            if self._protected:
                if self._evict_from_protected():
                    continue
            break

    def stats(self) -> Dict[str, float]:
        return {
            "capacity_bytes": float(self.capacity_bytes),
            "current_bytes": float(self.current_bytes),
            "hit_rate": self.metrics.hit_rate,
            "hits": float(self.metrics.hits),
            "misses": float(self.metrics.misses),
            "evictions": float(self.metrics.evictions),
            "bytes_h2d": float(self.metrics.bytes_h2d),
            "bytes_compressed": float(self.metrics.bytes_compressed),
        }

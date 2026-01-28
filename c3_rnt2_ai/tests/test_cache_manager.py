from __future__ import annotations

from c3rnt2.runtime.cache_manager import CacheManager


def test_cache_manager_basic():
    cache = CacheManager(capacity_bytes=10)
    assert cache.get("a") is None
    cache.put("a", 1, size_bytes=5)
    cache.put("b", 2, size_bytes=5)
    assert cache.get("a") == 1
    cache.put("c", 3, size_bytes=5)
    stats = cache.stats()
    assert stats["capacity_bytes"] == 10.0

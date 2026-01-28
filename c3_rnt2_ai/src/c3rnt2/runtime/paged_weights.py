from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .cache_manager import CacheManager
from .gpu_decompress import decompress_to_tensor
from .prefetch import Prefetcher


@dataclass
class PagedWeightsStats:
    page_faults: int = 0
    bytes_transferred: int = 0


class PagedWeights:
    """Tile-based weight manager with CPU storage and GPU cache (MVP)."""

    def __init__(self, tile_store: Dict[int, np.ndarray], cache: CacheManager, device: str = "cpu"):
        self.tile_store = tile_store
        self.cache = cache
        self.device = device
        self.stats = PagedWeightsStats()
        self.prefetcher = Prefetcher(self._load_tile, depth=2)

    def _load_tile(self, tile_id: int):
        tile = self.tile_store[tile_id]
        self.stats.bytes_transferred += tile.nbytes
        tensor = decompress_to_tensor(tile, device=self.device)
        self.cache.put((tile_id,), tensor, tile.nbytes)
        return tensor

    def request_tiles(self, tile_ids: Iterable[int]) -> List[object]:
        result = []
        for tile_id in tile_ids:
            cached = self.cache.get((tile_id,))
            if cached is not None:
                result.append(cached)
            else:
                self.stats.page_faults += 1
                result.append(self._load_tile(tile_id))
        return result

    def prefetch(self, tile_ids: Iterable[int]) -> None:
        self.prefetcher.schedule(tile_ids)
        self.prefetcher.run()

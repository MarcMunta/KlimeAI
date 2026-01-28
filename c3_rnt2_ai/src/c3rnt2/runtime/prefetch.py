from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Iterable, List


class Prefetcher:
    """Simple CPU->GPU prefetch scheduler (sync MVP)."""

    def __init__(self, loader: Callable[[int], object], depth: int = 2):
        self.loader = loader
        self.depth = depth
        self.queue: Deque[int] = deque()

    def schedule(self, tile_ids: Iterable[int]) -> None:
        for tile_id in tile_ids:
            if len(self.queue) >= self.depth:
                break
            self.queue.append(tile_id)

    def run(self) -> List[object]:
        loaded = []
        while self.queue:
            tile_id = self.queue.popleft()
            loaded.append(self.loader(tile_id))
        return loaded

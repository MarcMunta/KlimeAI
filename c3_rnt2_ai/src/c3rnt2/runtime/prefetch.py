from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Iterable, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class Prefetcher:
    """Simple CPU->GPU prefetch scheduler (sync MVP)."""

    def __init__(self, loader: Callable[[int], object], depth: int = 2, device: str = "cpu"):
        self.loader = loader
        self.depth = depth
        self.queue: Deque[int] = deque()
        self.device = device
        self.stream = None
        if torch is not None and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()

    def schedule(self, tile_ids: Iterable[int]) -> None:
        for tile_id in tile_ids:
            if len(self.queue) >= self.depth:
                break
            self.queue.append(tile_id)

    def run(self) -> List[object]:
        loaded = []
        while self.queue:
            tile_id = self.queue.popleft()
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    loaded.append(self.loader(tile_id))
            else:
                loaded.append(self.loader(tile_id))
        return loaded

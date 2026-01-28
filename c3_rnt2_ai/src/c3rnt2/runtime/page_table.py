from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PageEntry:
    tile_id: int
    location: str  # "gpu" or "cpu"


class PageTable:
    def __init__(self):
        self._entries: Dict[int, PageEntry] = {}

    def set(self, tile_id: int, location: str) -> None:
        self._entries[tile_id] = PageEntry(tile_id=tile_id, location=location)

    def get(self, tile_id: int) -> Optional[PageEntry]:
        return self._entries.get(tile_id)

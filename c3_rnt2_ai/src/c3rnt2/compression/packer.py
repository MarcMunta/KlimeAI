from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .entropy_coder import compress

MAGIC = b"C3PK"
VERSION = 1


@dataclass
class PackedTile:
    tile_id: int
    payload: bytes


def pack_tiles(tiles: Iterable[np.ndarray], tile_size: int, codec: str = "zstd") -> bytes:
    tiles_list: List[np.ndarray] = list(tiles)
    header = struct.pack("<4sBII", MAGIC, VERSION, tile_size, len(tiles_list))
    codec_bytes = codec.encode("ascii")
    header += struct.pack("<I", len(codec_bytes)) + codec_bytes
    body = bytearray()
    for tile_id, tile in enumerate(tiles_list):
        raw = tile.astype(np.float16).tobytes()
        comp = compress(raw, codec=codec).payload
        body.extend(struct.pack("<II", tile_id, len(comp)))
        body.extend(comp)
    return header + body

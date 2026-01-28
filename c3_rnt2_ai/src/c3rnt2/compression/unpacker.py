from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List

import numpy as np

from .entropy_coder import decompress

MAGIC = b"C3PK"


@dataclass
class UnpackedTile:
    tile_id: int
    tile: np.ndarray


def unpack_tiles(payload: bytes) -> List[UnpackedTile]:
    offset = 0
    magic, version, tile_size, num_tiles = struct.unpack_from("<4sBII", payload, offset)
    offset += struct.calcsize("<4sBII")
    if magic != MAGIC:
        raise ValueError("Invalid pack magic")
    (codec_len,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    codec = payload[offset : offset + codec_len].decode("ascii")
    offset += codec_len
    tiles: List[UnpackedTile] = []
    for _ in range(num_tiles):
        tile_id, comp_len = struct.unpack_from("<II", payload, offset)
        offset += 8
        comp = payload[offset : offset + comp_len]
        offset += comp_len
        raw = decompress(comp, codec=codec)
        tile = np.frombuffer(raw, dtype=np.float16).reshape(tile_size, tile_size)
        tiles.append(UnpackedTile(tile_id=tile_id, tile=tile))
    return tiles

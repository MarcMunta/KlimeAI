from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .rnt2_model import RNT2Model


@dataclass
class RNT2Stream:
    block_size: int
    codes: List[int]
    escapes: Dict[int, Tuple[int, bytes]]
    blocks: List[bytes]
    lengths: List[int]


def _split_blocks(data: bytes, block_size: int) -> List[bytes]:
    blocks = []
    for i in range(0, len(data), block_size):
        block = data[i : i + block_size]
        blocks.append(block)
    if not blocks:
        blocks = [b""]
    return blocks


def encode_text(text: str, model: RNT2Model) -> RNT2Stream:
    data = text.encode("utf-8")
    block_size = model.codebook.block_size
    blocks = _split_blocks(data, block_size)
    codes: List[int] = []
    escapes: Dict[int, Tuple[int, bytes]] = {}
    lengths: List[int] = []

    for idx, block in enumerate(blocks):
        orig_len = len(block)
        lengths.append(orig_len)
        padded = block.ljust(block_size, b"\x00")
        code = model.codebook.find(padded)
        if code is None:
            code = 0
            escapes[idx] = (orig_len, block)
        codes.append(code)

    return RNT2Stream(block_size=block_size, codes=codes, escapes=escapes, blocks=blocks, lengths=lengths)

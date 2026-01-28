from __future__ import annotations

from .rnt2_encode import RNT2Stream
from .rnt2_model import RNT2Model


def decode_stream(stream: RNT2Stream, model: RNT2Model) -> str:
    blocks = []
    for idx, code in enumerate(stream.codes):
        orig_len = stream.lengths[idx] if idx < len(stream.lengths) else stream.block_size
        if idx in stream.escapes:
            _, raw = stream.escapes[idx]
            blocks.append(raw[:orig_len])
        else:
            block = model.codebook.lookup(code)
            blocks.append(block[:orig_len])
    data = b"".join(blocks)
    return data.decode("utf-8", errors="strict")

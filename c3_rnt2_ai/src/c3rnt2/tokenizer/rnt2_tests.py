from __future__ import annotations

from .rnt2_model import RNT2Model, RNT2Codebook
from .rnt2_encode import encode_text
from .rnt2_decode import decode_stream


def roundtrip(text: str, block_size: int = 64) -> bool:
    model = RNT2Model(RNT2Codebook.from_builtin(block_size=block_size))
    stream = encode_text(text, model)
    out = decode_stream(stream, model)
    return out == text

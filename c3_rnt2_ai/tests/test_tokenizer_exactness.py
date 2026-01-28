from __future__ import annotations

from c3rnt2.tokenizer.rnt2_model import RNT2Codebook, RNT2Model
from c3rnt2.tokenizer.rnt2_encode import encode_text
from c3rnt2.tokenizer.rnt2_decode import decode_stream


def test_rnt2_exactness():
    model = RNT2Model(RNT2Codebook.from_builtin(block_size=64))
    samples = [
        "hola mundo",
        "emoji ????",
        "json {\"a\": 1, \"b\": [2,3]}",
        "code: for i in range(3): print(i)",
        "unicode: ???????",
    ]
    for text in samples:
        stream = encode_text(text, model)
        out = decode_stream(stream, model)
        assert out == text


def test_rnt2_metrics():
    model = RNT2Model(RNT2Codebook.from_builtin(block_size=64))
    text = "hello world" * 10
    stream = encode_text(text, model)
    byte_len = len(text.encode("utf-8"))
    ratio = len(stream.codes) / max(1, byte_len)
    escapes_pct = len(stream.escapes) / max(1, len(stream.blocks))
    assert ratio >= 0
    assert 0 <= escapes_pct <= 1

from __future__ import annotations

from c3rnt2.tokenizer.rnt2_model import RNT2Codebook, RNT2Model
from c3rnt2.tokenizer.rnt2_encode import encode_text
from c3rnt2.tokenizer.rnt2_decode import decode_stream
from c3rnt2.tokenizer.vortex_tok import (
    VortexTokModel,
    VortexMacroCodebook,
    encode as vortex_encode,
    decode as vortex_decode,
    encode_to_ids,
    decode_from_ids,
)


def test_rnt2_exactness():
    model = RNT2Model(RNT2Codebook.from_builtin(block_size=64))
    samples = [
        "hola mundo",
        "emoji \\U0001f600\\U0001f680",
        "json {\"a\": 1, \"b\": [2,3]}",
        "code: for i in range(3): print(i)",
        "unicode: \\u3053\\u3093\\u306b\\u3061\\u306f\\u4e16\\u754c",
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


def test_vortex_exactness_large(tmp_path):
    patch = RNT2Codebook.from_builtin(block_size=64)
    macro = VortexMacroCodebook(sequences=[[0, 1]])
    model = VortexTokModel(patch_codebook=patch, macro_codebook=macro)
    samples = [
        "def add(a, b):\\n    return a + b\\n",
        "{\"user\": \"alice\", \"items\": [1, 2, 3], \"ok\": true}",
        "long: " + ("abc123 " * 200),
    ]
    for text in samples:
        stream = vortex_encode(text, model)
        out = vortex_decode(stream, model)
        assert out == text

    file_path = tmp_path / "sample.txt"
    file_content = "line1\\nline2\\nline3\\n"
    file_path.write_text(file_content, encoding="utf-8")
    stream = vortex_encode(file_path.read_text(encoding="utf-8"), model)
    assert vortex_decode(stream, model) == file_content


def test_tokenizer_roundtrip_ids():
    patch = RNT2Codebook.from_builtin(block_size=64)
    macro = VortexMacroCodebook(sequences=[])
    model = VortexTokModel(patch_codebook=patch, macro_codebook=macro)
    text = "roundtrip: {\"a\": 1, \"b\": [2,3]}"
    ids, total_len = encode_to_ids(text, model)
    out = decode_from_ids(ids, model, total_len=total_len)
    assert out == text

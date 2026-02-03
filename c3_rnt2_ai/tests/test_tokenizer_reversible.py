import random

from c3rnt2.tokenizer.reversible import ReversibleTokenizer
from c3rnt2.tokenizer.vortex_tok import load_or_create


def _get_tokenizer(tmp_path):
    model_path = tmp_path / "vortex_tok.pt"
    model = load_or_create(model_path, block_size=16)
    return ReversibleTokenizer(model)


def test_roundtrip_unicode(tmp_path) -> None:
    tok = _get_tokenizer(tmp_path)
    text = "Hola ñ é ö 😀 — 東京"
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert out == text


def test_roundtrip_code_symbols(tmp_path) -> None:
    tok = _get_tokenizer(tmp_path)
    text = "def f(x):\n    return x * (x + 1) // 2  # ñ 😀"
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert out == text


def test_roundtrip_fuzz_small(tmp_path) -> None:
    tok = _get_tokenizer(tmp_path)
    random.seed(1337)
    alphabet = [chr(i) for i in range(32, 128)] + ["ñ", "é", "ö", "😀", "—", "東", "京"]
    for _ in range(50):
        text = "".join(random.choice(alphabet) for _ in range(random.randint(0, 64)))
        ids = tok.encode(text)
        out = tok.decode(ids)
        assert out == text

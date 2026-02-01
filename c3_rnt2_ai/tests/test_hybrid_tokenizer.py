from __future__ import annotations

from c3rnt2.tokenizer.hybrid import HybridTokenizer, TokenDebug


class DummyBackend:
    vocab_size = 10

    def encode(self, text: str):
        return [1, 2, 3]

    def decode(self, ids):
        return "x" * len(ids)

    def tokenize_debug(self, text: str):
        return [TokenDebug(token=0, text="", start=0, end=0, source="dummy")]


class AsciiBackend:
    vocab_size = 300

    def encode(self, text: str):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def tokenize_debug(self, text: str):
        out = []
        for i, ch in enumerate(text):
            out.append(TokenDebug(token=ord(ch), text=ch, start=i, end=i + 1, source="ascii"))
        return out


def test_hybrid_fallback_roundtrip():
    base = DummyBackend()
    tokenizer = HybridTokenizer(base)
    text = "Hola 🌍 Привет 你好"
    ids = tokenizer.encode(text)
    assert ids[0] == base.vocab_size
    assert tokenizer.decode(ids) == text
    debug = tokenizer.tokenize_debug(text)
    assert len(debug) == len(text.encode("utf-8"))


def test_hybrid_base_roundtrip():
    base = AsciiBackend()
    tokenizer = HybridTokenizer(base)
    text = "hello"
    ids = tokenizer.encode(text)
    assert ids[0] != base.vocab_size
    assert tokenizer.decode(ids) == text

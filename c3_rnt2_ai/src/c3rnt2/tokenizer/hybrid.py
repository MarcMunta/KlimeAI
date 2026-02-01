from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class TokenDebug:
    token: int
    text: str
    start: int
    end: int
    source: str


class TokenizerBackend(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> List[int]: ...

    def decode(self, ids: List[int]) -> str: ...

    def tokenize_debug(self, text: str) -> List[TokenDebug]: ...


class HFTokenizerBackend:
    def __init__(self, tokenizer: object):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        try:
            return int(self.tokenizer.vocab_size)  # type: ignore[attr-defined]
        except Exception:
            try:
                return len(self.tokenizer.get_vocab())  # type: ignore[attr-defined]
            except Exception:
                return 0

    def encode(self, text: str) -> List[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        return list(encoded["input_ids"])

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def tokenize_debug(self, text: str) -> List[TokenDebug]:
        try:
            encoded = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            ids = encoded.get("input_ids", [])
            offsets = encoded.get("offset_mapping", [])
            tokens = []
            for tok, (start, end) in zip(ids, offsets):
                tokens.append(TokenDebug(token=int(tok), text=text[start:end], start=int(start), end=int(end), source="hf"))
            return tokens
        except Exception:
            ids = self.encode(text)
            return [TokenDebug(token=int(tok), text="", start=0, end=0, source="hf") for tok in ids]


class HybridTokenizer:
    def __init__(self, base: TokenizerBackend):
        self.base = base
        base_vocab = max(0, int(base.vocab_size))
        # Reserve a sentinel plus 256 byte ids beyond base vocab.
        self._byte_sentinel = base_vocab
        self._byte_start = base_vocab + 1
        self._byte_end = self._byte_start + 256

    @property
    def vocab_size(self) -> int:
        return self._byte_end

    def encode(self, text: str) -> List[int]:
        ids = self.base.encode(text)
        try:
            if self.base.decode(ids) == text:
                return ids
        except Exception:
            pass
        data = text.encode("utf-8")
        return [self._byte_sentinel] + [self._byte_start + b for b in data]

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        if ids[0] == self._byte_sentinel:
            data = bytes([tok - self._byte_start for tok in ids[1:] if self._byte_start <= tok < self._byte_end])
            return data.decode("utf-8", errors="strict")
        return self.base.decode(ids)

    def tokenize_debug(self, text: str) -> List[TokenDebug]:
        ids = self.encode(text)
        if ids and ids[0] == self._byte_sentinel:
            data = text.encode("utf-8")
            out: List[TokenDebug] = []
            offset = 0
            for b in data:
                out.append(
                    TokenDebug(
                        token=self._byte_start + b,
                        text=bytes([b]).decode("utf-8", errors="ignore"),
                        start=offset,
                        end=offset + 1,
                        source="byte",
                    )
                )
                offset += 1
            return out
        return self.base.tokenize_debug(text)

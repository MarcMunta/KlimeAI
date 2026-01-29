from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .rnt2_model import RNT2Codebook, RNT2Model

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class VortexToken:
    kind: str  # PATCH, MACRO, SUBPATCH, ESC
    value: object


@dataclass
class VortexStream:
    tokens: List[VortexToken]
    total_len: int
    block_size: int


@dataclass
class VortexMacroCodebook:
    sequences: List[List[int]]
    _trie: dict = field(default_factory=dict, init=False, repr=False)
    _trie_ready: bool = field(default=False, init=False, repr=False)

    def lookup(self, idx: int) -> List[int]:
        return self.sequences[idx]

    def _build_trie(self) -> None:
        trie: dict = {}
        for idx, seq in enumerate(self.sequences):
            if len(seq) < 2:
                continue
            node = trie
            for tok in seq:
                node = node.setdefault(int(tok), {})
            node["_id"] = idx
        self._trie = trie
        self._trie_ready = True

    def _ensure_trie(self) -> None:
        if not self._trie_ready:
            self._build_trie()

    def match_longest(self, patch_ids: List[int], start: int) -> Optional[Tuple[int, int]]:
        if not self.sequences:
            return None
        self._ensure_trie()
        node = self._trie
        best: Optional[Tuple[int, int]] = None
        i = start
        while i < len(patch_ids):
            tok = patch_ids[i]
            if tok not in node:
                break
            node = node[tok]
            i += 1
            macro_id = node.get("_id")
            if isinstance(macro_id, int):
                best = (macro_id, i - start)
        return best

    @property
    def size(self) -> int:
        return len(self.sequences)


@dataclass
class VortexTokModel:
    patch_codebook: RNT2Codebook
    macro_codebook: VortexMacroCodebook
    sub_codebook: Optional[RNT2Codebook] = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "block_size": self.patch_codebook.block_size,
            "patch_entries": self.patch_codebook.entries,
            "macro_sequences": self.macro_codebook.sequences,
            "sub_block_size": self.sub_codebook.block_size if self.sub_codebook else None,
            "sub_entries": self.sub_codebook.entries if self.sub_codebook else None,
        }
        if torch:
            torch.save(payload, path)
        else:
            import pickle

            with path.open("wb") as f:
                pickle.dump(payload, f)

    @staticmethod
    def load(path: str | Path) -> "VortexTokModel":
        path = Path(path)
        if torch:
            payload = torch.load(path, map_location="cpu")
        else:
            import pickle

            with path.open("rb") as f:
                payload = pickle.load(f)
        patch = RNT2Codebook(block_size=payload["block_size"], entries=list(payload["patch_entries"]))
        macro = VortexMacroCodebook(sequences=list(payload.get("macro_sequences", [])))
        sub_entries = payload.get("sub_entries")
        sub_block_size = payload.get("sub_block_size")
        sub = None
        if sub_entries and sub_block_size:
            sub = RNT2Codebook(block_size=int(sub_block_size), entries=list(sub_entries))
        return VortexTokModel(patch_codebook=patch, macro_codebook=macro, sub_codebook=sub)


def _split_blocks(data: bytes, block_size: int) -> List[bytes]:
    return [data[i : i + block_size] for i in range(0, len(data), block_size)] or [b""]


def encode(text: str, model: VortexTokModel) -> VortexStream:
    data = text.encode("utf-8")
    blocks = _split_blocks(data, model.patch_codebook.block_size)
    patch_ids: List[int] = []
    for block in blocks:
        padded = block.ljust(model.patch_codebook.block_size, b"\x00")
        code = model.patch_codebook.find(padded)
        patch_ids.append(-1 if code is None else code)

    tokens: List[VortexToken] = []
    macro_codebook = model.macro_codebook

    i = 0
    while i < len(blocks):
        if patch_ids[i] >= 0:
            matched = macro_codebook.match_longest(patch_ids, i)
            if matched:
                macro_id, size = matched
                tokens.append(VortexToken(kind="MACRO", value=macro_id))
                i += size
                continue
            tokens.append(VortexToken(kind="PATCH", value=patch_ids[i]))
            i += 1
        else:
            # fallback: try sub-blocks if available
            sub = model.sub_codebook
            if sub is not None and sub.block_size > 0:
                sub_tokens: List[VortexToken] = []
                ok = True
                block = blocks[i]
                for j in range(0, len(block), sub.block_size):
                    chunk = block[j : j + sub.block_size]
                    padded = chunk.ljust(sub.block_size, b"\x00")
                    code = sub.find(padded)
                    if code is None:
                        ok = False
                        break
                    sub_tokens.append(VortexToken(kind="SUBPATCH", value=code))
                if ok and sub_tokens:
                    tokens.extend(sub_tokens)
                    i += 1
                    continue
            tokens.append(VortexToken(kind="ESC", value=blocks[i]))
            i += 1

    return VortexStream(tokens=tokens, total_len=len(data), block_size=model.patch_codebook.block_size)


def decode(stream: VortexStream, model: VortexTokModel) -> str:
    out = bytearray()
    for token in stream.tokens:
        if token.kind == "PATCH":
            out.extend(model.patch_codebook.lookup(int(token.value)))
        elif token.kind == "MACRO":
            for pid in model.macro_codebook.lookup(int(token.value)):
                out.extend(model.patch_codebook.lookup(int(pid)))
        elif token.kind == "SUBPATCH":
            if model.sub_codebook is None:
                raise ValueError("SUBPATCH token without sub_codebook")
            out.extend(model.sub_codebook.lookup(int(token.value)))
        elif token.kind == "ESC":
            out.extend(bytes(token.value))
        else:
            raise ValueError(f"Unknown token kind: {token.kind}")
    data = bytes(out[: stream.total_len])
    return data.decode("utf-8", errors="strict")


def metrics(stream: VortexStream) -> dict:
    tokens = len(stream.tokens)
    esc = sum(1 for t in stream.tokens if t.kind == "ESC")
    macro = sum(1 for t in stream.tokens if t.kind == "MACRO")
    ratio = stream.total_len / max(1, tokens)
    return {
        "tokens": tokens,
        "bytes": stream.total_len,
        "bytes_per_token": round(ratio, 4),
        "escapes_pct": round((esc / max(1, tokens)) * 100.0, 2),
        "macro_hit_rate": round((macro / max(1, tokens)) * 100.0, 2),
    }


def encode_to_ids(text: str, model: VortexTokModel) -> Tuple[List[int], int]:
    stream = encode(text, model)
    ids: List[int] = []
    patch_size = model.patch_codebook.size
    macro_size = model.macro_codebook.size
    sub_size = model.sub_codebook.size if model.sub_codebook else 0
    for token in stream.tokens:
        if token.kind == "PATCH":
            ids.append(int(token.value))
        elif token.kind == "MACRO":
            ids.append(patch_size + int(token.value))
        elif token.kind == "SUBPATCH":
            ids.append(patch_size + macro_size + int(token.value))
        elif token.kind == "ESC":
            for b in bytes(token.value):
                ids.append(patch_size + macro_size + sub_size + b)
    return ids, stream.total_len


def decode_from_ids(ids: List[int], model: VortexTokModel, total_len: Optional[int] = None) -> str:
    patch_size = model.patch_codebook.size
    macro_size = model.macro_codebook.size
    sub_size = model.sub_codebook.size if model.sub_codebook else 0
    out = bytearray()
    for token_id in ids:
        if token_id < patch_size:
            out.extend(model.patch_codebook.lookup(token_id))
        elif token_id < patch_size + macro_size:
            seq = model.macro_codebook.lookup(token_id - patch_size)
            for pid in seq:
                out.extend(model.patch_codebook.lookup(pid))
        elif token_id < patch_size + macro_size + sub_size:
            if model.sub_codebook is None:
                raise ValueError("SUBPATCH id without sub_codebook")
            out.extend(model.sub_codebook.lookup(token_id - patch_size - macro_size))
        else:
            out.append(token_id - patch_size - macro_size - sub_size)
    if total_len is not None:
        out = out[:total_len]
    else:
        out = out.rstrip(b"\x00")
    return bytes(out).decode("utf-8", errors="replace")


def load_or_create(model_path: Path, block_size: int) -> VortexTokModel:
    if model_path.exists():
        return VortexTokModel.load(model_path)
    rnt2 = RNT2Model(codebook=RNT2Codebook.from_builtin(block_size=block_size))
    macro = VortexMacroCodebook(sequences=[])
    return VortexTokModel(patch_codebook=rnt2.codebook, macro_codebook=macro, sub_codebook=None)

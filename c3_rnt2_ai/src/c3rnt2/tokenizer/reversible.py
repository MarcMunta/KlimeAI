from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List

from .vortex_tok import VortexTokModel, encode_to_ids, decode_from_ids


@dataclass
class ReversibleTokenizer:
    model: VortexTokModel

    def _byte_offset(self) -> int:
        return self.model.patch_codebook.size + self.model.macro_codebook.size + self.model.sub_size_total

    def encode(self, text: str) -> List[int]:
        ids, total_len = encode_to_ids(text, self.model)
        header = struct.pack('<I', int(total_len))
        offset = self._byte_offset()
        header_ids = [offset + b for b in header]
        return header_ids + ids

    def decode(self, ids: List[int]) -> str:
        if len(ids) < 4:
            return ""
        offset = self._byte_offset()
        header = bytes([(int(tok) - offset) & 0xFF for tok in ids[:4]])
        total_len = struct.unpack('<I', header)[0]
        payload = ids[4:]
        return decode_from_ids(payload, self.model, total_len=total_len)

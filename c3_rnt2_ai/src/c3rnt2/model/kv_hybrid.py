from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class QuantizedKV:
    values: np.ndarray
    scale: float


@dataclass
class KVHybridCache:
    window_size: int
    kv_quant_bits: int
    latent_slots: int
    exact_k: List[object] = field(default_factory=list)
    exact_v: List[object] = field(default_factory=list)
    quantized: List[QuantizedKV] = field(default_factory=list)
    latent: List[object] = field(default_factory=list)
    _latent_idx: int = 0

    def add(self, k, v):
        self.exact_k.append(k)
        self.exact_v.append(v)
        if len(self.exact_k) > self.window_size:
            old_k = self.exact_k.pop(0)
            old_v = self.exact_v.pop(0)
            self._quantize_and_store(old_k, old_v)

    def _quantize_and_store(self, k, v):
        if torch is None:
            return
        vec = torch.cat([k.flatten(), v.flatten()]).detach().cpu().numpy()
        scale = float(np.max(np.abs(vec)) / 127.0) if vec.size else 1.0
        if scale == 0:
            scale = 1.0
        q = np.clip(vec / scale, -127, 127).astype(np.int8)
        self.quantized.append(QuantizedKV(values=q, scale=scale))
        self._update_latent(vec)

    def _update_latent(self, vec: np.ndarray):
        if torch is None:
            return
        tensor = torch.tensor(vec, dtype=torch.float32)
        if len(self.latent) < self.latent_slots:
            self.latent.append(tensor)
        else:
            self.latent[self._latent_idx] = tensor
            self._latent_idx = (self._latent_idx + 1) % self.latent_slots

    def get_context(self) -> Tuple[List[object], List[object], List[object]]:
        return self.exact_k, self.exact_v, self.latent

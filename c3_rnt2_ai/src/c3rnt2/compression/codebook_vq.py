from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class VQResult:
    codes: np.ndarray
    codebook: np.ndarray


def vq_quantize(vectors: np.ndarray, codebook_size: int) -> VQResult:
    """Naive VQ: pick first K vectors as codebook, assign by L2."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D [n, d]")
    n, _ = vectors.shape
    if n == 0:
        raise ValueError("empty vectors")
    k = min(codebook_size, n)
    codebook = vectors[:k].copy()
    # assign nearest
    dists = ((vectors[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    codes = dists.argmin(axis=1)
    return VQResult(codes=codes.astype(np.int32), codebook=codebook)


def vq_dequantize(codes: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    return codebook[codes]

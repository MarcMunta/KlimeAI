from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def decompress_to_tensor(tile: np.ndarray, device: str = "cpu"):
    if torch is None:
        raise RuntimeError("PyTorch not available")
    tensor = torch.from_numpy(tile.copy())
    return tensor.to(device)

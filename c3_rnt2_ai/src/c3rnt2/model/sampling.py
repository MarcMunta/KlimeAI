from __future__ import annotations

import torch


def greedy_sample(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())

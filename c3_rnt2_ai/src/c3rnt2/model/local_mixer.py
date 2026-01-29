from __future__ import annotations

import torch
from torch import nn


class LocalMixer(nn.Module):
    """Depthwise conv local mixer for syntax patterns."""

    def __init__(self, hidden_size: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        y = x.transpose(1, 2)
        y = self.conv(y)
        return y.transpose(1, 2)

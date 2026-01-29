from __future__ import annotations

import torch
from torch import nn


class SSMTrack(nn.Module):
    """Minimal recurrent SSM block with O(n) scan."""

    def __init__(self, hidden_size: int, state_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.in_proj = nn.Linear(hidden_size, state_size)
        self.state_proj = nn.Linear(state_size, state_size)
        self.out_proj = nn.Linear(state_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        state = torch.zeros(batch, self.state_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq):
            inp = self.in_proj(x[:, t])
            state = torch.tanh(inp + self.state_proj(state))
            outputs.append(self.out_proj(state))
        return torch.stack(outputs, dim=1)

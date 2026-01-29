from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, cast

import torch
from torch import nn

from ..device import autocast_context


@dataclass
class LavaStats:
    reads: int = 0
    writes: int = 0


class LAVAMemory(nn.Module):
    """Latent-Addressed Vector Attention with fixed slots."""

    def __init__(self, hidden_size: int, latent_slots: int = 128, top_k: int = 4, dtype: str | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_slots = latent_slots
        self.top_k = top_k
        self.dtype = dtype

        self.addresses: torch.Tensor
        self.contents: torch.Tensor
        self.age: torch.Tensor
        self.importance: torch.Tensor

        self.register_buffer("addresses", torch.randn(latent_slots, hidden_size) * 0.02)
        self.register_buffer("contents", torch.randn(latent_slots, hidden_size) * 0.02)
        self.addr_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.register_buffer("age", torch.zeros(latent_slots))
        self.register_buffer("importance", torch.zeros(latent_slots))
        self.stats = LavaStats()

    def reset_state(self) -> None:
        with torch.no_grad():
            self.addresses.copy_(torch.randn_like(self.addresses) * 0.02)
            self.contents.copy_(torch.randn_like(self.contents) * 0.02)
            self.age.zero_()
            self.importance.zero_()
            self.stats = LavaStats()

    def reset_stats(self) -> None:
        self.stats = LavaStats()

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "addresses": self.addresses.detach().cpu(),
            "contents": self.contents.detach().cpu(),
            "age": self.age.detach().cpu(),
            "importance": self.importance.detach().cpu(),
        }
        torch.save(payload, path)

    def load_state(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        payload = torch.load(path, map_location="cpu")
        payload_t = cast(Dict[str, torch.Tensor], payload)
        with torch.no_grad():
            self.addresses.copy_(payload_t["addresses"].to(self.addresses.device))
            self.contents.copy_(payload_t["contents"].to(self.contents.device))
            self.age.copy_(payload_t["age"].to(self.age.device))
            self.importance.copy_(payload_t["importance"].to(self.importance.device))
        return True

    def read(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.read_step(x)
        with autocast_context(enabled=x.is_cuda, dtype=self.dtype):
            q = self.addr_proj(x)
            q_norm = torch.nn.functional.normalize(q, dim=-1)
            k_norm = torch.nn.functional.normalize(self.addresses, dim=-1)
            scores = torch.matmul(q_norm, k_norm.t())
            top_k = min(self.top_k, self.latent_slots)
            vals, idx = torch.topk(scores, k=top_k, dim=-1)
            attn = torch.softmax(vals, dim=-1)
            selected = self.contents[idx]
            mem = (attn.unsqueeze(-1) * selected).sum(dim=-2)
            self.stats.reads += int(x.numel() // self.hidden_size)
            return self.read_proj(mem)

    def write(self, x: torch.Tensor) -> None:
        if x.dim() == 2:
            return self.write_step(x)
        with torch.no_grad():
            batch_mean = x.mean(dim=(0, 1))
            # choose slot with max age or min importance
            score = self.age - self.importance
            slot = int(torch.argmax(score).item())
            self.contents[slot].mul_(0.9).add_(0.1 * batch_mean)
            self.addresses[slot].mul_(0.95).add_(0.05 * batch_mean)
            self.importance[slot] = min(1.0, float(self.importance[slot] + 0.05))
            self.age.add_(1.0)
            self.age[slot] = 0.0
            self.stats.writes += 1

    def read_step(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H]
        with autocast_context(enabled=x.is_cuda, dtype=self.dtype):
            q = self.addr_proj(x)
            q_norm = torch.nn.functional.normalize(q, dim=-1)
            k_norm = torch.nn.functional.normalize(self.addresses, dim=-1)
            scores = torch.matmul(q_norm, k_norm.t())
            top_k = min(self.top_k, self.latent_slots)
            vals, idx = torch.topk(scores, k=top_k, dim=-1)
            attn = torch.softmax(vals, dim=-1)
            selected = self.contents[idx]
            mem = (attn.unsqueeze(-1) * selected).sum(dim=-2)
            self.stats.reads += int(x.numel() // self.hidden_size)
            return self.read_proj(mem)

    def write_step(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            batch_mean = x.mean(dim=0)
            score = self.age - self.importance
            slot = int(torch.argmax(score).item())
            self.contents[slot].mul_(0.9).add_(0.1 * batch_mean)
            self.addresses[slot].mul_(0.95).add_(0.05 * batch_mean)
            self.importance[slot] = min(1.0, float(self.importance[slot] + 0.05))
            self.age.add_(1.0)
            self.age[slot] = 0.0
            self.stats.writes += 1

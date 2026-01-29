from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 1.0


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base = base
        self.config = config
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.A = nn.Parameter(torch.zeros(config.rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, config.rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.A.t()) @ self.B.t()
        return base_out + lora_out * (self.config.alpha / self.config.rank)


def inject_lora(model: nn.Module, config: LoRAConfig) -> Dict[str, LoRALinear]:
    wrapped: Dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent = model
            if "." in name:
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                leaf_name = parts[-1]
            else:
                leaf_name = name
            wrapped_module = LoRALinear(module, config)
            setattr(parent, leaf_name, wrapped_module)
            wrapped[name] = wrapped_module
    return wrapped

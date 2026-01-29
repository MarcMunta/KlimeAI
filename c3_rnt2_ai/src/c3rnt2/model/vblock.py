from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .local_mixer import LocalMixer, LocalMixerState
from .ssm_track import SSMTrack, SSMState
from .lava_memory import LAVAMemory


@dataclass
class VBlockConfig:
    hidden_size: int
    window_size: int
    latent_slots: int
    lava_top_k: int
    local_mixer_kernel: int
    ssm_state_size: int
    gated_mlp_ratio: int
    dtype: str | None = None


@dataclass
class VBlockState:
    local: LocalMixerState
    ssm: SSMState


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, ratio: int):
        super().__init__()
        inner = hidden_size * ratio
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)
        self.gate = nn.Linear(hidden_size, inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        ff = torch.nn.functional.gelu(self.fc1(x))
        return self.fc2(ff * gate)


class VBlock(nn.Module):
    """LocalMixer -> SSM -> LAVA -> GatedMLP with residuals."""

    def __init__(self, config: VBlockConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        self.norm4 = nn.LayerNorm(config.hidden_size)

        self.local = LocalMixer(config.hidden_size, kernel_size=config.local_mixer_kernel)
        self.ssm = SSMTrack(config.hidden_size, state_size=config.ssm_state_size)
        self.lava = LAVAMemory(
            hidden_size=config.hidden_size,
            latent_slots=config.latent_slots,
            top_k=config.lava_top_k,
            dtype=config.dtype,
        )
        self.mlp = GatedMLP(config.hidden_size, ratio=config.gated_mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(self.norm1(x))
        x = x + self.ssm(self.norm2(x))
        mem = self.lava.read(self.norm3(x))
        x = x + mem
        self.lava.write(x)
        x = x + self.mlp(self.norm4(x))
        return x

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> VBlockState:
        return VBlockState(
            local=self.local.init_state(batch, device, dtype),
            ssm=self.ssm.init_state(batch, device, dtype),
        )

    def step(self, x: torch.Tensor, state: VBlockState, write_memory: bool = True) -> tuple[torch.Tensor, VBlockState]:
        # x: [B, H]
        local_out, local_state = self.local.step(self.norm1(x), state.local)
        x = x + local_out
        ssm_out, ssm_state = self.ssm.step(self.norm2(x), state.ssm)
        x = x + ssm_out
        mem = self.lava.read_step(self.norm3(x))
        x = x + mem
        if write_memory:
            self.lava.write_step(x)
        x = x + self.mlp(self.norm4(x))
        return x, VBlockState(local=local_state, ssm=ssm_state)

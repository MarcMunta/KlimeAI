from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

from ..device import detect_device
from ..tokenizer.vortex_tok import VortexTokModel, load_or_create, encode_to_ids, decode_from_ids
from .vblock import VBlock, VBlockConfig
from .bad_decode import bad_decode
from .kv_hybrid import KVHybridCache


@dataclass
class VortexXConfig:
    hidden_size: int
    layers: int
    heads: int
    vocab_size: int
    window_size: int
    latent_slots: int
    lava_top_k: int
    local_mixer_kernel: int
    ssm_state_size: int
    gated_mlp_ratio: int
    draft_layers: int
    dtype: str | None = None


class CoreTransformer(nn.Module):
    """VORTEX-X core using V-Blocks and LAVA memory."""

    def __init__(self, config: VortexXConfig, tokenizer: VortexTokModel):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.byte_token_start = tokenizer.patch_codebook.size + tokenizer.macro_codebook.size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                VBlock(
                    VBlockConfig(
                        hidden_size=config.hidden_size,
                        window_size=config.window_size,
                        latent_slots=config.latent_slots,
                        lava_top_k=config.lava_top_k,
                        local_mixer_kernel=config.local_mixer_kernel,
                        ssm_state_size=config.ssm_state_size,
                        gated_mlp_ratio=config.gated_mlp_ratio,
                        dtype=config.dtype,
                    )
                )
                for _ in range(config.layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.kv_cache = KVHybridCache(window_size=config.window_size, kv_quant_bits=8, latent_slots=32)

    @staticmethod
    def from_settings(settings: dict) -> "CoreTransformer":
        core = settings.get("core", {})
        tok_cfg = settings.get("tokenizer", {})
        vx_cfg = settings.get("vortex_model", {})
        bad_cfg = settings.get("bad", {})
        device_info = detect_device()

        model_path = Path(tok_cfg.get("vortex_model_path", "data/runs/vortex_tok.pt"))
        block_size = int(tok_cfg.get("block_size", 64))
        tokenizer = load_or_create(model_path, block_size)

        patch_size = tokenizer.patch_codebook.size
        macro_size = tokenizer.macro_codebook.size
        vocab_size = max(int(core.get("vocab_size", 1024)), patch_size + macro_size + 256)
        layers = int(core.get("layers", 4))
        draft_layers = max(1, layers // 2)

        cfg = VortexXConfig(
            hidden_size=int(core.get("hidden_size", 256)),
            layers=layers,
            heads=int(core.get("heads", 4)),
            vocab_size=vocab_size,
            window_size=int(vx_cfg.get("window_size", 128)),
            latent_slots=int(vx_cfg.get("latent_slots", 64)),
            lava_top_k=int(vx_cfg.get("lava_top_k", 4)),
            local_mixer_kernel=int(vx_cfg.get("local_mixer_kernel", 5)),
            ssm_state_size=int(vx_cfg.get("ssm_state_size", 128)),
            gated_mlp_ratio=int(vx_cfg.get("gated_mlp_ratio", 4)),
            draft_layers=int(vx_cfg.get("draft_layers", draft_layers)),
            dtype=device_info.dtype,
        )
        model = CoreTransformer(cfg, tokenizer=tokenizer)
        model.bad_block_size = int(bad_cfg.get("block_size", 8))
        model.bad_entropy = float(bad_cfg.get("entropy_threshold", 3.5))
        return model

    def forward(self, input_ids: torch.Tensor, num_layers: int | None = None) -> torch.Tensor:
        x = self.embed(input_ids)
        depth = num_layers or self.config.layers
        for block in self.blocks[:depth]:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

    def full_next_logits(self, ids: List[int]) -> torch.Tensor:
        input_ids = torch.tensor([ids], dtype=torch.long)
        logits = self.forward(input_ids)
        return logits[:, -1, :]

    def draft_next_tokens(self, ids: List[int], count: int) -> List[int]:
        working = list(ids)
        tokens = []
        for _ in range(count):
            input_ids = torch.tensor([working], dtype=torch.long)
            logits = self.forward(input_ids, num_layers=self.config.draft_layers)[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            working.append(next_id)
            tokens.append(next_id)
        return tokens

    def encode_prompt(self, prompt: str) -> Tuple[List[int], int]:
        return encode_to_ids(prompt, self.tokenizer)

    def decode_ids(self, ids: List[int], total_len: int | None = None) -> str:
        return decode_from_ids(ids, self.tokenizer, total_len=total_len)

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        bad_cfg = {
            "block_size": getattr(self, "bad_block_size", 8),
            "entropy_threshold": getattr(self, "bad_entropy", 3.5),
        }
        text, _stats = bad_decode(
            self,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            block_size=bad_cfg["block_size"],
            entropy_threshold=bad_cfg["entropy_threshold"],
        )
        return text

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from .kv_hybrid import KVHybridCache


@dataclass
class CoreConfig:
    hidden_size: int
    layers: int
    heads: int
    vocab_size: int


class SimpleByteTokenizer:
    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        return bytes([t % 256 for t in tokens]).decode("utf-8", errors="ignore")


class CoreTransformer(nn.Module):
    """Minimal transformer for dev_small. Uses byte-level tokens for demo."""

    def __init__(self, config: CoreConfig):
        super().__init__()
        self.config = config
        self.tokenizer = SimpleByteTokenizer()
        vocab = max(256, config.vocab_size)
        self.embed = nn.Embedding(vocab, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        self.lm_head = nn.Linear(config.hidden_size, vocab)
        self.kv_cache = KVHybridCache(window_size=128, kv_quant_bits=8, latent_slots=32)

    @staticmethod
    def from_settings(settings: dict) -> "CoreTransformer":
        core = settings.get("core", {})
        cfg = CoreConfig(
            hidden_size=int(core.get("hidden_size", 256)),
            layers=int(core.get("layers", 4)),
            heads=int(core.get("heads", 4)),
            vocab_size=int(core.get("vocab_size", 1024)),
        )
        return CoreTransformer(cfg)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.encoder(x)
        return self.lm_head(x)

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        self.eval()
        tokens = self.tokenizer.encode(prompt)
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([tokens], dtype=torch.long)
            logits = self.forward(input_ids)[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            tokens.append(next_id)
            # update KV cache with dummy vectors
            with torch.no_grad():
                emb = self.embed(torch.tensor([next_id]))
                self.kv_cache.add(emb, emb)
        return self.tokenizer.decode(tokens)

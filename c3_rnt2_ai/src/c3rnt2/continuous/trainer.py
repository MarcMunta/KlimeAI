from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from ..model.core_transformer import CoreTransformer
from .dataset import collect_samples
from .lora import LoRAConfig, inject_lora
from .registry import begin_run, finalize_run, load_registry, rollback


@dataclass
class TrainResult:
    run_id: str
    promoted: bool
    loss: float
    samples: int


class ContinualTrainer:
    def __init__(self, settings: dict, base_dir: Path):
        self.settings = settings
        self.base_dir = base_dir
        self.state = load_registry(base_dir)

    def _build_model(self) -> CoreTransformer:
        return CoreTransformer.from_settings(self.settings)

    def run_tick(self) -> TrainResult:
        run_id, run_path = begin_run(self.base_dir)
        start = time.time()
        try:
            allowlist = self.settings.get("agent", {}).get("web_allowlist", ["docs.python.org"])
            samples = collect_samples(self.base_dir, allowlist)
            model = self._build_model()
            lora_cfg = LoRAConfig(rank=int(self.settings.get("continuous", {}).get("adapter_rank", 4)))
            inject_lora(model, lora_cfg)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

            loss_val = 0.0
            if samples:
                text = samples[0].response
                ids, _ = model.encode_prompt(text)
                if len(ids) < 2:
                    ids = ids + [0, 1]
                input_ids = torch.tensor([ids[:-1]], dtype=torch.long)
                targets = torch.tensor([ids[1:]], dtype=torch.long)
                logits = model.forward(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                mem_cost = sum(block.lava.stats.reads + block.lava.stats.writes for block in model.blocks)
                loss = loss + 1e-6 * mem_cost
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = float(loss.item())

            improved = loss_val > 0.0
            finalize_run(self.base_dir, run_id, promote=improved, meta={"loss": loss_val, "samples": len(samples)})
            return TrainResult(run_id=run_id, promoted=improved, loss=loss_val, samples=len(samples))
        except Exception:
            rollback(self.base_dir)
            raise

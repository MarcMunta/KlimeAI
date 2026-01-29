from __future__ import annotations

import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn

from ..model.core_transformer import CoreTransformer
from .dataset import collect_samples
from .lora import LoRAConfig, inject_lora, load_lora_state, save_lora_state
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
        try:
            allowlist = self.settings.get("agent", {}).get("web_allowlist", ["docs.python.org"])
            samples = collect_samples(self.base_dir, allowlist)
            if len(samples) < 2:
                finalize_run(self.base_dir, run_id, promote=False, meta={"loss": None, "samples": len(samples)})
                return TrainResult(run_id=run_id, promoted=False, loss=0.0, samples=len(samples))

            # split train/holdout
            random.shuffle(samples)
            split_idx = max(1, int(len(samples) * 0.9))
            train_samples = samples[:split_idx]
            holdout = samples[split_idx:]

            model = self._build_model()
            adapter_cfg = self.settings.get("continuous", {}).get("adapters", {})
            lora_cfg = LoRAConfig(
                rank=int(adapter_cfg.get("rank", self.settings.get("continuous", {}).get("adapter_rank", 4))),
                alpha=float(adapter_cfg.get("alpha", 1.0)),
            )
            target_modules = adapter_cfg.get("target_modules", [])
            inject_lora(model, lora_cfg, target_modules=target_modules)

            # Load current adapter if exists
            current = load_registry(self.base_dir).current_run_id
            if current:
                load_lora_state(model, self._adapter_path(current))

            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(self.settings.get("continuous", {}).get("lr", 1e-4)))
            max_steps = int(self.settings.get("continuous", {}).get("max_steps_per_tick", self.settings.get("continuous", {}).get("max_steps", 50)))

            base_loss = self._eval_loss(model, holdout)
            model.train()
            loss_val = 0.0
            for _ in range(max_steps):
                sample = random.choice(train_samples)
                ids, _ = model.encode_prompt(sample.response)
                if len(ids) < 2:
                    continue
                input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=model.device)
                targets = torch.tensor([ids[1:]], dtype=torch.long, device=model.device)
                logits = model.forward(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                mem_cost = sum(block.lava.stats.reads + block.lava.stats.writes for block in model.blocks)
                loss = loss + 1e-6 * mem_cost
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = float(loss.item())

            new_loss = self._eval_loss(model, holdout)
            min_improve = float(self.settings.get("continuous", {}).get("eval", {}).get("min_improvement", 0.0))
            improved = base_loss is None or (base_loss - new_loss) / max(1e-6, base_loss) >= min_improve

            # Save adapter for this run
            save_lora_state(model, self._adapter_path(run_id))
            finalize_run(
                self.base_dir,
                run_id,
                promote=improved,
                meta={"loss": new_loss, "base_loss": base_loss, "samples": len(samples)},
            )
            return TrainResult(run_id=run_id, promoted=improved, loss=new_loss, samples=len(samples))
        except Exception:
            rollback(self.base_dir)
            raise

    def _adapter_path(self, run_id: str) -> Path:
        return self.base_dir / "data" / "registry" / "adapters" / f"{run_id}.pt"

    def _eval_loss(self, model: CoreTransformer, samples: List) -> Optional[float]:
        if not samples:
            return None
        model.eval()
        losses = []
        with torch.no_grad():
            for sample in samples[: max(1, len(samples))]:
                ids, _ = model.encode_prompt(sample.response)
                if len(ids) < 2:
                    continue
                input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=model.device)
                targets = torch.tensor([ids[1:]], dtype=torch.long, device=model.device)
                logits = model.forward(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                losses.append(float(loss.item()))
        return sum(losses) / max(1, len(losses))

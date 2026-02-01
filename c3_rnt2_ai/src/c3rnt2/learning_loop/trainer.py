from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ..training.hf_qlora import train_once


def train_qlora(settings: dict, base_dir: Path, steps: int | None = None, reuse_dataset: bool = True):
    local = deepcopy(settings)
    if steps is not None:
        hf_cfg = dict(local.get("hf_train", {}) or {})
        hf_cfg["max_steps"] = int(steps)
        local["hf_train"] = hf_cfg
    return train_once(local, base_dir, reuse_dataset=reuse_dataset)

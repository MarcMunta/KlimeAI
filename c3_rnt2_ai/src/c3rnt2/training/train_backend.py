from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .hf_qlora import train_once as train_hf_once
from .core_train import train_once as train_core_once


def train_once_backend(settings: dict, base_dir: Path, reuse_dataset: bool, max_steps: int | None = None) -> Any:
    core_cfg = settings.get("core", {}) or {}
    backend = str(core_cfg.get("backend", "vortex")).lower()
    if backend == "hf":
        local = deepcopy(settings)
        if max_steps is not None and int(max_steps) > 0:
            hf_cfg = dict(local.get("hf_train", {}) or {})
            hf_cfg["max_steps"] = int(max_steps)
            local["hf_train"] = hf_cfg
        return train_hf_once(local, base_dir, reuse_dataset=reuse_dataset)
    return train_core_once(settings, base_dir, reuse_dataset=reuse_dataset, max_steps=max_steps)


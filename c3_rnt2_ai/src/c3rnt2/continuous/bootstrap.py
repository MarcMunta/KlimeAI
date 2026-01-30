from __future__ import annotations
import json

import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from ..device import autocast_context
from ..model.core_transformer import CoreTransformer
from .anchors import load_anchors
from .formatting import format_chat_sample
from .lora import LoRAConfig, inject_lora, resolve_target_modules, save_lora_state
from .registry import load_registry, save_registry, mark_bootstrapped
from .types import Sample


def _default_prompts() -> list[str]:
    return [
        "Write a short summary of the HTTP protocol.",
        "Explain a binary search in Python.",
        "Draft a docstring for a function that parses JSON.",
        "Describe the difference between a list and a tuple.",
        "Give a quick example of a SQL SELECT query.",
    ]


def _load_prompts(settings: dict, max_prompts: int) -> list[str]:
    anchors_path = Path(settings.get("continuous", {}).get("eval", {}).get("anchors_path", "data/continuous/anchors.jsonl"))
    anchors = load_anchors(anchors_path)
    prompts = [a.prompt for a in anchors if a.prompt]
    if not prompts:
        prompts = _default_prompts()
    return prompts[: max(1, max_prompts)]


def _distill_teacher(
    settings: dict,
    base_dir: Path,
    teacher: str,
    max_prompts: int,
    max_new_tokens: int,
    steps: int,
) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": f"transformers not available: {exc}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        teacher_tok = AutoTokenizer.from_pretrained(teacher)
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher, torch_dtype=dtype)
        teacher_model.to(device)
        teacher_model.eval()
    except Exception as exc:
        return {"ok": False, "error": f"failed to load teacher: {exc}"}

    prompts = _load_prompts(settings, max_prompts=max_prompts)
    samples: list[Sample] = []
    for prompt in prompts:
        try:
            inputs = teacher_tok(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = teacher_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
            )
            gen_ids = output[0][inputs["input_ids"].shape[1] :]
            completion = teacher_tok.decode(gen_ids, skip_special_tokens=True)
        except Exception:
            continue
        completion = completion.strip()
        if not completion:
            continue
        samples.append(Sample(prompt=prompt, response=completion, source_kind="bootstrap"))

    if not samples:
        return {"ok": False, "error": "no distillation samples produced"}

    seed = settings.get("continuous", {}).get("seed", settings.get("seed"))
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))

    model = CoreTransformer.from_settings(settings)
    adapter_cfg = settings.get("continuous", {}).get("adapters", {})
    lora_cfg = LoRAConfig(
        rank=int(adapter_cfg.get("rank", settings.get("continuous", {}).get("adapter_rank", 4))),
        alpha=float(adapter_cfg.get("alpha", 1.0)),
    )
    strict = bool(adapter_cfg.get("strict_target_modules", False))
    target_modules = resolve_target_modules(adapter_cfg, strict=strict)
    inject_lora(model, lora_cfg, target_modules=target_modules)

    lr = float(settings.get("continuous", {}).get("lr", 1e-4))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    last_loss = None
    for _ in range(max(1, steps)):
        sample = random.choice(samples)
        text = format_chat_sample(sample)
        ids, _ = model.encode_prompt(text)
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=model.device)
        target_ids = torch.tensor([ids[1:]], dtype=torch.long, device=model.device)
        with autocast_context(enabled=model.device.type == "cuda", dtype=model.config.dtype):
            logits = model.forward(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_loss = float(loss.item())

    adapter_path = base_dir / "data" / "registry" / "adapters" / "bootstrap.pt"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    save_lora_state(model, adapter_path)
    run_dir = base_dir / "data" / "registry" / "runs" / "bootstrap"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "loss": last_loss,
        "samples": len(samples),
        "source": "teacher",
        "teacher": teacher,
        "ts": time.time(),
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
    state = load_registry(base_dir)
    if state.current_run_id:
        state.history.append(state.current_run_id)
    state.current_run_id = "bootstrap"
    save_registry(base_dir, state)
    mark_bootstrapped(base_dir, {"source": "teacher", "teacher": teacher})
    return {"ok": True, "mode": "distill", "samples": len(samples), "loss": last_loss}


def run_bootstrap(
    settings: dict,
    base_dir: Path,
    checkpoint: str | None = None,
    teacher: str | None = None,
    max_prompts: int = 16,
    max_new_tokens: int = 64,
    steps: int = 50,
) -> dict[str, Any]:
    if teacher:
        return _distill_teacher(settings, base_dir, teacher, max_prompts, max_new_tokens, steps)

    core = settings.get("core", {}) or {}
    ckpt = checkpoint or core.get("checkpoint_path")
    if ckpt:
        path = Path(ckpt)
        if not path.exists():
            return {"ok": False, "error": f"checkpoint not found: {path}"}
        local_settings = deepcopy(settings)
        core = dict(core)
        core["checkpoint_path"] = str(path)
        local_settings["core"] = core
        _ = CoreTransformer.from_settings(local_settings)
        mark_bootstrapped(base_dir, {"source": "checkpoint", "path": str(path)})
        return {"ok": True, "mode": "checkpoint", "path": str(path)}

    return {"ok": False, "error": "no checkpoint or teacher provided"}


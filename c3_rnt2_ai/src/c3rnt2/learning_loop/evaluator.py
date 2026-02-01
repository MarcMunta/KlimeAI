from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch

from ..hf_model import load_hf_model


DEFAULT_CANARY = [
    {"prompt": "def add(a, b):", "response": " return a + b"},
    {"prompt": "JSON example:", "response": ' {"ok": true}'},
    {"prompt": "Explain Python list:", "response": " A list is a mutable sequence."},
    {"prompt": "Fix bug in loop:", "response": " Use range(len(items))"},
]


@dataclass
class EvalResult:
    ok: bool
    adapter_path: str | None
    base_loss: float | None
    adapter_loss: float | None
    improvement: float | None
    samples: int
    error: str | None = None


def _load_canary(path: Path) -> List[dict]:
    if not path.exists():
        return DEFAULT_CANARY
    records: List[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("prompt") and payload.get("response"):
            records.append(payload)
    return records or DEFAULT_CANARY


def _eval_loss(model, tokenizer, samples: Iterable[dict], max_length: int = 512) -> float:
    losses: List[float] = []
    with torch.inference_mode():
        for sample in samples:
            text = f"{sample['prompt']}{sample['response']}"
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            losses.append(float(out.loss.item()))
    return sum(losses) / max(1, len(losses))


def evaluate_adapter(base_dir: Path, settings: dict, adapter_path: Path | None = None) -> EvalResult:
    learning = settings.get("learning", {}) or {}
    canary_path = Path(learning.get("canary_path", base_dir / "data" / "learning" / "canary.jsonl"))
    if not canary_path.is_absolute():
        canary_path = base_dir / canary_path
    samples = _load_canary(canary_path)
    max_samples = int(learning.get("max_eval_samples", 8))
    samples = samples[:max_samples]

    try:
        local = dict(settings)
        core = dict(local.get("core", {}) or {})
        core.pop("hf_adapter_path", None)
        core.pop("hf_use_latest_adapter", None)
        local["core"] = core
        base = load_hf_model(local)
    except Exception as exc:
        return EvalResult(ok=False, adapter_path=str(adapter_path) if adapter_path else None, base_loss=None, adapter_loss=None, improvement=None, samples=0, error=str(exc))

    base_loss = _eval_loss(base.model, base.tokenizer, samples)
    adapter_loss = None
    improvement = None
    if adapter_path:
        try:
            from peft import PeftModel  # type: ignore

            adapted = PeftModel.from_pretrained(base.model, str(adapter_path))
            adapter_loss = _eval_loss(adapted, base.tokenizer, samples)
            improvement = base_loss - adapter_loss
        except Exception as exc:
            return EvalResult(ok=False, adapter_path=str(adapter_path), base_loss=base_loss, adapter_loss=None, improvement=None, samples=len(samples), error=str(exc))

    return EvalResult(
        ok=True,
        adapter_path=str(adapter_path) if adapter_path else None,
        base_loss=base_loss,
        adapter_loss=adapter_loss,
        improvement=improvement,
        samples=len(samples),
    )


def log_eval(base_dir: Path, settings: dict, result: EvalResult) -> None:
    learning = settings.get("learning", {}) or {}
    evals_path = Path(learning.get("evals_path", base_dir / "data" / "learning" / "evals.jsonl"))
    if not evals_path.is_absolute():
        evals_path = base_dir / evals_path
    evals_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "ok": result.ok,
        "adapter_path": result.adapter_path,
        "base_loss": result.base_loss,
        "adapter_loss": result.adapter_loss,
        "improvement": result.improvement,
        "samples": result.samples,
        "error": result.error,
    }
    with evals_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

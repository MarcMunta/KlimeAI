from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


APPROVAL_FILES = ("APPROVE.txt", "APPROVE.json")


@dataclass(frozen=True)
class PromotionResult:
    ok: bool
    promoted: bool
    reason: str
    run_id: str | None = None
    adapter_path: str | None = None
    current_adapter_path: str | None = None


def quarantine_root(base_dir: Path) -> Path:
    return base_dir / "data" / "continuous" / "quarantine"


def promoted_root(base_dir: Path) -> Path:
    return base_dir / "data" / "continuous" / "promoted"


def current_pointer_path(base_dir: Path) -> Path:
    return base_dir / "data" / "continuous" / "current_adapter.json"


def quarantine_run_dir(base_dir: Path, run_id: str) -> Path:
    return quarantine_root(base_dir) / str(run_id)


def promoted_run_dir(base_dir: Path, run_id: str) -> Path:
    return promoted_root(base_dir) / str(run_id)


def approval_present(run_dir: Path) -> bool:
    for name in APPROVAL_FILES:
        if (run_dir / name).exists():
            return True
    return False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_promotion_request(run_dir: Path, payload: dict[str, Any]) -> Path:
    out = run_dir / "PROMOTION_REQUEST.json"
    _atomic_write_json(out, payload)
    return out


def _update_current_pointer(base_dir: Path, *, run_id: str, adapter_path: Path, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    pointer = current_pointer_path(base_dir)
    state = _load_json(pointer)
    history = list(state.get("history", [])) if isinstance(state.get("history"), list) else []
    current = state.get("current_adapter_path")
    current_run = state.get("current_run_id")
    if current:
        history.append({"run_id": current_run, "adapter_path": current, "ts": state.get("ts")})
    state = {
        "current_run_id": str(run_id),
        "current_adapter_path": str(adapter_path),
        "ts": time.time(),
        "history": history[-50:],
    }
    if meta:
        state["meta"] = dict(meta)
    _atomic_write_json(pointer, state)
    return state


def _log_bench_promote(base_dir: Path, payload: dict[str, Any]) -> None:
    log_dir = base_dir / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "bench_promote.jsonl"
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")
    except Exception:
        pass


def _bench_thresholds(settings: dict) -> dict[str, Any]:
    cfg = settings.get("bench", {}) or {}
    bench_thresholds = settings.get("bench_thresholds", {}) or {}
    autopilot = settings.get("autopilot", {}) or {}
    min_tps = cfg.get("min_tokens_per_sec", bench_thresholds.get("min_tokens_per_sec", autopilot.get("bench_min_tokens_per_sec", 0.0)))
    try:
        min_tps = float(min_tps) if min_tps is not None else 0.0
    except Exception:
        min_tps = 0.0
    max_vram = cfg.get("max_vram_peak_mb")
    try:
        max_vram = float(max_vram) if max_vram is not None else None
    except Exception:
        max_vram = None
    max_reg_pct = cfg.get("max_regression_pct")
    if max_reg_pct is None:
        mr = bench_thresholds.get("max_regression", autopilot.get("bench_max_regression", 0.15))
        try:
            max_reg_pct = float(mr) * 100.0
        except Exception:
            max_reg_pct = 15.0
    else:
        try:
            max_reg_pct = float(max_reg_pct)
        except Exception:
            max_reg_pct = 15.0
    return {"min_tokens_per_sec": float(min_tps), "max_regression_pct": float(max_reg_pct), "max_vram_peak_mb": max_vram}


def _build_core_model_with_adapter(settings: dict, *, adapter_path: Path | None) -> object:
    from copy import deepcopy

    from ..model.core_transformer import CoreTransformer
    from .lora import LoRAConfig, inject_lora, load_lora_state, resolve_target_modules

    model = CoreTransformer.from_settings(deepcopy(settings))
    if adapter_path is None:
        return model
    adapter_cfg = (settings.get("continuous", {}) or {}).get("adapters", {}) or {}
    lora_cfg = LoRAConfig(rank=int(adapter_cfg.get("rank", adapter_cfg.get("adapter_rank", 4) or 4)), alpha=float(adapter_cfg.get("alpha", 1.0)))
    strict = bool(adapter_cfg.get("strict_target_modules", False))
    targets = resolve_target_modules(adapter_cfg, strict=strict)
    inject_lora(model, lora_cfg, target_modules=targets)
    load_lora_state(model, adapter_path)
    try:
        model.adapter_path = str(adapter_path)
    except Exception:
        pass
    return model


def _bench_short(settings: dict, *, adapter_path: Path | None, max_new_tokens: int = 64) -> dict[str, Any]:
    prompt = "Explain what a context manager is in Python and give a short example."
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    model = _build_core_model_with_adapter(settings, adapter_path=adapter_path)
    start = time.perf_counter()
    _ = model.generate(prompt, max_new_tokens=int(max_new_tokens)) if hasattr(model, "generate") else ""
    elapsed = max(1e-6, time.perf_counter() - start)
    vram_peak = None
    if torch is not None and torch.cuda.is_available():
        try:
            vram_peak = float(torch.cuda.max_memory_allocated() / (1024**2))
        except Exception:
            vram_peak = None
    tokens_per_sec = float(max_new_tokens) / elapsed
    return {
        "ok": True,
        "tokens_per_sec": float(tokens_per_sec),
        "vram_peak_mb": vram_peak,
        "elapsed_s": float(elapsed),
        "max_new_tokens": int(max_new_tokens),
    }


def promote_quarantine_run(
    base_dir: Path,
    *,
    run_id: str,
    require_approval: bool = True,
    settings: dict | None = None,
) -> PromotionResult:
    qdir = quarantine_run_dir(base_dir, run_id)
    adapter_src = qdir / "adapter.pt"
    manifest_path = qdir / "manifest.json"
    if not qdir.exists():
        return PromotionResult(ok=False, promoted=False, reason="quarantine_missing", run_id=run_id)
    if not adapter_src.exists():
        return PromotionResult(ok=False, promoted=False, reason="adapter_missing", run_id=run_id)

    manifest = _load_json(manifest_path)
    passed_eval = bool(manifest.get("passed_eval", False))
    if not passed_eval:
        return PromotionResult(ok=True, promoted=False, reason="passed_eval_false", run_id=run_id)

    if require_approval and not approval_present(qdir):
        return PromotionResult(ok=True, promoted=False, reason="approval_missing", run_id=run_id)

    if settings is not None:
        thresholds = _bench_thresholds(settings)
        min_tps = float(thresholds.get("min_tokens_per_sec", 0.0) or 0.0)
        max_reg_pct = float(thresholds.get("max_regression_pct", 0.0) or 0.0)
        max_vram = thresholds.get("max_vram_peak_mb")

        baseline_state = _load_json(current_pointer_path(base_dir))
        baseline_path = baseline_state.get("current_adapter_path")
        baseline_adapter = Path(str(baseline_path)) if baseline_path else None
        if baseline_adapter is not None and not baseline_adapter.is_absolute():
            baseline_adapter = base_dir / baseline_adapter
        if baseline_adapter is not None and not baseline_adapter.exists():
            baseline_adapter = None

        baseline = _bench_short(settings, adapter_path=baseline_adapter, max_new_tokens=64) if baseline_adapter else None
        candidate = _bench_short(settings, adapter_path=adapter_src, max_new_tokens=64)
        baseline_tps = float(baseline.get("tokens_per_sec")) if isinstance(baseline, dict) and baseline.get("tokens_per_sec") is not None else None
        candidate_tps = float(candidate.get("tokens_per_sec")) if isinstance(candidate, dict) and candidate.get("tokens_per_sec") is not None else None
        regression_pct = None
        if baseline_tps is not None and baseline_tps > 0 and candidate_tps is not None:
            regression_pct = max(0.0, (baseline_tps - candidate_tps) / baseline_tps * 100.0)

        failures: list[str] = []
        if candidate_tps is not None and min_tps > 0 and candidate_tps < min_tps:
            failures.append("below_min_tokens_per_sec")
        if regression_pct is not None and max_reg_pct > 0 and float(regression_pct) > float(max_reg_pct):
            failures.append("regression_exceeded")
        cand_vram = candidate.get("vram_peak_mb") if isinstance(candidate, dict) else None
        if max_vram is not None and cand_vram is not None:
            try:
                if float(cand_vram) > float(max_vram):
                    failures.append("vram_peak_exceeded")
            except Exception:
                pass

        bench_ok = not failures
        _log_bench_promote(
            base_dir,
            {
                "kind": "bench_promote",
                "backend": "core_quarantine",
                "run_id": str(run_id),
                "baseline_adapter_path": str(baseline_adapter) if baseline_adapter else None,
                "candidate_adapter_path": str(adapter_src),
                "baseline_tokens_per_sec": baseline_tps,
                "candidate_tokens_per_sec": candidate_tps,
                "regression_pct": regression_pct,
                "min_tokens_per_sec": min_tps,
                "max_regression_pct": max_reg_pct,
                "max_vram_peak_mb": max_vram,
                "failures": failures if failures else None,
                "bench_ok": bool(bench_ok),
                "ts": time.time(),
            },
        )
        if not bench_ok:
            reason = "bench_failed"
            if failures:
                reason = reason + ":" + ",".join(failures)
            return PromotionResult(ok=True, promoted=False, reason=reason, run_id=run_id)

    pdir = promoted_run_dir(base_dir, run_id)
    pdir.mkdir(parents=True, exist_ok=True)
    adapter_dst = pdir / "adapter.pt"
    shutil.copy2(adapter_src, adapter_dst)
    if manifest_path.exists():
        shutil.copy2(manifest_path, pdir / "manifest.json")
    promo_req = qdir / "PROMOTION_REQUEST.json"
    if promo_req.exists():
        shutil.copy2(promo_req, pdir / "PROMOTION_REQUEST.json")

    state = _update_current_pointer(base_dir, run_id=run_id, adapter_path=adapter_dst, meta={"source": "quarantine"})
    return PromotionResult(
        ok=True,
        promoted=True,
        reason="promoted",
        run_id=str(run_id),
        adapter_path=str(adapter_dst),
        current_adapter_path=str(state.get("current_adapter_path")),
    )

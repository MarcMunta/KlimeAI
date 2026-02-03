from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromoteResult:
    ok: bool
    promoted: bool
    adapter_path: str | None
    message: str


@dataclass(frozen=True)
class PromotionPolicy:
    min_improvement: float
    require_eval_ok: bool = True
    require_bench_ok: bool = True


def _load_latest_eval(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def promote_latest(base_dir: Path, settings: dict, min_improvement: float | None = None) -> PromoteResult:
    learning = settings.get("learning", {}) or {}
    evals_path = Path(learning.get("evals_path", base_dir / "data" / "learning" / "evals.jsonl"))
    if not evals_path.is_absolute():
        evals_path = base_dir / evals_path
    latest = _load_latest_eval(evals_path)
    if not latest:
        return PromoteResult(ok=False, promoted=False, adapter_path=None, message="no_evaluations")

    improvement = latest.get("improvement")
    adapter_path = latest.get("adapter_path")
    eval_ok = latest.get("eval_ok")
    bench_ok = latest.get("bench_ok")
    regression = latest.get("regression")
    threshold = float(min_improvement if min_improvement is not None else learning.get("promote_min_improvement", 0.0))
    policy = PromotionPolicy(
        min_improvement=threshold,
        require_eval_ok=bool(learning.get("require_eval_ok", True)),
        require_bench_ok=bool(learning.get("require_bench_ok", False)),
    )
    registry_dir = Path(settings.get("hf_train", {}).get("registry_dir", "data/registry/hf_train"))
    if not registry_dir.is_absolute():
        registry_dir = base_dir / registry_dir
    registry_path = registry_dir / "registry.json"
    promoted_path = registry_dir / "promoted.json"

    if improvement is None or adapter_path is None:
        return PromoteResult(ok=False, promoted=False, adapter_path=adapter_path, message="missing_improvement")
    if policy.require_eval_ok and eval_ok is False:
        # rollback to last good if available
        if promoted_path.exists():
            try:
                payload = json.loads(promoted_path.read_text(encoding="utf-8"))
                last = payload.get("adapter_path")
                if last:
                    registry = {"current_adapter": last, "ts": time.time()}
                    registry_path.parent.mkdir(parents=True, exist_ok=True)
                    registry_path.write_text(json.dumps(registry, ensure_ascii=True), encoding="utf-8")
                    return PromoteResult(ok=True, promoted=False, adapter_path=last, message="rolled_back")
            except Exception:
                pass
        return PromoteResult(ok=True, promoted=False, adapter_path=adapter_path, message="eval_not_ok")
    if policy.require_bench_ok and (bench_ok is not True or regression is True):
        if promoted_path.exists():
            try:
                payload = json.loads(promoted_path.read_text(encoding="utf-8"))
                last = payload.get("adapter_path")
                if last:
                    registry = {"current_adapter": last, "ts": time.time()}
                    registry_path.parent.mkdir(parents=True, exist_ok=True)
                    registry_path.write_text(json.dumps(registry, ensure_ascii=True), encoding="utf-8")
                    return PromoteResult(ok=True, promoted=False, adapter_path=last, message="rolled_back")
            except Exception:
                pass
        return PromoteResult(ok=True, promoted=False, adapter_path=adapter_path, message="bench_missing_or_regression")

    if float(improvement) >= policy.min_improvement:
        registry = {}
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
            except Exception:
                registry = {}
        registry["current_adapter"] = adapter_path
        registry["ts"] = time.time()
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(registry, ensure_ascii=True), encoding="utf-8")
        promoted_path.write_text(json.dumps({"adapter_path": adapter_path, "ts": time.time()}, ensure_ascii=True), encoding="utf-8")
        # Store baseline bench metrics for future regressions (best effort).
        if isinstance(latest.get("bench_tokens_per_sec"), (int, float)):
            try:
                (registry_dir / "bench_baseline.json").write_text(
                    json.dumps(
                        {
                            "adapter_path": adapter_path,
                            "bench_tokens_per_sec": float(latest.get("bench_tokens_per_sec")),
                            "ts": time.time(),
                        },
                        ensure_ascii=True,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return PromoteResult(ok=True, promoted=True, adapter_path=adapter_path, message="promoted")

    # rollback to last promoted if exists
    if promoted_path.exists():
        try:
            payload = json.loads(promoted_path.read_text(encoding="utf-8"))
            last = payload.get("adapter_path")
            if last:
                registry = {"current_adapter": last, "ts": time.time()}
                registry_path.parent.mkdir(parents=True, exist_ok=True)
                registry_path.write_text(json.dumps(registry, ensure_ascii=True), encoding="utf-8")
                return PromoteResult(ok=True, promoted=False, adapter_path=last, message="rolled_back")
        except Exception:
            pass
    return PromoteResult(ok=True, promoted=False, adapter_path=adapter_path, message="not_promoted")

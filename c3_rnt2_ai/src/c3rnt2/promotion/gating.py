from __future__ import annotations

# pylint: disable=broad-exception-caught
# ruff: noqa: BLE001

import json
import time
from pathlib import Path
from typing import Any


DEFAULT_BENCH_PROMPT = "Explain what a context manager is in Python and give a short example."


def log_promotion_decision(base_dir: Path, payload: dict[str, Any]) -> None:
    path = base_dir / "data" / "logs" / "promotions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record.setdefault("ts", time.time())
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except (OSError, TypeError, ValueError):
        # Never fail promotion on logging issues.
        pass


def resolve_bench_thresholds(settings: dict) -> dict[str, Any]:
    bench_cfg = settings.get("bench", {}) or {}
    bench_thresholds = settings.get("bench_thresholds", {}) or {}
    autopilot_cfg = settings.get("autopilot", {}) or {}

    min_tps = bench_cfg.get(
        "min_tokens_per_sec",
        bench_thresholds.get("min_tokens_per_sec", autopilot_cfg.get("bench_min_tokens_per_sec", 0.0)),
    )
    try:
        min_tps = float(min_tps) if min_tps is not None else 0.0
    except (TypeError, ValueError):
        min_tps = 0.0

    required_ctx = bench_cfg.get("required_ctx", bench_thresholds.get("required_ctx"))
    try:
        required_ctx = int(required_ctx) if required_ctx is not None else None
    except (TypeError, ValueError):
        required_ctx = None

    max_vram = bench_cfg.get("max_vram_peak_mb", bench_thresholds.get("max_vram_peak_mb"))
    try:
        max_vram = float(max_vram) if max_vram is not None else None
    except (TypeError, ValueError):
        max_vram = None

    max_regression = None
    if bench_cfg.get("max_regression_pct") is not None:
        try:
            max_regression = float(bench_cfg.get("max_regression_pct")) / 100.0
        except (TypeError, ValueError):
            max_regression = None
    if max_regression is None:
        mr = bench_thresholds.get("max_regression", autopilot_cfg.get("bench_max_regression", 0.15))
        try:
            max_regression = float(mr) if mr is not None else 0.15
        except (TypeError, ValueError):
            max_regression = 0.15

    max_latency_p95_ms = bench_cfg.get(
        "max_latency_p95_ms",
        bench_thresholds.get("max_latency_p95_ms", autopilot_cfg.get("bench_max_latency_p95_ms")),
    )
    try:
        max_latency_p95_ms = float(max_latency_p95_ms) if max_latency_p95_ms is not None else None
    except (TypeError, ValueError):
        max_latency_p95_ms = None

    max_latency_regression = bench_cfg.get(
        "max_latency_regression",
        bench_thresholds.get("max_latency_regression", autopilot_cfg.get("bench_max_latency_regression")),
    )
    try:
        max_latency_regression = (
            float(max_latency_regression) if max_latency_regression is not None else None
        )
    except (TypeError, ValueError):
        max_latency_regression = None

    max_error_rate = bench_cfg.get(
        "max_error_rate",
        bench_thresholds.get("max_error_rate", autopilot_cfg.get("bench_max_error_rate")),
    )
    try:
        max_error_rate = float(max_error_rate) if max_error_rate is not None else None
    except (TypeError, ValueError):
        max_error_rate = None

    return {
        "min_tokens_per_sec": float(min_tps),
        "required_ctx": required_ctx,
        "max_vram_peak_mb": max_vram,
        "max_regression": float(max_regression),
        "max_latency_p95_ms": max_latency_p95_ms,
        "max_latency_regression": max_latency_regression,
        "max_error_rate": max_error_rate,
    }


def _as_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def bench_gate(metrics: dict[str, Any], baseline: dict[str, Any] | None, thresholds: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    cand_tps = _as_float(metrics.get("tokens_per_sec"))
    cand_vram = _as_float(metrics.get("vram_peak_mb"))
    cand_ctx = _as_int(metrics.get("ctx"))
    cand_latency_p95 = _as_float(metrics.get("latency_p95_ms"))
    cand_error_rate = _as_float(metrics.get("error_rate"))

    min_tps = _as_float(thresholds.get("min_tokens_per_sec")) or 0.0
    required_ctx = _as_int(thresholds.get("required_ctx"))
    max_vram = _as_float(thresholds.get("max_vram_peak_mb"))
    max_regression = _as_float(thresholds.get("max_regression")) or 0.0
    max_latency_p95_ms = _as_float(thresholds.get("max_latency_p95_ms"))
    max_latency_regression = _as_float(thresholds.get("max_latency_regression"))
    max_error_rate = _as_float(thresholds.get("max_error_rate"))

    if min_tps > 0 and cand_tps is not None and cand_tps < min_tps:
        failures.append("below_min_tokens_per_sec")
    if required_ctx is not None and cand_ctx is not None and cand_ctx < required_ctx:
        failures.append("ctx_too_small")
    if max_vram is not None and cand_vram is not None and cand_vram > max_vram:
        failures.append("vram_peak_exceeded")
    if max_latency_p95_ms is not None and cand_latency_p95 is not None and cand_latency_p95 > max_latency_p95_ms:
        failures.append("latency_p95_exceeded")
    if max_error_rate is not None and cand_error_rate is not None and cand_error_rate > max_error_rate:
        failures.append("error_rate_exceeded")

    baseline_tps = _as_float(baseline.get("tokens_per_sec")) if isinstance(baseline, dict) else None
    baseline_latency_p95 = (
        _as_float(baseline.get("latency_p95_ms")) if isinstance(baseline, dict) else None
    )
    regression = None
    latency_regression = None
    if baseline_tps is not None and baseline_tps > 0 and cand_tps is not None:
        regression = max(0.0, (baseline_tps - cand_tps) / baseline_tps)
        if max_regression > 0 and float(regression) > float(max_regression):
            failures.append("regression_exceeded")
    if (
        baseline_latency_p95 is not None
        and baseline_latency_p95 > 0
        and cand_latency_p95 is not None
    ):
        latency_regression = max(0.0, (cand_latency_p95 - baseline_latency_p95) / baseline_latency_p95)
        if (
            max_latency_regression is not None
            and max_latency_regression > 0
            and float(latency_regression) > float(max_latency_regression)
        ):
            failures.append("latency_regression_exceeded")

    ok = not failures
    reason = "" if ok else ",".join(failures)
    return {
        "ok": bool(ok),
        "reason": reason,
        "failures": failures if failures else None,
        "candidate_tokens_per_sec": cand_tps,
        "baseline_tokens_per_sec": baseline_tps,
        "regression": regression,
        "candidate_latency_p95_ms": cand_latency_p95,
        "baseline_latency_p95_ms": baseline_latency_p95,
        "latency_regression": latency_regression,
        "candidate_error_rate": cand_error_rate,
        "min_tokens_per_sec": min_tps,
        "required_ctx": required_ctx,
        "max_vram_peak_mb": max_vram,
        "max_regression": max_regression,
        "max_latency_p95_ms": max_latency_p95_ms,
        "max_latency_regression": max_latency_regression,
        "max_error_rate": max_error_rate,
    }


def compare_to_baseline(metrics: dict[str, Any], baseline: dict[str, Any] | None, thresholds: dict[str, Any]) -> tuple[bool, str]:
    verdict = bench_gate(metrics, baseline, thresholds)
    return bool(verdict.get("ok", False)), str(verdict.get("reason") or "")


def run_bench_minimal(profile: str, base_dir: Path, *, max_new_tokens: int = 64) -> dict[str, Any]:
    from ..bench import BenchArgs, run_bench
    from ..config import load_settings

    settings = load_settings(profile)
    bench_cfg = settings.get("bench", {}) or {}
    required_ctx = bench_cfg.get("required_ctx")
    try:
        ctx = int(required_ctx) if required_ctx is not None else None
    except (TypeError, ValueError):
        ctx = None

    out_path = base_dir / "data" / "bench" / "promotion_minimal.json"
    args = BenchArgs(
        profile=str(profile),
        prompt=DEFAULT_BENCH_PROMPT,
        prompt_file=None,
        ctx=ctx,
        max_new=int(max_new_tokens),
        warmup=1,
        repeat=1,
        seed=0,
        json_out=out_path,
        jsonl_out=None,
    )
    return run_bench(settings, base_dir=base_dir, args=args)

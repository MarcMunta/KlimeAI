from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from c3rnt2.config import load_settings  # type: ignore[import-not-found]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result


def _parse_last_dict(output: str) -> dict:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return {}
    last = lines[-1].strip()
    try:
        payload = ast.literal_eval(last)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="rtx4080_16gb_vortexx_next")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--vram-margin-mb", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_tests:
        _run([sys.executable, "-m", "pytest", "-q"], check=True)

    warmup_cmd = [
        sys.executable,
        "scripts/bench_generate.py",
        "--profile",
        args.profile,
        "--max-new-tokens",
        str(args.warmup_tokens),
        "--latency-iters",
        "1",
    ]
    _run(warmup_cmd, check=True)

    bench_cmd = [
        sys.executable,
        "scripts/bench_generate.py",
        "--profile",
        args.profile,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    result = _run(bench_cmd, check=True)
    payload = _parse_last_dict(result.stdout)
    tokens_per_sec = float(payload.get("tokens_per_second", 0.0)) if payload else 0.0
    vram_peak_gb = payload.get("vram_peak_gb") if payload else None
    vram_peak_mb = float(vram_peak_gb) * 1024.0 if isinstance(vram_peak_gb, (int, float)) else None
    avg_latency_ms = None
    if tokens_per_sec > 0:
        avg_latency_ms = (float(args.max_new_tokens) / tokens_per_sec) * 1000.0

    summary = {
        "tokens_per_sec": round(tokens_per_sec, 3) if tokens_per_sec else None,
        "vram_peak_mb": round(vram_peak_mb, 3) if vram_peak_mb is not None else None,
        "avg_latency_ms": round(avg_latency_ms, 3) if avg_latency_ms is not None else None,
    }
    print(summary)

    settings = load_settings(args.profile)
    budget_mb = None
    runtime_cfg = settings.get("runtime", {}) or {}
    c3_cfg = settings.get("c3", {}) or {}
    if runtime_cfg.get("cache_vram_budget_mb") is not None:
        budget_mb = float(runtime_cfg.get("cache_vram_budget_mb"))
    elif c3_cfg.get("cache_vram_budget_mb") is not None:
        budget_mb = float(c3_cfg.get("cache_vram_budget_mb"))
    margin = args.vram_margin_mb
    if margin is None:
        margin = int(os.getenv("C3RNT2_VRAM_MARGIN_MB", "512"))
    if vram_peak_mb is not None and budget_mb is not None:
        if vram_peak_mb > budget_mb + margin:
            print(
                {
                    "ok": False,
                    "error": "vram_peak_exceeded",
                    "vram_peak_mb": vram_peak_mb,
                    "budget_mb": budget_mb,
                    "margin_mb": margin,
                }
            )
            sys.exit(2)


if __name__ == "__main__":
    main()

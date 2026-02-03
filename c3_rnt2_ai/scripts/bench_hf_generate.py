from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from c3rnt2.config import load_settings  # type: ignore[import-not-found]
from c3rnt2.hf_model import load_hf_model  # type: ignore[import-not-found]


def _rss_mb() -> float | None:
    # Best-effort RSS, no extra deps.
    try:
        if sys.platform.startswith("win"):
            import ctypes
            import ctypes.wintypes as wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            hproc = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(hproc, ctypes.byref(counters), counters.cb) == 0:
                return None
            return float(counters.WorkingSetSize) / (1024.0**2)
    except Exception:
        pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB, macOS reports bytes.
        if rss > 10_000_000:
            return rss / (1024.0**2)
        return rss / 1024.0
    except Exception:
        return None


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    k = (len(data) - 1) * (pct / 100.0)
    f = int(k)
    c = min(len(data) - 1, f + 1)
    if f == c:
        return float(data[f])
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return float(d0 + d1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="def f(x):")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    profile = args.profile or "dev_small"
    settings = load_settings(profile)
    core_cfg = dict(settings.get("core", {}) or {})
    core_cfg["backend"] = "hf"
    settings["core"] = core_cfg

    # Bench expects quantization on 16GB profiles; avoid silently falling back to full-precision downloads.
    if bool(core_cfg.get("hf_load_in_4bit") or core_cfg.get("hf_load_in_8bit")):
        try:
            import bitsandbytes  # type: ignore  # noqa: F401
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"bitsandbytes_missing: {exc}"}, ensure_ascii=True))
            raise SystemExit(1)

    try:
        hf = load_hf_model(settings)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"hf_load_failed: {exc}"}, ensure_ascii=True))
        raise SystemExit(1)

    tok = hf.tokenizer
    model = hf.model
    prompt_text = str(args.prompt)
    enc = tok(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"]
    device = getattr(model, "device", None) or torch.device(getattr(hf, "cfg", None).device if getattr(hf, "cfg", None) else "cpu")
    if isinstance(device, str):
        device = torch.device(device)
    input_ids = input_ids.to(device)
    ctx_len = int(input_ids.shape[1])

    # Warmup (does not count).
    for _ in range(max(0, int(args.warmup))):
        with torch.inference_mode():
            _ = model.generate(input_ids=input_ids, max_new_tokens=int(args.max_new_tokens), do_sample=False)
        if device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    lat_ms: list[float] = []
    new_tokens: list[int] = []
    total_s = 0.0

    iters = max(1, int(args.iters))
    for _ in range(iters):
        if device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        start = time.time()
        with torch.inference_mode():
            out = model.generate(input_ids=input_ids, max_new_tokens=int(args.max_new_tokens), do_sample=False)
        if device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        elapsed = max(1e-6, time.time() - start)
        total_s += elapsed
        lat_ms.append(elapsed * 1000.0)
        try:
            new_tokens.append(max(0, int(out.shape[1]) - int(input_ids.shape[1])))
        except Exception:
            new_tokens.append(int(args.max_new_tokens))

    total_new = int(sum(new_tokens))
    tokens_per_sec = float(total_new) / max(1e-6, total_s)
    p50 = _pct(lat_ms, 50.0)
    p95 = _pct(lat_ms, 95.0)

    vram_peak_mb = None
    if device.type == "cuda":
        try:
            vram_peak_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
        except Exception:
            vram_peak_mb = None
    ram_mb = _rss_mb()

    result = {
        "tokens_per_second": round(tokens_per_sec, 3),
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "iters": int(iters),
        "ctx_len": int(ctx_len),
        "prompt_chars": int(len(prompt_text)),
        "new_tokens_total": int(total_new),
    }
    bench = {
        "ts": time.time(),
        "profile": profile,
        "backend": "hf",
        "adapter": getattr(hf, "adapter_path", None),
        "ctx_len": int(ctx_len),
        "max_new_tokens": int(args.max_new_tokens),
        "tokens_per_sec": float(result.get("tokens_per_second", 0.0)),
        "vram_peak_mb": vram_peak_mb,
        "ram_mb": round(ram_mb, 3) if ram_mb is not None else None,
        "raw": result,
    }

    print(json.dumps(bench, ensure_ascii=True))
    bench_dir = ROOT / "data" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "latest.json").write_text(json.dumps(bench, ensure_ascii=True, indent=2), encoding="utf-8")
    (bench_dir / "latest.txt").write_text(str(result), encoding="utf-8")


if __name__ == "__main__":
    main()

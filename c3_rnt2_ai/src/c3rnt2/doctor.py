from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .config import validate_profile
from .device import detect_device
from .utils.locks import FileLock, LockUnavailable


def check_deps(modules: list[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for name in modules:
        try:
            __import__(name)
            status[name] = "ok"
        except Exception as exc:
            status[name] = f"missing ({exc.__class__.__name__})"
    return status


def run_deep_checks(settings: dict, base_dir: Path) -> dict[str, Any]:
    info = detect_device()
    deep_report: dict[str, Any] = {"deep_ok": True}
    torch_available = torch is not None
    if not torch_available:
        deep_report["deep_ok"] = False
        deep_report["error"] = "torch not available"
    else:
        from .model_loader import load_inference_model
        if info.cuda_available:
            torch.cuda.reset_peak_memory_stats()
        local_settings = deepcopy(settings)
        core_cfg = dict(local_settings.get("core", {}) or {})
        if core_cfg.get("cuda_graphs"):
            core_cfg["cuda_graphs"] = False
            local_settings["core"] = core_cfg
        if core_cfg.get("compile") or core_cfg.get("compile_step") or core_cfg.get("compile_local_mixer_step"):
            try:
                import triton  # type: ignore
            except Exception:
                core_cfg["compile"] = False
                core_cfg["compile_step"] = False
                core_cfg["compile_local_mixer_step"] = False
                local_settings["core"] = core_cfg

        try:
            model = load_inference_model(local_settings)
        except Exception as exc:
            msg = str(exc).lower()
            if "triton" in msg or "inductor" in msg:
                core = dict(local_settings.get("core", {}) or {})
                if core.get("compile") or core.get("compile_step") or core.get("compile_local_mixer_step"):
                    core["compile"] = False
                    core["compile_step"] = False
                    core["compile_local_mixer_step"] = False
                    local_settings["core"] = core
                    model = load_inference_model(local_settings)
                else:
                    raise
            else:
                raise
        prompt = "def add(a, b):"
        ids, _ = model.encode_prompt(prompt)
        if not ids:
            ids = [0]
        input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        start = time.time()
        with torch.inference_mode():
            if hasattr(model, "step_block"):
                _ = model.forward(input_ids)
                state = model.init_state(prompt_ids=ids)
                _ = model.step(ids[-1], state)
                block_ids = torch.tensor([ids[: max(1, min(2, len(ids)))]], dtype=torch.long, device=model.device)
                _ = model.step_block(block_ids, state)
                last_tok = ids[-1]
                gen_state = state
                gen_tokens = 16
                gen_start = time.time()
                for _ in range(gen_tokens):
                    logits, gen_state = model.step(last_tok, gen_state)
                    last_tok = int(torch.argmax(logits, dim=-1).item())
                gen_elapsed = max(1e-6, time.time() - gen_start)
                tokens_per_sec = gen_tokens / gen_elapsed
            else:
                gen_tokens = 16
                gen_start = time.time()
                _ = model.generate(prompt, max_new_tokens=gen_tokens)
                gen_elapsed = max(1e-6, time.time() - gen_start)
                tokens_per_sec = gen_tokens / gen_elapsed
        elapsed = time.time() - start
        vram_peak_gb = None
        if info.cuda_available:
            vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        deep_report.update(
            {
                "elapsed_sec": round(elapsed, 3),
                "vram_peak_gb": round(vram_peak_gb, 3) if vram_peak_gb is not None else None,
                "tokens_per_sec": round(tokens_per_sec, 3),
            }
        )
    runtime = settings.get("runtime", {}) or {}
    web_cfg = settings.get("tools", {}).get("web", {}) or {}
    self_patch_cfg = settings.get("self_patch", {}) or {}
    kv_quant = str(runtime.get("kv_quant", "none")).lower()
    gpu_decompress = str(runtime.get("gpu_decompress", "none")).lower()
    web_allow = web_cfg.get("allow_domains", [])
    cache_dir = Path(web_cfg.get("cache_dir", base_dir / "data" / "web_cache"))
    logs_dir = base_dir / "data" / "logs"
    queue_dir = Path(self_patch_cfg.get("queue_dir", base_dir / "data" / "self_patch" / "queue"))
    sandbox_dir = Path(self_patch_cfg.get("sandbox_dir", base_dir / "data" / "self_patch" / "sandbox"))
    bootstrap_path = base_dir / "data" / "registry" / "bootstrap" / "bootstrap_samples.jsonl"

    deep_report["kv_quant"] = kv_quant
    deep_report["gpu_decompress"] = gpu_decompress
    deep_report["web"] = {
        "enabled": bool(web_cfg.get("enabled", False)),
        "allow_domains": web_allow,
        "rate_limit_per_min": web_cfg.get("rate_limit_per_min", 30),
        "cache_dir": str(cache_dir),
        "logs_dir": str(logs_dir),
    }
    deep_report["self_patch"] = {
        "queue_dir": str(queue_dir),
        "sandbox_dir": str(sandbox_dir),
        "allowed_paths": self_patch_cfg.get("allowed_paths", []),
        "max_patch_kb": self_patch_cfg.get("max_patch_kb", 0),
    }
    deep_report["bootstrap_dataset"] = {
        "path": str(bootstrap_path),
        "exists": bootstrap_path.exists(),
    }
    # Triton availability if required.
    if gpu_decompress == "triton":
        try:
            import triton  # type: ignore  # noqa: F401
            deep_report["gpu_decompress_ready"] = True
        except Exception:
            deep_report["gpu_decompress_ready"] = False
            deep_report["deep_ok"] = False

    # Lock availability check.
    lock_dir = base_dir / "data" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_status = {}
    for role in ("serve", "train", "self_patch"):
        lock_path = lock_dir / f"{role}.lock"
        lock = FileLock(lock_path)
        try:
            lock.acquire(blocking=False)
            lock.release()
            lock_status[role] = "available"
        except LockUnavailable:
            lock_status[role] = "locked"
        except Exception as exc:
            lock_status[role] = f"error: {exc}"
    deep_report["locks"] = lock_status

    # Web allowlist sanity.
    if bool(web_cfg.get("enabled", False)) and not web_allow:
        deep_report["deep_ok"] = False

    return deep_report


def doctor_report(settings: dict, base_dir: Path, deep: bool = False) -> dict[str, Any]:
    info = detect_device()
    report: Dict[str, Any] = {
        "device": info.device,
        "cuda_available": info.cuda_available,
        "gpu": info.name,
        "vram_gb": info.vram_gb,
        "dtype": info.dtype,
        "python": sys.version.split()[0],
    }
    validate_profile(settings, base_dir=base_dir)
    if deep:
        report.update(run_deep_checks(settings, base_dir=base_dir))
    return report

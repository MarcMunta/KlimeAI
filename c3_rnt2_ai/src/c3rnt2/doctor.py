from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .config import DEFAULT_SETTINGS_PATH, load_settings, validate_profile
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


def _profile_checks(base_dir: Path) -> dict[str, str]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8")) or {}
        profiles = data.get("profiles", {}) or {}
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}
    results: dict[str, str] = {}
    for name in profiles.keys():
        try:
            settings = load_settings(name)
            validate_profile(settings, base_dir=base_dir)
            results[name] = "ok"
        except Exception as exc:
            results[name] = f"error: {exc}"
    return results


def run_deep_checks(settings: dict, base_dir: Path) -> dict[str, Any]:
    info = detect_device()
    report: dict[str, Any] = {"deep_ok": True}
    model_loaded = False
    tokens_per_sec = None
    vram_peak_gb = None

    if torch is None:
        report["deep_ok"] = False
        report["model_error"] = "torch not available"
    else:
        backend = str((settings.get("core", {}) or {}).get("backend", "vortex")).lower()
        full_model = os.getenv("C3RNT2_DOCTOR_FULL_MODEL", "").strip().lower() in {"1", "true", "yes"}

        if backend == "hf" and not full_model:
            report["model_load_mode"] = "dry"
            try:
                __import__("transformers")
                report["model_loaded"] = False
            except Exception as exc:
                report["deep_ok"] = False
                report["model_error"] = f"transformers not available: {exc}"
        else:
            from .model_loader import load_inference_model

            local_settings = deepcopy(settings)
            core_cfg = dict(local_settings.get("core", {}) or {})
            if core_cfg.get("cuda_graphs"):
                core_cfg["cuda_graphs"] = False
                local_settings["core"] = core_cfg
            if core_cfg.get("compile") or core_cfg.get("compile_step") or core_cfg.get("compile_local_mixer_step"):
                try:
                    import triton  # type: ignore  # noqa: F401
                except Exception:
                    core_cfg["compile"] = False
                    core_cfg["compile_step"] = False
                    core_cfg["compile_local_mixer_step"] = False
                    local_settings["core"] = core_cfg
            try:
                model = load_inference_model(local_settings)
                model_loaded = True
            except Exception as exc:
                report["deep_ok"] = False
                report["model_error"] = str(exc)

    if model_loaded:
        prompt = "def add(a, b):"
        try:
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
            if info.cuda_available:
                vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            report.update(
                {
                    "elapsed_sec": round(elapsed, 3),
                    "tokens_per_sec": round(tokens_per_sec, 3) if tokens_per_sec is not None else None,
                    "vram_peak_gb": round(vram_peak_gb, 3) if vram_peak_gb is not None else None,
                }
            )
        except Exception as exc:
            report["deep_ok"] = False
            report["model_error"] = str(exc)

    runtime = settings.get("runtime", {}) or {}
    tools_cfg = settings.get("tools", {}) or {}
    web_cfg = tools_cfg.get("web", {}) or {}
    self_patch_cfg = settings.get("self_patch", {}) or {}
    locks_dir = base_dir / "data" / "locks"
    bootstrap_path = base_dir / "data" / "registry" / "bootstrap" / "bootstrap_samples.jsonl"

    checks = {
        "windows_cuda": {"ok": (not sys.platform.startswith("win")) or bool(info.cuda_available)},
        "web_enabled": bool(web_cfg.get("enabled", False)),
        "web_allowlist_ok": bool(web_cfg.get("allow_domains")) if web_cfg.get("enabled") else True,
        "web_content_types_ok": bool(web_cfg.get("allow_content_types")) if web_cfg.get("enabled") else True,
        "web_cache_ttl_s": web_cfg.get("cache_ttl_s", None),
        "self_patch_enabled": bool(self_patch_cfg.get("enabled", False)),
        "self_patch_paths_ok": bool(self_patch_cfg.get("allowed_paths")) if self_patch_cfg.get("enabled") else True,
        "bootstrap_dataset_exists": bootstrap_path.exists(),
        "kv_quant": runtime.get("kv_quant", "none"),
        "kv_quant_2bit_experimental": runtime.get("kv_quant_2bit_experimental", False),
        "gpu_decompress": runtime.get("gpu_decompress", "none"),
        "locks_dir": str(locks_dir),
        "locks_exist": locks_dir.exists(),
        "profiles": _profile_checks(base_dir),
    }

    if sys.platform.startswith("win") and not bool(info.cuda_available):
        checks["windows_cuda"]["error"] = "CUDA not available on Windows (install CUDA-enabled torch + drivers)."
        report["deep_ok"] = False

    # HF training stack (QLoRA) gating (Windows-safe).
    try:
        backend = str((settings.get("core", {}) or {}).get("backend", "vortex")).lower()
        hf_train_cfg = settings.get("hf_train", {}) or {}
        if backend == "hf" and bool(hf_train_cfg.get("enabled", False)):
            quant_requested = bool(hf_train_cfg.get("load_in_4bit", True) or hf_train_cfg.get("load_in_8bit", False))
            required = ["transformers", "accelerate", "peft"]
            if quant_requested:
                required.append("bitsandbytes")
            deps = check_deps(required)
            missing = [name for name, status in deps.items() if status != "ok"]
            checks["hf_train_stack"] = {
                "ok": not missing,
                "required": required,
                "missing": missing or None,
                "status": deps,
                "quant_requested": bool(quant_requested),
            }
            if missing:
                checks["hf_train_stack"]["message"] = (
                    "En Windows, para training QLoRA se recomienda WSL2 o instalar wheels compatibles. "
                    "Inferencia puede seguir si hay fallback, pero training debe bloquearse."
                )
                report["deep_ok"] = False
    except Exception as exc:
        checks["hf_train_stack"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    # Tokenizer roundtrip (VORTEX-Tok).
    try:
        from .tokenizer.vortex_tok import load_or_create, encode, decode, metrics  # type: ignore

        tok_cfg = settings.get("tokenizer", {}) or {}
        model_path = Path(tok_cfg.get("vortex_tok_path", tok_cfg.get("vortex_model_path", base_dir / "data" / "runs" / "vortex_tok.pt")))
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        block_size = int(tok_cfg.get("block_size", 64))
        tok_model = load_or_create(model_path, block_size=block_size)
        cases = [
            "",
            "hello world",
            "ASCII: ~!@#$%^&*()_+-=[]{}|;:',.<>/?",
            "utf8: ñ é ö 😀 — 東京",
            "\tTabs\nNewlines\r\nWindows newlines\n\n",
            "code:\n```py\ndef f(x):\n    return x * (x + 1) // 2\n```\n",
            'json: {"ok": true, "text": "ñ 😀 \\\\ \\\" \\n", "n": 123}',
            "\x00\x01\t\ncontrol-chars",
            "a" * 4096,
            ("abc" * 1500) + " END",
        ]
        rng = random.Random(0)
        alphabet = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t-_=+()[]{}<>/\\'\".,:;!?"
            "Ã±Ã©Ã¶ðŸ˜€â€”æ±äº¬"
        )
        for _ in range(100):
            n = rng.randint(0, 200)
            cases.append("".join(rng.choice(alphabet) for _ in range(n)))
        start = time.time()
        failures: list[dict[str, Any]] = []
        for text in cases:
            try:
                stream = encode(text, tok_model)
                out = decode(stream, tok_model)
                if out != text:
                    failures.append({"case": text[:120], "error": "mismatch"})
            except Exception as exc:
                failures.append({"case": text[:120], "error": str(exc)})
        elapsed_ms = (time.time() - start) * 1000.0
        # Aggregate metrics on a representative sample.
        stream = encode("\n\n".join(cases), tok_model)
        tok_metrics = metrics(stream)
        checks["tokenizer_roundtrip"] = {
            "ok": not failures,
            "elapsed_ms": round(elapsed_ms, 3),
            **tok_metrics,
            "failures": failures[:3] if failures else None,
        }
        if failures:
            report["deep_ok"] = False
    except Exception as exc:
        checks["tokenizer_roundtrip"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    # Bench (ctx, VRAM peak, regression vs baseline).
    try:
        profile_name = str(settings.get("_profile") or os.getenv("C3RNT2_PROFILE") or "dev_small").strip()
        bench_cfg = settings.get("bench", {}) or {}
        required_ctx = int(bench_cfg.get("required_ctx", 4096) or 4096)
        max_new = int(bench_cfg.get("doctor_max_new_tokens", 32) or 32)
        timeout_s = float(bench_cfg.get("doctor_timeout_s", 1800) or 1800)
        env = dict(os.environ)
        # Avoid downloading weights during doctor runs by default.
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        cmd = [
            sys.executable,
            "-m",
            "c3rnt2",
            "bench",
            "--profile",
            profile_name,
            "--ctx",
            str(required_ctx),
            "--max-new-tokens",
            str(max_new),
        ]
        try:
            res = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True, env=env, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            checks["bench"] = {
                "ok": False,
                "error": f"timeout_after_s:{timeout_s}",
                "stdout_tail": (exc.stdout or "")[-400:] if exc.stdout else None,
                "stderr_tail": (exc.stderr or "")[-400:] if exc.stderr else None,
            }
            report["deep_ok"] = False
            res = None
        latest_path = base_dir / "data" / "bench" / "latest.json"
        latest = None
        if latest_path.exists():
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except Exception:
                latest = None
        if res is not None:
            checks["bench"] = {
                "ok": bool(res.returncode == 0 and isinstance(latest, dict) and latest.get("ok", False)),
                "returncode": int(res.returncode),
                "latest": latest,
                "stdout_tail": (res.stdout or "")[-400:] if res.stdout else None,
                "stderr_tail": (res.stderr or "")[-400:] if res.stderr else None,
            }
            if not bool(checks["bench"].get("ok", False)):
                report["deep_ok"] = False
    except Exception as exc:
        checks["bench"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    if str(runtime.get("gpu_decompress", "none")).lower() == "triton":
        try:
            import triton  # type: ignore  # noqa: F401
            checks["gpu_decompress_ready"] = True
        except Exception:
            checks["gpu_decompress_ready"] = False
            report["deep_ok"] = False

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
    checks["locks"] = lock_status

    report["checks"] = checks
    return report


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

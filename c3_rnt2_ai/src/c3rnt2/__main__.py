from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .config import load_settings, resolve_profile, validate_profile
from .continuous.bootstrap import run_bootstrap
from .device import detect_device
from .doctor import check_deps, run_deep_checks
from .model_loader import load_inference_model
from .prompting.chat_format import build_chat_prompt
from .server import run_server
from .utils.locks import LockUnavailable, acquire_exclusive_lock


def _load_and_validate(profile: str | None) -> dict:
    settings = load_settings(profile)
    validate_profile(settings, base_dir=Path("."))
    return settings


def cmd_doctor(args: argparse.Namespace) -> None:
    info = detect_device()
    print(
        {
            "device": info.device,
            "cuda_available": info.cuda_available,
            "gpu": info.name,
            "vram_gb": info.vram_gb,
            "dtype": info.dtype,
            "python": sys.version.split()[0],
        }
    )
    modules = [
        "torch",
        "bitsandbytes",
        "faiss",
        "triton",
        "fastapi",
        "zstandard",
        "lz4",
    ]
    status = check_deps(modules)
    print({"deps": status})

    base_dir = Path(".")
    try:
        settings = _load_and_validate(args.profile)
        print({"settings_ok": True, "profile": resolve_profile(args.profile)})
    except Exception as exc:
        print({"warning": "settings_invalid", "error": str(exc)})
        if not args.deep:
            return
        settings = load_settings(args.profile)

    if args.deep:
        try:
            deep_result = run_deep_checks(settings, base_dir=base_dir)
            print({"deep": deep_result})
        except Exception as exc:
            print({"deep": {"deep_ok": False, "error": str(exc)}})


def cmd_chat(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    model = load_inference_model(settings)
    info = detect_device()
    print({"device": info.device, "vram_gb": info.vram_gb, "dtype": info.dtype})
    decode_cfg = settings.get("decode", {}) or {}
    default_system = settings.get("core", {}).get("hf_system_prompt", "You are a helpful coding assistant.")
    backend = settings.get("core", {}).get("backend", "vortex")
    print("VORTEX-X chat. Type 'exit' to quit.")
    while True:
        prompt = input("> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        messages = [{"role": "user", "content": prompt}]
        prompt_text = build_chat_prompt(messages, backend, tokenizer=getattr(model, "tokenizer", None), default_system=default_system)
        max_new = args.max_new_tokens or int(decode_cfg.get("max_new_tokens", 64))
        temperature = args.temperature if args.temperature is not None else float(decode_cfg.get("temperature", 1.0))
        top_p = args.top_p if args.top_p is not None else float(decode_cfg.get("top_p", 1.0))
        response = model.generate(
            prompt_text,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
        )
        print(response)


def cmd_serve(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    try:
        lock = acquire_exclusive_lock(base_dir, "serve")
    except LockUnavailable:
        print({"ok": False, "error": "serve lock unavailable (train/self_patch running?)"})
        return
    try:
        run_server(settings, base_dir=base_dir, host=args.host, port=args.port)
    finally:
        lock.release()


def cmd_bootstrap(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    result = run_bootstrap(
        settings=settings,
        base_dir=base_dir,
        checkpoint=args.checkpoint,
        teacher=args.teacher,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        teacher_device=args.teacher_device,
        teacher_quant=args.teacher_quant,
        teacher_max_memory=args.teacher_max_memory,
        reuse_dataset=args.reuse_dataset,
        batch_tokens=args.batch_tokens,
        grad_accum_steps=args.grad_accum,
        profile_name=args.profile,
    )
    print(result)


def cmd_bench(args: argparse.Namespace) -> None:
    profile = args.profile or resolve_profile(None)
    _ = _load_and_validate(profile)
    script = Path(__file__).resolve().parents[2] / "scripts" / "bench_generate.py"
    if not script.exists():
        print({"ok": False, "error": "bench_generate.py not found"})
        return
    cmd = [sys.executable, str(script), "--profile", profile, "--max-new-tokens", str(args.max_new_tokens)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="c3rnt2")
    sub = parser.add_subparsers(dest="command")

    doc = sub.add_parser("doctor")
    doc.add_argument("--profile", default=None)
    doc.add_argument("--deep", action="store_true")
    doc.set_defaults(func=cmd_doctor)

    chat = sub.add_parser("chat")
    chat.add_argument("--profile", default=None)
    chat.add_argument("--max-new-tokens", type=int, default=None)
    chat.add_argument("--temperature", type=float, default=None)
    chat.add_argument("--top-p", type=float, default=None)
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve")
    serve.add_argument("--profile", default=None)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    bench = sub.add_parser("bench")
    bench.add_argument("--profile", default=None)
    bench.add_argument("--max-new-tokens", type=int, default=512)
    bench.set_defaults(func=cmd_bench)

    boot = sub.add_parser("bootstrap")
    boot.add_argument("--profile", default=None)
    boot.add_argument("--checkpoint", default=None)
    boot.add_argument("--teacher", default=None)
    boot.add_argument("--teacher-device", default="cuda")
    boot.add_argument("--teacher-quant", default="none", choices=["none", "8bit", "4bit"])
    boot.add_argument("--teacher-max-memory", default=None)
    boot.add_argument("--max-prompts", type=int, default=16)
    boot.add_argument("--max-new-tokens", type=int, default=64)
    boot.add_argument("--steps", type=int, default=50)
    boot.add_argument("--reuse-dataset", action="store_true")
    boot.add_argument("--batch-tokens", type=int, default=4096)
    boot.add_argument("--grad-accum", type=int, default=1)
    boot.set_defaults(func=cmd_bootstrap)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()

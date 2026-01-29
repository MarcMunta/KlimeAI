from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import load_settings
from .model.core_transformer import CoreTransformer
from .training import eval as eval_mod
from .tokenizer import rnt2_train
from .agent.agent_loop import run_demo_agent
from .continuous.trainer import ContinualTrainer
from .selfimprove.improve_loop import run_improve_loop
from .selfimprove.patch_ops import apply_patch


def _parse_interval(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    return int(interval)


def cmd_tokenizer_train(args: argparse.Namespace) -> None:
    rnt2_train.train(
        args.codebook_size,
        args.block_size,
        Path(args.corpus),
        Path(args.output),
        vortex_output=Path(args.vortex_output),
        macro_size=args.macro_size,
        macro_min_len=args.macro_min_len,
    )


def cmd_eval(_args: argparse.Namespace) -> None:
    eval_mod.main()


def cmd_chat(args: argparse.Namespace) -> None:
    settings = load_settings(args.profile)
    model = CoreTransformer.from_settings(settings)
    print("VORTEX-X chat. Type 'exit' to quit.")
    while True:
        prompt = input("> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        response = model.generate(prompt, max_new_tokens=args.max_new_tokens)
        print(response)


def cmd_agent_demo(args: argparse.Namespace) -> None:
    settings = load_settings(args.profile)
    report = run_demo_agent(settings)
    print(report)


def cmd_self_train(args: argparse.Namespace) -> None:
    settings = load_settings(args.profile)
    trainer = ContinualTrainer(settings=settings, base_dir=Path("."))
    interval_sec = _parse_interval(args.interval)
    while True:
        result = trainer.run_tick()
        print({"run_id": result.run_id, "promoted": result.promoted, "loss": result.loss})
        if args.once:
            break
        time.sleep(interval_sec)


def cmd_self_improve(_args: argparse.Namespace) -> None:
    report = run_improve_loop(Path("."))
    print(report)


def cmd_apply_patch(args: argparse.Namespace) -> None:
    diff = Path(args.diff).read_text(encoding="utf-8")
    result = apply_patch(Path("."), diff, approve=args.approve)
    print({"ok": result.ok, "message": result.message})


def main() -> None:
    parser = argparse.ArgumentParser(prog="c3rnt2")
    sub = parser.add_subparsers(dest="command")

    tok = sub.add_parser("tokenizer-train")
    tok.add_argument("--corpus", default="data/corpora")
    tok.add_argument("--output", default="data/runs/rnt2_dev.pt")
    tok.add_argument("--vortex-output", default="data/runs/vortex_tok.pt")
    tok.add_argument("--block-size", type=int, default=64)
    tok.add_argument("--codebook-size", type=int, default=1024)
    tok.add_argument("--macro-size", type=int, default=256)
    tok.add_argument("--macro-min-len", type=int, default=2)
    tok.set_defaults(func=cmd_tokenizer_train)

    ev = sub.add_parser("eval")
    ev.set_defaults(func=cmd_eval)

    chat = sub.add_parser("chat")
    chat.add_argument("--profile", default=None)
    chat.add_argument("--max-new-tokens", type=int, default=64)
    chat.set_defaults(func=cmd_chat)

    agent = sub.add_parser("agent-demo")
    agent.add_argument("--profile", default=None)
    agent.set_defaults(func=cmd_agent_demo)

    st = sub.add_parser("self-train")
    st.add_argument("--profile", default=None)
    st.add_argument("--interval", default="30m")
    st.add_argument("--once", action="store_true")
    st.set_defaults(func=cmd_self_train)

    si = sub.add_parser("self-improve")
    si.set_defaults(func=cmd_self_improve)

    ap = sub.add_parser("apply-patch")
    ap.add_argument("--diff", required=True)
    ap.add_argument("--approve", action="store_true")
    ap.set_defaults(func=cmd_apply_patch)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()

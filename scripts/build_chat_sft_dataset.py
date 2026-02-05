from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

try:
    from datasets import load_dataset  # type: ignore[import-untyped]
except Exception as exc:  # pragma: no cover
    raise RuntimeError("datasets package is required: pip install datasets") from exc


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _iter_open_orca(max_samples: int) -> Iterable[dict]:
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    seen = 0
    for item in ds:
        if max_samples and seen >= max_samples:
            break
        question = str(item.get("question", "")).strip()
        response = str(item.get("response", "")).strip()
        if not question or not response:
            continue
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
        yield {
            "prompt": "",
            "response": response,
            "source_kind": "open_orca",
            "messages": messages,
        }
        seen += 1


def _pick_arena_choice(item: dict) -> Optional[list[dict]]:
    winner = str(item.get("winner", "")).strip().lower()
    conv_a = item.get("conversation_a") or []
    conv_b = item.get("conversation_b") or []
    if winner == "model_a":
        return conv_a if isinstance(conv_a, list) else None
    if winner == "model_b":
        return conv_b if isinstance(conv_b, list) else None
    return None


def _iter_chat_messages(dataset_name: str, split: str, max_samples: int) -> Iterable[dict]:
    ds = load_dataset(dataset_name, split=split, streaming=True)
    seen = 0
    for item in ds:
        if max_samples and seen >= max_samples:
            break
        messages = item.get("messages")
        if not isinstance(messages, list):
            continue
        cleaned = []
        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role not in {"user", "assistant", "system"}:
                continue
            if not content:
                continue
            cleaned.append({"role": role, "content": content})
        if len(cleaned) < 2:
            continue
        last = cleaned[-1]
        if last.get("role") != "assistant":
            continue
        response = str(last.get("content", "")).strip()
        if not response:
            continue
        yield {
            "prompt": "",
            "response": response,
            "source_kind": dataset_name.split("/")[-1],
            "messages": cleaned,
        }
        seen += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/registry/hf_train/sft_samples.jsonl")
    parser.add_argument("--max-openorca", type=int, default=10000)
    parser.add_argument("--chat-dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--chat-split", default="train")
    parser.add_argument("--max-chat", type=int, default=10000)
    args = parser.parse_args()

    output = Path(args.output)
    rows = []
    rows.extend(list(_iter_open_orca(max_samples=max(0, int(args.max_openorca)))))
    try:
        rows.extend(list(_iter_chat_messages(args.chat_dataset, args.chat_split, max(0, int(args.max_chat)))))
    except Exception as exc:
        print(f"Skipping chat dataset {args.chat_dataset}: {exc}")

    total = _write_jsonl(output, rows)
    print(f"Wrote {total} samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

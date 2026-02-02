from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model_loader import load_inference_model
from ..prompting.chat_format import build_chat_prompt


def _extract_diff(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        for i in range(len(parts) - 1):
            if parts[i].strip().endswith("diff"):
                return parts[i + 1].strip() + "\n"
    for marker in ("diff --git", "--- a/"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:].strip() + "\n"
    return text.strip() + "\n"


def _format_context(context: dict, repo_root: Path) -> str:
    if not context:
        return ""
    lines = []
    for key, value in context.items():
        if key == "files" and isinstance(value, list):
            for path in value:
                try:
                    full = (repo_root / str(path)).resolve()
                    if repo_root.resolve() not in full.parents:
                        continue
                    text = full.read_text(encoding="utf-8", errors="ignore")
                    lines.append(f"[FILE] {path}\n{text[:4000]}")
                except Exception:
                    continue
        else:
            lines.append(f"[{key}] {value}")
    return "\n\n".join(lines)


def generate_diff_with_llm(goal: str, context: dict, repo_root: Path, settings: dict) -> str:
    model = load_inference_model(settings)
    system = (
        "You are a code patch generator. Return ONLY a unified diff (git apply compatible). "
        "Do not include explanations."
    )
    ctx = _format_context(context or {}, repo_root)
    user = f"Goal: {goal}\n\nContext:\n{ctx}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = build_chat_prompt(messages, backend=str(settings.get("core", {}).get("backend", "vortex")), tokenizer=getattr(model, "tokenizer", None), default_system=None)
    output = model.generate(prompt, max_new_tokens=512, temperature=0.0)
    diff = _extract_diff(output)
    return diff

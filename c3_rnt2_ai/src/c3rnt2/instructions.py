from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

DEFAULT_VORTEX_SYSTEM = "You are Vortex, a local technical assistant."


def _resolve_path(base_dir: Path, raw: object | None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_instruction_bundle(settings: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    base_dir = Path(base_dir or ".").resolve()
    instructions = settings.get("instructions", {}) or {}
    core = settings.get("core", {}) or {}
    parts: list[str] = []
    sources: list[dict[str, str]] = []

    candidates = [
        ("vortex_system", instructions.get("vortex_system_path") or core.get("system_prompt_path")),
        ("domain_policy", instructions.get("domain_policy_path")),
        ("operator_notes", instructions.get("operator_notes_path")),
    ]
    for kind, raw_path in candidates:
        path = _resolve_path(base_dir, raw_path)
        if path is None or not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parts.append(text)
        sources.append(
            {
                "kind": str(kind),
                "path": str(path),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    if parts:
        text = "\n\n".join(parts).strip()
    else:
        text = (
            str(core.get("hf_system_prompt") or "").strip()
            or DEFAULT_VORTEX_SYSTEM
        )
        sources.append(
            {
                "kind": "inline_fallback",
                "path": "",
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"text": text, "digest": digest, "sources": sources}

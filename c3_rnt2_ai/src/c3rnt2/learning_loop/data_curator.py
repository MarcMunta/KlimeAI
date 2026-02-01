from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..continuous.types import Sample


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3})?[-.\s]?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
_INJECTION_RE = re.compile(r"(ignore previous|system prompt|developer message|exfiltrate|jailbreak|sudo)", re.IGNORECASE)
_DANGEROUS_URL_RE = re.compile(r"\b(file://|data:|javascript:|ftp://)", re.IGNORECASE)


@dataclass
class CurateResult:
    ok: bool
    written: int
    skipped: int
    output_path: Path
    error: str | None = None


def _iter_raw(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            yield payload


def _has_pii(text: str) -> bool:
    return bool(_EMAIL_RE.search(text) or _PHONE_RE.search(text))


def _has_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def _has_dangerous_url(text: str) -> bool:
    return bool(_DANGEROUS_URL_RE.search(text))


def _quality_ok(text: str, min_chars: int, max_chars: int | None) -> bool:
    cleaned = " ".join(text.split())
    if len(cleaned) < min_chars:
        return False
    if max_chars is not None and len(cleaned) > max_chars:
        return False
    return True


def curate_dataset(base_dir: Path, settings: dict) -> CurateResult:
    learning = settings.get("learning", {}) or {}
    raw_path = Path(learning.get("raw_path", base_dir / "data" / "learning" / "raw.jsonl"))
    curated_path = Path(learning.get("curated_path", base_dir / "data" / "learning" / "curated.jsonl"))
    if not raw_path.is_absolute():
        raw_path = base_dir / raw_path
    if not curated_path.is_absolute():
        curated_path = base_dir / curated_path
    curated_path.parent.mkdir(parents=True, exist_ok=True)

    min_chars = int(learning.get("min_chars", 20))
    max_chars = learning.get("max_chars")
    max_chars = int(max_chars) if max_chars is not None else None

    seen: set[str] = set()
    written = 0
    skipped = 0
    try:
        with curated_path.open("w", encoding="utf-8") as handle:
            for payload in _iter_raw(raw_path):
                prompt = str(payload.get("prompt", "")).strip()
                response = str(payload.get("response", "")).strip()
                if not prompt or not response:
                    skipped += 1
                    continue
                joined = f"{prompt}\n{response}"
                if _has_pii(joined) or _has_injection(joined) or _has_dangerous_url(joined):
                    skipped += 1
                    continue
                if not _quality_ok(response, min_chars=min_chars, max_chars=max_chars):
                    skipped += 1
                    continue
                digest = f"{prompt}\n{response}"
                if digest in seen:
                    skipped += 1
                    continue
                seen.add(digest)
                sample = Sample(prompt=prompt, response=response, source_kind=str(payload.get("source", "unknown")))
                out = {
                    "prompt": sample.prompt,
                    "response": sample.response,
                    "source_kind": sample.source_kind,
                    "messages": sample.messages,
                }
                handle.write(json.dumps(out, ensure_ascii=True) + "\n")
                written += 1
        return CurateResult(ok=True, written=written, skipped=skipped, output_path=curated_path)
    except Exception as exc:
        return CurateResult(ok=False, written=written, skipped=skipped, output_path=curated_path, error=str(exc))

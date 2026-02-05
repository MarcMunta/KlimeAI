from __future__ import annotations

import re
from dataclasses import dataclass

from .store import SkillRecord, SkillStore


@dataclass(frozen=True)
class SelectedSkill:
    record: SkillRecord
    score: float


def _extract_match_text(messages: list[dict], *, max_chars: int = 4000) -> str:
    parts: list[str] = []
    for msg in reversed(messages or []):
        if str(msg.get("role") or "").lower() != "user":
            continue
        content = str(msg.get("content") or "")
        if content:
            parts.append(content)
        if len(parts) >= 2:
            break
    text = "\n".join(reversed(parts)).strip()
    if len(text) > int(max_chars):
        text = text[-int(max_chars) :]
    return text


class SkillsRouter:
    def __init__(self, store: SkillStore):
        self.store = store
        self._regex_cache: dict[str, re.Pattern] = {}

    def _compile(self, pattern: str) -> re.Pattern | None:
        key = str(pattern)
        if key in self._regex_cache:
            return self._regex_cache[key]
        try:
            compiled = re.compile(key, flags=re.IGNORECASE)
        except re.error:
            return None
        self._regex_cache[key] = compiled
        return compiled

    def select(
        self,
        messages: list[dict],
        *,
        model: str | None,
        max_k: int,
        token_budget_total: int,
        strict: bool,
    ) -> list[SkillRecord]:
        _ = model  # reserved for future filtering
        text = _extract_match_text(messages)
        if not text:
            return []
        lowered = text.lower()

        candidates: list[SelectedSkill] = []
        for rec in self.store.list():
            if not rec.enabled:
                continue
            if strict:
                if not rec.trusted:
                    continue
                safety = rec.spec.safety
                if safety.network or safety.filesystem_write or safety.shell:
                    continue

            trig = rec.spec.triggers
            score = 0.0
            for kw in trig.keywords:
                needle = str(kw).strip().lower()
                if needle and needle in lowered:
                    score += 1.0
            for pat in trig.regex:
                compiled = self._compile(pat)
                if compiled is None:
                    continue
                try:
                    if compiled.search(text):
                        score += 1.5
                except Exception:
                    continue

            if score <= 0:
                continue
            score = score * 10.0 + float(rec.spec.priority)
            candidates.append(SelectedSkill(record=rec, score=score))

        candidates.sort(key=lambda s: (-float(s.score), s.record.ref))

        selected: list[SkillRecord] = []
        used = 0
        max_k = max(1, int(max_k))
        token_budget_total = max(1, int(token_budget_total))
        overhead_per_skill = 12  # wrapper + id header

        for item in candidates:
            if len(selected) >= max_k:
                break
            need = int(item.record.prompt_tokens_est) + int(overhead_per_skill)
            if used + need > token_budget_total:
                continue
            selected.append(item.record)
            used += need

        return selected


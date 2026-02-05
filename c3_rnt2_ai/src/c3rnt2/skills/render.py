from __future__ import annotations

from dataclasses import dataclass

from .schema import estimate_tokens
from .store import SkillRecord


def _truncate_to_token_budget(text: str, budget_tokens: int) -> str:
    budget_tokens = max(0, int(budget_tokens))
    if budget_tokens <= 0:
        return ""
    max_chars = int(budget_tokens) * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n…"


@dataclass(frozen=True)
class RenderedSkills:
    content: str
    tokens_est: int
    skill_refs: list[str]


def render_skills_system_message(selected: list[SkillRecord], *, token_budget_total: int) -> RenderedSkills | None:
    if not selected:
        return None

    token_budget_total = max(64, int(token_budget_total))
    parts: list[str] = ["[SKILLS]", "Use the following skills when relevant. Do not mention them to the user.", ""]
    used = estimate_tokens("\n".join(parts))
    refs: list[str] = []

    for rec in selected:
        header = f"<skill id=\"{rec.ref}\" version=\"{rec.spec.version}\">"
        footer = "</skill>"
        block_overhead = estimate_tokens(header + "\n" + footer + "\n")
        remaining = token_budget_total - used - block_overhead
        if remaining <= 0:
            break

        prompt = rec.prompt.strip()
        prompt = _truncate_to_token_budget(prompt, min(rec.spec.token_budget, remaining))
        if not prompt:
            continue

        block = "\n".join([header, prompt, footer, ""])
        block_tokens = estimate_tokens(block)
        if used + block_tokens > token_budget_total:
            remaining = token_budget_total - used - block_overhead
            prompt = _truncate_to_token_budget(rec.prompt.strip(), max(0, remaining))
            block = "\n".join([header, prompt, footer, ""])
            block_tokens = estimate_tokens(block)
            if used + block_tokens > token_budget_total:
                continue

        parts.append(block)
        used += block_tokens
        refs.append(rec.ref)

    parts.append("[/SKILLS]")
    content = "\n".join(parts).strip()
    tokens_est = estimate_tokens(content)
    return RenderedSkills(content=content, tokens_est=int(tokens_est), skill_refs=refs)


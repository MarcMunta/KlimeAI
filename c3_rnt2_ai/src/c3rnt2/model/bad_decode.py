from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class BADStats:
    proposed: int = 0
    accepted: int = 0
    rejected: int = 0
    entropy_high: int = 0


def _entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    return float(ent.mean().item())


def _apply_repetition_penalty(logits: torch.Tensor, generated: List[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0:
        return logits
    logits = logits.clone()
    for token_id in set(generated):
        val = logits[0, token_id]
        if val > 0:
            logits[0, token_id] = val / penalty
        else:
            logits[0, token_id] = val * penalty
    return logits


def _no_repeat_ngram_mask(generated: List[int], no_repeat_ngram: int) -> set[int]:
    if no_repeat_ngram <= 0 or len(generated) < no_repeat_ngram - 1:
        return set()
    n = no_repeat_ngram
    ngrams = {}
    for i in range(len(generated) - n + 1):
        prev = tuple(generated[i : i + n - 1])
        nxt = generated[i + n - 1]
        ngrams.setdefault(prev, set()).add(nxt)
    prefix = tuple(generated[-(n - 1) :])
    return ngrams.get(prefix, set())


def _sample_logits(logits: torch.Tensor, temperature: float, top_p: float, generated: List[int], repetition_penalty: float, no_repeat_ngram: int) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits / max(1e-6, temperature)
    logits = _apply_repetition_penalty(logits, generated, repetition_penalty)
    banned = _no_repeat_ngram_mask(generated, no_repeat_ngram)
    if banned:
        logits[0, list(banned)] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > top_p
        if mask.any():
            cutoff = mask.float().argmax().item()
            sorted_probs[cutoff + 1 :] = 0.0
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return int(torch.multinomial(probs, num_samples=1).item())


def bad_decode(
    model,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    entropy_threshold: float,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    adaptive_granularity: bool = True,
) -> Tuple[str, BADStats]:
    ids, total_len = model.encode_prompt(prompt)
    stats = BADStats()
    generated = list(ids)

    model.reset_state()
    state_full = model.init_state(batch=1)
    state_draft = model.init_state(batch=1)

    # warmup states with prompt
    last_logits_full = None
    last_logits_draft = None
    for tok in ids:
        last_logits_full, state_full = model.step(tok, state_full, write_memory=True)
        last_logits_draft, state_draft = model.step(tok, state_draft, num_layers=model.config.draft_layers, write_memory=False)

    remaining = max_new_tokens
    while remaining > 0:
        draft_block = min(block_size, remaining)
        draft_tokens: List[int] = []
        for _ in range(draft_block):
            if last_logits_draft is None:
                last_logits_draft, state_draft = model.step(generated[-1], state_draft, num_layers=model.config.draft_layers, write_memory=False)
            next_tok = _sample_logits(last_logits_draft, temperature, top_p, generated + draft_tokens, repetition_penalty, no_repeat_ngram)
            draft_tokens.append(next_tok)
            last_logits_draft, state_draft = model.step(next_tok, state_draft, num_layers=model.config.draft_layers, write_memory=False)
        stats.proposed += len(draft_tokens)

        accepted = 0
        for tok in draft_tokens:
            if last_logits_full is None:
                last_logits_full, state_full = model.step(generated[-1], state_full, write_memory=True)
            ent = _entropy(last_logits_full)
            if adaptive_granularity and ent > entropy_threshold and getattr(model, "escape_mode", "") == "exact":
                stats.entropy_high += 1
                start = getattr(model, "byte_token_start", 0)
                byte_slice = last_logits_full[:, start : start + 256]
                next_tok = _sample_logits(byte_slice, temperature, top_p, generated, repetition_penalty, no_repeat_ngram) + start
            else:
                next_tok = _sample_logits(last_logits_full, temperature, top_p, generated, repetition_penalty, no_repeat_ngram)
            if next_tok == tok:
                generated.append(tok)
                accepted += 1
                stats.accepted += 1
                last_logits_full, state_full = model.step(tok, state_full, write_memory=True)
            else:
                generated.append(next_tok)
                stats.rejected += 1
                last_logits_full, state_full = model.step(next_tok, state_full, write_memory=True)
                break
        remaining -= max(1, accepted)

    text = model.decode_ids(generated, total_len=None)
    return text, stats

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


def bad_decode(model, prompt: str, max_new_tokens: int, block_size: int, entropy_threshold: float) -> Tuple[str, BADStats]:
    ids, total_len = model.encode_prompt(prompt)
    stats = BADStats()

    generated = list(ids)
    remaining = max_new_tokens

    while remaining > 0:
        draft_block = min(block_size, remaining)
        draft_tokens = model.draft_next_tokens(generated, draft_block)
        stats.proposed += len(draft_tokens)

        accepted = 0
        for tok in draft_tokens:
            logits = model.full_next_logits(generated)
            ent = _entropy(logits)
            if ent > entropy_threshold:
                draft_block = 1
            if ent > entropy_threshold:
                stats.entropy_high += 1
                # drop to byte-level granularity when entropy is high
                start = getattr(model, "byte_token_start", 0)
                byte_slice = logits[:, start : start + 256]
                next_tok = int(torch.argmax(byte_slice, dim=-1).item()) + start
            else:
                next_tok = int(torch.argmax(logits, dim=-1).item())
            if next_tok == tok:
                generated.append(tok)
                accepted += 1
                stats.accepted += 1
            else:
                generated.append(next_tok)
                stats.rejected += 1
                break
        remaining -= max(1, accepted)

    text = model.decode_ids(generated, total_len=None)
    return text, stats

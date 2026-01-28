from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class RouterConfig:
    top_k: int = 2
    stability_threshold: float = 0.1
    mem_cost_weight: float = 0.01


@dataclass
class RouterState:
    last_selection: List[int]


class Router:
    """Coherent router with hysteresis (MVP)."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self.state = RouterState(last_selection=[])

    def route(self, scores: Iterable[float]) -> List[int]:
        scores = np.asarray(list(scores), dtype=np.float32)
        if scores.size == 0:
            return []
        top_idx = scores.argsort()[::-1][: self.config.top_k].tolist()
        if not self.state.last_selection:
            self.state.last_selection = top_idx
            return top_idx

        # hysteresis: if top selection is close, keep previous
        prev = self.state.last_selection
        prev_score = scores[prev].mean() if prev else 0.0
        new_score = scores[top_idx].mean()
        if new_score - prev_score < self.config.stability_threshold:
            return prev
        self.state.last_selection = top_idx
        return top_idx

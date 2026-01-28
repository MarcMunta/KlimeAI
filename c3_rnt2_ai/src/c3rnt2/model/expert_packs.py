from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ExpertPack:
    name: str
    deltas: Dict[str, np.ndarray]

    def apply(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        updated = dict(weights)
        for key, delta in self.deltas.items():
            if key in updated:
                updated[key] = updated[key] + delta
        return updated


def load_expert_packs(pack_dir: str) -> List[ExpertPack]:
    # MVP: placeholder loader; returns empty list when no packs exist.
    return []

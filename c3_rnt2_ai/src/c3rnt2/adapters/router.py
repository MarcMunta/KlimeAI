from __future__ import annotations

import math
from dataclasses import dataclass

from ..continuous.knowledge_store import EmbeddingBackend


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-9:
        return vec
    return [v / norm for v in vec]


@dataclass(frozen=True)
class RouterDecision:
    selected_adapter: str | None
    reason: str
    score: float | None = None
    selected_adapters: list[str] | None = None
    scores: list[float] | None = None


class AdapterRouter:
    def __init__(
        self,
        *,
        mode: str = "keyword_map",
        keyword_map: dict[str, str] | None = None,
        default_adapter: str | None = None,
        embedding_backend: str = "hash",
        embedding_dim: int = 128,
        embedding_min_score: float = 0.0,
        top_k: int = 1,
    ):
        self.mode = str(mode or "keyword_map").lower()
        self.keyword_map = {str(k).lower(): str(v) for k, v in (keyword_map or {}).items() if k and v}
        self.default_adapter = str(default_adapter) if default_adapter else None
        self.embedding_min_score = float(embedding_min_score or 0.0)
        try:
            self.top_k = max(1, int(top_k))
        except Exception:
            self.top_k = 1
        self._embedder = EmbeddingBackend(backend=str(embedding_backend or "hash"), dim=int(embedding_dim))

    @classmethod
    def from_settings(cls, settings: dict) -> "AdapterRouter":
        cfg = settings.get("adapters", {}) or {}
        router_cfg = cfg.get("router", {}) or {}
        knowledge = settings.get("knowledge", {}) or {}
        return cls(
            mode=router_cfg.get("mode", "keyword_map"),
            keyword_map=dict(router_cfg.get("keyword_map", {}) or {}),
            default_adapter=router_cfg.get("default") or cfg.get("default"),
            embedding_backend=router_cfg.get("embedding_backend", knowledge.get("embedding_backend", "hash")),
            embedding_dim=int(router_cfg.get("embedding_dim", 128)),
            embedding_min_score=float(router_cfg.get("embedding_min_score", 0.0)),
            top_k=int(router_cfg.get("top_k", 1) or 1),
        )

    def _embedding_rank(self, prompt: str, adapter_names: list[str]) -> list[tuple[str, float]]:
        # Embed prompt and per-adapter routing text, then rank by cosine.
        by_adapter: dict[str, list[str]] = {}
        for kw, adapter in self.keyword_map.items():
            by_adapter.setdefault(adapter, []).append(kw)
        texts = []
        for name in adapter_names:
            kws = " ".join(sorted(set(by_adapter.get(name, []))))
            texts.append(f"{name} {kws}".strip())
        vecs = self._embedder.encode([prompt] + texts)
        qvec = _normalize(vecs[0])
        scored: list[tuple[str, float]] = []
        for name, vec in zip(adapter_names, vecs[1:]):
            score = _dot(qvec, _normalize(vec))
            scored.append((name, float(score)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    def select(self, prompt: str, adapter_names: list[str], *, top_k: int | None = None) -> RouterDecision:
        if not adapter_names:
            return RouterDecision(None, reason="no_adapters")
        try:
            k = max(1, int(top_k) if top_k is not None else int(getattr(self, "top_k", 1)))
        except Exception:
            k = 1
        k = min(k, len(adapter_names))
        prompt_l = (prompt or "").lower()

        if self.mode in {"keyword_map", "keyword", "hybrid"}:
            # Prefer longest matches to reduce accidental collisions.
            for kw, adapter in sorted(self.keyword_map.items(), key=lambda kv: len(kv[0]), reverse=True):
                if adapter not in adapter_names:
                    continue
                if kw and kw in prompt_l:
                    if self.mode != "hybrid" or k <= 1:
                        return RouterDecision(adapter, reason=f"keyword:{kw}")
                    ranked = self._embedding_rank(prompt, adapter_names)
                    picks = [adapter]
                    scores = [1.0]
                    for name, score in ranked:
                        if name == adapter:
                            continue
                        picks.append(name)
                        scores.append(float(score))
                        if len(picks) >= k:
                            break
                    return RouterDecision(
                        picks[0],
                        reason=f"keyword:{kw}+embedding",
                        score=scores[0] if scores else None,
                        selected_adapters=picks,
                        scores=scores,
                    )
            if self.mode != "hybrid":
                fallback = self.default_adapter if self.default_adapter in adapter_names else adapter_names[0]
                return RouterDecision(fallback, reason="default")

        if self.mode in {"embedding", "hybrid"}:
            ranked = self._embedding_rank(prompt, adapter_names)
            best_name, best_score = ranked[0]
            if best_score < self.embedding_min_score:
                fallback = self.default_adapter if self.default_adapter in adapter_names else adapter_names[0]
                return RouterDecision(fallback, reason="embedding_below_threshold", score=float(best_score))
            picks = [name for name, _score in ranked[:k]] if k > 1 else None
            scores = [float(score) for _name, score in ranked[:k]] if k > 1 else None
            return RouterDecision(
                best_name,
                reason="embedding",
                score=float(best_score),
                selected_adapters=picks,
                scores=scores,
            )

        # Unknown mode -> safe fallback.
        fallback = self.default_adapter if self.default_adapter in adapter_names else adapter_names[0]
        return RouterDecision(fallback, reason="unknown_mode_default")

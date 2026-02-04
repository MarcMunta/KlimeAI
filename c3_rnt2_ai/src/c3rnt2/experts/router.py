from __future__ import annotations

from ..adapters.router import AdapterRouter


class ExpertRouter(AdapterRouter):
    @classmethod
    def from_settings(cls, settings: dict) -> "ExpertRouter":
        cfg = settings.get("experts", {}) or {}
        router_cfg = cfg.get("router", {}) or {}
        knowledge = settings.get("knowledge", {}) or {}
        return cls(
            mode=router_cfg.get("mode", "keyword"),
            keyword_map=dict(router_cfg.get("keyword_map", {}) or {}),
            default_adapter=router_cfg.get("default") or cfg.get("default"),
            embedding_backend=router_cfg.get("embedding_backend", knowledge.get("embedding_backend", "hash")),
            embedding_dim=int(router_cfg.get("embedding_dim", 128)),
            embedding_min_score=float(router_cfg.get("embedding_min_score", 0.0)),
        )


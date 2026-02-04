from __future__ import annotations

from c3rnt2.adapters.router import AdapterRouter


def test_adapter_router_embedding_topk_returns_ranked() -> None:
    router = AdapterRouter(mode="embedding", keyword_map={}, default_adapter=None, top_k=2)

    def _fake_rank(_prompt: str, _names: list[str]) -> list[tuple[str, float]]:
        return [("a", 0.9), ("b", 0.2), ("c", -0.1)]

    router._embedding_rank = _fake_rank  # type: ignore[assignment]

    decision = router.select("prompt", ["a", "b", "c"])
    assert decision.selected_adapter == "a"
    assert decision.score == 0.9
    assert decision.selected_adapters == ["a", "b"]
    assert decision.scores == [0.9, 0.2]


def test_adapter_router_hybrid_keyword_plus_embedding_topk() -> None:
    router = AdapterRouter(mode="hybrid", keyword_map={"python": "prog"}, default_adapter="general", top_k=2)

    def _fake_rank(_prompt: str, _names: list[str]) -> list[tuple[str, float]]:
        return [("general", 0.8), ("prog", 0.1)]

    router._embedding_rank = _fake_rank  # type: ignore[assignment]

    decision = router.select("write python code", ["general", "prog"])
    assert decision.selected_adapter == "prog"
    assert decision.reason.startswith("keyword:")
    assert decision.selected_adapters == ["prog", "general"]


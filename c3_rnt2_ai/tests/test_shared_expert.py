from __future__ import annotations

from pathlib import Path

from c3rnt2.adapters.registry import AdapterRegistry
from c3rnt2.server import _apply_hf_adapter_selection


class _DummyHFModel:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.calls: list[tuple] = []
        self.active_adapter_name: str | None = None
        self.weights: dict[str, float] | None = None

    def add_adapter(self, name: str, path: str) -> bool:  # type: ignore[no-untyped-def]
        self.calls.append(("add", name, path))
        return True

    def set_weighted_adapters(self, weights: dict[str, float]) -> bool:  # type: ignore[no-untyped-def]
        self.calls.append(("mix", dict(weights)))
        self.weights = dict(weights)
        self.active_adapter_name = "mixed"
        return True

    def set_adapter(self, name: str) -> None:  # type: ignore[no-untyped-def]
        self.calls.append(("set", name))
        self.active_adapter_name = str(name)


def test_shared_expert_forces_weighted_mix(tmp_path: Path) -> None:
    adapter_a = tmp_path / "a"
    adapter_a.mkdir(parents=True, exist_ok=True)
    shared = tmp_path / "shared"
    shared.mkdir(parents=True, exist_ok=True)

    settings = {
        "experts": {
            "enabled": True,
            "router": {"mix_mode": "weighted"},
            "shared_expert_path": str(shared),
            "shared_expert_name": "shared_expert",
            "shared_expert_weight": 0.25,
        }
    }

    registry = AdapterRegistry(enabled=True, paths={"a": str(adapter_a)}, max_loaded=6, default=None)
    model = _DummyHFModel(base_dir=tmp_path)

    out = _apply_hf_adapter_selection(model, settings, registry, {"adapters": ["a"], "scores": [1.0]})
    assert out["ok"] is True
    assert out["shared_expert"] == "shared_expert"
    assert out["shared_used"] is True
    assert set(out["active_adapters"]) == {"a", "shared_expert"}
    assert isinstance(out.get("weights"), dict)
    assert out["weights"]["shared_expert"] > 0.0


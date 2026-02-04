from __future__ import annotations

from pathlib import Path

from c3rnt2.experts.registry import ExpertRegistry


def test_expert_registry_discovers_hf_train_runs(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "data" / "registry" / "hf_train" / "run1" / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "experts": {"enabled": True, "paths": {}, "max_loaded": 6},
        "hf_train": {"registry_dir": str(tmp_path / "data" / "registry" / "hf_train")},
    }
    reg = ExpertRegistry.from_settings(settings, base_dir=tmp_path)
    assert reg.enabled is True
    assert reg.max_loaded == 6
    assert reg.get_path("expert_run1") == str(adapter_dir)


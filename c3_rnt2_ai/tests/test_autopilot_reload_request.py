from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from c3rnt2 import autopilot as ap


def test_autopilot_writes_reload_request_on_promote(tmp_path: Path, monkeypatch) -> None:
    def _fake_train(_profile: str, reuse_dataset: bool, max_steps: int | None):
        _ = reuse_dataset, max_steps
        return {
            "ok": True,
            "ok_train": True,
            "ok_eval": True,
            "eval_ok": True,
            "improvement": 1.0,
            "adapter_dir": str(tmp_path / "data" / "registry" / "hf_train" / "run" / "adapter"),
            "vram_peak_mb": 123.0,
            "tokens_per_sec": 456.0,
        }

    def _fake_promote(_base_dir: Path, _settings: dict, min_improvement: float | None = None):
        _ = min_improvement
        return SimpleNamespace(ok=True, promoted=True, adapter_path=str(tmp_path / "adapter_promoted"), message="promoted")

    monkeypatch.setattr(ap, "_train_subprocess", _fake_train)
    monkeypatch.setattr(ap, "promote_latest", _fake_promote)

    settings = {
        "_profile": "safe_selftrain_4080_hf",
        "autopilot": {
            "enabled": True,
            "train_cooldown_minutes": 0,
            "eval_cooldown_minutes": 0,
            "ingest_cooldown_minutes": 9999,
            "patch_cooldown_minutes": 9999,
            "min_improvement": 0.0,
        },
        "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "knowledge": {"embedding_backend": "hash"},
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "learning": {"evals_path": str(tmp_path / "data" / "learning" / "evals.jsonl")},
        "hf_train": {"enabled": False},
    }

    result = ap.run_autopilot_tick(settings, tmp_path, no_web=True, mock=False, force=False)
    assert result.ok is True
    reload_path = tmp_path / "data" / "state" / "reload.json"
    assert reload_path.exists()
    payload = json.loads(reload_path.read_text(encoding="utf-8"))
    assert payload.get("adapter_path") == str(tmp_path / "adapter_promoted")

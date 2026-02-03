from __future__ import annotations

from pathlib import Path

from c3rnt2.autopilot import run_autopilot_tick


def test_autopilot_tick_skips_when_disabled(tmp_path: Path) -> None:
    settings = {
        "_profile": "safe_selftrain_4080_hf",
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "knowledge": {"embedding_backend": "hash"},
        "autopilot": {"enabled": False},
    }
    result = run_autopilot_tick(settings, tmp_path, no_web=True, mock=True)
    assert result.ok is True
    assert result.steps.get("skipped") == "disabled"


def test_autopilot_force_runs_when_disabled(tmp_path: Path) -> None:
    settings = {
        "_profile": "safe_selftrain_4080_hf",
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "knowledge": {"embedding_backend": "hash"},
        "autopilot": {"enabled": False, "training_jsonl_max_items": 0},
        "hf_train": {"enabled": False},
    }
    result = run_autopilot_tick(settings, tmp_path, no_web=True, mock=True, force=True)
    assert result.ok is True
    assert result.steps.get("ingest_sources", {}).get("ok") is True

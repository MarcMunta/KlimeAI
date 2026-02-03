from __future__ import annotations

from pathlib import Path

from c3rnt2.autopilot import run_autopilot_tick


def test_autopilot_once_no_web_mock(tmp_path: Path) -> None:
    settings = {
        "_profile": "safe_selftrain_4080_hf",
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "knowledge": {"embedding_backend": "hash"},
    }
    result = run_autopilot_tick(settings, tmp_path, no_web=True, mock=True)
    assert result.ok is True
    assert "ingest_sources" in result.steps

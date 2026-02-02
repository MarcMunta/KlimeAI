from __future__ import annotations

import os
from pathlib import Path

from c3rnt2.agent.runner import run_agent, Action


def test_agent_runner_dry(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_dummy.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    settings = {"tools": {"web": {"enabled": False, "allow_domains": []}}, "agent": {"web_allowlist": []}}

    calls = {"count": 0}

    def provider(_messages):
        if calls["count"] == 0:
            calls["count"] += 1
            return Action(type="run_tests", args={})
        return Action(type="finish", args={"summary": "done"})

    report = run_agent("Run tests", settings, tmp_path, max_iters=2, action_provider=provider)
    assert report["ok"]
    episodes = tmp_path / "data" / "episodes" / "agent.jsonl"
    assert episodes.exists()

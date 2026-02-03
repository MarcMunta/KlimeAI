from __future__ import annotations

import subprocess
from pathlib import Path

from c3rnt2 import autopilot as ap


def test_autopatch_subprocess_cpu_sets_cuda_visible_devices(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def _fake_run(cmd, cwd=None, check=False, capture_output=False, text=False, env=None):
        _ = cwd, check, capture_output, text
        seen["cmd"] = cmd
        seen["env"] = dict(env or {})
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok": true, "promoted": false, "skipped": "not_triggered"}\n', stderr="")

    monkeypatch.setattr(ap.subprocess, "run", _fake_run)

    settings = {
        "_profile": "autonomous_4080_hf",
        "autopilot": {"autopatch_enabled": True, "autopatch_strategy": "subprocess_cpu"},
        "self_patch": {"enabled": True},
    }
    res = ap._maybe_autopatch(tmp_path, settings, eval_short=None, profile="autonomous_4080_hf", mock=False)
    assert res.get("ok") is True
    assert seen["env"].get("CUDA_VISIBLE_DEVICES") == ""
    assert "autopatch-once" in " ".join(seen["cmd"])

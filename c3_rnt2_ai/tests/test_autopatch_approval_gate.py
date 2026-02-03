from __future__ import annotations

from pathlib import Path

from c3rnt2 import autopilot as ap


def test_autopatch_requires_approval_file(tmp_path: Path, monkeypatch) -> None:
    # Force triggers without actually running subprocesses.
    monkeypatch.setattr(ap, "_run_cmd", lambda *_args, **_kwargs: (False, "fail"))
    monkeypatch.setattr(ap, "_scan_todo_priority", lambda *_args, **_kwargs: 0)

    settings = {
        "_profile": "autonomous_4080_hf",
        "autopilot": {
            "autopatch_enabled": True,
            "autopatch_on_test_fail": True,
            "autopatch_on_doctor_fail": False,
            "autopatch_require_approval": True,
            "approval_file": "data/APPROVE_AUTOPATCH",
        },
        "self_patch": {"enabled": True, "allowed_paths": ["src/", "tests/"]},
    }

    res = ap._maybe_autopatch(tmp_path, settings, eval_short=None, profile="autonomous_4080_hf", mock=False)
    assert res.get("ok") is True
    assert res.get("skipped") == "approval_required"
    assert "tests_failed" in (res.get("triggers") or [])
    assert Path(res.get("approval_file")).name == "APPROVE_AUTOPATCH"


from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from c3rnt2 import __main__ as main_mod


def test_cmd_doctor_exits_on_deep_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def _fake_load_and_validate(_profile, override=None):
        settings = {
            "_profile": "p",
            "agent": {"tools_enabled": list(main_mod._supported_agent_tools())},
            "tools": {"web": {"enabled": False}},
            "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
            "tokenizer": {"vortex_tok_path": str(tmp_path / "tok.pt")},
            "runtime": {"cache_vram_budget_mb": 1},
            "core": {"backend": "hf", "hf_model": "Qwen/Qwen2.5-7B-Instruct", "hf_device": "cpu"},
            "decode": {"max_new_tokens": 8},
        }
        return override(settings) if override else settings

    monkeypatch.setattr(main_mod, "_load_and_validate", _fake_load_and_validate)
    monkeypatch.setattr(main_mod, "_run_doctor_checks", lambda *_args, **_kwargs: {"ok": True, "errors": [], "info": {}})
    monkeypatch.setattr(main_mod, "run_deep_checks", lambda *_args, **_kwargs: {"deep_ok": False, "checks": {}})

    args = SimpleNamespace(profile=None, deep=True)
    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_doctor(args)
    assert int(exc.value.code) == 1


from __future__ import annotations

import importlib.util

from c3rnt2 import doctor as doctor_mod


def test_external_engine_doctor_fails_closed_when_pkg_missing(monkeypatch) -> None:
    real_find = importlib.util.find_spec

    def _fake_find(name: str):  # type: ignore[no-untyped-def]
        if name in {"vllm", "sglang"}:
            return None
        return real_find(name)

    monkeypatch.setattr(doctor_mod.importlib.util, "find_spec", _fake_find)

    settings = {
        "_profile": "p",
        "core": {"backend": "external", "external_engine": "sglang", "external_base_url": "http://127.0.0.1:30000"},
    }
    out = doctor_mod._external_engine_deep_check(settings)
    assert out["ok"] is False
    assert out["error"] == "external_engine_not_installed"
    assert "install" in out


from __future__ import annotations

from pathlib import Path

import pytest


def _base_setup(monkeypatch, start_calls: list[str]):
    pytest.importorskip("fastapi")
    from c3rnt2 import server as server_mod

    class DummyModel:
        def __init__(self):
            self.tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return "ok"

        def stream_generate(self, _prompt: str, **_kwargs):
            yield "ok"

    dummy = DummyModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    def _fake_start_autolearn_background(*_args, **_kwargs):
        start_calls.append("started")
        return object()

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)
    monkeypatch.setattr(
        "c3rnt2.autolearn.start_autolearn_background", _fake_start_autolearn_background
    )
    return server_mod


def test_server_does_not_start_autolearn_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    start_calls: list[str] = []
    server_mod = _base_setup(monkeypatch, start_calls)
    settings = {
        "core": {"backend": "vortex", "hf_system_prompt": "SYS"},
        "rag": {"enabled": False},
        "autolearn": {"enabled": False},
    }
    server_mod.create_app(settings, base_dir=tmp_path)
    assert start_calls == []


def test_server_starts_autolearn_when_enabled(monkeypatch, tmp_path: Path) -> None:
    start_calls: list[str] = []
    server_mod = _base_setup(monkeypatch, start_calls)
    settings = {
        "core": {"backend": "vortex", "hf_system_prompt": "SYS"},
        "rag": {"enabled": False},
        "autolearn": {"enabled": True},
    }
    server_mod.create_app(settings, base_dir=tmp_path)
    assert start_calls == ["started"]

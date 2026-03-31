from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pylint: disable=import-error,no-name-in-module

import importlib
import json
from pathlib import Path

import pytest


class _ExternalFailModel:
    is_external = True

    def generate(self, _prompt: str, **_kwargs):
        raise RuntimeError("external_engine_circuit_open")

    def stream_generate(self, _prompt: str, **_kwargs):
        raise RuntimeError("external_engine_circuit_open")


class _FallbackModel:
    is_hf = True

    def generate(self, _prompt: str, **_kwargs):
        return "fallback-ok"

    def stream_generate(self, _prompt: str, **_kwargs):
        yield "fallback-ok"


@pytest.mark.usefixtures("monkeypatch")
def test_chat_non_stream_falls_back_from_external(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    server_mod = importlib.import_module("c3rnt2.server")

    external = _ExternalFailModel()
    fallback = _FallbackModel()

    def _fake_load_backend_model(_settings, _base_dir, backend):  # type: ignore[no-untyped-def]
        b = str(backend)
        if b == "external":
            return external
        if b == "hf":
            return fallback
        return fallback

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    settings = {
        "core": {
            "backend": "external",
            "external_engine": "ollama",
            "external_base_url": "http://127.0.0.1:11434",
            "backend_fallback": "hf",
            "hf_system_prompt": "SYS",
        },
        "rag": {"enabled": False},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "external", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("choices")[0]["message"]["content"] == "fallback-ok"
    assert data.get("model") == "hf"


@pytest.mark.usefixtures("monkeypatch")
def test_chat_stream_falls_back_from_external(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    server_mod = importlib.import_module("c3rnt2.server")

    external = _ExternalFailModel()
    fallback = _FallbackModel()

    def _fake_load_backend_model(_settings, _base_dir, backend):  # type: ignore[no-untyped-def]
        b = str(backend)
        if b == "external":
            return external
        if b == "hf":
            return fallback
        return fallback

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    settings = {
        "core": {
            "backend": "external",
            "external_engine": "ollama",
            "external_base_url": "http://127.0.0.1:11434",
            "backend_fallback": "hf",
            "hf_system_prompt": "SYS",
        },
        "rag": {"enabled": False},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "external", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200

    deltas: list[str] = []
    done_evt = None
    for raw in resp.text.splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        evt = json.loads(payload)
        choice0 = (evt.get("choices") or [{}])[0]
        delta = (choice0.get("delta") or {}).get("content")
        if isinstance(delta, str) and delta:
            deltas.append(delta)
        if choice0.get("finish_reason") == "stop":
            done_evt = evt

    assert "".join(deltas) == "fallback-ok"
    assert isinstance(done_evt, dict)
    assert done_evt.get("model") == "hf"

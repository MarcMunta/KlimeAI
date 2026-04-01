from __future__ import annotations

from pathlib import Path

import pytest


def test_external_model_name_routes_to_external_backend(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class ExternalDummyModel:
        is_external = True

        def __init__(self) -> None:
            self.tokenizer = None
            self.last_prompt = None
            self.last_messages = None

        def generate(self, prompt: str, **kwargs):
            self.last_prompt = prompt
            self.last_messages = kwargs.get("messages")
            return "external-ok"

    external = ExternalDummyModel()

    def _fake_load_backend_model(_settings, _base_dir, backend):
        if str(backend).lower() == "external":
            return external
        raise AssertionError(f"unexpected backend load: {backend}")

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    settings = {
        "core": {
            "backend": "external",
            "external_engine": "sglang",
            "external_base_url": "http://127.0.0.1:30000",
            "external_model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
            "hf_system_prompt": "SYS",
        },
        "rag": {"enabled": False},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
            "messages": [{"role": "user", "content": "Hola"}],
            "max_tokens": 8,
        },
    )

    assert resp.status_code == 200
    assert resp.headers.get("X-Vortex-Backend") == "external"
    assert external.last_messages == [
        {"role": "user", "content": "Hola"},
    ]
    assert "Q: Hola" in str(external.last_prompt)

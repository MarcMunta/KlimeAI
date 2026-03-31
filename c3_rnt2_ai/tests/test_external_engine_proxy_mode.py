from __future__ import annotations

from c3rnt2 import doctor as doctor_mod


def test_external_engine_doctor_checks_sglang_endpoint(monkeypatch) -> None:
    class DummyResponse:
        status_code = 200

        def json(self):  # pragma: no cover - defensive
            return {"data": [{"id": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"}]}

    def _fake_get(url: str, timeout: float = 0.0):  # type: ignore[no-untyped-def]
        assert url == "http://127.0.0.1:30000/v1/models"
        assert timeout == 2.0
        return DummyResponse()

    monkeypatch.setattr("requests.get", _fake_get)
    settings = {
        "_profile": "p",
        "core": {
            "backend": "external",
            "external_engine": "sglang",
            "external_base_url": "http://127.0.0.1:30000",
            "external_model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        },
    }
    out = doctor_mod._external_engine_deep_check(settings)
    assert out["ok"] is True
    assert out["engine"] == "sglang"
    assert out["model"] == "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"


def test_external_engine_doctor_supports_ollama(monkeypatch) -> None:
    class DummyResponse:
        status_code = 200

        def json(self):  # pragma: no cover - defensive
            return {"models": []}

    def _fake_get(url: str, timeout: float = 0.0):  # type: ignore[no-untyped-def]
        assert url == "http://127.0.0.1:11434/api/tags"
        assert timeout == 2.0
        return DummyResponse()

    monkeypatch.setattr("requests.get", _fake_get)

    settings = {
        "_profile": "p",
        "core": {
            "backend": "external",
            "external_engine": "ollama",
            "external_base_url": "http://127.0.0.1:11434",
        },
    }
    out = doctor_mod._external_engine_deep_check(settings)
    assert out["ok"] is True
    assert out["engine"] == "ollama"

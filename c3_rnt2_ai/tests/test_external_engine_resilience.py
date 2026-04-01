from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pylint: disable=import-error,no-name-in-module

from typing import Any, cast
import requests

from c3rnt2.external_engine import ExternalEngineConfig, ExternalEngineModel, load_external_engine_model


class _DummyResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = int(status_code)
        self._payload = payload or {"choices": [{"message": {"content": "ok"}}]}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"status={self.status_code}", response=cast(Any, self)
            )

    def json(self) -> dict:
        return dict(self._payload)

    def iter_lines(self, decode_unicode: bool = False):
        _ = decode_unicode
        yield "data: [DONE]"

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        return None


def test_external_engine_generate_retries_on_timeout_then_succeeds(monkeypatch) -> None:
    cfg = ExternalEngineConfig(
        engine="ollama",
        base_url="http://127.0.0.1:11434",
        retry_max_attempts=2,
        retry_backoff_base_s=0.0,
        retry_backoff_max_s=0.0,
        circuit_enabled=False,
    )
    model = ExternalEngineModel(cfg)

    calls = {"n": 0}

    def _fake_post(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.Timeout("timeout")
        return _DummyResponse(status_code=200, payload={"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr(model.session, "post", _fake_post)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    out = model.generate("ping", max_new_tokens=8)
    assert out == "hello"
    assert calls["n"] == 2

    stats = model.runtime_stats()
    assert stats["requests_succeeded"] == 1
    assert stats["retries_total"] == 1


def test_external_engine_generate_circuit_breaker_opens(monkeypatch) -> None:
    cfg = ExternalEngineConfig(
        engine="ollama",
        base_url="http://127.0.0.1:11434",
        retry_max_attempts=1,
        circuit_enabled=True,
        circuit_fail_threshold=2,
        circuit_open_s=60.0,
        retry_backoff_base_s=0.0,
        retry_backoff_max_s=0.0,
    )
    model = ExternalEngineModel(cfg)

    def _always_fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise requests.ConnectionError("conn down")

    monkeypatch.setattr(model.session, "post", _always_fail)

    for _ in range(2):
        try:
            model.generate("ping", max_new_tokens=8)
        except requests.ConnectionError:
            pass

    try:
        model.generate("ping", max_new_tokens=8)
        assert False, "expected circuit open error"
    except RuntimeError as exc:
        assert "external_engine_circuit_open" in str(exc)

    stats = model.runtime_stats()
    assert stats["circuit_open"] is True
    assert stats["consecutive_failures"] >= 2


def test_external_engine_generate_retries_on_retryable_http_status(monkeypatch) -> None:
    cfg = ExternalEngineConfig(
        engine="ollama",
        base_url="http://127.0.0.1:11434",
        retry_max_attempts=2,
        retry_backoff_base_s=0.0,
        retry_backoff_max_s=0.0,
        circuit_enabled=False,
    )
    model = ExternalEngineModel(cfg)

    calls = {"n": 0}

    def _fake_post(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            return _DummyResponse(status_code=503)
        return _DummyResponse(status_code=200, payload={"choices": [{"message": {"content": "world"}}]})

    monkeypatch.setattr(model.session, "post", _fake_post)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    out = model.generate("ping", max_new_tokens=8)
    assert out == "world"
    assert calls["n"] == 2
    assert model.runtime_stats()["retries_total"] == 1


def test_external_engine_generate_prefers_chat_messages(monkeypatch) -> None:
    cfg = ExternalEngineConfig(
        engine="sglang",
        base_url="http://127.0.0.1:30000",
        circuit_enabled=False,
    )
    model = ExternalEngineModel(cfg)

    captured: dict[str, Any] = {}

    def _fake_post(*_args, **kwargs):  # type: ignore[no-untyped-def]
        captured["json"] = kwargs.get("json")
        return _DummyResponse(status_code=200, payload={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(model.session, "post", _fake_post)

    out = model.generate(
        "Q: flat prompt",
        messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Hola"},
        ],
    )

    assert out == "ok"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Hola"},
    ]


def test_load_external_engine_model_parses_resilience_config() -> None:
    settings = {
        "core": {
            "backend": "external",
            "external_engine": "ollama",
            "external_base_url": "http://127.0.0.1:11434",
            "external_retry_max_attempts": 3,
            "external_retry_backoff_base_s": 0.1,
            "external_retry_backoff_max_s": 1.5,
            "external_retry_on_statuses": [408, 429, 500],
            "external_circuit_enabled": True,
            "external_circuit_fail_threshold": 5,
            "external_circuit_open_s": 12,
        }
    }
    model = load_external_engine_model(settings)
    assert model.cfg.retry_max_attempts == 3
    assert model.cfg.retry_backoff_base_s == 0.1
    assert model.cfg.retry_backoff_max_s == 1.5
    assert model.cfg.retry_on_statuses == (408, 429, 500)
    assert model.cfg.circuit_enabled is True
    assert model.cfg.circuit_fail_threshold == 5
    assert model.cfg.circuit_open_s == 12.0

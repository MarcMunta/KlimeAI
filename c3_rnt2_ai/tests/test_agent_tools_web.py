from __future__ import annotations

from pathlib import Path

import c3rnt2.agent.tools as tools_mod
from c3rnt2.agent.tools import AgentTools


class _DummyFetch:
    def __init__(self, ok: bool, text: str, error: str | None = None):
        self.ok = ok
        self.text = text
        self.error = error


def _tools(tmp_path: Path) -> AgentTools:
    return AgentTools(
        allowlist=[],
        web_cfg={"enabled": True, "allow_domains": ["duckduckgo.com"]},
        cache_root=tmp_path,
    )


def test_search_web_encodes_query(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_fetch(url: str, **_kwargs):
        captured["url"] = url
        return _DummyFetch(ok=True, text='<a class="result__a" href="https://example.com">Title</a>')

    monkeypatch.setattr(tools_mod, "web_fetch", fake_fetch)
    tools = _tools(tmp_path)
    result = tools.search_web("hello world")
    assert result.ok
    assert "hello+world" in captured.get("url", "")
    assert " " not in captured.get("url", "")


def test_search_web_disabled(tmp_path: Path) -> None:
    tools = AgentTools(allowlist=[], web_cfg={"enabled": False}, cache_root=tmp_path)
    result = tools.search_web("test")
    assert not result.ok
    assert result.output == "web disabled"


def test_search_web_parses_results(monkeypatch, tmp_path: Path) -> None:
    def fake_fetch(url: str, **_kwargs):
        _ = url
        html = '<a class="result__a" href="https://example.com">Example Site</a>'
        return _DummyFetch(ok=True, text=html)

    monkeypatch.setattr(tools_mod, "web_fetch", fake_fetch)
    tools = _tools(tmp_path)
    result = tools.search_web("example")
    assert result.ok
    assert result.output.strip() == "Example Site - https://example.com"


def test_search_web_fallback_anchor(monkeypatch, tmp_path: Path) -> None:
    def fake_fetch(url: str, **_kwargs):
        _ = url
        html = '<a href="https://example.com">Example</a>'
        return _DummyFetch(ok=True, text=html)

    monkeypatch.setattr(tools_mod, "web_fetch", fake_fetch)
    tools = _tools(tmp_path)
    result = tools.search_web("example")
    assert result.ok
    assert result.output.strip() == "Example - https://example.com"

from __future__ import annotations

from c3rnt2.config import load_settings


def test_safe_selftrain_4080_hf_autopilot_is_top_level() -> None:
    settings = load_settings("safe_selftrain_4080_hf")
    assert "autopilot" in settings
    tools = settings.get("tools", {}) or {}
    assert "autopilot" not in tools
    server = settings.get("server", {}) or {}
    assert server.get("auto_reload_adapter") is True
    assert int(server.get("reload_interval_s")) == 30

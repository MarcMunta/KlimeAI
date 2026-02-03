from __future__ import annotations

from c3rnt2.config import load_settings


def test_profile_autonomous_4080_hf_loads() -> None:
    settings = load_settings("autonomous_4080_hf")
    assert (settings.get("autopilot", {}) or {}).get("enabled") is True
    assert (settings.get("autopilot", {}) or {}).get("autopatch_enabled") is True
    assert (settings.get("autopilot", {}) or {}).get("restart_after_patch") is True
    assert (settings.get("self_patch", {}) or {}).get("enabled") is True
    assert ((settings.get("continuous", {}) or {}).get("web_discovery", {}) or {}).get("enabled") is True


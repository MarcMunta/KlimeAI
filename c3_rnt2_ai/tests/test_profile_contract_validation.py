from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pylint: disable=import-error,no-name-in-module

from pathlib import Path

import pytest

from c3rnt2.config import load_settings, validate_profile


def test_profile_contract_offline_requires_no_backend_fallback() -> None:
    settings = load_settings("rtx4080_16gb_programming_local")
    settings["core"]["backend_fallback"] = "hf"

    with pytest.raises(ValueError, match="core.backend_fallback=null"):
        validate_profile(settings, base_dir=Path("."))


def test_profile_contract_offline_external_requires_localhost_base_url() -> None:
    settings = load_settings("rtx4080_16gb_programming_local")
    settings["core"]["external_base_url"] = "http://10.1.2.3:11434"

    with pytest.raises(ValueError, match="localhost external_base_url"):
        validate_profile(settings, base_dir=Path("."))


def test_profile_contract_offline_requires_empty_tools_allow_domains() -> None:
    settings = load_settings("rtx4080_16gb_programming_local")
    settings["tools"]["web"]["allow_domains"] = ["docs.python.org"]

    with pytest.raises(ValueError, match=r"tools\.web\.allow_domains=\[\]"):
        validate_profile(settings, base_dir=Path("."))


def test_profile_contract_offline_requires_empty_agent_web_allowlist() -> None:
    settings = load_settings("rtx4080_16gb_programming_local")
    settings["agent"]["web_allowlist"] = ["docs.python.org"]

    with pytest.raises(ValueError, match=r"agent\.web_allowlist=\[\]"):
        validate_profile(settings, base_dir=Path("."))

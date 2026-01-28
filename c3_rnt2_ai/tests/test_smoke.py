from __future__ import annotations

from c3rnt2.config import load_settings
from c3rnt2.device import detect_device


def test_smoke_settings():
    settings = load_settings("dev_small")
    assert "tokenizer" in settings


def test_smoke_device():
    info = detect_device()
    assert info.device

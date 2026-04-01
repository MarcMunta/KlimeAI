from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pylint: disable=import-error,no-name-in-module

from c3rnt2.continuous.trainer import _normalize_sample_weights
from c3rnt2.continuous.types import Sample


def test_normalize_sample_weights_returns_none_when_all_non_positive() -> None:
    samples = [
        Sample(prompt="p1", response="r1", source_kind="episode"),
        Sample(prompt="p2", response="r2", source_kind="logs"),
    ]
    weights = _normalize_sample_weights(samples, {"episode": 0.0, "logs": -3.0})
    assert weights is None


def test_normalize_sample_weights_parses_invalid_values_safely() -> None:
    samples = [
        Sample(prompt="p1", response="r1", source_kind="episode"),
        Sample(prompt="p2", response="r2", source_kind="unknown"),
    ]
    weights = _normalize_sample_weights(samples, {"episode": "bad-number", "unknown": 2.0})
    assert weights is not None
    assert weights[0] == 1.0
    assert weights[1] == 2.0

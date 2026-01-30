from __future__ import annotations

import json
from pathlib import Path

import pytest

_ = pytest.importorskip("torch")

from c3rnt2.continuous.bootstrap import _hash_messages, _save_dataset
from c3rnt2.continuous.types import Sample


def test_bootstrap_dataset_format(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "registry" / "bootstrap" / "bootstrap_samples.jsonl"
    sample = Sample(prompt="Hi", response="Hello")
    meta = {
        "teacher": "Qwen/Qwen2.5-8B-Instruct",
        "quant": "4bit",
        "seed": 123,
        "profile": "qwen8b_base",
        "ts": 1.23,
        "params": {},
    }
    _save_dataset(dataset_path, [sample], meta, default_system="System")

    payload = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["messages"][0]["role"] == "system"
    assert payload["prompt_hash"] == _hash_messages(payload["messages"])
    assert payload["teacher"] == "Qwen/Qwen2.5-8B-Instruct"
    assert payload["quant"] == "4bit"
    assert payload["seed"] == 123
    assert payload["profile"] == "qwen8b_base"
    assert "ts" in payload

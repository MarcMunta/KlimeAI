from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.learning_loop.data_collector import collect_from_episodes
from c3rnt2.learning_loop.data_curator import curate_dataset


def test_collect_and_curate(tmp_path: Path):
    base_dir = tmp_path
    episodes = base_dir / "data" / "episodes" / "agent.jsonl"
    episodes.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "Fix add",
        "prompt": "def add(a, b):",
        "patch": "return a + b",
        "tests_ok": True,
    }
    episodes.write_text(json.dumps(payload), encoding="utf-8")

    settings = {
        "learning": {
            "raw_path": "data/learning/raw.jsonl",
            "curated_path": "data/learning/curated.jsonl",
            "state_path": "data/learning/state.sqlite",
            "min_chars": 5,
        }
    }
    first = collect_from_episodes(base_dir, settings, max_events=10)
    assert first.ok
    assert first.added == 1

    second = collect_from_episodes(base_dir, settings, max_events=10)
    assert second.ok
    assert second.added == 0

    curated = curate_dataset(base_dir, settings)
    assert curated.ok
    assert curated.output_path.exists()
    lines = curated.output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

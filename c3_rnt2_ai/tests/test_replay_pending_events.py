from __future__ import annotations

import sqlite3
from pathlib import Path

from c3rnt2.continuous.replay_buffer import ReplayBuffer, ReplayItem
from c3rnt2.continuous.types import Sample


def _success_count(db_path: Path, digest: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT success_count FROM replay WHERE hash = ?", (digest,)).fetchone()
        return int(row[0]) if row else 0


def test_replay_pending_event_applied_on_add(tmp_path: Path) -> None:
    db_path = tmp_path / "replay.sqlite"
    replay = ReplayBuffer(db_path)
    sample = Sample(prompt="p", response="r")
    digest = replay.hash_sample(sample.prompt, sample.response)

    assert replay.bump_success_once(digest, "event-1", delta=1) is False
    assert _success_count(db_path, digest) == 0

    item = ReplayItem(sample=sample, source_kind="episode", quality_score=0.9, novelty_score=0.9, success_count=0)
    replay.add(item)
    assert _success_count(db_path, digest) == 1

    # duplicate event should not count twice
    assert replay.bump_success_once(digest, "event-1", delta=1) is False
    assert _success_count(db_path, digest) == 1

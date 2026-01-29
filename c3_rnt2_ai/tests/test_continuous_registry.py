from __future__ import annotations

from pathlib import Path

from c3rnt2.continuous.registry import begin_run, finalize_run, load_registry, rollback


def test_continuous_registry_rollbacks(tmp_path: Path):
    run1, _ = begin_run(tmp_path)
    finalize_run(tmp_path, run1, promote=True, meta={"loss": 1.0})
    run2, _ = begin_run(tmp_path)
    finalize_run(tmp_path, run2, promote=True, meta={"loss": 0.5})

    state = load_registry(tmp_path)
    assert state.current_run_id == run2
    prev = rollback(tmp_path)
    assert prev == run1
    state = load_registry(tmp_path)
    assert state.current_run_id == run1

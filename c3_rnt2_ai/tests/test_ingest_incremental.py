from __future__ import annotations

from pathlib import Path

from c3rnt2.continuous.dataset import ingest_sources


def test_ingest_incremental_logs_and_episodes(tmp_path: Path) -> None:
    base_dir = tmp_path
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "sample.log"
    log_path.write_text("hello log\n", encoding="utf-8")

    episodes_dir = data_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = episodes_dir / "agent.jsonl"
    episodes_path.write_text('{"task":"t","prompt":"ctx","patch":"diff","tests_ok":true}\n', encoding="utf-8")

    settings = {
        "continuous": {
            "knowledge_path": str(data_dir / "continuous" / "knowledge.sqlite"),
            "ingest_web": False,
            "ingest": {
                "max_files_per_tick": 10,
                "max_bytes_per_file": 1024 * 1024,
                "max_total_bytes_per_tick": 1024 * 1024,
                "web": {"cooldown_minutes": 60},
            },
        },
        "rag": {"enabled": False},
    }

    first = ingest_sources(base_dir, allowlist=[], settings=settings)
    second = ingest_sources(base_dir, allowlist=[], settings=settings)
    assert first > 0
    assert second == 0

    log_path.write_text("hello log\nmore\n", encoding="utf-8")
    with episodes_path.open("a", encoding="utf-8") as handle:
        handle.write('{"task":"t2","prompt":"ctx2","patch":"diff2","tests_ok":true}\n')

    third = ingest_sources(base_dir, allowlist=[], settings=settings)
    assert third > 0

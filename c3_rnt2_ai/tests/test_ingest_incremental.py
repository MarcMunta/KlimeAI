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


def test_ingest_local_sources_excludes_web_cache_and_noise(tmp_path: Path) -> None:
    base_dir = tmp_path
    (base_dir / "src").mkdir(parents=True, exist_ok=True)
    (base_dir / "src" / "main.py").write_text("print('repo file')\n", encoding="utf-8")
    (base_dir / "data" / "corpora" / "programming" / "python").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "corpora" / "programming" / "python" / "tips.md").write_text("Python tips\n", encoding="utf-8")
    (base_dir / "data" / "web_cache").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "web_cache" / "cached.txt").write_text("do not ingest\n", encoding="utf-8")
    (base_dir / "data" / "sample.log").write_text("ignore logs\n", encoding="utf-8")

    settings = {
        "continuous": {
            "knowledge_path": str(base_dir / "data" / "continuous" / "knowledge.sqlite"),
            "ingest_web": False,
            "local_sources": {
                "enabled": True,
                "include_repo": True,
                "include_local_corpus": True,
                "include_lessons": False,
                "include_logs": False,
                "include_memory": False,
                "repo_paths": ["src"],
                "corpus_paths": ["data/corpora/programming"],
                "lesson_paths": [],
                "include_globs": ["*.py", "*.md", "*.txt"],
                "exclude_globs": ["data/web_cache/**"],
            },
            "ingest": {
                "max_files_per_tick": 20,
                "max_bytes_per_file": 1024 * 1024,
                "max_total_bytes_per_tick": 1024 * 1024,
                "web": {"cooldown_minutes": 60},
            },
        },
        "rag": {"enabled": False},
    }

    first = ingest_sources(base_dir, allowlist=[], settings=settings)
    second = ingest_sources(base_dir, allowlist=[], settings=settings)
    assert first >= 2
    assert second == 0

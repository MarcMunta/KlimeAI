from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pylint: disable=import-error,no-name-in-module

from pathlib import Path

from c3rnt2.continuous import dataset as dataset_mod
from c3rnt2.continuous.knowledge_store import KnowledgeStore


def test_collect_samples_filters_low_novelty_items(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path
    cont_dir = base_dir / "data" / "continuous"
    cont_dir.mkdir(parents=True, exist_ok=True)

    knowledge_path = cont_dir / "knowledge.sqlite"
    replay_path = cont_dir / "replay.sqlite"

    store = KnowledgeStore(knowledge_path, embedding_backend="hash", index_backend="none")
    store.ingest_text("memory", "unit", "Deterministic chunk for novelty filtering.", quality=0.9)

    settings = {
        "continuous": {
            "knowledge_path": str(knowledge_path),
            "ingest_web": False,
            "filter": {"min_quality": 0.0, "min_novelty": 0.9, "max_repeat_ratio": 1.0},
            "replay": {"path": str(replay_path), "seed_chunks": 1, "sample_size": 4},
        },
        "knowledge": {"embedding_backend": "hash", "index_backend": "none"},
        "rag": {"enabled": False},
    }

    monkeypatch.setattr(dataset_mod, "_novelty_score_from_vec", lambda *_args, **_kwargs: 0.0)
    collected = dataset_mod.collect_samples(base_dir, allowlist=[], settings=settings, ingest=False)

    assert collected.samples == []
    assert collected.stats.filtered >= 1


def test_collect_samples_filters_semantic_duplicates_in_same_batch(
    tmp_path: Path, monkeypatch
) -> None:
    base_dir = tmp_path
    cont_dir = base_dir / "data" / "continuous"
    cont_dir.mkdir(parents=True, exist_ok=True)

    knowledge_path = cont_dir / "knowledge.sqlite"
    replay_path = cont_dir / "replay.sqlite"

    store = KnowledgeStore(knowledge_path, embedding_backend="hash", index_backend="none")
    store.ingest_text("docs", "a", "Local duplicate sample A.", quality=0.95)
    store.ingest_text("docs", "b", "Local duplicate sample B.", quality=0.95)

    settings = {
        "continuous": {
            "knowledge_path": str(knowledge_path),
            "ingest_web": False,
            "filter": {
                "min_quality": 0.0,
                "min_novelty": 0.0,
                "max_repeat_ratio": 1.0,
                "semantic_dedup_threshold": 0.9,
            },
            "replay": {"path": str(replay_path), "seed_chunks": 2, "sample_size": 8},
        },
        "knowledge": {"embedding_backend": "hash", "index_backend": "none"},
        "rag": {"enabled": False},
    }

    monkeypatch.setattr(dataset_mod, "embed_text", lambda _text: [1.0, 0.0])
    collected = dataset_mod.collect_samples(base_dir, allowlist=[], settings=settings, ingest=False)

    assert len(collected.samples) == 1
    assert collected.stats.filtered >= 1

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..continuous.knowledge_store import KnowledgeChunk, EmbeddingBackend, embed_text
from ..continuous.types import Sample


@dataclass
class BuildStats:
    total: int
    written: int
    skipped: int
    deduped: int
    semantic_deduped: int


def _hash_sample(sample: Sample) -> str:
    payload = json.dumps(sample.messages or [], ensure_ascii=True) + "\n" + (sample.response or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repeat_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0
    unique = len(set(words))
    return 1.0 - (unique / max(1, len(words)))


def _cosine(a: List[float], b: List[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denom <= 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def _embed_texts(texts: List[str], backend: EmbeddingBackend | None) -> List[List[float]]:
    if backend is None:
        return [embed_text(text) for text in texts]
    return backend.encode(texts)


def _doc_examples(chunks: Iterable[KnowledgeChunk], system: str) -> List[Sample]:
    samples: List[Sample] = []
    for chunk in chunks:
        user = f"Read the following documentation excerpt and answer clearly:\n\n{chunk.text}\n\nQuestion: Summarize the key points."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        samples.append(Sample(prompt="", response=chunk.text, source_kind=chunk.source_kind, messages=messages))
    return samples


def _episode_examples(episodes_path: Path, system: str) -> List[Sample]:
    if not episodes_path.exists():
        return []
    samples: List[Sample] = []
    for line in episodes_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        task = str(payload.get("task", "")).strip()
        diff = str(payload.get("patch", "")).strip()
        context = str(payload.get("prompt", "")).strip()
        if not diff or not task:
            continue
        user = f"Task: {task}"
        if context:
            user = f"{user}\n\nContext:\n{context}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        samples.append(Sample(prompt="", response=diff, source_kind="episode", messages=messages))
    return samples


def build_sft_dataset(
    *,
    chunks: Iterable[KnowledgeChunk],
    episodes_path: Path,
    output_path: Path,
    system_prompt: str,
    min_chars: int = 40,
    max_repeat_ratio: float = 0.8,
    semantic_dedup_threshold: float = 0.97,
    embedding_backend: EmbeddingBackend | None = None,
) -> BuildStats:
    examples = _doc_examples(chunks, system_prompt) + _episode_examples(episodes_path, system_prompt)
    seen_hashes: set[str] = set()
    seen_vecs: List[List[float]] = []
    written = 0
    skipped = 0
    deduped = 0
    semantic_deduped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in examples:
            text = (sample.response or "").strip()
            if len(text) < min_chars:
                skipped += 1
                continue
            if _repeat_ratio(text) > max_repeat_ratio:
                skipped += 1
                continue
            if text.startswith("{") or text.startswith("["):
                try:
                    json.loads(text)
                except Exception:
                    skipped += 1
                    continue
            digest = _hash_sample(sample)
            if digest in seen_hashes:
                deduped += 1
                continue
            combined = (sample.messages or []) + [{"role": "assistant", "content": text}]
            combined_text = json.dumps(combined, ensure_ascii=True)
            vec = _embed_texts([combined_text], embedding_backend)[0]
            if seen_vecs:
                sims = [_cosine(vec, prev) for prev in seen_vecs]
                if sims and max(sims) >= semantic_dedup_threshold:
                    semantic_deduped += 1
                    continue
            seen_hashes.add(digest)
            seen_vecs.append(vec)
            payload = {
                "messages": sample.messages,
                "prompt": sample.prompt,
                "response": sample.response,
                "source_kind": sample.source_kind,
                "source_ref": getattr(sample, "source_ref", None),
                "ts": time.time(),
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            written += 1
    return BuildStats(
        total=len(examples),
        written=written,
        skipped=skipped,
        deduped=deduped,
        semantic_deduped=semantic_deduped,
    )

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


def embed_text(text: str, dim: int = 128) -> List[float]:
    vec = [0.0] * dim
    for tok in _tokenize(text):
        idx = hash(tok) % dim
        vec[idx] += 1.0
    return vec


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class MemoryItem:
    text: str
    score: float


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, text TEXT, vec TEXT)"
            )
            conn.commit()

    def add(self, text: str) -> None:
        vec = embed_text(text)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO memory (text, vec) VALUES (?, ?)", (text, json.dumps(vec)))
            conn.commit()

    def query(self, text: str, top_k: int = 5) -> List[MemoryItem]:
        qvec = embed_text(text)
        items: List[MemoryItem] = []
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT text, vec FROM memory")
            for row in cur.fetchall():
                vec = json.loads(row[1])
                score = _dot(qvec, vec)
                items.append(MemoryItem(text=row[0], score=score))
        items.sort(key=lambda x: x.score, reverse=True)
        return items[:top_k]

    def summarize(self, text: str) -> str:
        # MVP: naive summary.
        return text[:200] + ("..." if len(text) > 200 else "")

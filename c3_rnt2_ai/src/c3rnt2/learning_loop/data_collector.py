from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class CollectResult:
    ok: bool
    added: int
    skipped: int
    total: int
    output_path: Path
    error: str | None = None


class CollectorState:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, ts REAL)")
            conn.commit()

    def seen(self, digest: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("SELECT hash FROM seen WHERE hash = ?", (digest,))
            return cur.fetchone() is not None

    def add(self, digest: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO seen (hash, ts) VALUES (?, ?)", (digest, time.time()))
            conn.commit()


def _hash_record(prompt: str, response: str, source: str) -> str:
    payload = f"{source}\n{prompt}\n{response}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def _iter_episode_records(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            yield payload


def collect_from_episodes(base_dir: Path, settings: dict, max_events: int | None = None) -> CollectResult:
    learning = settings.get("learning", {}) or {}
    output_path = Path(learning.get("raw_path", base_dir / "data" / "learning" / "raw.jsonl"))
    state_path = Path(learning.get("state_path", base_dir / "data" / "learning" / "state.sqlite"))
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    if not state_path.is_absolute():
        state_path = base_dir / state_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = CollectorState(state_path)
    episodes_path = base_dir / "data" / "episodes" / "agent.jsonl"
    added = 0
    skipped = 0
    total = 0
    max_events = int(max_events) if max_events is not None else int(learning.get("max_events", 500))

    try:
        with output_path.open("a", encoding="utf-8") as handle:
            for payload in _iter_episode_records(episodes_path):
                if max_events and total >= max_events:
                    break
                total += 1
                prompt = str(payload.get("prompt", "")).strip()
                response = str(payload.get("patch", payload.get("response", ""))).strip()
                if not prompt or not response:
                    skipped += 1
                    continue
                digest = _hash_record(prompt, response, "episode")
                if state.seen(digest):
                    skipped += 1
                    continue
                record = {
                    "version": 1,
                    "ts": time.time(),
                    "source": "episode",
                    "prompt": prompt,
                    "response": response,
                    "meta": {
                        "tests_ok": bool(payload.get("tests_ok", False)),
                        "tools_ok": bool(payload.get("tools_ok", False)),
                        "task": payload.get("task"),
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                state.add(digest)
                added += 1
        return CollectResult(ok=True, added=added, skipped=skipped, total=total, output_path=output_path)
    except Exception as exc:
        return CollectResult(ok=False, added=added, skipped=skipped, total=total, output_path=output_path, error=str(exc))

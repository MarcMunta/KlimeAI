from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..agent.memory import MemoryStore
from ..agent.tools import AgentTools


@dataclass
class Sample:
    prompt: str
    response: str


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_logs(data_dir: Path) -> Iterable[str]:
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".log", ".txt"}:
            yield path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield json.dumps(payload)
                except Exception:
                    continue


def collect_samples(base_dir: Path, allowlist: List[str]) -> List[Sample]:
    samples: List[Sample] = []
    seen: set[str] = set()

    # Memory store
    memory_path = base_dir / "data" / "memory" / "agent_memory.sqlite"
    if memory_path.exists():
        store = MemoryStore(memory_path)
        for item in store.query("summary", top_k=50):
            key = _hash(item.text)
            if key not in seen:
                samples.append(Sample(prompt="Summarize", response=item.text))
                seen.add(key)

    # Logs
    for text in _load_logs(base_dir / "data"):
        key = _hash(text)
        if key not in seen:
            samples.append(Sample(prompt="Continue", response=text[:1000]))
            seen.add(key)

    # Web docs (best effort)
    tools = AgentTools(allowlist=allowlist)
    for url in ["https://docs.python.org/3/", "https://pytorch.org/docs/stable/"]:
        doc = tools.open_docs(url)
        if doc.ok:
            key = _hash(doc.output)
            if key not in seen:
                samples.append(Sample(prompt="Read docs", response=doc.output[:1000]))
                seen.add(key)

    return samples

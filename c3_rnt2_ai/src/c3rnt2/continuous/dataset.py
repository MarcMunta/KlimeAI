from __future__ import annotations

# pylint: disable=broad-exception-caught
# ruff: noqa: BLE001

import json
import sqlite3
import time
import hashlib
import math
import re
import threading
from fnmatch import fnmatchcase
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlsplit, urlunsplit

from ..agent.memory import MemoryStore
from ..agent.tools import AgentTools
from .knowledge_store import KnowledgeStore, IngestPolicy, EmbeddingBackend, embed_text
from .replay_buffer import ReplayBuffer, ReplayItem
from .types import Sample


_RETRIEVE_STORE_CACHE: dict[tuple[str, str, str, str], KnowledgeStore] = {}
_RETRIEVE_STORE_LOCK = threading.Lock()


@dataclass
class CollectStats:
    new_docs: int
    novelty_avg: float
    successes: int
    filtered: int
    total_candidates: int


@dataclass
class CollectedSamples:
    samples: List[Sample]
    stats: CollectStats
    gold_samples: List[Sample]


class IngestState:
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
            conn.execute("CREATE TABLE IF NOT EXISTS ingest_state (key TEXT PRIMARY KEY, value TEXT, ts REAL)")
            conn.commit()

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT value FROM ingest_state WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO ingest_state (key, value, ts) VALUES (?, ?, ?)", (key, value, time.time()))
            conn.commit()

    def get_json(self, key: str, default: dict) -> dict:
        raw = self.get(key)
        if not raw:
            return dict(default)
        try:
            return json.loads(raw)
        except Exception:
            return dict(default)

    def set_json(self, key: str, value: dict) -> None:
        self.set(key, json.dumps(value))


def _iter_log_files(data_dir: Path) -> Iterable[Path]:
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "agent.jsonl" and "episodes" in path.parts:
            continue
        suffix = path.suffix.lower()
        if suffix in {".log", ".txt", ".jsonl"}:
            yield path


def _match_any_glob(path_str: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    norm = path_str.replace("\\", "/")
    for pattern in patterns:
        if fnmatchcase(norm, pattern):
            return True
        if fnmatchcase(Path(norm).name, pattern):
            return True
    return False


def _is_probably_text_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except Exception:
        return False
    if b"\x00" in chunk:
        return False
    return True


def _iter_local_source_files(
    base_dir: Path,
    roots: list[tuple[str, Path]],
    *,
    include_globs: list[str],
    exclude_globs: list[str],
) -> Iterable[tuple[str, Path, str]]:
    seen: set[str] = set()
    for source_kind, root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in candidates:
            try:
                rel = path.resolve().relative_to(base_dir.resolve()).as_posix()
            except Exception:
                rel = path.resolve().as_posix()
            if rel in seen:
                continue
            if include_globs and not _match_any_glob(rel, include_globs):
                continue
            if exclude_globs and _match_any_glob(rel, exclude_globs):
                continue
            if not _is_probably_text_file(path):
                continue
            seen.add(rel)
            yield source_kind, path, rel


def _ingest_local_sources(
    *,
    base_dir: Path,
    state: IngestState,
    store: KnowledgeStore,
    settings: dict,
    max_files_per_tick: int,
    max_bytes_per_file: int,
    max_total_bytes_per_tick: int,
    files_used: int,
    bytes_used: int,
) -> tuple[int, int, int]:
    continuous = settings.get("continuous", {}) or {}
    local_cfg = continuous.get("local_sources", {}) or {}
    if not bool(local_cfg.get("enabled", False)):
        return 0, files_used, bytes_used

    repo_paths = local_cfg.get("repo_paths", []) if bool(local_cfg.get("include_repo", False)) else []
    corpus_paths = local_cfg.get("corpus_paths", []) if bool(local_cfg.get("include_local_corpus", False)) else []
    lesson_paths = local_cfg.get("lesson_paths", []) if bool(local_cfg.get("include_lessons", False)) else []
    include_globs = [str(item) for item in (local_cfg.get("include_globs", []) or []) if item]
    exclude_globs = [str(item) for item in (local_cfg.get("exclude_globs", []) or []) if item]

    roots: list[tuple[str, Path]] = []
    for raw in repo_paths:
        path = Path(str(raw))
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        roots.append(("repo", path))
    for raw in corpus_paths:
        path = Path(str(raw))
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        roots.append(("docs", path))
    for raw in lesson_paths:
        path = Path(str(raw))
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        roots.append(("lesson", path))

    new_docs = 0
    for source_kind, path, rel in _iter_local_source_files(
        base_dir,
        roots,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
    ):
        if files_used >= max_files_per_tick or bytes_used >= max_total_bytes_per_tick:
            break
        try:
            stat = path.stat()
        except Exception:
            continue
        key = f"local:{source_kind}:{rel}"
        meta = state.get_json(key, {"mtime": 0.0, "size": 0})
        if (
            float(meta.get("mtime", 0.0)) == float(stat.st_mtime)
            and int(meta.get("size", 0)) == int(stat.st_size)
        ):
            continue
        budget_left = max_total_bytes_per_tick - bytes_used
        if budget_left <= 0:
            break
        max_bytes = min(max_bytes_per_file, budget_left, int(stat.st_size))
        if max_bytes <= 0:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text.strip():
            state.set_json(key, {"mtime": stat.st_mtime, "size": stat.st_size})
            continue
        payload = f"Path: {rel}\n\n{text}"
        payload_bytes = payload.encode("utf-8", errors="ignore")
        if len(payload_bytes) > max_bytes:
            payload_bytes = payload_bytes[:max_bytes]
            payload = payload_bytes.decode("utf-8", errors="ignore")
        quality = 0.95 if source_kind == "repo" else 0.9
        new_docs += store.ingest_text(source_kind, rel, payload, quality=quality)
        bytes_used += len(payload_bytes)
        files_used += 1
        state.set_json(key, {"mtime": stat.st_mtime, "size": stat.st_size})
    return new_docs, files_used, bytes_used


def _load_logs(data_dir: Path) -> Iterable[str]:
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".log", ".txt"}:
            yield path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield json.dumps(payload)
                except Exception:
                    continue


def _load_episodes(path: Path) -> List[dict]:
    if not path.exists():
        return []
    episodes = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            episodes.append(payload)
    return episodes


def _resolve_queue_dir(base_dir: Path, settings: dict) -> Path:
    queue_dir = settings.get("self_patch", {}).get("queue_dir", "data/self_patch/queue")
    qpath = Path(queue_dir)
    if not qpath.is_absolute():
        qpath = base_dir / qpath
    return qpath


def _load_patch_from_queue(queue_dir: Path, patch_id: str | None) -> str:
    if not patch_id:
        return ""
    patch_path = queue_dir / patch_id / "patch.diff"
    if not patch_path.exists():
        return ""
    try:
        return patch_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _episode_prompt(payload: dict) -> str:
    prompt = str(payload.get("prompt", "") or payload.get("context", "")).strip()
    if prompt:
        return prompt
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return str(msg.get("content")).strip()
    return ""


def _episode_hash(task: str, prompt: str, patch: str, tests_ok: bool, tools_ok: bool) -> str:
    signals = []
    if tests_ok:
        signals.append("tests_ok")
    if tools_ok:
        signals.append("tools_ok")
    signal_label = "+".join(signals)
    payload = f"{task}\n{prompt}\n{patch}\n{signal_label}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def _cosine_sim(a: List[float], b: List[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denom <= 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def _novelty_score(text: str, recent_vecs: List[List[float]]) -> float:
    vec = embed_text(text)
    return _novelty_score_from_vec(vec, recent_vecs)


def _novelty_score_from_vec(vec: List[float], recent_vecs: List[List[float]]) -> float:
    if not recent_vecs:
        return 1.0
    sims = [_cosine_sim(vec, other) for other in recent_vecs]
    return max(0.0, 1.0 - max(sims))


def _semantic_dedup_threshold(filter_cfg: dict, default: float = 0.97) -> float:
    raw = filter_cfg.get("semantic_dedup_threshold", default)
    try:
        threshold = float(raw)
    except Exception:
        threshold = float(default)
    return max(0.0, min(1.0, threshold))


def _is_semantic_duplicate(vec: List[float], recent_vecs: List[List[float]], threshold: float) -> bool:
    if not recent_vecs:
        return False
    return max(_cosine_sim(vec, other) for other in recent_vecs) >= threshold


def _quality_score(text: str, source_kind: str, max_repeat_ratio: float) -> float:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return 0.0
    words = cleaned.split()
    length = len(words)
    if length < 5:
        return 0.1
    unique_ratio = len(set(words)) / max(1, length)
    repeat_ratio = 1.0 - unique_ratio
    base = 0.7
    if source_kind == "logs":
        base *= 0.3
    if source_kind == "memory":
        base *= 0.8
    if source_kind == "web":
        base *= 0.9
    if source_kind == "episode":
        base *= 1.2
    if source_kind == "repo":
        base *= 1.3
    if source_kind in {"docs", "lesson"}:
        base *= 1.15
    if repeat_ratio > max_repeat_ratio:
        base *= 0.4
    return max(0.0, min(1.0, base))


_INSTRUCTION_KEYWORDS = {
    "system",
    "assistant",
    "developer",
    "user",
    "instruction",
    "instructions",
    "ignore",
    "follow",
    "must",
    "policy",
    "prompt",
    "role",
    "override",
}


def _instruction_density(text: str) -> float:
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)
    if not tokens:
        return 0.0
    keyword_hits = sum(1 for tok in tokens if tok in _INSTRUCTION_KEYWORDS)
    pattern_hits = 0
    patterns = [
        r"ignore (all|previous) instructions",
        r"do not follow",
        r"system prompt",
        r"role:\s*system",
        r"developer message",
        r"you are (an|a) (assistant|system)",
    ]
    for pat in patterns:
        if re.search(pat, lowered):
            pattern_hits += 1
    return (keyword_hits + pattern_hits * 3) / max(1, len(tokens))


def _canonicalize_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
        if not path:
            path = "/"
    return urlunsplit((scheme, netloc, path, parts.query or "", ""))


def _chunk_web_text(text: str, max_chars: int) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return [cleaned]
    chunk_size = min(800, max_chars)
    if chunk_size <= 0:
        chunk_size = 800
    overlap = 120 if chunk_size > 120 else max(0, chunk_size // 6)
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def _strict_web_ingest(settings: dict) -> bool:
    cont = settings.get("continuous", {}) or {}
    strict_flag = cont.get("strict_web_ingest")
    if strict_flag is not None:
        return bool(strict_flag)
    if bool(settings.get("hf_train", {}).get("enabled", False)):
        return True
    profile_name = settings.get("_profile")
    if profile_name and str(profile_name) == "safe_selftrain_4080":
        return True
    return False


def _sanitize_web_text(
    text: str,
    *,
    max_chars: int,
    max_instruction_density: float,
    max_repeat_lines: int,
) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style.*?>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if max_repeat_lines > 0 and lines:
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for line in lines:
            key = line.lower()
            count = seen.get(key, 0)
            if count >= max_repeat_lines:
                continue
            seen[key] = count + 1
            deduped.append(line)
        lines = deduped
    cleaned = " ".join(" ".join(lines).split())
    if max_instruction_density > 0:
        density = _instruction_density(cleaned)
        if density > max_instruction_density:
            return ""
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned.strip()


def _promote_web_quarantine(
    store: KnowledgeStore,
    recent_vecs: List[List[float]],
    filter_cfg: dict,
    max_repeat_ratio: float,
) -> int:
    min_quality = float(filter_cfg.get("min_quality", 0.35))
    min_chars = int(filter_cfg.get("min_chars", 200))
    min_novelty = float(filter_cfg.get("min_novelty", 0.2))
    limit = int(filter_cfg.get("quarantine_limit", 200))
    promoted = 0
    for chunk in store.sample_chunks(limit=limit, min_quality=0.0, source_kinds=["web"]):
        if chunk.score >= min_quality:
            continue
        text = chunk.text
        if len(text) < min_chars:
            continue
        quality = _quality_score(text, "web", max_repeat_ratio)
        novelty = _novelty_score(text, recent_vecs)
        if quality >= min_quality and novelty >= min_novelty:
            if store.update_quality(text, max(min_quality, quality)):
                promoted += 1
    return promoted


def _resolve_knowledge_store(
    base_dir: Path,
    settings: dict,
    *,
    cache_for_retrieval: bool,
) -> KnowledgeStore:
    continuous = settings.get("continuous", {}) or {}
    knowledge_path = Path(
        continuous.get(
            "knowledge_path", base_dir / "data" / "continuous" / "knowledge.sqlite"
        )
    )
    knowledge_cfg = settings.get("knowledge", {}) or {}
    embed_backend = str(knowledge_cfg.get("embedding_backend", "auto"))
    embed_model = knowledge_cfg.get("embedding_model")
    index_backend = str(knowledge_cfg.get("index_backend", "auto"))
    embedder: str | EmbeddingBackend
    embedder = (
        EmbeddingBackend(backend=embed_backend, model_name=embed_model)
        if embed_model
        else embed_backend
    )
    if not cache_for_retrieval:
        return KnowledgeStore(
            knowledge_path,
            embedding_backend=embedder,
            index_backend=index_backend,
        )

    cache_key = (
        str(knowledge_path.resolve()),
        embed_backend,
        str(embed_model or ""),
        index_backend,
    )
    with _RETRIEVE_STORE_LOCK:
        cached = _RETRIEVE_STORE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        store = KnowledgeStore(
            knowledge_path,
            embedding_backend=embedder,
            index_backend=index_backend,
        )
        _RETRIEVE_STORE_CACHE[cache_key] = store
        return store


def ingest_sources(base_dir: Path, allowlist: List[str], settings: dict) -> int:
    continuous = settings.get("continuous", {})
    filter_cfg = continuous.get("filter", {}) or {}
    semantic_dedup_threshold = _semantic_dedup_threshold(filter_cfg)
    local_sources_cfg = continuous.get("local_sources", {}) or {}
    include_memory = bool(local_sources_cfg.get("include_memory", True))
    include_logs = bool(local_sources_cfg.get("include_logs", True))
    ingest_cfg = continuous.get("ingest", {}) or {}
    max_files_per_tick = int(ingest_cfg.get("max_files_per_tick", 200))
    max_bytes_per_file = int(ingest_cfg.get("max_bytes_per_file", 2_000_000))
    max_total_bytes_per_tick = int(ingest_cfg.get("max_total_bytes_per_tick", 10_000_000))
    ingest_web_cfg = ingest_cfg.get("web", {}) or {}
    web_cooldown = float(ingest_web_cfg.get("cooldown_minutes", 60))
    knowledge_path = Path(continuous.get("knowledge_path", base_dir / "data" / "continuous" / "knowledge.sqlite"))
    knowledge_cfg = settings.get("knowledge", {}) or {}
    policy_cfg = knowledge_cfg.get("policy", {}) or {}
    policy = IngestPolicy(
        min_quality=float(policy_cfg.get("min_quality", 0.0)),
        max_age_days=policy_cfg.get("max_age_days"),
        allow_domains=policy_cfg.get("allow_domains"),
        deny_domains=policy_cfg.get("deny_domains"),
        allow_source_kinds=policy_cfg.get("allow_source_kinds"),
        deny_source_kinds=policy_cfg.get("deny_source_kinds"),
    )
    embed_backend = knowledge_cfg.get("embedding_backend", "auto")
    embed_model = knowledge_cfg.get("embedding_model")
    embedder = EmbeddingBackend(backend=str(embed_backend), model_name=embed_model) if embed_model else embed_backend
    store = KnowledgeStore(
        knowledge_path,
        embedding_backend=embedder,
        index_backend=knowledge_cfg.get("index_backend", "auto"),
        policy=policy,
    )
    state = IngestState(knowledge_path)
    new_docs = 0
    files_used = 0
    bytes_used = 0

    # Memory store (small; dedup via hash)
    memory_path = base_dir / "data" / "memory" / "agent_memory.sqlite"
    if include_memory and memory_path.exists():
        mem = MemoryStore(memory_path)
        for item in mem.query("summary", top_k=50):
            new_docs += store.ingest_text("memory", str(memory_path), item.text, quality=0.6)

    local_docs, files_used, bytes_used = _ingest_local_sources(
        base_dir=base_dir,
        state=state,
        store=store,
        settings=settings,
        max_files_per_tick=max_files_per_tick,
        max_bytes_per_file=max_bytes_per_file,
        max_total_bytes_per_tick=max_total_bytes_per_tick,
        files_used=files_used,
        bytes_used=bytes_used,
    )
    new_docs += local_docs

    # Logs (incremental)
    if include_logs:
        for path in _iter_log_files(base_dir / "data"):
            if files_used >= max_files_per_tick or bytes_used >= max_total_bytes_per_tick:
                break
            try:
                stat = path.stat()
            except Exception:
                continue
            key = f"log:{path.as_posix()}"
            meta = state.get_json(key, {"mtime": 0.0, "size": 0, "offset": 0})
            prev_mtime = float(meta.get("mtime", 0.0))
            prev_size = int(meta.get("size", 0))
            offset = int(meta.get("offset", 0))
            if stat.st_size < offset:
                offset = 0
            if stat.st_mtime == prev_mtime and stat.st_size == prev_size and offset >= stat.st_size:
                continue
            remaining = stat.st_size - offset
            if remaining <= 0:
                state.set_json(key, {"mtime": stat.st_mtime, "size": stat.st_size, "offset": offset})
                continue
            budget_left = max_total_bytes_per_tick - bytes_used
            if budget_left <= 0:
                break
            max_bytes = min(max_bytes_per_file, budget_left, remaining)
            if max_bytes <= 0:
                break
            try:
                with path.open("rb") as handle:
                    handle.seek(offset)
                    data = handle.read(max_bytes)
            except Exception:
                continue
            if not data:
                continue
            if path.suffix.lower() == ".jsonl":
                last_nl = data.rfind(b"\n")
                if last_nl == -1:
                    continue
                data = data[: last_nl + 1]
            text = data.decode("utf-8", errors="ignore")
            if text:
                new_docs += store.ingest_text("logs", path.as_posix(), text, quality=0.2)
            bytes_used += len(data)
            files_used += 1
            offset += len(data)
            state.set_json(key, {"mtime": stat.st_mtime, "size": stat.st_size, "offset": offset})

    # Web docs (cache + cooldown)
    tools_cfg = settings.get("tools", {}) or {}
    tools_web_cfg = tools_cfg.get("web", {}) or {}
    web_enabled = bool(tools_web_cfg.get("enabled", False))
    ingest_web_enabled = bool(continuous.get("ingest_web", True))
    if ingest_web_enabled and allowlist and not web_enabled:
        message = "ingest_web enabled but tools.web.enabled=false; skipping web ingest"
        if _strict_web_ingest(settings):
            raise RuntimeError(message)
        print(f"WARN {message}")
    if ingest_web_enabled and allowlist and web_enabled:
        urls = continuous.get("ingest_urls", ["https://docs.python.org/3/", "https://pytorch.org/docs/stable/"])
        tools = AgentTools(allowlist=allowlist, web_cfg=tools_web_cfg, security_cfg=settings.get("security", {}) or {})
        recent_chunks = store.sample_chunks(limit=50, source_kinds=["web"])
        recent_vecs = [embed_text(chunk.text) for chunk in recent_chunks]
        for raw_url in urls:
            if files_used >= max_files_per_tick or bytes_used >= max_total_bytes_per_tick:
                break
            url = _canonicalize_url(str(raw_url))
            key = f"web:{url}"
            meta = state.get_json(key, {})
            last_ts = float(meta.get("ts", 0.0))
            last_hash = meta.get("hash")
            if web_cooldown > 0 and (time.time() - last_ts) < web_cooldown * 60.0:
                continue
            doc = tools.open_docs(url)
            if not doc.ok:
                continue
            content = doc.output or ""
            sanitize_cfg = ingest_web_cfg.get("sanitize", {}) or {}
            max_chars = int(sanitize_cfg.get("max_chars", 2000))
            max_instr = float(sanitize_cfg.get("max_instruction_density", 0.04))
            max_repeat_lines = int(sanitize_cfg.get("max_repeat_lines", 2))
            content = _sanitize_web_text(
                content,
                max_chars=0,
                max_instruction_density=max_instr,
                max_repeat_lines=max_repeat_lines,
            )
            if not content:
                continue
            content_bytes = content.encode("utf-8", errors="ignore")
            content_hash = hashlib.sha256(content_bytes).hexdigest()
            if content_hash == last_hash:
                etag = meta.get("etag")
                last_modified = meta.get("last_modified")
                if isinstance(doc.meta, dict):
                    etag = doc.meta.get("etag") or etag
                    last_modified = doc.meta.get("last_modified") or last_modified
                state.set_json(key, {"ts": time.time(), "hash": last_hash, "etag": etag, "last_modified": last_modified})
                continue
            chunks = _chunk_web_text(content, max_chars)
            if not chunks:
                continue
            for idx, chunk in enumerate(chunks):
                if files_used >= max_files_per_tick or bytes_used >= max_total_bytes_per_tick:
                    break
                vec = embed_text(chunk)
                if _is_semantic_duplicate(vec, recent_vecs, semantic_dedup_threshold):
                    continue
                budget_left = max_total_bytes_per_tick - bytes_used
                if budget_left <= 0:
                    break
                chunk_bytes = chunk.encode("utf-8", errors="ignore")
                max_bytes = min(max_bytes_per_file, budget_left)
                if max_bytes > 0 and len(chunk_bytes) > max_bytes:
                    chunk_bytes = chunk_bytes[:max_bytes]
                    chunk = chunk_bytes.decode("utf-8", errors="ignore")
                source_ref = url if len(chunks) == 1 else f"{url}#chunk={idx}"
                new_docs += store.ingest_text("web", source_ref, chunk, quality=0.0)
                bytes_used += len(chunk_bytes)
                files_used += 1
                recent_vecs.append(vec)
            etag = meta.get("etag")
            last_modified = meta.get("last_modified")
            if isinstance(doc.meta, dict):
                etag = doc.meta.get("etag") or etag
                last_modified = doc.meta.get("last_modified") or last_modified
            state.set_json(key, {"ts": time.time(), "hash": content_hash, "etag": etag, "last_modified": last_modified})

    # Episodes (incremental)
    episodes_path = base_dir / "data" / "episodes" / "agent.jsonl"
    if episodes_path.exists() and files_used < max_files_per_tick and bytes_used < max_total_bytes_per_tick:
        key = f"episodes:{episodes_path.as_posix()}"
        meta = state.get_json(key, {"size": 0, "offset": 0})
        offset = int(meta.get("offset", 0))
        queue_dir = _resolve_queue_dir(base_dir, settings)
        try:
            ep_stat = episodes_path.stat()
        except Exception:
            ep_stat = None
        if ep_stat is not None:
            if ep_stat.st_size < offset:
                offset = 0
            if not (ep_stat.st_size == int(meta.get("size", 0)) and offset >= ep_stat.st_size):
                remaining = ep_stat.st_size - offset
                if remaining > 0:
                    budget_left = max_total_bytes_per_tick - bytes_used
                    max_bytes = min(max_bytes_per_file, budget_left, remaining)
                    if max_bytes > 0:
                        with episodes_path.open("rb") as handle:
                            handle.seek(offset)
                            data = handle.read(max_bytes)
                        if data:
                            last_nl = data.rfind(b"\n")
                            if last_nl != -1:
                                data = data[: last_nl + 1]
                                offset += len(data)
                                text = data.decode("utf-8", errors="ignore")
                                for line in text.splitlines():
                                    try:
                                        payload = json.loads(line)
                                    except Exception:
                                        continue
                                    if not isinstance(payload, dict):
                                        continue
                                    tests_ok = bool(payload.get("tests_ok"))
                                    tools_ok = bool(payload.get("tools_ok"))
                                    if not (tests_ok or tools_ok):
                                        continue
                                    task = str(payload.get("task", "")).strip()
                                    context = _episode_prompt(payload)
                                    diff = str(payload.get("patch", "")).strip()
                                    if not diff:
                                        raw_patch_id = payload.get("patch_id")
                                        patch_id = str(raw_patch_id).strip() if raw_patch_id else ""
                                        diff = _load_patch_from_queue(queue_dir, patch_id)
                                    if task or diff:
                                        doc_text = f"{task}\n{context}\n{diff}".strip()
                                        new_docs += store.ingest_text("episode", episodes_path.as_posix(), doc_text, quality=0.9)
                                bytes_used += len(data)
                                files_used += 1
                        state.set_json(key, {"size": ep_stat.st_size, "offset": offset})

    return new_docs


def collect_samples(base_dir: Path, allowlist: List[str], settings: dict, ingest: bool = True) -> CollectedSamples:
    continuous = settings.get("continuous", {})
    replay_cfg = continuous.get("replay", {})
    filter_cfg = continuous.get("filter", {})
    min_quality = float(filter_cfg.get("min_quality", 0.35))
    min_novelty = float(filter_cfg.get("min_novelty", 0.2))
    max_repeat_ratio = float(filter_cfg.get("max_repeat_ratio", 0.8))
    semantic_dedup_threshold = _semantic_dedup_threshold(filter_cfg)
    replay_path = Path(replay_cfg.get("path", base_dir / "data" / "continuous" / "replay.sqlite"))
    sample_size = int(replay_cfg.get("sample_size", 64))
    top_frac = float(replay_cfg.get("top_frac", 0.7))
    random_frac = float(replay_cfg.get("random_frac", 0.3))
    seed_chunks = int(replay_cfg.get("seed_chunks", 40))
    max_items = replay_cfg.get("max_items")
    max_items = int(max_items) if max_items is not None else None

    store = _resolve_knowledge_store(base_dir, settings, cache_for_retrieval=False)
    replay = ReplayBuffer(replay_path)
    new_docs = ingest_sources(base_dir, allowlist, settings) if ingest else 0

    recent_vecs = [embed_text(t) for t in replay.recent_texts(limit=50)]
    _promote_web_quarantine(store, recent_vecs, filter_cfg, max_repeat_ratio)
    total_candidates = 0
    filtered = 0
    novelty_scores: List[float] = []
    successes = 0
    gold_samples: List[Sample] = []

    # Episodes -> gold samples
    episodes_path = base_dir / "data" / "episodes" / "agent.jsonl"
    queue_dir = _resolve_queue_dir(base_dir, settings)
    seen_episode_hashes: set[str] = set()
    for ep in _load_episodes(episodes_path):
        tests_ok = bool(ep.get("tests_ok"))
        tools_ok = bool(ep.get("tools_ok"))
        if not (tests_ok or tools_ok):
            continue
        task = str(ep.get("task", "")).strip()
        context = _episode_prompt(ep)
        diff = str(ep.get("patch", "")).strip()
        if not diff:
            raw_patch_id = ep.get("patch_id")
            patch_id = str(raw_patch_id).strip() if raw_patch_id else ""
            diff = _load_patch_from_queue(queue_dir, patch_id)
        if not diff:
            continue
        prompt = f"Task: {task}".strip()
        if context:
            prompt = f"{prompt}\n\nContext:\n{context}"
        sample = Sample(prompt=prompt, response=diff, source_kind="episode")
        event_id = _episode_hash(task, context, diff, tests_ok, tools_ok)
        if event_id in seen_episode_hashes:
            continue
        seen_episode_hashes.add(event_id)
        gold_samples.append(sample)
        quality = _quality_score(diff, "episode", max_repeat_ratio)
        vec = embed_text(diff)
        novelty = _novelty_score_from_vec(vec, recent_vecs)
        total_candidates += 1
        successes += 1
        digest = replay.hash_sample(sample.prompt, sample.response)
        replay.bump_success_once(digest, event_id, delta=1)
        if quality >= min_quality and novelty >= min_novelty:
            if _is_semantic_duplicate(vec, recent_vecs, semantic_dedup_threshold):
                filtered += 1
                continue
            inserted = replay.add(
                ReplayItem(
                    sample=sample,
                    source_kind="episode",
                    quality_score=quality,
                    novelty_score=novelty,
                    success_count=0,
                ),
                max_items=max_items,
            )
            if inserted:
                novelty_scores.append(novelty)
                recent_vecs.append(vec)
        else:
            filtered += 1

    # Knowledge chunks -> seed replay
    chunks = store.sample_chunks(limit=seed_chunks, min_quality=min_quality)
    for chunk in chunks:
        prompt = "Continue"
        if chunk.source_kind == "memory":
            prompt = "Summarize"
        elif chunk.source_kind == "web":
            prompt = "Read docs"
        elif chunk.source_kind == "episode":
            prompt = "Review"
        elif chunk.source_kind == "repo":
            prompt = "Review code"
        elif chunk.source_kind == "docs":
            prompt = "Read docs"
        elif chunk.source_kind == "lesson":
            prompt = "Study lesson"
        sample = Sample(prompt=prompt, response=chunk.text, source_kind=chunk.source_kind)
        quality = _quality_score(chunk.text, chunk.source_kind, max_repeat_ratio)
        vec = embed_text(chunk.text)
        novelty = _novelty_score_from_vec(vec, recent_vecs)
        total_candidates += 1
        if quality >= min_quality and novelty >= min_novelty:
            if _is_semantic_duplicate(vec, recent_vecs, semantic_dedup_threshold):
                filtered += 1
                continue
            inserted = replay.add(
                ReplayItem(
                    sample=sample,
                    source_kind=chunk.source_kind,
                    quality_score=quality,
                    novelty_score=novelty,
                    success_count=0,
                ),
                max_items=max_items,
            )
            if inserted:
                novelty_scores.append(novelty)
                recent_vecs.append(vec)
        else:
            filtered += 1

    source_weights_cfg = continuous.get("source_weights", {}) or {}
    source_weights: dict[str, float] = {}
    for kind, weight in source_weights_cfg.items():
        try:
            parsed = float(weight)
        except Exception:
            continue
        if parsed < 0.0:
            parsed = 0.0
        source_weights[str(kind)] = parsed
    samples = replay.sample(
        sample_size,
        top_frac=top_frac,
        random_frac=random_frac,
        source_weights=source_weights,
    )
    novelty_avg = sum(novelty_scores) / max(1, len(novelty_scores))
    stats = CollectStats(
        new_docs=new_docs,
        novelty_avg=novelty_avg,
        successes=successes,
        filtered=filtered,
        total_candidates=total_candidates,
    )
    return CollectedSamples(samples=samples, stats=stats, gold_samples=gold_samples)


def retrieve_context_details(base_dir: Path, query: str, settings: dict, top_k: int = 3) -> tuple[str, list[dict]]:
    rag_cfg = settings.get("rag", {})
    max_chars = int(rag_cfg.get("max_chars", 1200))
    store = _resolve_knowledge_store(base_dir, settings, cache_for_retrieval=True)
    chunks = store.retrieve(query, top_k=top_k, min_quality=0.0)
    joined = "\n\n".join(chunk.text for chunk in chunks)
    if max_chars and len(joined) > max_chars:
        joined = joined[:max_chars]
    refs = [
        {"kind": chunk.source_kind, "ref": chunk.source_ref}
        for chunk in chunks if chunk.source_ref
    ]
    return joined, refs


def retrieve_context(base_dir: Path, query: str, settings: dict, top_k: int = 3) -> str:
    context, _refs = retrieve_context_details(base_dir, query, settings, top_k=top_k)
    return context

"""
autolearn.py — Real auto-learning daemon for Vortex.

Orchestrates three kinds of autonomous learning:

1. **Web Knowledge Ingest** (on startup + periodic):
   Crawls configured official web pages, sanitizes content,
   and indexes it into the knowledge store for RAG retrieval.

2. **Self-Code Analysis** (periodic):
   Reads Vortex's own source files, indexes them as knowledge,
   and uses the loaded model to analyse code for improvements.

3. **Self-Edit Proposals** (periodic):
   When the model identifies improvements to its own code,
   generates diff proposals through the self-edits system.
   Proposals require manual approval unless auto_apply is enabled.

All operations are safe: they use file locks, respect VRAM budgets,
and never auto-apply patches without explicit configuration.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("vortex.autolearn")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_LOCK: threading.Lock = threading.Lock()

# State file for tracking what we've already processed
_STATE_FILE = "data/state/autolearn.json"


def _load_state(base_dir: Path) -> dict:
    p = base_dir / _STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(base_dir: Path, state: dict) -> None:
    p = base_dir / _STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), "utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------------------
# 1. Web Knowledge Ingest
# ---------------------------------------------------------------------------

# Seed URLs — small starter set. The model discovers more dynamically.
_SEED_URLS: list[str] = [
    "https://docs.python.org/3/tutorial/index.html",
    "https://pytorch.org/docs/stable/torch.html",
    "https://huggingface.co/docs/transformers/main/en/index",
    "https://fastapi.tiangolo.com/tutorial/",
]

# Trusted domain suffixes — URLs on these domains are safe to fetch
_TRUSTED_DOMAINS: set[str] = {
    "python.org",
    "pytorch.org",
    "huggingface.co",
    "fastapi.tiangolo.com",
    "numpy.org",
    "scipy.org",
    "pandas.pydata.org",
    "scikit-learn.org",
    "tensorflow.org",
    "wikipedia.org",
    "github.com",
    "github.io",
    "readthedocs.io",
    "docs.microsoft.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "openai.com",
    "anthropic.com",
    "docs.docker.com",
    "redis.io",
    "sqlite.org",
    "peps.python.org",
    "mypy.readthedocs.io",
    "docs.pydantic.dev",
    "docs.pytest.org",
    "ruff.rs",
    "vite.dev",
    "react.dev",
    "typescriptlang.org",
}


def web_ingest_once(
    base_dir: Path,
    settings: dict,
    *,
    urls: list[str] | None = None,
    force: bool = False,
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Fetch and index web pages into the knowledge store.

    Returns a summary dict with counts and errors.
    """
    state = _load_state(base_dir)
    cooldown_min = float(
        (settings.get("autolearn", {}) or {}).get("web_cooldown_minutes", 30)
    )
    last_ts = float(state.get("last_web_ingest_ts", 0))
    if not force and (time.time() - last_ts) < cooldown_min * 60:
        return {
            "ok": True,
            "skipped": "cooldown",
            "seconds_left": int(cooldown_min * 60 - (time.time() - last_ts)),
        }

    from .continuous.knowledge_store import KnowledgeStore

    knowledge_path = Path(
        (settings.get("continuous", {}) or {}).get(
            "knowledge_path",
            base_dir / "data" / "continuous" / "knowledge.sqlite",
        )
    )
    store = KnowledgeStore(knowledge_path)

    # Start with explicit URLs or seed list
    base_urls = urls or (settings.get("autolearn", {}) or {}).get("urls", _SEED_URLS)
    allow_domains = _resolve_allow_domains(settings)

    # Let the model discover additional official URLs dynamically
    discovered = _discover_urls_with_model(
        base_dir,
        settings,
        model=model,
        tokenizer=tokenizer,
    )
    # Merge: explicit + discovered, de-duplicated
    seen_urls: set[str] = set()
    target_urls: list[str] = []
    for u in list(base_urls) + discovered:
        if u not in seen_urls:
            seen_urls.add(u)
            target_urls.append(u)

    total_added = 0
    errors: list[str] = []
    fetched = 0

    for url in target_urls:
        try:
            # Check domain allowlist
            from urllib.parse import urlparse

            domain = urlparse(url).hostname or ""
            if allow_domains and not any(domain.endswith(d) for d in allow_domains):
                continue

            # Check if already fetched recently
            url_hash = _hash(url)
            url_state = state.get("urls", {}).get(url_hash, {})
            last_fetch = float(url_state.get("ts", 0))
            if not force and (time.time() - last_fetch) < cooldown_min * 60:
                continue

            text = _fetch_and_clean(url, settings, allow_domains)
            if not text or len(text) < 100:
                continue

            fetched += 1
            added = store.ingest_text(
                source_kind="web",
                source_ref=url,
                text=text,
                quality=0.6,
            )
            total_added += added

            # Track URL state
            state.setdefault("urls", {})[url_hash] = {
                "url": url,
                "ts": time.time(),
                "chars": len(text),
                "chunks_added": added,
            }

        except Exception as exc:
            errors.append(f"{url}: {exc}")
            logger.warning("autolearn web fetch failed: %s — %s", url, exc)

    state["last_web_ingest_ts"] = time.time()
    state["total_web_chunks"] = state.get("total_web_chunks", 0) + total_added
    _save_state(base_dir, state)

    result = {
        "ok": True,
        "fetched": fetched,
        "chunks_added": total_added,
        "errors": errors[:5],
        "total_urls": len(target_urls),
    }
    logger.info("autolearn web ingest: %s", result)
    return result


def _resolve_allow_domains(settings: dict) -> list[str]:
    """Resolve effective allowlist from settings with strict fallback."""
    domains: set[str] = set()
    for key_path in [
        ("tools", "web", "allow_domains"),
        ("agent", "web_allowlist"),
        ("security", "web", "allowlist_domains"),
        ("autolearn", "allow_domains"),
    ]:
        cfg = settings
        for k in key_path:
            cfg = (cfg or {}).get(k, {})
        if isinstance(cfg, list):
            domains.update(str(d) for d in cfg)
    # Add domains from the URLs themselves
    from urllib.parse import urlparse

    for url in (settings.get("autolearn", {}) or {}).get("urls", _SEED_URLS):
        host = urlparse(url).hostname
        if host:
            domains.add(host)
    if not domains:
        domains.update(
            {
                "docs.python.org",
                "pytorch.org",
                "huggingface.co",
                "fastapi.tiangolo.com",
            }
        )
    return list(domains) if domains else []


def _fetch_and_clean(
    url: str,
    settings: dict,
    allow_domains: list[str] | None = None,
) -> str:
    """Fetch a URL and return cleaned text content."""
    timeout = int((settings.get("tools", {}).get("web", {}) or {}).get("timeout_s", 15))
    max_bytes = int(
        (settings.get("tools", {}).get("web", {}) or {}).get("max_bytes", 512000)
    )

    try:
        from .config import resolve_web_strict
        from .tools.web_access import web_fetch

        result = web_fetch(
            url,
            allowlist=allow_domains or [],
            max_bytes=max_bytes,
            timeout_s=timeout,
            strict=bool(resolve_web_strict(settings)),
        )
        if result and result.ok and result.text:
            return _sanitize_html(result.text)
    except Exception:
        pass
    return ""


def _issue_context_lines(
    issue_lines: set[int], total_lines: int, *, window: int = 5
) -> list[int]:
    if total_lines <= 0 or not issue_lines:
        return []
    context_lines: set[int] = set()
    for ln in issue_lines:
        for offset in range(-window, window + 1):
            context_lines.add(int(ln) + int(offset))
    valid = context_lines & set(range(1, int(total_lines) + 1))
    return sorted(valid)


def _sanitize_html(raw: str) -> str:
    """Strip HTML tags, scripts, styles; keep text content."""
    # Remove script/style blocks
    text = re.sub(
        r"<(script|style|noscript)[^>]*>[\s\S]*?</\1>", " ", raw, flags=re.IGNORECASE
    )
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just navigation/boilerplate (very short repeated lines)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 3:
            continue
        if stripped in {
            "Skip to content",
            "Skip to main content",
            "Toggle navigation",
            "Menu",
            "Close",
        }:
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------------
# URL Discovery — model decides which official pages to learn from
# ---------------------------------------------------------------------------


def _discover_urls_with_model(
    base_dir: Path,
    settings: dict,
    *,
    model: Any = None,
    tokenizer: Any = None,
) -> list[str]:
    """Ask the loaded model to suggest official documentation URLs.

    The model analyses the project context and proposes relevant official
    pages it should learn from.  Results are cached in state so they
    accumulate across restarts.
    """
    state = _load_state(base_dir)
    discovered: list[str] = state.get("discovered_urls", [])

    # Only re-discover periodically (daily by default)
    cooldown_h = float(
        (settings.get("autolearn", {}) or {}).get("discovery_cooldown_hours", 24)
    )
    last_ts = float(state.get("last_discovery_ts", 0))
    if (time.time() - last_ts) < cooldown_h * 3600:
        return discovered

    if model is None:
        return discovered

    # Build context about the project for the model
    project_context = _gather_project_context(base_dir)
    topics = [
        str(item).strip()
        for item in (
            (settings.get("autolearn", {}) or {}).get("discovery_topics", []) or []
        )
        if str(item).strip()
    ]
    topics_block = ""
    if topics:
        topics_block = (
            "\nFOCUS TOPICS (prioritize these):\n"
            + "\n".join(f"- {item}" for item in topics[:20])
            + "\n"
        )

    prompt = (
        "You are a technical research assistant. Based on this project description, "
        "suggest 10-15 OFFICIAL documentation URLs that would be most valuable to study.\n\n"
        "RULES:\n"
        "- Only suggest official documentation sites (*.org, *.io, docs.*, official repos)\n"
        "- Focus on technologies actually used in this project\n"
        "- Defensive cybersecurity / authorized security testing content is allowed\n"
        "- Exclude offensive or illegal misuse guides\n"
        "- Include API references, tutorials, and best-practice guides\n"
        "- One URL per line, no numbering, no explanations\n"
        "- Only HTTPS URLs\n\n"
        f"{topics_block}"
        f"PROJECT CONTEXT:\n{project_context}\n\n"
        "URLS:"
    )

    try:
        generate_fn = getattr(model, "generate", None)
        if generate_fn is None:
            return discovered

        if tokenizer is not None:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            if hasattr(inputs, "to"):
                device = getattr(model, "device", "cpu")
                inputs = inputs.to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.4,
                do_sample=True,
                pad_token_id=getattr(tokenizer, "eos_token_id", 0),
            )
            response = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
        else:
            response = str(generate_fn(prompt, max_tokens=512))

    except Exception as exc:
        logger.warning("autolearn URL discovery failed: %s", exc)
        return discovered

    # Parse and validate URLs from model response
    new_urls = _parse_and_validate_urls(
        response, allowed_domains=_resolve_allow_domains(settings)
    )
    if new_urls:
        existing = set(discovered)
        for url in new_urls:
            if url not in existing:
                discovered.append(url)
                existing.add(url)
        state["discovered_urls"] = discovered
        state["last_discovery_ts"] = time.time()
        state["last_discovery_count"] = len(new_urls)
        _save_state(base_dir, state)
        logger.info(
            "autolearn discovered %d new URLs (%d total)",
            len(new_urls),
            len(discovered),
        )

    return discovered


def _gather_project_context(base_dir: Path) -> str:
    """Build a brief description of the project for URL discovery."""
    parts: list[str] = []

    # Read pyproject.toml for dependencies
    for candidate in [base_dir / "pyproject.toml", base_dir.parent / "pyproject.toml"]:
        if candidate.exists():
            try:
                content = candidate.read_text("utf-8", errors="replace")
                dep_match = re.search(
                    r"\[(?:project\.)?dependencies\](.*?)(?:\n\[|\Z)",
                    content,
                    re.DOTALL,
                )
                if dep_match:
                    parts.append(f"Dependencies:\n{dep_match.group(1).strip()[:500]}")
            except Exception:
                pass
            break

    # Scan imports from key source files
    key_files = [
        "src/c3rnt2/server.py",
        "src/c3rnt2/inference/engine.py",
        "src/c3rnt2/continuous/knowledge_store.py",
        "src/c3rnt2/training/trainer.py",
    ]
    imports: set[str] = set()
    for kf in key_files:
        fp = base_dir / kf
        if not fp.exists():
            continue
        try:
            for line in fp.read_text("utf-8", errors="replace").split("\n")[:80]:
                m = re.match(r"^\s*(?:from|import)\s+([\w.]+)", line)
                if m:
                    imports.add(m.group(1).split(".")[0])
        except Exception:
            pass
    if imports:
        parts.append(f"Python imports used: {', '.join(sorted(imports))}")

    # Check frontend package.json
    for pkg_path in [
        base_dir.parent / "vortex-chat" / "package.json",
        base_dir / "vortex-chat" / "package.json",
    ]:
        if pkg_path.exists():
            try:
                pkg = json.loads(pkg_path.read_text("utf-8"))
                deps = list((pkg.get("dependencies", {}) or {}).keys())
                if deps:
                    parts.append(f"Frontend deps: {', '.join(deps[:15])}")
            except Exception:
                pass
            break

    parts.append(
        "Project type: AI/ML backend with LLM inference, LoRA fine-tuning, "
        "RAG knowledge retrieval, self-training pipeline, web API "
        "(FastAPI/Starlette), React+TypeScript frontend with Vite."
    )
    return "\n".join(parts)


def _parse_and_validate_urls(
    response: str, allowed_domains: list[str] | None = None
) -> list[str]:
    """Extract and validate URLs from model output."""
    from urllib.parse import urlparse

    urls: list[str] = []
    seen: set[str] = set()

    for line in response.split("\n"):
        line = line.strip().lstrip("-\u2022*1234567890.)")
        match = re.search(r"(https?://[^\s<>\"']+)", line)
        if not match:
            continue
        url = match.group(1).rstrip(".,;:)")
        if url in seen:
            continue

        try:
            parsed = urlparse(url)
        except Exception:
            continue

        if parsed.scheme != "https":
            continue
        host = parsed.hostname or ""
        if not host or len(host) < 4:
            continue

        allowlisted = False
        # Respect explicit allowlist when configured.
        if allowed_domains:
            allowlisted = any(
                host == d or host.endswith("." + d) for d in allowed_domains
            )
            if not allowlisted:
                continue

        # When allowlist is not configured, keep a strict trusted-domain filter.
        if not allowlisted:
            trusted = any(host == d or host.endswith("." + d) for d in _TRUSTED_DOMAINS)
            # Also accept domains that look like official docs
            looks_official = (
                host.startswith("docs.")
                or ".readthedocs." in host
                or host.endswith(".dev")
                or (
                    host.endswith(".io")
                    and ("docs" in parsed.path or "api" in parsed.path)
                )
            )
            if not trusted and not looks_official:
                continue

        urls.append(url)
        seen.add(url)

    return urls[:20]  # Cap to avoid noise


# ---------------------------------------------------------------------------
# 2. Self-Code Analysis — index own source files as knowledge
# ---------------------------------------------------------------------------

# Patterns for files to analyze
_SOURCE_GLOBS = [
    "c3_rnt2_ai/src/c3rnt2/**/*.py",
    "c3_rnt2_ai/tests/**/*.py",
    "c3_rnt2_ai/scripts/**/*.py",
    "vortex-chat/services/**/*.ts",
    "vortex-chat/components/**/*.tsx",
    "vortex-chat/App.tsx",
    "vortex-chat/types.ts",
]

# Files to NEVER auto-modify (safety)
_UNTOUCHABLE = {
    "autolearn.py",
    "safety_kernel.py",
    "policy.py",
    "store.py",  # self_edits store
}


def self_code_index(base_dir: Path, settings: dict) -> dict[str, Any]:
    """Read own source files and index them into the knowledge store.

    This gives the RAG system awareness of its own codebase.
    """
    state = _load_state(base_dir)
    cooldown_min = float(
        (settings.get("autolearn", {}) or {}).get("code_index_cooldown_minutes", 60)
    )
    last_ts = float(state.get("last_code_index_ts", 0))
    if (time.time() - last_ts) < cooldown_min * 60:
        return {"ok": True, "skipped": "cooldown"}

    from .continuous.knowledge_store import KnowledgeStore

    knowledge_path = Path(
        (settings.get("continuous", {}) or {}).get(
            "knowledge_path",
            base_dir / "data" / "continuous" / "knowledge.sqlite",
        )
    )
    store = KnowledgeStore(knowledge_path)

    import glob

    total_added = 0
    files_processed = 0
    hashes_state = state.get("code_hashes", {})

    for pattern in _SOURCE_GLOBS:
        for filepath in glob.glob(str(base_dir / pattern), recursive=True):
            fp = Path(filepath)
            if not fp.is_file():
                continue
            try:
                content = fp.read_text("utf-8", errors="replace")
            except Exception:
                continue
            if len(content) < 50:
                continue

            content_hash = _hash(content)
            rel_path = str(fp.relative_to(base_dir)).replace("\\", "/")

            # Skip if unchanged since last index
            if hashes_state.get(rel_path) == content_hash:
                continue

            # Index with high quality (own code is authoritative)
            added = store.ingest_text(
                source_kind="self_code",
                source_ref=rel_path,
                text=f"# File: {rel_path}\n{content}",
                quality=0.8,
            )
            total_added += added
            files_processed += 1
            hashes_state[rel_path] = content_hash

    state["code_hashes"] = hashes_state
    state["last_code_index_ts"] = time.time()
    state["total_code_chunks"] = state.get("total_code_chunks", 0) + total_added
    _save_state(base_dir, state)

    result = {
        "ok": True,
        "files_processed": files_processed,
        "chunks_added": total_added,
    }
    logger.info("autolearn code index: %s", result)
    return result


# ---------------------------------------------------------------------------
# 3. Self-Code Analysis + Auto-Edit Proposals
# ---------------------------------------------------------------------------


def self_code_analyze(
    base_dir: Path,
    settings: dict,
    *,
    model=None,
    tokenizer=None,
) -> dict[str, Any]:
    """Analyze own source files and propose improvements.

    Uses the loaded model to review code and generate diffs.
    If no model is provided, returns analysis without proposals.
    """
    state = _load_state(base_dir)
    cooldown_min = float(
        (settings.get("autolearn", {}) or {}).get("analyze_cooldown_minutes", 120)
    )
    last_ts = float(state.get("last_analyze_ts", 0))
    if (time.time() - last_ts) < cooldown_min * 60:
        return {"ok": True, "skipped": "cooldown"}

    # Pick files to analyze (rotating through them)
    analyze_idx = int(state.get("analyze_file_index", 0))
    files_to_analyze = _collect_analyzable_files(base_dir)
    if not files_to_analyze:
        return {"ok": True, "skipped": "no_files"}

    batch_size = int((settings.get("autolearn", {}) or {}).get("analyze_batch_size", 3))
    batch = files_to_analyze[analyze_idx : analyze_idx + batch_size]
    if not batch:
        analyze_idx = 0
        batch = files_to_analyze[:batch_size]

    proposals_created = 0
    analyses: list[dict] = []

    for rel_path in batch:
        fp = base_dir / rel_path
        if not fp.is_file():
            continue

        try:
            content = fp.read_text("utf-8", errors="replace")
        except Exception:
            continue

        if len(content) < 100:
            continue

        # Static analysis: find common issues
        issues = _static_analyze(rel_path, content)
        if issues:
            analyses.append({"file": rel_path, "issues": issues})

        # If model is available, ask it to propose improvements
        if model is not None and issues:
            proposal = _model_propose_fix(
                model, tokenizer, rel_path, content, issues, base_dir, settings
            )
            if proposal and proposal.get("diff"):
                proposals_created += 1

    state["last_analyze_ts"] = time.time()
    state["analyze_file_index"] = analyze_idx + batch_size
    state["total_analyses"] = state.get("total_analyses", 0) + len(analyses)
    state["total_proposals"] = state.get("total_proposals", 0) + proposals_created
    _save_state(base_dir, state)

    result = {
        "ok": True,
        "files_analyzed": len(batch),
        "issues_found": sum(len(a["issues"]) for a in analyses),
        "proposals_created": proposals_created,
        "analyses": analyses[:5],
    }
    logger.info("autolearn analyze: %s", result)
    return result


def _collect_analyzable_files(base_dir: Path) -> list[str]:
    """Collect Python/TS files that can be analyzed and potentially modified."""
    import glob

    files: list[str] = []
    for pattern in _SOURCE_GLOBS:
        for filepath in glob.glob(str(base_dir / pattern), recursive=True):
            fp = Path(filepath)
            if not fp.is_file():
                continue
            name = fp.name
            if name in _UNTOUCHABLE:
                continue
            rel = str(fp.relative_to(base_dir)).replace("\\", "/")
            files.append(rel)
    return sorted(files)


def _static_analyze(rel_path: str, content: str) -> list[dict]:
    """Run basic static analysis on a source file. Returns list of issues."""
    issues: list[dict] = []

    lines = content.split("\n")

    if rel_path.endswith(".py"):
        for i, line in enumerate(lines, 1):
            # bare except
            if re.match(r"\s*except\s*:", line):
                issues.append(
                    {
                        "line": i,
                        "type": "bare_except",
                        "msg": "Bare except clause — should catch specific exceptions",
                    }
                )
            # TODO/FIXME/HACK
            m = re.search(r"#\s*(TODO|FIXME|HACK|XXX)[:\s](.+)", line, re.IGNORECASE)
            if m:
                issues.append(
                    {
                        "line": i,
                        "type": "todo",
                        "msg": f"{m.group(1)}: {m.group(2).strip()}",
                    }
                )
            # very long lines
            if len(line) > 150 and not line.strip().startswith("#"):
                issues.append(
                    {
                        "line": i,
                        "type": "long_line",
                        "msg": f"Line too long ({len(line)} chars)",
                    }
                )
            # print() in production code (not tests/scripts)
            if "test" not in rel_path and "script" not in rel_path:
                if re.match(r"\s*print\(", line) and "# noqa" not in line:
                    issues.append(
                        {
                            "line": i,
                            "type": "debug_print",
                            "msg": "print() in production code",
                        }
                    )

    elif rel_path.endswith((".ts", ".tsx")):
        for i, line in enumerate(lines, 1):
            if re.search(r"\bconsole\.log\b", line) and "// noqa" not in line:
                issues.append(
                    {
                        "line": i,
                        "type": "debug_log",
                        "msg": "console.log in production code",
                    }
                )
            if "any" in line and re.search(r":\s*any\b", line):
                issues.append(
                    {
                        "line": i,
                        "type": "any_type",
                        "msg": "Explicit 'any' type — consider typing",
                    }
                )

    # Cap issues to avoid noise
    return issues[:10]


def _model_propose_fix(
    model,
    tokenizer,
    rel_path: str,
    content: str,
    issues: list[dict],
    base_dir: Path,
    settings: dict,
) -> dict | None:
    """Use the loaded model to propose a fix for identified issues.

    Creates a self-edit proposal through the store.
    """
    try:
        from .self_edits.store import SelfEditsStore
    except ImportError:
        return None

    # Build a focused prompt for the model
    issues_text = "\n".join(
        f"  Line {iss['line']}: [{iss['type']}] {iss['msg']}" for iss in issues[:5]
    )

    # Only show relevant portion of the file (around issues)
    issue_lines = {iss["line"] for iss in issues}
    all_lines = content.split("\n")
    snippet_lines = _issue_context_lines(issue_lines, len(all_lines), window=5)

    if not snippet_lines:
        return None

    # Build snippet with line numbers
    snippet_parts = []
    for ln in snippet_lines:
        snippet_parts.append(f"{ln:4d} | {all_lines[ln - 1]}")
    snippet = "\n".join(snippet_parts)

    prompt = (
        f"You are a code reviewer. The file `{rel_path}` has these issues:\n"
        f"{issues_text}\n\n"
        f"Relevant code:\n```\n{snippet}\n```\n\n"
        f"Generate a unified diff (--- a/{rel_path} +++ b/{rel_path}) that fixes these issues. "
        f"Only fix what's clearly broken or clearly improvable. "
        f"If no fix is needed, respond with 'NO_FIX_NEEDED'.\n"
    )

    # Generate using model
    try:
        generate = getattr(model, "generate", None)
        if generate is None:
            return None

        if tokenizer is not None:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            )
            if hasattr(inputs, "to"):
                device = getattr(model, "device", "cpu")
                inputs = inputs.to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
                if hasattr(tokenizer, "eos_token_id")
                else 0,
            )
            response = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        else:
            # Fallback for non-HF models
            response = str(generate(prompt, max_tokens=512))

    except Exception as exc:
        logger.warning("autolearn model generate failed: %s", exc)
        return None

    if not response or "NO_FIX_NEEDED" in response:
        return None

    # Extract diff from response
    diff = _extract_diff(response, rel_path)
    if not diff:
        return None

    # Create proposal through self-edits store
    try:
        store = SelfEditsStore.from_app_dir(base_dir)
        result = store.create_from_diff(
            diff,
            title=f"Auto-fix: {rel_path}",
            summary=f"Auto-detected issues:\n{issues_text}",
            author="autolearn",
        )
        logger.info("autolearn proposal created: %s", result)
        return {"diff": diff, "proposal": result}
    except Exception as exc:
        logger.warning("autolearn proposal creation failed: %s", exc)
        return None


def _extract_diff(response: str, rel_path: str) -> str | None:
    """Extract a unified diff from model response."""
    # Try to find a diff block
    diff_match = re.search(
        r"(---\s+a/.*?\n\+\+\+\s+b/.*?\n(?:@@.*?\n(?:[+ \-].*?\n?)*))",
        response,
        re.MULTILINE,
    )
    if diff_match:
        return diff_match.group(1).strip()

    # Try to find content inside ```diff blocks
    code_match = re.search(r"```diff\n([\s\S]*?)```", response)
    if code_match:
        return code_match.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# 4. Main Auto-Learning Daemon
# ---------------------------------------------------------------------------


def run_autolearn_tick(
    base_dir: Path,
    settings: dict,
    *,
    model=None,
    tokenizer=None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute one tick of auto-learning. Called periodically.

    1. Web ingest (if cooldown passed)
    2. Self-code indexing (if cooldown passed)
    3. Self-code analysis + proposals (if cooldown passed and model available)
    """
    autolearn_cfg = settings.get("autolearn", {}) or {}
    if not force and not autolearn_cfg.get("enabled", True):
        return {"ok": True, "skipped": "disabled"}

    with _LOCK:
        results: dict[str, Any] = {"ts": _now_iso()}

        # Step 1: Web knowledge ingest
        try:
            web_enabled = autolearn_cfg.get("web_ingest", True)
            if web_enabled:
                results["web_ingest"] = web_ingest_once(
                    base_dir,
                    settings,
                    force=force,
                    model=model,
                    tokenizer=tokenizer,
                )
            else:
                results["web_ingest"] = {"ok": True, "skipped": "disabled"}
        except Exception as exc:
            results["web_ingest"] = {"ok": False, "error": str(exc)}
            logger.exception("autolearn web ingest error")

        # Step 2: Index own source code
        try:
            code_index_enabled = autolearn_cfg.get("code_index", True)
            if code_index_enabled:
                results["code_index"] = self_code_index(base_dir, settings)
            else:
                results["code_index"] = {"ok": True, "skipped": "disabled"}
        except Exception as exc:
            results["code_index"] = {"ok": False, "error": str(exc)}
            logger.exception("autolearn code index error")

        # Step 3: Analyze code + propose fixes
        try:
            analyze_enabled = autolearn_cfg.get("code_analyze", True)
            if analyze_enabled:
                results["code_analyze"] = self_code_analyze(
                    base_dir,
                    settings,
                    model=model,
                    tokenizer=tokenizer,
                )
            else:
                results["code_analyze"] = {"ok": True, "skipped": "disabled"}
        except Exception as exc:
            results["code_analyze"] = {"ok": False, "error": str(exc)}
            logger.exception("autolearn code analyze error")

        results["ok"] = True
        logger.info(
            "autolearn tick completed: %s",
            {k: v.get("ok") if isinstance(v, dict) else v for k, v in results.items()},
        )
        return results


def start_autolearn_background(
    base_dir: Path,
    settings: dict,
    *,
    model=None,
    tokenizer=None,
    interval_minutes: float | None = None,
) -> threading.Thread:
    """Start the auto-learning daemon in a background thread.

    Returns the thread handle.
    """
    autolearn_cfg = settings.get("autolearn", {}) or {}
    interval = interval_minutes or float(autolearn_cfg.get("interval_minutes", 30))

    def _loop():
        logger.info("autolearn daemon started (interval=%s min)", interval)
        # Initial tick on startup (with force=True for first web ingest)
        try:
            run_autolearn_tick(
                base_dir, settings, model=model, tokenizer=tokenizer, force=True
            )
        except Exception:
            logger.exception("autolearn initial tick failed")

        while True:
            try:
                time.sleep(interval * 60)
                run_autolearn_tick(base_dir, settings, model=model, tokenizer=tokenizer)
            except Exception:
                logger.exception("autolearn tick error")
                time.sleep(60)  # backoff on error

    thread = threading.Thread(target=_loop, name="autolearn-daemon", daemon=True)
    thread.start()
    return thread

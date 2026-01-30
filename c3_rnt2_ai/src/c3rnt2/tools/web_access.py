from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests


_BUCKET_LOCK = threading.Lock()
_BUCKETS: dict[tuple[str, int], "TokenBucket"] = {}


@dataclass
class WebFetchResult:
    ok: bool
    url: str
    status: int | None
    text: str
    from_cache: bool
    error: str | None = None


class TokenBucket:
    def __init__(self, rate_per_min: int):
        self.capacity = max(1, int(rate_per_min))
        self.tokens = float(self.capacity)
        self.refill_per_sec = float(self.capacity) / 60.0
        self.updated = time.time()

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.time()
        elapsed = max(0.0, now - self.updated)
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            self.updated = now
        if self.tokens < tokens:
            return False
        self.tokens -= tokens
        return True


def _normalize_allowlist(allowlist: Iterable[str]) -> list[str]:
    return [a.strip().lower() for a in allowlist if str(a).strip()]


def _domain_allowed(url: str, allowlist: Iterable[str]) -> bool:
    domain = urlparse(url).netloc.lower()
    if not domain:
        return False
    domain = domain.split("@")[-1]
    if domain.startswith("[") and domain.endswith("]"):
        domain = domain[1:-1]
    domain = domain.split(":")[0]
    allowed = _normalize_allowlist(allowlist)
    return any(domain == a or domain.endswith("." + a) for a in allowed)


def _cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _log_event(payload: dict) -> None:
    log_path = Path("data") / "logs" / "web_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _get_bucket(cache_dir: Path, rate_limit_per_min: int) -> TokenBucket:
    key = (str(cache_dir.resolve()), int(rate_limit_per_min))
    with _BUCKET_LOCK:
        bucket = _BUCKETS.get(key)
        if bucket is None:
            bucket = TokenBucket(rate_per_min=rate_limit_per_min)
            _BUCKETS[key] = bucket
        return bucket


def web_fetch(
    url: str,
    *,
    allowlist: Iterable[str],
    max_bytes: int,
    timeout_s: float,
    cache_dir: str | Path,
    rate_limit_per_min: int = 30,
) -> WebFetchResult:
    if os.getenv("C3RNT2_WEB_DISABLED") or os.getenv("C3RNT2_SANDBOX_NO_NET"):
        return WebFetchResult(ok=False, url=url, status=None, text="", from_cache=False, error="web disabled by policy")
    scheme = urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        return WebFetchResult(ok=False, url=url, status=None, text="", from_cache=False, error="unsupported URL scheme")
    if not _domain_allowed(url, allowlist):
        return WebFetchResult(ok=False, url=url, status=None, text="", from_cache=False, error="domain not in allowlist")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bucket = _get_bucket(cache_dir, rate_limit_per_min)
    if not bucket.consume(1.0):
        return WebFetchResult(ok=False, url=url, status=None, text="", from_cache=False, error="rate limit exceeded")

    cache_path = _cache_path(cache_dir, url)
    cached = _load_cache(cache_path)
    headers: dict[str, str] = {}
    if cached:
        if cached.get("etag"):
            headers["If-None-Match"] = str(cached.get("etag"))
        if cached.get("last_modified"):
            headers["If-Modified-Since"] = str(cached.get("last_modified"))
        if not headers:
            payload = {
                "ts": time.time(),
                "url": url,
                "status": 200,
                "ok": True,
                "from_cache": True,
                "bytes": len(str(cached.get("text", "")).encode("utf-8", errors="ignore")),
                "error": None,
            }
            _log_event(payload)
            return WebFetchResult(
                ok=True,
                url=url,
                status=200,
                text=str(cached.get("text", "")),
                from_cache=True,
                error=None,
            )

    start = time.time()
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s, stream=True)
    except Exception as exc:
        _log_event(
            {
                "ts": time.time(),
                "url": url,
                "status": None,
                "ok": False,
                "from_cache": False,
                "bytes": 0,
                "error": f"request error: {exc}",
            }
        )
        return WebFetchResult(ok=False, url=url, status=None, text="", from_cache=False, error=f"request error: {exc}")

    if resp.status_code == 304 and cached:
        payload = {
            "ts": time.time(),
            "url": url,
            "status": 304,
            "ok": True,
            "from_cache": True,
            "bytes": len(str(cached.get("text", "")).encode("utf-8", errors="ignore")),
            "error": None,
        }
        _log_event(payload)
        return WebFetchResult(
            ok=True,
            url=url,
            status=304,
            text=str(cached.get("text", "")),
            from_cache=True,
        )

    if not resp.ok:
        _log_event(
            {
                "ts": time.time(),
                "url": url,
                "status": resp.status_code,
                "ok": False,
                "from_cache": False,
                "bytes": 0,
                "error": f"http {resp.status_code}",
            }
        )
        return WebFetchResult(ok=False, url=url, status=resp.status_code, text="", from_cache=False, error=f"http {resp.status_code}")

    content_type = str(resp.headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
    if content_type not in {"text/html", "text/plain"}:
        _log_event(
            {
                "ts": time.time(),
                "url": url,
                "status": resp.status_code,
                "ok": False,
                "from_cache": False,
                "bytes": 0,
                "error": f"unsupported content type: {content_type}",
            }
        )
        return WebFetchResult(
            ok=False,
            url=url,
            status=resp.status_code,
            text="",
            from_cache=False,
            error=f"unsupported content type: {content_type}",
        )

    max_bytes = max(1, int(max_bytes))
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=16384):
        if not chunk:
            continue
        remaining = max_bytes - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            total += remaining
            break
        chunks.append(chunk)
        total += len(chunk)
    raw = b"".join(chunks)
    encoding = resp.encoding or "utf-8"
    text = raw.decode(encoding, errors="ignore")

    cache_payload = {
        "url": url,
        "etag": resp.headers.get("ETag"),
        "last_modified": resp.headers.get("Last-Modified"),
        "text": text,
        "content_type": content_type,
        "ts": time.time(),
    }
    _write_cache(cache_path, cache_payload)

    elapsed_ms = (time.time() - start) * 1000.0
    _log_event(
        {
            "ts": time.time(),
            "url": url,
            "status": resp.status_code,
            "ok": True,
            "from_cache": False,
            "bytes": len(raw),
            "elapsed_ms": round(elapsed_ms, 2),
            "error": None,
        }
    )
    return WebFetchResult(ok=True, url=url, status=resp.status_code, text=text, from_cache=False)

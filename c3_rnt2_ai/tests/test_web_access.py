<<<<<<< HEAD
﻿import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from c3rnt2.tools.web_access import web_fetch, reset_rate_limits


def _run_server(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
=======
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from c3rnt2.tools.web_access import web_fetch


class _Handler(BaseHTTPRequestHandler):
    cache_hits = 0
    etag_hits = 0

    def log_message(self, format, *args):  # noqa: N802
        return

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/cache"):
            _Handler.cache_hits += 1
            body = b"cached-response"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/etag"):
            _Handler.etag_hits += 1
            if self.headers.get("If-None-Match") == '"v1"':
                self.send_response(304)
                self.end_headers()
                return
            body = b"etag-response"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("ETag", '"v1"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
>>>>>>> 7ef3a231663391568cb83c4c686642e75f55c974
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


<<<<<<< HEAD
def test_web_fetch_allowlist_and_cache(tmp_path: Path):
    reset_rate_limits()
    hits = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits["count"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"hello world")

        def log_message(self, format, *args):
            return

    server = _run_server(Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    cache_dir = tmp_path / "cache"
    res = web_fetch(url, allowlist=["127.0.0.1"], cache_dir=cache_dir, rate_limit_per_min=10, cache_ttl_s=3600)
    assert res.ok
    res2 = web_fetch(url, allowlist=["127.0.0.1"], cache_dir=cache_dir, rate_limit_per_min=10, cache_ttl_s=3600)
    assert res2.ok
    assert hits["count"] == 1
    server.shutdown()


def test_web_fetch_rate_limit(tmp_path: Path):
    reset_rate_limits()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            return

    server = _run_server(Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    cache_dir = tmp_path / "cache"
    res = web_fetch(url, allowlist=["127.0.0.1"], cache_dir=cache_dir, rate_limit_per_min=1, cache_ttl_s=0)
    assert res.ok
    res2 = web_fetch(url, allowlist=["127.0.0.1"], cache_dir=cache_dir, rate_limit_per_min=1, cache_ttl_s=0)
    assert not res2.ok
    server.shutdown()


def test_web_fetch_allowlist_block(tmp_path: Path):
    reset_rate_limits()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            return

    server = _run_server(Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    cache_dir = tmp_path / "cache"
    res = web_fetch(url, allowlist=["example.com"], cache_dir=cache_dir, rate_limit_per_min=10)
    assert not res.ok
    server.shutdown()
=======
def test_web_allowlist_blocks(tmp_path):
    server = _start_server()
    try:
        url = f"http://127.0.0.1:{server.server_port}/cache"
        result = web_fetch(
            url,
            allowlist=["example.com"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=5,
        )
        assert not result.ok
        err = result.error or ""
        assert ("allowlist" in err) or ("web disabled" in err)
    finally:
        server.shutdown()


def test_web_cache_hits(tmp_path):
    server = _start_server()
    _Handler.cache_hits = 0
    try:
        url = f"http://127.0.0.1:{server.server_port}/cache"
        first = web_fetch(
            url,
            allowlist=["127.0.0.1"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=5,
        )
        second = web_fetch(
            url,
            allowlist=["127.0.0.1"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=5,
        )
        if not first.ok:
            assert "web disabled" in (first.error or "")
            return
        assert second.ok
        assert second.from_cache
        assert _Handler.cache_hits == 1
    finally:
        server.shutdown()


def test_web_rate_limit(tmp_path):
    server = _start_server()
    try:
        url = f"http://127.0.0.1:{server.server_port}/etag"
        first = web_fetch(
            url,
            allowlist=["127.0.0.1"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=1,
        )
        second = web_fetch(
            url,
            allowlist=["127.0.0.1"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=1,
        )
        if not first.ok:
            assert "web disabled" in (first.error or "")
            return
        assert not second.ok
        err = second.error or ""
        assert ("rate limit" in err) or ("web disabled" in err)
    finally:
        server.shutdown()
>>>>>>> 7ef3a231663391568cb83c4c686642e75f55c974

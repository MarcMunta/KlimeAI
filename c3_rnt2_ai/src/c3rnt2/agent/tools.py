from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .policies import WebPolicy


@dataclass
class ToolResult:
    ok: bool
    output: str


class AgentTools:
    def __init__(self, allowlist: List[str]):
        self.policy = WebPolicy(allowlist=allowlist)

    def run_tests(self, repo_path: Path) -> ToolResult:
        try:
            result = subprocess.run(
                ["pytest", "-q"],
                cwd=str(repo_path),
                check=False,
                capture_output=True,
                text=True,
            )
            out = result.stdout + result.stderr
            return ToolResult(ok=result.returncode == 0, output=out.strip())
        except Exception as exc:
            return ToolResult(ok=False, output=f"pytest failed: {exc}")

    def search_web(self, query: str) -> ToolResult:
        if not self.policy.check_rate():
            return ToolResult(ok=False, output="rate limit exceeded")
        # MVP: naive GET to duckduckgo html if allowed (unlikely). Return stub if blocked.
        url = f"https://duckduckgo.com/html/?q={query}"
        if not self.policy.allow_url(url):
            return ToolResult(ok=False, output="domain not in allowlist")
        try:
            resp = requests.get(url, timeout=10)
            return ToolResult(ok=resp.ok, output=resp.text[:1000])
        except Exception as exc:
            return ToolResult(ok=False, output=f"web error: {exc}")

    def open_docs(self, url: str) -> ToolResult:
        if not self.policy.check_rate():
            return ToolResult(ok=False, output="rate limit exceeded")
        if not self.policy.allow_url(url):
            return ToolResult(ok=False, output="domain not in allowlist")
        try:
            resp = requests.get(url, timeout=10)
            return ToolResult(ok=resp.ok, output=resp.text[:1200])
        except Exception as exc:
            return ToolResult(ok=False, output=f"web error: {exc}")

    def edit_repo(self, file_path: Path, new_text: str) -> ToolResult:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(new_text, encoding="utf-8")
            return ToolResult(ok=True, output=str(file_path))
        except Exception as exc:
            return ToolResult(ok=False, output=f"edit failed: {exc}")

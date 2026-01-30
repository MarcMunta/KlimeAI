from __future__ import annotations

import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..tools.web_access import web_fetch


@dataclass
class ToolResult:
    ok: bool
    output: str


class AgentTools:
    def __init__(
        self,
        allowlist: List[str],
        sandbox_root: Path | None = None,
        cache_root: Path | None = None,
        rate_limit_per_min: int = 30,
        web_enabled: bool | None = None,
        web_cfg: dict | None = None,
    ):
        self.web_cfg = dict(web_cfg or {})
        self.web_enabled = bool(web_enabled) if web_enabled is not None else bool(self.web_cfg.get("enabled", False))
        allow_domains = self.web_cfg.get("allow_domains") or self.web_cfg.get("allowlist") or allowlist
        self.allowlist = [str(a) for a in (allow_domains or [])]
        self.sandbox_root = sandbox_root or Path("data") / "workspaces"
        self.cache_root = Path(self.web_cfg.get("cache_dir") or cache_root or Path("data") / "web_cache")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.rate_limit_per_min = int(self.web_cfg.get("rate_limit_per_min", rate_limit_per_min))
        self.max_bytes = int(self.web_cfg.get("max_bytes", 1_000_000))
        self.timeout_s = float(self.web_cfg.get("timeout_s", 10))

    def _sanitize_text(self, text: str) -> str:
        text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    def _fetch(self, url: str) -> ToolResult:
        if not self.web_enabled:
            return ToolResult(ok=False, output="web tool disabled")
        result = web_fetch(
            url,
            allowlist=self.allowlist,
            max_bytes=self.max_bytes,
            timeout_s=self.timeout_s,
            cache_dir=self.cache_root,
            rate_limit_per_min=self.rate_limit_per_min,
        )
        if not result.ok:
            return ToolResult(ok=False, output=result.error or "web fetch failed")
        text = self._sanitize_text(result.text) if result.text else ""
        return ToolResult(ok=True, output=text)

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
        if not self.web_enabled:
            return ToolResult(ok=False, output="web tool disabled")
        # MVP: naive GET to duckduckgo html if allowlisted.
        url = f"https://duckduckgo.com/html/?q={query}"
        result = self._fetch(url)
        if not result.ok:
            return result
        return ToolResult(ok=True, output=result.output[:1000])

    def open_docs(self, url: str) -> ToolResult:
        result = self._fetch(url)
        if not result.ok:
            return result
        return ToolResult(ok=True, output=result.output[:1200])

    def web_fetch(self, url: str) -> ToolResult:
        return self._fetch(url)

    def edit_repo(self, file_path: Path, new_text: str) -> ToolResult:
        try:
            self.sandbox_root.mkdir(parents=True, exist_ok=True)
            # Ensure edits only happen in sandbox workspace
            if not str(file_path).startswith(str(self.sandbox_root)):
                file_path = self.sandbox_root / file_path.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(new_text, encoding="utf-8")
            return ToolResult(ok=True, output=str(file_path))
        except Exception as exc:
            return ToolResult(ok=False, output=f"edit failed: {exc}")

    def propose_patch(self, repo_root: Path, changes: Dict[Path, str]) -> ToolResult:
        try:
            from ..selfimprove.patch_ops import propose_patch

            diff = propose_patch(repo_root, changes)
            return ToolResult(ok=True, output=diff)
        except Exception as exc:
            return ToolResult(ok=False, output=f"propose failed: {exc}")

    def validate_patch(self, repo_root: Path, diff_text: str) -> ToolResult:
        try:
            from ..selfimprove.patch_ops import validate_patch
            from ..selfimprove.safety_kernel import SafetyPolicy

            result = validate_patch(repo_root, diff_text, SafetyPolicy())
            return ToolResult(ok=result.ok, output=result.message)
        except Exception as exc:
            return ToolResult(ok=False, output=f"validate failed: {exc}")

    def apply_patch(self, repo_root: Path, diff_text: str, approve: bool = False) -> ToolResult:
        try:
            from ..selfimprove.patch_ops import apply_patch

            result = apply_patch(repo_root, diff_text, approve=approve)
            return ToolResult(ok=result.ok, output=result.message)
        except Exception as exc:
            return ToolResult(ok=False, output=f"apply failed: {exc}")

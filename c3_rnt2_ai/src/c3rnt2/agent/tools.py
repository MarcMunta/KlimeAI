from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..tools.web_access import web_fetch
from .sandbox import run_sandbox_command


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
        web_cfg: dict | None = None,
        agent_cfg: dict | None = None,
        self_patch_cfg: dict | None = None,
    ):
        raw_cfg = dict(web_cfg or {})
        if "web" in raw_cfg:
            raw_cfg = raw_cfg.get("web", {}) or {}
        self.web_cfg = raw_cfg
        self.agent_cfg = dict(agent_cfg or {})
        self.self_patch_cfg = dict(self_patch_cfg or {})
        self.web_enabled = bool(self.web_cfg.get("enabled", False))
        self.allowlist = list(self.web_cfg.get("allow_domains", allowlist) or [])
        self.cache_root = Path(self.web_cfg.get("cache_dir") or cache_root or Path("data") / "web_cache")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.rate_limit_per_min = int(self.web_cfg.get("rate_limit_per_min", rate_limit_per_min))
        self.max_bytes = int(self.web_cfg.get("max_bytes", 512_000))
        self.timeout_s = float(self.web_cfg.get("timeout_s", 10))
        cache_ttl = self.web_cfg.get("cache_ttl_s", None)
        self.cache_ttl_s = int(cache_ttl) if cache_ttl is not None else None
        self.allow_content_types = self.web_cfg.get("allow_content_types")
        self.sandbox_root = sandbox_root or Path("data") / "workspaces"
        self.allow_git = bool(self.agent_cfg.get("allow_git", False))

    def _sanitize_text(self, text: str) -> str:
        text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    def web_fetch(self, url: str) -> ToolResult:
        if not self.web_enabled:
            return ToolResult(ok=False, output="web disabled")
        result = web_fetch(
            url,
            allowlist=self.allowlist,
            max_bytes=self.max_bytes,
            timeout_s=self.timeout_s,
            cache_dir=self.cache_root,
            rate_limit_per_min=self.rate_limit_per_min,
            cache_ttl_s=self.cache_ttl_s,
            allow_content_types=self.allow_content_types,
        )
        if not result.ok:
            return ToolResult(ok=False, output=result.error or "fetch failed")
        text = self._sanitize_text(result.text)
        return ToolResult(ok=True, output=text)

    def fetch_page(self, url: str) -> ToolResult:
        return self.web_fetch(url)

    def search_web(self, query: str) -> ToolResult:
        url = f"https://duckduckgo.com/html/?q={query}"
        result = self.web_fetch(url)
        if not result.ok:
            return result
        return ToolResult(ok=True, output=result.output[:1000])

    def open_docs(self, url: str) -> ToolResult:
        result = self.fetch_page(url)
        if not result.ok:
            return result
        return ToolResult(ok=True, output=result.output[:1200])

    def run_tests(self, repo_path: Path) -> ToolResult:
        result = run_sandbox_command(repo_path, ["pytest", "-q"], self.sandbox_root, timeout_s=300)
        out = (result.get("stdout", "") + result.get("stderr", "")).strip()
        return ToolResult(ok=bool(result.get("ok")), output=out)

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

    def propose_patch(
        self,
        repo_root: Path,
        changes: Dict[Path, str],
        goal: str = "agent_patch",
        *,
        llm_generate_diff: bool = False,
        llm_context: dict | None = None,
    ) -> ToolResult:
        try:
            from ..self_patch.propose_patch import propose_patch

            context = {"changes": {str(k): v for k, v in changes.items()}}
            if llm_context:
                context.update(llm_context)
            proposal = propose_patch(
                goal,
                context,
                repo_root,
                settings={"self_patch": self.self_patch_cfg} if self.self_patch_cfg else {},
                llm_generate_diff=llm_generate_diff,
            )
            return ToolResult(ok=True, output=proposal.patch_id)
        except Exception as exc:
            return ToolResult(ok=False, output=f"propose failed: {exc}")

    def sandbox_patch(self, repo_root: Path, patch_id: str) -> ToolResult:
        try:
            from ..self_patch.sandbox_run import sandbox_run

            result = sandbox_run(repo_root, patch_id, settings={"self_patch": self.self_patch_cfg} if self.self_patch_cfg else {})
            return ToolResult(ok=bool(result.get("ok")), output=json.dumps(result))
        except Exception as exc:
            return ToolResult(ok=False, output=f"sandbox failed: {exc}")

    def apply_patch(self, repo_root: Path, patch_id: str, approve: bool = False) -> ToolResult:
        try:
            if not approve:
                return ToolResult(ok=False, output="approval required")
            from ..self_patch.apply_patch import apply_patch

            result = apply_patch(patch_id, repo_root, settings={"self_patch": self.self_patch_cfg} if self.self_patch_cfg else {})
            return ToolResult(ok=result.ok, output=result.error or "applied")
        except Exception as exc:
            return ToolResult(ok=False, output=f"apply failed: {exc}")

    def write_patch(self, repo_root: Path, diff_text: str, goal: str = "agent_patch", create_branch: bool = False) -> ToolResult:
        if create_branch and not self.allow_git:
            return ToolResult(ok=False, output="git disabled")
        try:
            from ..self_patch.propose_patch import propose_patch

            proposal = propose_patch(goal, {"changes": {}}, repo_root, settings={"self_patch": self.self_patch_cfg} if self.self_patch_cfg else {}, diff_text=diff_text)
            branch = None
            if create_branch and self.allow_git:
                branch = f"agent/{proposal.patch_id}"
                subprocess.run(["git", "checkout", "-b", branch], cwd=str(repo_root), check=False)
            payload = {"patch_id": proposal.patch_id, "branch": branch}
            return ToolResult(ok=True, output=json.dumps(payload))
        except Exception as exc:
            return ToolResult(ok=False, output=f"write_patch failed: {exc}")

    def summarize_diff(self, repo_root: Path) -> ToolResult:
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
            )
            return ToolResult(ok=True, output=result.stdout.strip())
        except Exception as exc:
            return ToolResult(ok=False, output=f"diff failed: {exc}")

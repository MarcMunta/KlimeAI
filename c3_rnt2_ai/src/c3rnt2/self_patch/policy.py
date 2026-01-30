from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ALLOWED_PATHS = (
    "src/c3rnt2/",
    "tests/",
    "config/",
    "README.md",
    "TODO.md",
    "pyproject.toml",
    "scripts/",
)
DEFAULT_FORBIDDEN_GLOBS = (
    ".env",
    ".env.*",
    "data/**",
    "*.key",
    "*.pem",
    "*.p12",
    "*.sqlite",
    "*.db",
)
DEFAULT_MAX_PATCH_KB = 128


@dataclass(frozen=True)
class SelfPatchPolicy:
    allowed_paths: tuple[str, ...]
    forbidden_globs: tuple[str, ...]
    max_patch_kb: int


def normalize_path(repo_root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return str(rel).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def parse_diff_paths(diff_text: str) -> list[str]:
    paths: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            rel = line[6:].strip()
            if rel and rel != "/dev/null":
                paths.append(rel)
    return paths


def _matches_allowed(rel: str, allowlist: Iterable[str]) -> bool:
    if not allowlist:
        return False
    for prefix in allowlist:
        if not prefix:
            continue
        if rel == prefix.rstrip("/"):
            return True
        if rel.startswith(prefix):
            return True
    return False


def is_forbidden(rel: str, forbidden_globs: Iterable[str]) -> bool:
    rel_norm = rel.replace("\\", "/")
    for pattern in forbidden_globs:
        if fnmatch.fnmatch(rel_norm, pattern):
            return True
    return False


def validate_patch(repo_root: Path, diff_text: str, policy: SelfPatchPolicy) -> tuple[bool, str, list[str]]:
    if len(diff_text.encode("utf-8")) > policy.max_patch_kb * 1024:
        return False, "patch too large", []
    paths = parse_diff_paths(diff_text)
    for rel in paths:
        if is_forbidden(rel, policy.forbidden_globs):
            return False, f"forbidden path: {rel}", paths
        if not _matches_allowed(rel, policy.allowed_paths):
            return False, f"path not in allowlist: {rel}", paths
    return True, "ok", paths

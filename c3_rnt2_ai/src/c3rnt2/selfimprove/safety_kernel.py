from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class SafetyPolicy:
    max_patch_kb: int = 64
    allow_commands: tuple[str, ...] = ("pytest",)


SAFETY_KERNEL_PATHS = {
    "src/c3rnt2/selfimprove/safety_kernel.py",
    "src/c3rnt2/selfimprove/patch_ops.py",
    "src/c3rnt2/selfimprove/improve_loop.py",
}

FORBIDDEN_PATH_PREFIXES = {
    ".git",
    "data/registry",
}


def normalize_path(repo_root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return str(rel).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def is_forbidden(repo_root: Path, path: Path) -> bool:
    rel = normalize_path(repo_root, path)
    if rel in SAFETY_KERNEL_PATHS:
        return True
    for prefix in FORBIDDEN_PATH_PREFIXES:
        if rel.startswith(prefix):
            return True
    return False

<<<<<<< HEAD
﻿from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Iterable

from .utils import diff_paths


DEFAULT_ALLOWED = [
    "src/",
    "tests/",
]
DEFAULT_FORBIDDEN = [
    ".env",
    ".env.",
    ".git/",
    "data/",
    "data/db",
    "data/keys",
    "keys",
    "secrets",
]


def _normalize(path: Path) -> str:
    return str(path).replace("\\", "/")


def _is_allowed(path: Path, allowed: Iterable[str], forbidden: Iterable[str]) -> bool:
    rel = _normalize(path)
    for forb in forbidden:
        if rel.startswith(forb):
            return False
    for allow in allowed:
        if rel.startswith(allow) or rel == allow:
            return True
    return False


def apply_patch(
    base_dir: Path,
    patch_id: str,
    settings: dict | None = None,
    *,
    human_approved: bool = False,
) -> dict:
    queue_dir = base_dir / "data" / "self_patch" / "queue" / patch_id
    patch_path = queue_dir / "patch.diff"
    sandbox_path = queue_dir / "sandbox.json"
    if not human_approved:
        return {"ok": False, "error": "human approval required"}
    if not patch_path.exists():
        return {"ok": False, "error": "patch.diff not found"}
    if not sandbox_path.exists():
        return {"ok": False, "error": "sandbox.json not found"}
    sandbox = json.loads(sandbox_path.read_text(encoding="utf-8"))
    if not sandbox.get("ok"):
        return {"ok": False, "error": "sandbox not ok"}

    diff_text = patch_path.read_text(encoding="utf-8")
    paths = diff_paths(diff_text)
    cfg = settings.get("self_patch", {}) if settings else {}
    allowed = cfg.get("allowed_paths", DEFAULT_ALLOWED)
    forbidden = cfg.get("forbidden_paths", DEFAULT_FORBIDDEN)
    for rel in paths:
        if not _is_allowed(rel, allowed, forbidden):
            return {"ok": False, "error": f"forbidden path: {rel}"}

    try:
        subprocess.run(["git", "apply", str(patch_path)], cwd=str(base_dir), check=True)
    except Exception as exc:
        return {"ok": False, "error": f"apply failed: {exc}"}
    meta_path = queue_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "applied"
            meta["applied_ts"] = time.time()
            meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
        except Exception:
            pass
    return {"ok": True, "patch_id": patch_id}
=======
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .policy import (
    SelfPatchPolicy,
    validate_patch,
    DEFAULT_ALLOWED_PATHS,
    DEFAULT_FORBIDDEN_GLOBS,
    DEFAULT_MAX_PATCH_KB,
)


@dataclass
class PatchApplyResult:
    ok: bool
    message: str


def _make_policy(settings: dict | None) -> SelfPatchPolicy:
    cfg = (settings or {}).get("self_patch", {}) or {}
    allowed = tuple(str(p) for p in cfg.get("allowed_paths", DEFAULT_ALLOWED_PATHS))
    forbidden = tuple(str(p) for p in cfg.get("forbidden_globs", DEFAULT_FORBIDDEN_GLOBS))
    max_patch_kb = int(cfg.get("max_patch_kb", DEFAULT_MAX_PATCH_KB))
    return SelfPatchPolicy(allowed_paths=allowed, forbidden_globs=forbidden, max_patch_kb=max_patch_kb)


def apply_patch(patch_id: str, repo_root: Path, settings: dict | None = None) -> PatchApplyResult:
    queue_dir = repo_root / "data" / "self_patch" / "queue" / patch_id
    patch_path = queue_dir / "patch.diff"
    meta_path = queue_dir / "meta.json"
    sandbox_path = queue_dir / "sandbox.json"

    if not patch_path.exists():
        return PatchApplyResult(ok=False, message=f"patch not found: {patch_id}")
    if not sandbox_path.exists():
        return PatchApplyResult(ok=False, message="sandbox.json missing")
    try:
        sandbox = json.loads(sandbox_path.read_text(encoding="utf-8"))
    except Exception:
        return PatchApplyResult(ok=False, message="invalid sandbox.json")
    if not sandbox.get("ok"):
        return PatchApplyResult(ok=False, message="sandbox not ok")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if not meta.get("ready_for_review", False):
            return PatchApplyResult(ok=False, message="patch not ready_for_review")

    diff_text = patch_path.read_text(encoding="utf-8")
    policy = _make_policy(settings)
    ok, message, _paths = validate_patch(repo_root, diff_text, policy)
    if not ok:
        return PatchApplyResult(ok=False, message=message)

    if not diff_text.strip():
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            meta["status"] = "applied"
            meta["applied_at"] = time.time()
            meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
        return PatchApplyResult(ok=True, message="applied (empty patch)")

    diff_file = Path(tempfile.mkstemp(suffix=".diff")[1])
    diff_file.write_text(diff_text, encoding="utf-8")
    try:
        subprocess.run(["git", "apply", str(diff_file)], cwd=str(repo_root), check=True)
    except Exception as exc:
        return PatchApplyResult(ok=False, message=f"apply failed: {exc}")
    finally:
        try:
            diff_file.unlink()
        except Exception:
            pass

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta["status"] = "applied"
        meta["applied_at"] = time.time()
        meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
    return PatchApplyResult(ok=True, message="applied")
>>>>>>> 7ef3a231663391568cb83c4c686642e75f55c974

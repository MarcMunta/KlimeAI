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

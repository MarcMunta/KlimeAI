<<<<<<< HEAD
﻿from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

from .utils import PatchMeta


def propose_patch(
    base_dir: Path,
    goal: str,
    context: Optional[str] = None,
    diff_text: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    patch_id = uuid.uuid4().hex[:10]
    queue_dir = base_dir / "data" / "self_patch" / "queue" / patch_id
    queue_dir.mkdir(parents=True, exist_ok=True)
    patch_path = queue_dir / "patch.diff"
    if diff_text is None:
        diff_text = ""
    if not dry_run:
        patch_path.write_text(diff_text, encoding="utf-8")
    else:
        patch_path.write_text(diff_text, encoding="utf-8")
    meta = PatchMeta(
        patch_id=patch_id,
        goal=goal,
        context=context,
        created_ts=time.time(),
        status="proposed",
    )
    meta_path = queue_dir / "meta.json"
    meta_path.write_text(json.dumps(meta.__dict__, ensure_ascii=True), encoding="utf-8")
    return {
        "ok": True,
        "patch_id": patch_id,
        "queue_dir": str(queue_dir),
        "diff_bytes": len(diff_text.encode("utf-8")),
    }
=======
from __future__ import annotations

import difflib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .policy import (
    SelfPatchPolicy,
    normalize_path,
    validate_patch,
    DEFAULT_ALLOWED_PATHS,
    DEFAULT_FORBIDDEN_GLOBS,
    DEFAULT_MAX_PATCH_KB,
)


@dataclass
class PatchProposal:
    patch_id: str
    queue_dir: Path
    patch_path: Path
    meta_path: Path
    meta: dict


def _build_diff(repo_root: Path, changes: dict[str, str]) -> str:
    diff_chunks: list[str] = []
    for rel_path, new_text in changes.items():
        abs_path = (repo_root / rel_path).resolve()
        old_text = ""
        if abs_path.exists():
            old_text = abs_path.read_text(encoding="utf-8", errors="ignore")
        rel = normalize_path(repo_root, abs_path)
        diff = difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
        diff_text = "\n".join(diff)
        if diff_text and not diff_text.endswith("\n"):
            diff_text += "\n"
        diff_chunks.append(diff_text)
    return "\n".join(diff_chunks)


def _parse_context(
    context: str | dict | None,
    repo_root: Path,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {"source": "none"}
    if context is None:
        return "", payload
    if isinstance(context, dict):
        payload = {"source": "dict"}
        if "patch" in context:
            return str(context["patch"]), payload
        if "changes" in context and isinstance(context["changes"], dict):
            return _build_diff(repo_root, {str(k): str(v) for k, v in context["changes"].items()}), payload
        return "", payload
    ctx_str = str(context)
    ctx_path = Path(ctx_str)
    if ctx_path.exists():
        payload = {"source": "file", "path": str(ctx_path)}
        ctx_str = ctx_path.read_text(encoding="utf-8", errors="ignore")
    # Attempt to parse JSON if possible.
    try:
        parsed = json.loads(ctx_str)
        if isinstance(parsed, dict):
            payload = {"source": "json"}
            if "patch" in parsed:
                return str(parsed["patch"]), payload
            if "changes" in parsed and isinstance(parsed["changes"], dict):
                return _build_diff(repo_root, {str(k): str(v) for k, v in parsed["changes"].items()}), payload
    except Exception:
        pass
    # If looks like a diff, return as-is.
    if "+++ b/" in ctx_str or ctx_str.lstrip().startswith("diff --git"):
        payload = {"source": "diff_text"}
        return ctx_str, payload
    payload = {"source": "text"}
    return "", payload


def _make_policy(settings: dict | None) -> SelfPatchPolicy:
    cfg = (settings or {}).get("self_patch", {}) or {}
    allowed = tuple(str(p) for p in cfg.get("allowed_paths", DEFAULT_ALLOWED_PATHS))
    forbidden = tuple(str(p) for p in cfg.get("forbidden_globs", DEFAULT_FORBIDDEN_GLOBS))
    max_patch_kb = int(cfg.get("max_patch_kb", DEFAULT_MAX_PATCH_KB))
    return SelfPatchPolicy(allowed_paths=allowed, forbidden_globs=forbidden, max_patch_kb=max_patch_kb)


def propose_patch(
    goal: str,
    context: str | dict | None,
    repo_root: Path,
    settings: dict | None = None,
) -> PatchProposal:
    patch_id = uuid.uuid4().hex[:12]
    queue_dir = repo_root / "data" / "self_patch" / "queue" / patch_id
    queue_dir.mkdir(parents=True, exist_ok=True)
    patch_path = queue_dir / "patch.diff"
    meta_path = queue_dir / "meta.json"

    diff_text, ctx_meta = _parse_context(context, repo_root)
    policy = _make_policy(settings)
    ok, message, paths = validate_patch(repo_root, diff_text, policy) if diff_text else (True, "empty patch", [])

    meta = {
        "id": patch_id,
        "goal": goal,
        "created_at": time.time(),
        "status": "proposed" if ok else "blocked",
        "validation": {"ok": ok, "message": message},
        "paths": paths,
        "context": ctx_meta,
        "patch_bytes": len(diff_text.encode("utf-8")),
        "ready_for_review": False,
    }
    patch_path.write_text(diff_text, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
    return PatchProposal(
        patch_id=patch_id,
        queue_dir=queue_dir,
        patch_path=patch_path,
        meta_path=meta_path,
        meta=meta,
    )
>>>>>>> 7ef3a231663391568cb83c4c686642e75f55c974

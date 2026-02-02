from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .policy import policy_from_settings, validate_patch
from .sandbox_run import sandbox_run
from .apply_patch import _resolve_queue_root


@dataclass
class JudgeResult:
    ok: bool
    patch_id: str
    message: str


class PatchJudge:
    def __init__(self, settings: dict | None = None):
        self.settings = settings or {}
        self.policy = policy_from_settings(self.settings)

    def evaluate(self, repo_root: Path, patch_id: str) -> JudgeResult:
        queue_root = _resolve_queue_root(repo_root, self.settings)
        queue_dir = queue_root / patch_id
        patch_path = queue_dir / "patch.diff"
        meta_path = queue_dir / "meta.json"
        if not patch_path.exists():
            return JudgeResult(ok=False, patch_id=patch_id, message="patch.diff not found")
        diff_text = patch_path.read_text(encoding="utf-8")
        ok, message, _paths = validate_patch(repo_root, diff_text, self.policy)
        if not ok:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["status"] = "blocked"
                meta["error"] = message
                meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
            return JudgeResult(ok=False, patch_id=patch_id, message=message)

        sandbox = sandbox_run(repo_root, patch_id, settings=self.settings)
        if not sandbox.get("ok"):
            return JudgeResult(ok=False, patch_id=patch_id, message="sandbox_failed")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "ready_for_review"
            meta["ready_for_review"] = True
            meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
        return JudgeResult(ok=True, patch_id=patch_id, message="ready_for_review")

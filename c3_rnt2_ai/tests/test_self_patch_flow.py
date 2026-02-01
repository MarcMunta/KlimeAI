from __future__ import annotations

import json
import subprocess
from pathlib import Path

from c3rnt2.self_patch.propose_patch import propose_patch
from c3rnt2.self_patch.apply_patch import apply_patch


def test_self_patch_propose_and_apply(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    src_dir = repo / "src" / "c3rnt2"
    src_dir.mkdir(parents=True, exist_ok=True)
    target = src_dir / "demo.txt"
    target.write_text("old\n", encoding="utf-8")

    settings = {
        "self_patch": {
            "allowed_paths": ["src/"],
            "forbidden_globs": ["data/**"],
            "max_patch_kb": 64,
        }
    }
    context = {"changes": {"src/c3rnt2/demo.txt": "new\n"}}
    proposal = propose_patch("update demo", context, repo, settings=settings)
    assert proposal.patch_path.exists()
    meta = json.loads(proposal.meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "proposed"

    sandbox = proposal.queue_dir / "sandbox.json"
    sandbox.write_text(json.dumps({"ok": True}), encoding="utf-8")
    meta["ready_for_review"] = True
    meta["status"] = "ready_for_review"
    proposal.meta_path.write_text(json.dumps(meta), encoding="utf-8")
    result = apply_patch(proposal.patch_id, repo, settings=settings)
    assert result.ok
    assert target.read_text(encoding="utf-8") == "new\n"

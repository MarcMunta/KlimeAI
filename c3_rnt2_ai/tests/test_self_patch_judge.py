from __future__ import annotations

import json
import subprocess
from pathlib import Path

from c3rnt2.self_patch.propose_patch import propose_patch
from c3rnt2.self_patch.judge import PatchJudge


def test_patch_judge_marks_ready(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    src_dir = repo / "src" / "c3rnt2"
    src_dir.mkdir(parents=True, exist_ok=True)
    target = src_dir / "demo.txt"
    target.write_text("old\n", encoding="utf-8")

    settings = {"self_patch": {"allowed_paths": ["src/"], "forbidden_globs": ["data/**"], "max_patch_kb": 64}}
    proposal = propose_patch("update demo", {"changes": {"src/c3rnt2/demo.txt": "new\n"}}, repo, settings=settings)
    judge = PatchJudge(settings=settings)
    result = judge.evaluate(repo, proposal.patch_id)
    assert result.ok
    meta = json.loads(proposal.meta_path.read_text(encoding="utf-8"))
    assert meta.get("status") == "ready_for_review"

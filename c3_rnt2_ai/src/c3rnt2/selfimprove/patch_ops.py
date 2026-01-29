from __future__ import annotations

import difflib
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json

from .safety_kernel import SafetyPolicy, is_forbidden, normalize_path
def _log_episode(repo_root: Path, payload: dict) -> None:
    path = repo_root / "data" / "episodes" / "agent.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")



@dataclass
class PatchResult:
    ok: bool
    message: str


def propose_patch(repo_root: Path, changes: Dict[Path, str]) -> str:
    diff_chunks: List[str] = []
    for path, new_text in changes.items():
        abs_path = (repo_root / path).resolve()
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


def _paths_from_diff(diff_text: str) -> List[Path]:
    paths: List[Path] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            rel = line[6:].strip()
            paths.append(Path(rel))
    return paths


def validate_patch(repo_root: Path, diff_text: str, policy: SafetyPolicy) -> PatchResult:
    if len(diff_text.encode("utf-8")) > policy.max_patch_kb * 1024:
        return PatchResult(ok=False, message="patch too large")
    for rel in _paths_from_diff(diff_text):
        if is_forbidden(repo_root, repo_root / rel):
            return PatchResult(ok=False, message=f"forbidden path: {rel}")

    sandbox_root = repo_root / "data" / "workspaces" / f"validate_{int(time.time())}"
    if sandbox_root.exists():
        shutil.rmtree(sandbox_root)
    shutil.copytree(repo_root, sandbox_root, ignore=shutil.ignore_patterns(".git", "data/registry"))

    diff_file = sandbox_root / "_patch.diff"
    diff_file.write_text(diff_text, encoding="utf-8")

    try:
        subprocess.run(["git", "apply", str(diff_file)], cwd=str(sandbox_root), check=True)
    except Exception as exc:
        return PatchResult(ok=False, message=f"apply failed: {exc}")

    try:
        result = subprocess.run(["pytest", "-q"], cwd=str(sandbox_root), capture_output=True, text=True)
        if result.returncode != 0:
            return PatchResult(ok=False, message="tests failed")
    except Exception as exc:
        return PatchResult(ok=False, message=f"tests error: {exc}")

    return PatchResult(ok=True, message="ok")


def apply_patch(repo_root: Path, diff_text: str, approve: bool = False) -> PatchResult:
    approve_flag = repo_root / "data" / "APPROVE_SELF_PATCH"
    if not approve and not approve_flag.exists():
        return PatchResult(ok=False, message="approval required")
    for rel in _paths_from_diff(diff_text):
        if is_forbidden(repo_root, repo_root / rel):
            return PatchResult(ok=False, message=f"forbidden path: {rel}")
    diff_file = Path(tempfile.mkstemp(suffix=".diff")[1])
    diff_file.write_text(diff_text, encoding="utf-8")
    try:
        subprocess.run(["git", "apply", str(diff_file)], cwd=str(repo_root), check=True)
        tests_ok = None
        tests_output = ""
        if approve:
            result = subprocess.run(["pytest", "-q"], cwd=str(repo_root), capture_output=True, text=True)
            tests_ok = result.returncode == 0
            tests_output = (result.stdout + result.stderr)[:500]
            _log_episode(repo_root, {
                "task": "self-improve apply_patch",
                "prompt": "pytest -q",
                "patch": diff_text,
                "tests_ok": tests_ok,
                "tests_output_excerpt": tests_output,
                "timestamp": time.time(),
            })
    except Exception as exc:
        return PatchResult(ok=False, message=f"apply failed: {exc}")
    finally:
        try:
            diff_file.unlink()
        except Exception:
            pass
    return PatchResult(ok=True, message="applied")

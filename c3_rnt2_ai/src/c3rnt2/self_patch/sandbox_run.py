from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .policy import (
    SelfPatchPolicy,
    validate_patch,
    DEFAULT_ALLOWED_PATHS,
    DEFAULT_FORBIDDEN_GLOBS,
    DEFAULT_MAX_PATCH_KB,
)


@dataclass
class SandboxResult:
    ok: bool
    sandbox_path: Path
    ruff: dict
    pytest: dict
    bench: dict | None
    network_disabled: bool


def _make_policy(settings: dict | None) -> SelfPatchPolicy:
    cfg = (settings or {}).get("self_patch", {}) or {}
    allowed = tuple(str(p) for p in cfg.get("allowed_paths", DEFAULT_ALLOWED_PATHS))
    forbidden = tuple(str(p) for p in cfg.get("forbidden_globs", DEFAULT_FORBIDDEN_GLOBS))
    max_patch_kb = int(cfg.get("max_patch_kb", DEFAULT_MAX_PATCH_KB))
    return SelfPatchPolicy(allowed_paths=allowed, forbidden_globs=forbidden, max_patch_kb=max_patch_kb)


def _copy_repo(repo_root: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "data")
    shutil.copytree(repo_root, dest, ignore=ignore)


def _run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> dict:
    try:
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
        output = (result.stdout + result.stderr).strip()
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "output": output[:1200],
            "cmd": " ".join(cmd),
        }
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": None, "output": f"command not found: {exc}", "cmd": " ".join(cmd)}
    except Exception as exc:
        return {"ok": False, "returncode": None, "output": f"error: {exc}", "cmd": " ".join(cmd)}


def _run_ruff(cwd: Path, env: dict[str, str]) -> dict:
    if shutil.which("ruff"):
        return _run_cmd(["ruff", "check", "."], cwd=cwd, env=env)
    # Fallback to python -m ruff if installed as module.
    module_check = _run_cmd([sys.executable, "-m", "ruff", "--version"], cwd=cwd, env=env)
    if module_check.get("ok"):
        return _run_cmd([sys.executable, "-m", "ruff", "check", "."], cwd=cwd, env=env)
    return {"ok": None, "returncode": None, "output": "ruff not available", "cmd": "ruff check ."}


def run_sandbox(
    patch_id: str,
    repo_root: Path,
    settings: dict | None = None,
    profile: str | None = None,
) -> SandboxResult:
    queue_dir = repo_root / "data" / "self_patch" / "queue" / patch_id
    patch_path = queue_dir / "patch.diff"
    meta_path = queue_dir / "meta.json"
    if not patch_path.exists():
        raise FileNotFoundError(f"patch not found: {patch_path}")
    diff_text = patch_path.read_text(encoding="utf-8")

    policy = _make_policy(settings)
    ok, message, _paths = validate_patch(repo_root, diff_text, policy)
    if not ok:
        sandbox_payload = {
            "ok": False,
            "error": f"validation failed: {message}",
            "ruff": None,
            "pytest": None,
            "bench": None,
            "network_disabled": True,
            "ts": time.time(),
        }
        (queue_dir / "sandbox.json").write_text(json.dumps(sandbox_payload, ensure_ascii=True), encoding="utf-8")
        return SandboxResult(ok=False, sandbox_path=repo_root, ruff={}, pytest={}, bench=None, network_disabled=True)

    sandbox_root = Path(settings.get("self_patch", {}).get("sandbox_dir", "data/self_patch/sandbox")) if settings else Path("data/self_patch/sandbox")
    sandbox_path = sandbox_root / patch_id
    _copy_repo(repo_root, sandbox_path)

    apply_result = {"ok": True, "output": "empty patch"}
    if diff_text.strip():
        diff_file = sandbox_path / "_patch.diff"
        diff_file.write_text(diff_text, encoding="utf-8")
        apply_result = _run_cmd(["git", "apply", str(diff_file.resolve())], cwd=sandbox_path, env=os.environ.copy())
        if not apply_result.get("ok"):
            sandbox_payload = {
                "ok": False,
                "error": f"apply failed: {apply_result.get('output')}",
                "ruff": None,
                "pytest": None,
                "bench": None,
                "network_disabled": True,
                "ts": time.time(),
            }
            (queue_dir / "sandbox.json").write_text(json.dumps(sandbox_payload, ensure_ascii=True), encoding="utf-8")
            return SandboxResult(ok=False, sandbox_path=sandbox_path, ruff={}, pytest={}, bench=None, network_disabled=True)

    env = os.environ.copy()
    env["C3RNT2_SANDBOX_NO_NET"] = "1"
    env["C3RNT2_WEB_DISABLED"] = "1"
    env["HTTP_PROXY"] = "http://127.0.0.1:9"
    env["HTTPS_PROXY"] = "http://127.0.0.1:9"
    if profile:
        env["C3RNT2_PROFILE"] = profile

    ruff_result = _run_ruff(sandbox_path, env)
    pytest_result = _run_cmd([sys.executable, "-m", "pytest", "-q"], cwd=sandbox_path, env=env)

    bench_result = None
    if torch is not None and torch.cuda.is_available():
        bench_cmd = [sys.executable, "-m", "c3rnt2", "bench", "--max-new-tokens", "16"]
        if profile:
            bench_cmd.extend(["--profile", profile])
        bench_result = _run_cmd(bench_cmd, cwd=sandbox_path, env=env)

    ok_all = bool(pytest_result.get("ok"))

    sandbox_payload = {
        "ok": ok_all,
        "ruff": ruff_result,
        "pytest": pytest_result,
        "bench": bench_result,
        "network_disabled": True,
        "ts": time.time(),
    }
    (queue_dir / "sandbox.json").write_text(json.dumps(sandbox_payload, ensure_ascii=True), encoding="utf-8")

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta["ready_for_review"] = bool(ok_all)
        meta["status"] = "ready_for_review" if ok_all else "sandbox_failed"
        meta["sandbox"] = {"ok": ok_all, "ts": time.time()}
        meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")

    return SandboxResult(
        ok=ok_all,
        sandbox_path=sandbox_path,
        ruff=ruff_result,
        pytest=pytest_result,
        bench=bench_result,
        network_disabled=True,
    )

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .continuous.dataset import ingest_sources
from .continuous.knowledge_store import KnowledgeStore, EmbeddingBackend
from .learning_loop.promoter import promote_latest
from .self_patch.policy import policy_from_settings, validate_patch
from .tools.web_ingest import ingest_urls
from .utils.locks import FileLock, LockUnavailable, is_lock_held


@dataclass
class AutopilotResult:
    ok: bool
    steps: dict[str, Any]
    error: str | None = None


def _state_path(base_dir: Path) -> Path:
    return base_dir / "data" / "state" / "autopilot.json"


def _log_path(base_dir: Path) -> Path:
    return base_dir / "data" / "logs" / "autopilot.jsonl"


def _load_state(base_dir: Path) -> dict:
    path = _state_path(base_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(base_dir: Path, state: dict) -> None:
    path = _state_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")


def _log_event(base_dir: Path, payload: dict) -> None:
    path = _log_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("ts", time.time())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _resolve_allowlist(settings: dict) -> list[str]:
    tools_web = settings.get("tools", {}).get("web", {}) or {}
    allow = tools_web.get("allow_domains")
    if allow:
        return list(allow)
    agent = settings.get("agent", {}) or {}
    return list(agent.get("web_allowlist", []))


def _cooldown_ok(last_ts: float, cooldown_min: float) -> bool:
    if cooldown_min <= 0:
        return True
    return (time.time() - last_ts) >= cooldown_min * 60.0


def _ingest_web(base_dir: Path, settings: dict, state: dict) -> dict[str, Any]:
    allowlist = _resolve_allowlist(settings)
    tools_web = settings.get("tools", {}).get("web", {}) or {}
    if not bool(tools_web.get("enabled", False)):
        return {"ok": False, "error": "tools.web.enabled=false"}
    if not allowlist:
        return {"ok": False, "error": "allow_domains required"}
    cont = settings.get("continuous", {}) or {}
    urls = cont.get("ingest_urls", [])
    if not urls:
        return {"ok": False, "error": "ingest_urls empty"}

    items = ingest_urls(urls, allowlist, base_dir=base_dir, settings=settings, state=state)
    if not items:
        return {"ok": True, "ingested": 0}

    knowledge_path = Path(cont.get("knowledge_path", base_dir / "data" / "continuous" / "knowledge.sqlite"))
    knowledge_cfg = settings.get("knowledge", {}) or {}
    embed_backend = knowledge_cfg.get("embedding_backend", "auto")
    embed_model = knowledge_cfg.get("embedding_model")
    embedder = EmbeddingBackend(backend=str(embed_backend), model_name=embed_model) if embed_model else embed_backend
    store = KnowledgeStore(
        knowledge_path,
        embedding_backend=embedder,
        index_backend=knowledge_cfg.get("index_backend", "auto"),
    )
    ingested = 0
    for item in items:
        ingested += store.ingest_text("web", item.source_ref, item.text, quality=0.0)
    return {"ok": True, "ingested": ingested, "items": len(items)}


def _train_subprocess(profile: str, reuse_dataset: bool, max_steps: int | None) -> dict:
    cmd = [sys.executable, "-m", "c3rnt2", "train-once", "--profile", profile]
    if reuse_dataset:
        cmd.append("--reuse-dataset")
    env = dict(os.environ)
    if max_steps is not None and int(max_steps) > 0:
        env["C3RNT2_TRAIN_MAX_STEPS"] = str(int(max_steps))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return {"ok": False, "ok_train": False, "ok_eval": False, "error": result.stderr.strip() or "train subprocess failed"}
    payload = None
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
    if payload is None:
        return {"ok": False, "ok_train": False, "ok_eval": False, "error": "train subprocess output not parseable"}
    return payload


def validate_autopatch_diff(base_dir: Path, settings: dict, diff_text: str) -> tuple[bool, str, list[str]]:
    policy = policy_from_settings(settings)
    return validate_patch(base_dir, diff_text, policy)


def _export_adapter_marker(base_dir: Path, adapter_path: str | None) -> None:
    if not adapter_path:
        return
    out_dir = base_dir / "data" / "registry" / "hf_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "adapter_current.pt"
    marker.write_text(str(adapter_path), encoding="utf-8")


def _run_cmd(cmd: list[str], base_dir: Path, env: dict | None = None, ok_codes: set[int] | None = None) -> tuple[bool, str]:
    result = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True, env=env)
    out = result.stdout.strip() if result.stdout else ""
    err = result.stderr.strip() if result.stderr else ""
    ok_codes = ok_codes or {0}
    return result.returncode in ok_codes, out or err


def _scan_todo_priority(base_dir: Path, pattern: str, allowed_roots: list[str]) -> int:
    regex = re.compile(pattern)
    count = 0
    for root in allowed_roots:
        root_path = base_dir / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if regex.search(text):
                count += 1
    return count


def _git_cmd(base_dir: Path, args: list[str]) -> tuple[bool, str]:
    return _run_cmd(["git"] + args, base_dir)


def _git_current_branch(base_dir: Path) -> str | None:
    ok, out = _git_cmd(base_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    return out if ok else None


def _git_checkout(base_dir: Path, branch: str, create: bool = False) -> tuple[bool, str]:
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(branch)
    return _git_cmd(base_dir, args)


def _git_commit_all(base_dir: Path, message: str) -> tuple[bool, str]:
    ok, out = _git_cmd(base_dir, ["add", "-A"])
    if not ok:
        return ok, out
    return _git_cmd(base_dir, ["commit", "-m", message])


def _git_tag(base_dir: Path, name: str) -> tuple[bool, str]:
    return _git_cmd(base_dir, ["tag", name])


def _git_merge_ff(base_dir: Path, branch: str) -> tuple[bool, str]:
    return _git_cmd(base_dir, ["merge", "--ff-only", branch])


def _maybe_autopatch(
    base_dir: Path,
    settings: dict,
    *,
    eval_short: dict | None,
    profile: str,
    mock: bool,
) -> dict[str, Any]:
    cfg = settings.get("autopilot", {}) or {}
    if not bool(cfg.get("autopatch_enabled", False)) or mock:
        return {"ok": True, "skipped": "disabled_or_mock"}
    if is_lock_held(base_dir, "self_patch"):
        return {"ok": False, "skipped": "self_patch_locked"}
    triggers: list[str] = []
    if bool(cfg.get("autopatch_on_test_fail", True)):
        ok, _msg = _run_cmd(["pytest", "-q"], base_dir, ok_codes={0, 5})
        if not ok:
            triggers.append("tests_failed")
    todo_pattern = cfg.get("todo_regex", r"TODO\((P1|PRIORITY)\)|TODO!|TODO:HIGH|TODO:CRITICAL")
    allowed = list((settings.get("self_patch", {}) or {}).get("allowed_paths", ["src/", "tests/"]))
    if todo_pattern:
        todo_hits = _scan_todo_priority(base_dir, str(todo_pattern), allowed)
        if todo_hits:
            triggers.append("todo_priority")
    if eval_short and eval_short.get("ok") is False:
        triggers.append("eval_regression")
    if not triggers:
        return {"ok": True, "skipped": "not_triggered"}

    if not bool((settings.get("self_patch", {}) or {}).get("enabled", False)):
        return {"ok": True, "skipped": "self_patch_disabled", "triggers": triggers}

    if not (base_dir / ".git").exists():
        return {"ok": False, "error": "git_repo_required", "triggers": triggers}

    goal = str(cfg.get("autopatch_goal", "autopilot fix"))
    branch_name = f"autopilot/{time.strftime('%Y%m%d_%H%M%S')}"
    original_branch = _git_current_branch(base_dir)
    if not original_branch:
        return {"ok": False, "error": "git_branch_unavailable", "triggers": triggers}

    ok, msg = _git_checkout(base_dir, branch_name, create=True)
    if not ok:
        return {"ok": False, "error": f"git_checkout_failed: {msg}", "triggers": triggers}

    try:
        from .self_patch.propose_patch import propose_patch
        from .self_patch.sandbox_run import sandbox_run
        from .self_patch.apply_patch import apply_patch
    except Exception as exc:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": f"self_patch_import_failed: {exc}", "triggers": triggers}

    try:
        proposal = propose_patch(goal, {"changes": {}}, base_dir, settings=settings, llm_generate_diff=True)
    except Exception as exc:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": f"propose_patch_failed: {exc}", "triggers": triggers}

    diff_text = ""
    try:
        diff_text = proposal.patch_path.read_text(encoding="utf-8")
    except Exception:
        diff_text = ""
    ok_diff, msg_diff, _paths = validate_autopatch_diff(base_dir, settings, diff_text)
    if not ok_diff:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": f"patch_rejected: {msg_diff}", "triggers": triggers}

    sandbox = sandbox_run(base_dir, proposal.patch_id, settings=settings, allow_empty=True)
    if not sandbox.get("ok"):
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": "sandbox_failed", "triggers": triggers}

    try:
        meta = json.loads(proposal.meta_path.read_text(encoding="utf-8"))
        meta["ready_for_review"] = True
        meta["status"] = "ready_for_review"
        proposal.meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass

    apply_result = apply_patch(proposal.patch_id, base_dir, settings=settings)
    if not apply_result.ok:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": f"apply_failed: {apply_result.error}", "triggers": triggers}

    ok_commit, msg_commit = _git_commit_all(base_dir, f"autopilot: {goal}")
    if not ok_commit:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": f"commit_failed: {msg_commit}", "triggers": triggers}

    ok_tests, _msg_tests = _run_cmd(["pytest", "-q"], base_dir, ok_codes={0, 5})
    if not ok_tests:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": "tests_failed", "triggers": triggers}

    env = dict(os.environ)
    env["C3RNT2_NO_NET"] = "1"
    ok_doc, _msg_doc = _run_cmd([sys.executable, "-m", "c3rnt2", "doctor", "--deep", "--profile", profile], base_dir, env=env)
    if not ok_doc:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": "doctor_failed", "triggers": triggers}

    ok_checkout, _msg_checkout = _git_checkout(base_dir, "main")
    if not ok_checkout:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": "checkout_main_failed", "triggers": triggers}

    ok_tag, _msg_tag = _git_tag(base_dir, f"autopilot/backup-{time.strftime('%Y%m%d_%H%M%S')}")
    if not ok_tag:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": "tag_failed", "triggers": triggers}

    ok_merge, _msg_merge = _git_merge_ff(base_dir, branch_name)
    if not ok_merge:
        _git_checkout(base_dir, original_branch)
        return {"ok": False, "error": "merge_failed", "triggers": triggers}

    _git_checkout(base_dir, original_branch)
    return {"ok": True, "promoted": True, "branch": branch_name, "triggers": triggers}


def run_autopilot_tick(
    settings: dict,
    base_dir: Path,
    *,
    no_web: bool = False,
    mock: bool = False,
) -> AutopilotResult:
    state = _load_state(base_dir)
    steps: dict[str, Any] = {}
    state.setdefault("last_tick_ts", 0.0)
    state.setdefault("last_ingest_ts", 0.0)
    state.setdefault("last_train_ts", 0.0)
    state.setdefault("last_eval_ts", 0.0)
    state.setdefault("last_patch_ts", 0.0)

    lock = FileLock(base_dir / "data" / "locks" / "autopilot.lock")
    try:
        lock.acquire(blocking=False)
    except LockUnavailable as exc:
        return AutopilotResult(ok=False, steps={"lock": "unavailable"}, error=str(exc))

    try:
        autopilot_cfg = settings.get("autopilot", {}) or {}
        ingest_cooldown = float(autopilot_cfg.get("ingest_cooldown_minutes", 10))
        train_cooldown = float(autopilot_cfg.get("train_cooldown_minutes", 60))
        eval_cooldown = float(autopilot_cfg.get("eval_cooldown_minutes", 60))
        patch_cooldown = float(autopilot_cfg.get("patch_cooldown_minutes", 120))
        reuse_dataset = bool(autopilot_cfg.get("reuse_dataset", False))
        max_steps = autopilot_cfg.get("train_max_steps")
        min_improvement = float(autopilot_cfg.get("min_improvement", settings.get("hf_train", {}).get("eval", {}).get("min_improvement", 0.0)))

        # ingest (web + logs/episodes)
        if no_web:
            steps["web_ingest"] = {"ok": True, "skipped": "no_web"}
        else:
            cont = settings.get("continuous", {}) or {}
            ingest_web = bool(cont.get("ingest_web", True))
            if ingest_web and _cooldown_ok(float(state.get("last_ingest_ts", 0.0)), ingest_cooldown):
                steps["web_ingest"] = _ingest_web(base_dir, settings, state)
                state["last_ingest_ts"] = time.time()
            else:
                steps["web_ingest"] = {"ok": True, "skipped": "cooldown_or_disabled"}

        settings_no_web = json.loads(json.dumps(settings))
        settings_no_web.setdefault("continuous", {})["ingest_web"] = False
        try:
            new_docs = ingest_sources(base_dir, _resolve_allowlist(settings_no_web), settings_no_web)
        except Exception as exc:
            steps["ingest_sources"] = {"ok": False, "error": str(exc)}
        else:
            steps["ingest_sources"] = {"ok": True, "new_docs": new_docs}

        # training
        train_result = None
        if not mock and _cooldown_ok(float(state.get("last_train_ts", 0.0)), train_cooldown):
            if is_lock_held(base_dir, "train") or is_lock_held(base_dir, "self_patch"):
                steps["train"] = {"ok": False, "skipped": "lock_held"}
            else:
                profile = settings.get("_profile") or os.getenv("C3RNT2_PROFILE") or "dev_small"
                train_result = _train_subprocess(str(profile), reuse_dataset=reuse_dataset, max_steps=int(max_steps) if max_steps else None)
                steps["train"] = train_result
                state["last_train_ts"] = time.time()
        else:
            steps["train"] = {"ok": True, "skipped": "mock_or_cooldown"}

        # eval + promote
        if train_result and train_result.get("ok_train") and _cooldown_ok(float(state.get("last_eval_ts", 0.0)), eval_cooldown):
            eval_ok = bool(train_result.get("eval_ok", train_result.get("ok_eval", False)))
            improvement = train_result.get("improvement")
            steps["eval_short"] = {"ok": eval_ok, "improvement": improvement}
            state["last_eval_ts"] = time.time()
            if eval_ok and improvement is not None and float(improvement) >= min_improvement:
                steps["promote"] = promote_latest(base_dir, settings, min_improvement=min_improvement).__dict__
            else:
                steps["promote"] = {"ok": True, "promoted": False, "reason": "eval_not_ok_or_no_improvement"}
            if eval_ok:
                _export_adapter_marker(base_dir, train_result.get("adapter_dir"))
        else:
            steps["eval_short"] = {"ok": True, "skipped": "no_train_or_cooldown"}
            steps["promote"] = {"ok": True, "promoted": False, "skipped": "no_eval"}

        # autopatch (gated)
        if _cooldown_ok(float(state.get("last_patch_ts", 0.0)), patch_cooldown):
            profile = settings.get("_profile") or os.getenv("C3RNT2_PROFILE") or "dev_small"
            steps["autopatch"] = _maybe_autopatch(
                base_dir,
                settings,
                eval_short=steps.get("eval_short") if isinstance(steps.get("eval_short"), dict) else None,
                profile=str(profile),
                mock=mock,
            )
            state["last_patch_ts"] = time.time()
        else:
            steps["autopatch"] = {"ok": True, "skipped": "cooldown"}

        state["last_tick_ts"] = time.time()
        _save_state(base_dir, state)
        _log_event(base_dir, {"ok": True, "steps": steps})
        return AutopilotResult(ok=True, steps=steps)
    finally:
        lock.release()


def run_autopilot_loop(
    settings: dict,
    base_dir: Path,
    *,
    once: bool = False,
    interval_minutes: float | None = None,
    no_web: bool = False,
    mock: bool = False,
) -> None:
    interval = interval_minutes
    if interval is None:
        interval = float((settings.get("autopilot", {}) or {}).get("interval_minutes", settings.get("continuous", {}).get("interval_minutes", 30)))
    while True:
        result = run_autopilot_tick(settings, base_dir, no_web=no_web, mock=mock)
        print({"autopilot": {"ok": result.ok, "steps": result.steps, "error": result.error}})
        if once:
            break
        time.sleep(max(5.0, interval * 60.0))

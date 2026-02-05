from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .installer import approve as approve_stage
from .installer import stage as stage_source
from .proposals import apply_proposal, create_proposal
from .store import SkillStore, SkillsConfig


def _skills_root_from_args(args: argparse.Namespace) -> Path:
    raw = getattr(args, "skills_dir", None) or "skills"
    return Path(str(raw))


def _strict_from_args(args: argparse.Namespace) -> bool:
    cfg = SkillsConfig.from_env()
    val = getattr(args, "strict", None)
    if val is None:
        return bool(cfg.strict)
    return bool(val)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=True))


def cmd_skills_list(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    store = SkillStore(root)
    records, errors = store.refresh()
    payload = {
        "ok": not bool(errors),
        "errors": errors or None,
        "object": "list",
        "data": [r.to_public_dict(include_prompt=False) for r in store.list()],
    }
    _print_json(payload)


def cmd_skills_show(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    store = SkillStore(root)
    store.refresh()
    ref = str(getattr(args, "skill_ref") or "").strip()
    rec = store.get(ref)
    if rec is None:
        _print_json({"ok": False, "error": "skill_not_found"})
        sys.exit(1)
    _print_json({"ok": True, "skill": rec.to_public_dict(include_prompt=True)})


def cmd_skills_validate(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    store = SkillStore(root)
    strict = _strict_from_args(args)
    target = str(getattr(args, "skill_ref", "") or "").strip()
    validate_all = bool(getattr(args, "all", False)) or not target

    if validate_all:
        report = store.validate_all(strict=bool(strict))
        report["strict"] = bool(strict)
        _print_json(report)
        if not bool(report.get("ok", False)):
            sys.exit(1)
        return

    store.refresh()
    rec = store.get(target)
    if rec is None:
        _print_json({"ok": False, "error": "skill_not_found"})
        sys.exit(1)
    from .scanner import scan_tree

    errors: list[str] = []
    scan = scan_tree(rec.path, strict=bool(strict), max_files=500, max_total_bytes=2 * 1024 * 1024)
    if not scan.ok:
        errors.extend(scan.errors)
    if strict:
        safety = rec.spec.safety
        if safety.network or safety.filesystem_write or safety.shell:
            errors.append("unsafe_safety_flags")
        if not rec.trusted:
            errors.append("not_trusted")
    payload = {"ok": not errors, "strict": bool(strict), "errors": errors or None, "skill": rec.to_public_dict(include_prompt=False)}
    _print_json(payload)
    if errors:
        sys.exit(1)


def cmd_skills_stage(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    strict = _strict_from_args(args)
    source = str(getattr(args, "source") or "").strip()
    ref = getattr(args, "ref", None)
    subdir = getattr(args, "subdir", None)
    result = stage_source(root, source, ref=str(ref).strip() if ref else None, subdir=str(subdir).strip() if subdir else None, strict=bool(strict))
    payload = {"ok": bool(result.ok), "staged_id": result.staged_id, "errors": result.errors, "found": result.found, "strict": bool(strict)}
    _print_json(payload)
    if not bool(result.ok):
        sys.exit(1)


def cmd_skills_approve(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    strict = _strict_from_args(args)
    staged_id = str(getattr(args, "staged_id") or "").strip()
    namespace = str(getattr(args, "namespace") or "community").strip() or "community"
    payload = approve_stage(root, staged_id, namespace=namespace, strict=bool(strict))
    _print_json(payload)
    if not bool(payload.get("ok", False)):
        sys.exit(1)


def cmd_skills_install(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    strict = _strict_from_args(args)
    source = str(getattr(args, "source") or "").strip()
    ref = getattr(args, "ref", None)
    subdir = getattr(args, "subdir", None)
    namespace = str(getattr(args, "namespace") or "community").strip() or "community"
    stage_res = stage_source(root, source, ref=str(ref).strip() if ref else None, subdir=str(subdir).strip() if subdir else None, strict=bool(strict))
    if not bool(stage_res.ok) or not stage_res.staged_id:
        _print_json({"ok": False, "error": "stage_failed", "details": stage_res.errors})
        sys.exit(1)
    approve_res = approve_stage(root, stage_res.staged_id, namespace=namespace, strict=bool(strict))
    _print_json(approve_res)
    if not bool(approve_res.get("ok", False)):
        sys.exit(1)


def cmd_skills_remove(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    store = SkillStore(root)
    store.refresh()
    ref = str(getattr(args, "skill_ref") or "").strip()
    payload = store.remove(ref)
    _print_json(payload)
    if not bool(payload.get("ok", False)):
        sys.exit(1)


def cmd_skills_enable(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    store = SkillStore(root)
    store.refresh()
    ref = str(getattr(args, "skill_ref") or "").strip()
    store.set_enabled(ref, True)
    _print_json({"ok": True, "skill_ref": ref, "enabled": True})


def cmd_skills_disable(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    store = SkillStore(root)
    store.refresh()
    ref = str(getattr(args, "skill_ref") or "").strip()
    store.set_enabled(ref, False)
    _print_json({"ok": True, "skill_ref": ref, "enabled": False})


def cmd_skills_propose(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    ref = str(getattr(args, "skill_ref") or "").strip()
    change = str(getattr(args, "change") or "").strip()
    res = create_proposal(root, skill_ref=ref, change=change)
    payload = {"ok": bool(res.ok), "proposal_path": str(res.proposal_path) if res.proposal_path else None, "error": res.error}
    _print_json(payload)
    if not bool(res.ok):
        sys.exit(1)


def cmd_skills_apply_proposal(args: argparse.Namespace) -> None:
    root = _skills_root_from_args(args)
    strict = _strict_from_args(args)
    proposal_path = Path(str(getattr(args, "proposal") or "").strip())
    run_tests = bool(getattr(args, "run_tests", False))
    payload = apply_proposal(Path("."), root, proposal_path, strict=bool(strict), run_tests=bool(run_tests))
    _print_json(payload)
    if not bool(payload.get("ok", False)):
        sys.exit(1)


def register_skills_cli(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    skills = subparsers.add_parser("skills")
    skills.add_argument("--skills-dir", default="skills")
    skills.add_argument("--strict", dest="strict", action="store_true", default=None)
    skills.add_argument("--no-strict", dest="strict", action="store_false")

    skills_sub = skills.add_subparsers(dest="skills_command")

    ls = skills_sub.add_parser("list")
    ls.set_defaults(func=cmd_skills_list)

    show = skills_sub.add_parser("show")
    show.add_argument("skill_ref")
    show.set_defaults(func=cmd_skills_show)

    val = skills_sub.add_parser("validate")
    val.add_argument("skill_ref", nargs="?", default=None)
    val.add_argument("--all", action="store_true")
    val.set_defaults(func=cmd_skills_validate)

    stage = skills_sub.add_parser("stage")
    stage.add_argument("source")
    stage.add_argument("--ref", default=None)
    stage.add_argument("--subdir", default=None)
    stage.set_defaults(func=cmd_skills_stage)

    approve = skills_sub.add_parser("approve")
    approve.add_argument("staged_id")
    approve.add_argument("--namespace", default="community")
    approve.set_defaults(func=cmd_skills_approve)

    install = skills_sub.add_parser("install")
    install.add_argument("source")
    install.add_argument("--ref", default=None)
    install.add_argument("--subdir", default=None)
    install.add_argument("--namespace", default="community")
    install.set_defaults(func=cmd_skills_install)

    rm = skills_sub.add_parser("remove")
    rm.add_argument("skill_ref")
    rm.set_defaults(func=cmd_skills_remove)

    en = skills_sub.add_parser("enable")
    en.add_argument("skill_ref")
    en.set_defaults(func=cmd_skills_enable)

    dis = skills_sub.add_parser("disable")
    dis.add_argument("skill_ref")
    dis.set_defaults(func=cmd_skills_disable)

    prop = skills_sub.add_parser("propose")
    prop.add_argument("skill_ref")
    prop.add_argument("--change", required=True)
    prop.set_defaults(func=cmd_skills_propose)

    app = skills_sub.add_parser("apply-proposal")
    app.add_argument("proposal")
    app.add_argument("--run-tests", dest="run_tests", action="store_true")
    app.set_defaults(func=cmd_skills_apply_proposal)

    return skills

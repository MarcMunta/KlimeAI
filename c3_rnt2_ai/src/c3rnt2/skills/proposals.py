from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import parse_skill_ref
from .store import SkillStore


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class ProposalResult:
    ok: bool
    proposal_path: Path | None
    error: str | None


def create_proposal(skills_root: Path, *, skill_ref: str, change: str) -> ProposalResult:
    root = Path(skills_root)
    proposals_dir = root / "_proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_skill_ref(skill_ref)
    if parsed is None:
        return ProposalResult(ok=False, proposal_path=None, error="skill_ref_invalid")
    ns, sid = parsed

    change_text = str(change or "").strip()
    if not change_text:
        return ProposalResult(ok=False, proposal_path=None, error="change_missing")

    proposal_id = uuid.uuid4().hex[:12]
    proposal = {
        "proposal_id": proposal_id,
        "created_at": _utc_iso(),
        "skill_ref": f"{ns}/{sid}",
        "action": "append_prompt",
        "append_text": change_text,
    }
    path = proposals_dir / f"{proposal_id}.yaml"
    path.write_text(yaml.safe_dump(proposal, sort_keys=False), encoding="utf-8")
    return ProposalResult(ok=True, proposal_path=path, error=None)


def apply_proposal(
    base_dir: Path,
    skills_root: Path,
    proposal_path: Path,
    *,
    strict: bool,
    run_tests: bool,
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    skills_root = Path(skills_root)
    proposal_path = Path(proposal_path)

    if not proposal_path.exists():
        return {"ok": False, "error": "proposal_not_found"}
    try:
        payload = yaml.safe_load(proposal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"proposal_load_failed:{exc}"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "proposal_invalid"}

    skill_ref = str(payload.get("skill_ref") or "").strip()
    parsed = parse_skill_ref(skill_ref)
    if parsed is None:
        return {"ok": False, "error": "skill_ref_invalid"}
    ns, sid = parsed

    action = str(payload.get("action") or "").strip()
    if action != "append_prompt":
        return {"ok": False, "error": "action_not_supported"}

    append_text = str(payload.get("append_text") or "").strip()
    if not append_text:
        return {"ok": False, "error": "append_text_missing"}

    skill_dir = skills_root / ns / sid
    prompt_path = skill_dir / "prompt.md"
    if not prompt_path.exists():
        return {"ok": False, "error": "prompt_missing"}

    before = prompt_path.read_text(encoding="utf-8")
    backup_path = prompt_path.with_suffix(".prompt.bak")
    backup_path.write_text(before, encoding="utf-8")

    try:
        prompt_path.write_text(before.rstrip() + "\n\n# Proposal\n" + append_text.strip() + "\n", encoding="utf-8")

        store = SkillStore(skills_root)
        report = store.validate_all(strict=bool(strict))
        if not bool(report.get("ok", False)):
            raise RuntimeError("skills_validate_failed")

        if run_tests:
            result = subprocess.run(["pytest", "-q"], cwd=str(base_dir), check=False, capture_output=True, text=True)
            if int(result.returncode) != 0:
                raise RuntimeError("tests_failed")

    except Exception as exc:
        try:
            prompt_path.write_text(before, encoding="utf-8")
        except Exception:
            pass
        return {"ok": False, "error": str(exc)}
    finally:
        try:
            backup_path.unlink()
        except Exception:
            pass

    return {"ok": True, "skill_ref": skill_ref}


from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import SkillSafety, SkillSpec, build_skill_ref, estimate_tokens, parse_skill_yaml, validate_namespace
from .scanner import scan_tree


def _read_env_bool(*keys: str, default: bool) -> bool:
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        val = str(raw).strip().lower()
        if val in {"1", "true", "yes", "y", "on"}:
            return True
        if val in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _read_env_int(*keys: str, default: int, min_value: int, max_value: int) -> int:
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            val = int(str(raw).strip())
        except Exception:
            continue
        val = max(int(min_value), min(int(max_value), int(val)))
        return int(val)
    return int(default)


@dataclass(frozen=True)
class SkillsConfig:
    enabled: bool
    max_k: int
    token_budget_total: int
    strict: bool

    @staticmethod
    def from_env() -> "SkillsConfig":
        enabled = _read_env_bool("KLIMEAI_SKILLS_ENABLED", "VORTEX_SKILLS_ENABLED", "C3RNT2_SKILLS_ENABLED", default=False)
        strict = _read_env_bool("KLIMEAI_SKILLS_STRICT", "VORTEX_SKILLS_STRICT", "C3RNT2_SKILLS_STRICT", default=True)
        max_k = _read_env_int("KLIMEAI_SKILLS_MAX_K", "VORTEX_SKILLS_MAX_K", "C3RNT2_SKILLS_MAX_K", default=3, min_value=1, max_value=10)
        token_budget_total = _read_env_int(
            "KLIMEAI_SKILLS_TOKEN_BUDGET",
            "VORTEX_SKILLS_TOKEN_BUDGET",
            "C3RNT2_SKILLS_TOKEN_BUDGET",
            default=600,
            min_value=64,
            max_value=4096,
        )
        return SkillsConfig(enabled=bool(enabled), max_k=int(max_k), token_budget_total=int(token_budget_total), strict=bool(strict))


@dataclass(frozen=True)
class SkillRecord:
    namespace: str
    spec: SkillSpec
    path: Path
    prompt_path: Path
    prompt: str
    prompt_tokens_est: int
    enabled: bool
    trusted: bool
    install_meta: dict[str, Any] | None

    @property
    def ref(self) -> str:
        return build_skill_ref(self.namespace, self.spec.id)

    def to_public_dict(self, *, include_prompt: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.ref,
            "object": "skill",
            "name": self.spec.name,
            "version": self.spec.version,
            "tags": list(self.spec.tags),
            "enabled": bool(self.enabled),
            "trusted": bool(self.trusted),
            "token_budget": int(self.spec.token_budget),
            "priority": int(self.spec.priority),
            "safety": {
                "network": bool(self.spec.safety.network),
                "filesystem_write": bool(self.spec.safety.filesystem_write),
                "shell": bool(self.spec.safety.shell),
            },
        }
        if self.install_meta:
            data["install"] = dict(self.install_meta)
        if include_prompt:
            data["prompt"] = self.prompt
        return data


class SkillStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._skills: dict[str, SkillRecord] = {}
        self._last_refresh_ts: float = 0.0
        self.state_path = self.root / ".state.json"
        self.index_path = self.root / ".index.json"

    @property
    def staging_root(self) -> Path:
        return self.root / "_staging"

    @property
    def proposals_root(self) -> Path:
        return self.root / "_proposals"

    def _load_state(self) -> dict[str, bool]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, bool] = {}
        for k, v in data.items():
            key = str(k)
            out[key] = bool(v)
        return out

    def _write_state(self, state: dict[str, bool]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.state_path)

    def refresh(self, *, strict_scan: bool = False) -> tuple[list[SkillRecord], list[str]]:
        errors: list[str] = []
        state = self._load_state()
        skills: dict[str, SkillRecord] = {}

        if not self.root.exists():
            with self._lock:
                self._skills = {}
                self._last_refresh_ts = time.time()
            return [], []

        for namespace_dir in sorted(self.root.iterdir(), key=lambda p: p.name):
            if not namespace_dir.is_dir():
                continue
            namespace = namespace_dir.name.strip()
            if namespace.startswith("_") or namespace.startswith("."):
                continue
            ns_errors = validate_namespace(namespace)
            if ns_errors:
                errors.append(f"namespace_invalid:{namespace}:{','.join(ns_errors)}")
                continue

            for skill_dir in sorted(namespace_dir.iterdir(), key=lambda p: p.name):
                if not skill_dir.is_dir():
                    continue
                if skill_dir.name.startswith("_") or skill_dir.name.startswith("."):
                    continue
                yaml_path = skill_dir / "skill.yaml"
                if not yaml_path.exists():
                    yaml_path = skill_dir / "skill.yml"
                if not yaml_path.exists():
                    continue
                prompt_path = skill_dir / "prompt.md"
                if not prompt_path.exists():
                    errors.append(f"prompt_missing:{namespace}/{skill_dir.name}")
                    continue
                try:
                    raw_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"yaml_load_failed:{namespace}/{skill_dir.name}:{exc}")
                    continue
                spec, spec_errors = parse_skill_yaml(raw_yaml)
                if spec_errors:
                    errors.append(f"yaml_invalid:{namespace}/{skill_dir.name}:{','.join(spec_errors)}")
                    continue
                if spec is None:
                    errors.append(f"yaml_invalid:{namespace}/{skill_dir.name}:unknown")
                    continue
                if spec.id != skill_dir.name:
                    errors.append(f"id_path_mismatch:{namespace}/{skill_dir.name}:{spec.id}")
                    continue
                try:
                    prompt = prompt_path.read_text(encoding="utf-8")
                except Exception as exc:
                    errors.append(f"prompt_read_failed:{namespace}/{skill_dir.name}:{exc}")
                    continue
                prompt_tokens_est = estimate_tokens(prompt)
                if prompt_tokens_est > int(spec.token_budget):
                    errors.append(f"prompt_over_budget:{namespace}/{skill_dir.name}:{prompt_tokens_est}>{int(spec.token_budget)}")
                    continue

                ref = build_skill_ref(namespace, spec.id)
                enabled = bool(state.get(ref, True))
                install_meta = None
                install_path = skill_dir / "install.json"
                if install_path.exists():
                    try:
                        install_meta = json.loads(install_path.read_text(encoding="utf-8"))
                        if not isinstance(install_meta, dict):
                            install_meta = None
                    except Exception:
                        install_meta = None
                trusted = bool(namespace == "vortex-core" or (install_meta and install_meta.get("approved_at")))

                if strict_scan:
                    scan = scan_tree(skill_dir, strict=True, max_files=500, max_total_bytes=2 * 1024 * 1024)
                    if not scan.ok:
                        errors.append(f"scan_failed:{ref}:{','.join(scan.errors[:5])}")
                        continue

                skills[ref] = SkillRecord(
                    namespace=namespace,
                    spec=spec,
                    path=skill_dir,
                    prompt_path=prompt_path,
                    prompt=prompt,
                    prompt_tokens_est=int(prompt_tokens_est),
                    enabled=bool(enabled),
                    trusted=bool(trusted),
                    install_meta=install_meta,
                )

        with self._lock:
            self._skills = skills
            self._last_refresh_ts = time.time()

        try:
            self.index_path.write_text(
                json.dumps(
                    {
                        "ts": time.time(),
                        "skills": [rec.to_public_dict(include_prompt=False) for rec in skills.values()],
                    },
                    ensure_ascii=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

        return list(skills.values()), errors

    def list(self) -> list[SkillRecord]:
        with self._lock:
            return list(self._skills.values())

    def get(self, ref: str) -> SkillRecord | None:
        with self._lock:
            return self._skills.get(str(ref))

    def set_enabled(self, ref: str, enabled: bool) -> None:
        ref = str(ref)
        state = self._load_state()
        state[ref] = bool(enabled)
        self._write_state(state)
        with self._lock:
            if ref in self._skills:
                rec = self._skills[ref]
                self._skills[ref] = SkillRecord(
                    namespace=rec.namespace,
                    spec=rec.spec,
                    path=rec.path,
                    prompt_path=rec.prompt_path,
                    prompt=rec.prompt,
                    prompt_tokens_est=rec.prompt_tokens_est,
                    enabled=bool(enabled),
                    trusted=rec.trusted,
                    install_meta=rec.install_meta,
                )

    def remove(self, ref: str) -> dict[str, Any]:
        rec = self.get(ref)
        if rec is None:
            return {"ok": False, "error": "skill_not_found"}
        if rec.namespace == "vortex-core":
            return {"ok": False, "error": "cannot_remove_core_skill"}
        try:
            rec.path.resolve().relative_to(self.root.resolve())
        except Exception:
            return {"ok": False, "error": "path_outside_store"}
        for path in sorted(rec.path.rglob("*"), key=lambda p: len(p.as_posix()), reverse=True):
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            except Exception:
                continue
        try:
            rec.path.rmdir()
        except Exception:
            pass
        try:
            parent = rec.path.parent
            if parent != self.root and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass
        self.refresh()
        return {"ok": True}

    def validate_all(self, *, strict: bool) -> dict[str, Any]:
        records, refresh_errors = self.refresh()
        errors: list[str] = list(refresh_errors or [])
        records = self.list()
        for rec in records:
            scan = scan_tree(rec.path, strict=bool(strict), max_files=500, max_total_bytes=2 * 1024 * 1024)
            if not scan.ok:
                errors.append(f"{rec.ref}:{','.join(scan.errors[:6])}")
            if strict:
                safety: SkillSafety = rec.spec.safety
                if safety.network or safety.filesystem_write or safety.shell:
                    errors.append(f"{rec.ref}:unsafe_safety_flags")
                if not rec.trusted:
                    errors.append(f"{rec.ref}:not_trusted")
        return {"ok": not errors, "errors": errors or None, "skills": [r.to_public_dict() for r in records]}

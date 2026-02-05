from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

_MAX_TAGS = 32
_MAX_TRIGGERS = 64
_MAX_KEYWORD_LEN = 64
_MAX_REGEX_LEN = 160
_MAX_NAME_LEN = 96
_MAX_VERSION_LEN = 32


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (chars/4 heuristic)."""
    if not text:
        return 0
    return (len(text) + 3) // 4


def build_skill_ref(namespace: str, skill_id: str) -> str:
    return f"{namespace}/{skill_id}"


def parse_skill_ref(value: str) -> tuple[str, str] | None:
    raw = str(value or "").strip()
    if not raw or "/" not in raw:
        return None
    ns, sid = raw.split("/", 1)
    ns = ns.strip()
    sid = sid.strip()
    if not ns or not sid:
        return None
    return ns, sid


@dataclass(frozen=True)
class SkillSafety:
    network: bool = False
    filesystem_write: bool = False
    shell: bool = False


@dataclass(frozen=True)
class SkillTriggers:
    keywords: tuple[str, ...] = ()
    regex: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillSpec:
    id: str
    name: str
    version: str
    tags: tuple[str, ...] = ()
    triggers: SkillTriggers = field(default_factory=SkillTriggers)
    token_budget: int = 256
    priority: int = 0
    safety: SkillSafety = field(default_factory=SkillSafety)
    requires_approval: bool = False
    source: str | None = None


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _coerce_int(value: object, *, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        out = int(value) if value is not None else int(default)
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), int(out))
    if max_value is not None:
        out = min(int(max_value), int(out))
    return int(out)


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def validate_namespace(namespace: str) -> list[str]:
    ns = str(namespace or "").strip()
    if not ns:
        return ["namespace_missing"]
    if not _NAMESPACE_RE.match(ns):
        return ["namespace_invalid"]
    return []


def validate_skill_id(skill_id: str) -> list[str]:
    sid = str(skill_id or "").strip()
    if not sid:
        return ["id_missing"]
    if not _SKILL_ID_RE.match(sid):
        return ["id_invalid"]
    return []


_REDOS_SUSPECT_RE = re.compile(r"\\([^)]*[+*][^)]*\\)[+*{]")


def _validate_regex_trigger(pattern: str) -> list[str]:
    errors: list[str] = []
    raw = str(pattern or "")
    if not raw.strip():
        return ["trigger_regex_empty"]
    if len(raw) > _MAX_REGEX_LEN:
        errors.append("trigger_regex_too_long")
    if "\x00" in raw:
        errors.append("trigger_regex_nul")
    # Heuristic ReDoS guard: reject nested quantifiers like (.+)+
    if _REDOS_SUSPECT_RE.search(raw):
        errors.append("trigger_regex_redos_suspect")
    try:
        re.compile(raw, flags=re.IGNORECASE)
    except re.error:
        errors.append("trigger_regex_invalid")
    return errors


def parse_skill_yaml(data: Any) -> tuple[SkillSpec | None, list[str]]:
    if not isinstance(data, dict):
        return None, ["skill_yaml_not_mapping"]

    errors: list[str] = []

    sid = _safe_str(data.get("id")) or ""
    errors.extend(validate_skill_id(sid))

    name = _safe_str(data.get("name")) or ""
    if not name:
        errors.append("name_missing")
    elif len(name) > _MAX_NAME_LEN:
        errors.append("name_too_long")

    version = _safe_str(data.get("version")) or ""
    if not version:
        errors.append("version_missing")
    elif len(version) > _MAX_VERSION_LEN:
        errors.append("version_too_long")

    tags_raw = data.get("tags") or []
    tags: list[str] = []
    if tags_raw is None:
        tags_raw = []
    if not isinstance(tags_raw, list):
        errors.append("tags_invalid")
        tags_raw = []
    for item in tags_raw[: _MAX_TAGS + 1]:
        if len(tags) >= _MAX_TAGS:
            errors.append("tags_too_many")
            break
        val = _safe_str(item)
        if not val:
            continue
        tags.append(val)

    triggers_raw = data.get("triggers") or {}
    if triggers_raw is None:
        triggers_raw = {}
    if not isinstance(triggers_raw, dict):
        errors.append("triggers_invalid")
        triggers_raw = {}

    keywords_raw = triggers_raw.get("keywords") or []
    regex_raw = triggers_raw.get("regex") or []
    keywords: list[str] = []
    regex: list[str] = []

    if keywords_raw is None:
        keywords_raw = []
    if not isinstance(keywords_raw, list):
        errors.append("trigger_keywords_invalid")
        keywords_raw = []
    for item in keywords_raw[: _MAX_TRIGGERS + 1]:
        if len(keywords) >= _MAX_TRIGGERS:
            errors.append("trigger_keywords_too_many")
            break
        val = _safe_str(item)
        if not val:
            continue
        if len(val) > _MAX_KEYWORD_LEN:
            errors.append("trigger_keyword_too_long")
            continue
        keywords.append(val)

    if regex_raw is None:
        regex_raw = []
    if not isinstance(regex_raw, list):
        errors.append("trigger_regex_invalid_type")
        regex_raw = []
    for item in regex_raw[: _MAX_TRIGGERS + 1]:
        if len(regex) >= _MAX_TRIGGERS:
            errors.append("trigger_regex_too_many")
            break
        val = _safe_str(item)
        if not val:
            continue
        regex.append(val)
        errors.extend(_validate_regex_trigger(val))

    token_budget = _coerce_int(data.get("token_budget"), default=256, min_value=1, max_value=4096)
    priority = _coerce_int(data.get("priority"), default=0, min_value=-1000, max_value=1000)

    safety_raw = data.get("safety") or {}
    if safety_raw is None:
        safety_raw = {}
    if not isinstance(safety_raw, dict):
        errors.append("safety_invalid")
        safety_raw = {}
    safety = SkillSafety(
        network=_coerce_bool(safety_raw.get("network"), default=False),
        filesystem_write=_coerce_bool(safety_raw.get("filesystem_write"), default=False),
        shell=_coerce_bool(safety_raw.get("shell"), default=False),
    )

    requires_approval = _coerce_bool(data.get("requires_approval"), default=False)
    source = _safe_str(data.get("source"))

    if errors:
        return None, errors

    spec = SkillSpec(
        id=sid,
        name=name,
        version=version,
        tags=tuple(tags),
        triggers=SkillTriggers(keywords=tuple(keywords), regex=tuple(regex)),
        token_budget=int(token_budget),
        priority=int(priority),
        safety=safety,
        requires_approval=bool(requires_approval),
        source=source,
    )
    return spec, []


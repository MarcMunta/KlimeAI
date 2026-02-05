from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

from .scanner import ALLOWED_BASENAMES, ALLOWED_EXTENSIONS, BLOCKED_BASENAMES, BLOCKED_EXTENSIONS, scan_tree
from .schema import parse_skill_yaml, validate_namespace

# NOTE: GitHub archive downloads typically redirect from github.com -> codeload.github.com.
DEFAULT_ALLOW_DOMAINS: tuple[str, ...] = ("github.com", "raw.githubusercontent.com", "codeload.github.com")

_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_allowed_https_url(url: str, *, allow_domains: tuple[str, ...]) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme.lower() != "https":
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    return host in {d.lower() for d in allow_domains}


def _stage_id_for(source: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(str(time.time()).encode("utf-8"))
    return h.hexdigest()[:12]


def _zipinfo_is_symlink(info: zipfile.ZipInfo) -> bool:
    # Unix symlink bit is 0o120000 in high 16 bits.
    try:
        mode = (info.external_attr >> 16) & 0o170000
        return mode == 0o120000
    except Exception:
        return False


def _validate_member_name(name: str) -> bool:
    raw = str(name or "")
    if not raw or raw.startswith("/") or raw.startswith("\\"):
        return False
    parts = Path(raw).parts
    if any(part in {"..", ""} for part in parts):
        return False
    if ":" in parts[0]:
        return False
    return True


def _precheck_filename(rel_posix: str) -> list[str]:
    p = Path(rel_posix)
    base = p.name.lower()
    ext = p.suffix.lower()
    errors: list[str] = []
    if base in BLOCKED_BASENAMES:
        errors.append("blocked_file")
    if ext in BLOCKED_EXTENSIONS:
        errors.append("blocked_extension")
    if ext and ext not in ALLOWED_EXTENSIONS:
        errors.append("extension_not_allowed")
    if not ext and p.name not in ALLOWED_BASENAMES:
        errors.append("basename_not_allowed")
    return errors


def _download_to(
    url: str,
    dest: Path,
    *,
    max_total_bytes: int,
    allow_domains: tuple[str, ...],
    timeout_s: float = 15.0,
) -> tuple[bool, str | None]:
    try:
        resp = requests.get(url, stream=True, timeout=float(timeout_s), allow_redirects=True)
    except Exception as exc:
        return False, f"download_failed:{exc}"
    if not _is_allowed_https_url(str(resp.url), allow_domains=allow_domains):
        return False, "redirect_not_allowed"
    if int(resp.status_code) != 200:
        return False, f"http_{int(resp.status_code)}"

    total = 0
    try:
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > int(max_total_bytes):
                    return False, "download_too_large"
                handle.write(chunk)
    except Exception as exc:
        return False, f"write_failed:{exc}"
    return True, None


def _extract_zip(
    zip_path: Path,
    out_dir: Path,
    *,
    max_files: int,
    max_total_bytes: int,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    files = 0
    total_bytes = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception as exc:
        return False, [f"zip_open_failed:{exc}"]

    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if _zipinfo_is_symlink(info):
                errors.append("zip_symlink_blocked")
                continue
            if not _validate_member_name(info.filename):
                errors.append("zip_path_invalid")
                continue
            rel_posix = Path(info.filename).as_posix()
            pre = _precheck_filename(rel_posix)
            if pre:
                errors.append(f"{rel_posix}:{','.join(pre)}")
                continue

            files += 1
            if files > int(max_files):
                errors.append("too_many_files")
                break
            total_bytes += max(0, int(info.file_size or 0))
            if total_bytes > int(max_total_bytes):
                errors.append("zip_total_size_exceeded")
                break

            dest_path = (out_dir / rel_posix).resolve()
            try:
                dest_path.relative_to(out_dir.resolve())
            except Exception:
                errors.append("zip_path_escape")
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with zf.open(info, "r") as src, dest_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            except Exception as exc:
                errors.append(f"zip_extract_failed:{rel_posix}:{exc}")
                continue

    return (not errors), errors


def _find_repo_root(extracted: Path) -> Path:
    if not extracted.exists():
        return extracted
    entries = [p for p in extracted.iterdir() if p.name not in {"__MACOSX"}]
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return extracted


def _find_skill_dirs(root: Path, *, max_skills: int = 100) -> list[Path]:
    found: list[Path] = []
    for yaml_path in root.rglob("skill.yaml"):
        if len(found) >= int(max_skills):
            break
        if not yaml_path.is_file():
            continue
        skill_dir = yaml_path.parent
        if (skill_dir / "prompt.md").exists():
            found.append(skill_dir)
    return sorted(found, key=lambda p: p.as_posix())


@dataclass(frozen=True)
class StageResult:
    ok: bool
    staged_id: str | None
    errors: list[str] | None
    found: list[str] | None


def stage(
    skills_root: Path,
    source: str,
    *,
    ref: str | None = None,
    subdir: str | None = None,
    strict: bool = True,
    allow_domains: tuple[str, ...] = DEFAULT_ALLOW_DOMAINS,
    max_files: int = 500,
    max_total_bytes: int = 20 * 1024 * 1024,
) -> StageResult:
    skills_root = Path(skills_root)
    staging_root = skills_root / "_staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    src = str(source or "").strip()
    if not src:
        return StageResult(ok=False, staged_id=None, errors=["source_missing"], found=None)

    url = None
    used_ref = ref
    attempted: list[str] = []
    if "://" in src:
        url = src
        if not _is_allowed_https_url(url, allow_domains=allow_domains):
            return StageResult(ok=False, staged_id=None, errors=["url_not_allowed"], found=None)
    else:
        if not _OWNER_REPO_RE.match(src):
            return StageResult(ok=False, staged_id=None, errors=["source_invalid"], found=None)
        owner, repo = src.split("/", 1)
        refs = [str(used_ref)] if used_ref else ["main", "master"]
        candidates: list[tuple[str, str]] = []
        for r in refs:
            candidate = f"https://github.com/{owner}/{repo}/archive/refs/heads/{r}.zip"
            if _is_allowed_https_url(candidate, allow_domains=allow_domains):
                candidates.append((r, candidate))
        if not candidates:
            return StageResult(ok=False, staged_id=None, errors=["no_download_url"], found=None)

    staged_id = _stage_id_for(f"{src}@{used_ref or ''}:{subdir or ''}")
    stage_dir = staging_root / staged_id
    if stage_dir.exists():
        try:
            shutil.rmtree(stage_dir)
        except Exception:
            pass
    stage_dir.mkdir(parents=True, exist_ok=True)

    zip_path = stage_dir / "source.zip"
    ok = False
    err = None
    if "://" in src:
        ok, err = _download_to(str(url), zip_path, max_total_bytes=int(max_total_bytes), allow_domains=allow_domains)
    else:
        # Try common default branches when ref isn't provided.
        for r, candidate_url in candidates:
            ok, err = _download_to(candidate_url, zip_path, max_total_bytes=int(max_total_bytes), allow_domains=allow_domains)
            attempted.append(f"{r}:{candidate_url}:{err or 'ok'}")
            if ok:
                used_ref = r
                url = candidate_url
                break
            if err == "http_404":
                continue
        if not ok:
            err = ";".join(attempted) or str(err or "download_failed")
    if not ok:
        try:
            shutil.rmtree(stage_dir)
        except Exception:
            pass
        return StageResult(ok=False, staged_id=None, errors=[str(err or "download_failed")], found=None)

    extracted_dir = stage_dir / "extracted"
    ok, ex_errors = _extract_zip(zip_path, extracted_dir, max_files=int(max_files), max_total_bytes=int(max_total_bytes))
    if not ok:
        try:
            shutil.rmtree(stage_dir)
        except Exception:
            pass
        return StageResult(ok=False, staged_id=None, errors=ex_errors, found=None)

    repo_root = _find_repo_root(extracted_dir)
    if subdir:
        repo_root = (repo_root / str(subdir)).resolve()
        try:
            repo_root.relative_to(extracted_dir.resolve())
        except Exception:
            try:
                shutil.rmtree(stage_dir)
            except Exception:
                pass
            return StageResult(ok=False, staged_id=None, errors=["subdir_invalid"], found=None)

    scan = scan_tree(repo_root, strict=bool(strict), max_files=int(max_files), max_total_bytes=int(max_total_bytes))
    if not scan.ok:
        try:
            shutil.rmtree(stage_dir)
        except Exception:
            pass
        return StageResult(ok=False, staged_id=None, errors=scan.errors, found=None)

    skill_dirs = _find_skill_dirs(repo_root)
    if not skill_dirs:
        try:
            shutil.rmtree(stage_dir)
        except Exception:
            pass
        return StageResult(ok=False, staged_id=None, errors=["no_skills_found"], found=None)

    found_rel: list[str] = []
    for d in skill_dirs:
        try:
            found_rel.append(str(d.relative_to(repo_root).as_posix()))
        except Exception:
            continue

    meta = {
        "staged_id": staged_id,
        "source": src,
        "url": str(url),
        "ref": used_ref,
        "subdir": subdir,
        "created_at": _utc_iso(),
        "root": str(repo_root.relative_to(stage_dir).as_posix()),
        "found": found_rel,
        "strict": bool(strict),
        "scan": {"files": int(scan.files), "bytes": int(scan.bytes)},
    }
    (stage_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    return StageResult(ok=True, staged_id=staged_id, errors=None, found=found_rel)


def approve(
    skills_root: Path,
    staged_id: str,
    *,
    namespace: str,
    strict: bool = True,
) -> dict[str, Any]:
    skills_root = Path(skills_root)
    ns_errors = validate_namespace(namespace)
    if ns_errors:
        return {"ok": False, "error": "namespace_invalid", "details": ns_errors}

    stage_dir = skills_root / "_staging" / str(staged_id)
    meta_path = stage_dir / "meta.json"
    if not meta_path.exists():
        return {"ok": False, "error": "staged_not_found"}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    root_rel = str(meta.get("root") or "extracted").strip()
    repo_root = (stage_dir / root_rel).resolve()
    if not repo_root.exists():
        return {"ok": False, "error": "staged_root_missing"}

    scan = scan_tree(repo_root, strict=bool(strict), max_files=500, max_total_bytes=20 * 1024 * 1024)
    if not scan.ok:
        return {"ok": False, "error": "scan_failed", "details": scan.errors}

    found = meta.get("found") or []
    if not isinstance(found, list):
        found = []
    installed: list[str] = []
    errors: list[str] = []

    for rel in found:
        skill_dir = (repo_root / str(rel)).resolve()
        try:
            skill_dir.relative_to(repo_root.resolve())
        except Exception:
            errors.append("skill_path_escape")
            continue
        yaml_path = skill_dir / "skill.yaml"
        if not yaml_path.exists():
            errors.append(f"{rel}:skill_yaml_missing")
            continue
        try:
            raw_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{rel}:yaml_load_failed:{exc}")
            continue
        spec, spec_errors = parse_skill_yaml(raw_yaml)
        if spec is None or spec_errors:
            errors.append(f"{rel}:yaml_invalid:{','.join(spec_errors)}")
            continue
        if spec.id != skill_dir.name:
            errors.append(f"{rel}:id_path_mismatch:{spec.id}!={skill_dir.name}")
            continue

        dest_dir = (skills_root / namespace / spec.id).resolve()
        try:
            dest_dir.relative_to(skills_root.resolve())
        except Exception:
            errors.append(f"{rel}:dest_escape")
            continue
        if dest_dir.exists():
            errors.append(f"{rel}:dest_exists")
            continue

        dest_dir.mkdir(parents=True, exist_ok=False)
        for path in skill_dir.rglob("*"):
            if path.is_dir():
                continue
            try:
                rel_path = path.relative_to(skill_dir)
            except Exception:
                continue
            rel_posix = rel_path.as_posix()
            pre = _precheck_filename(rel_posix)
            if pre:
                errors.append(f"{rel}:{rel_posix}:{','.join(pre)}")
                continue
            out_path = dest_dir / rel_posix
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, out_path)
            except Exception as exc:
                errors.append(f"{rel}:{rel_posix}:copy_failed:{exc}")
                continue

        install_meta = {
            "source": str(meta.get("source") or ""),
            "url": str(meta.get("url") or ""),
            "ref": meta.get("ref"),
            "subdir": meta.get("subdir"),
            "staged_id": str(staged_id),
            "installed_at": _utc_iso(),
            "approved_at": _utc_iso(),
        }
        try:
            (dest_dir / "install.json").write_text(json.dumps(install_meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

        installed.append(f"{namespace}/{spec.id}")

    if errors:
        # Roll back anything copied for this approval.
        for ref in installed:
            ns, sid = ref.split("/", 1)
            try:
                shutil.rmtree(skills_root / ns / sid)
            except Exception:
                pass
        return {"ok": False, "error": "approve_failed", "details": errors}

    try:
        shutil.rmtree(stage_dir)
    except Exception:
        pass

    return {"ok": True, "installed": installed}

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ALLOWED_EXTENSIONS: set[str] = {".md", ".markdown", ".yaml", ".yml", ".json", ".txt"}
ALLOWED_BASENAMES: set[str] = {".gitkeep"}

BLOCKED_BASENAMES: set[str] = {
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "pipfile",
    "pipfile.lock",
}

BLOCKED_EXTENSIONS: set[str] = {
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
}

DANGEROUS_TEXT_SNIPPETS: tuple[str, ...] = (
    "rm -rf",
    "rmdir /s",
    "del /f",
    "format c:",
    "invoke-webrequest",
    "iwr ",
    "iex(",
    "curl | bash",
    "wget | sh",
    "powershell -enc",
    "set-executionpolicy",
    "add-mppreference",
)


def _is_probably_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _scan_text_for_danger(text: str) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for needle in DANGEROUS_TEXT_SNIPPETS:
        if needle in lowered:
            hits.append(needle)
    return hits


@dataclass(frozen=True)
class ScanResult:
    ok: bool
    errors: list[str]
    files: int
    bytes: int


def scan_tree(
    root: Path,
    *,
    strict: bool,
    max_files: int = 500,
    max_total_bytes: int = 20 * 1024 * 1024,
) -> ScanResult:
    root = Path(root)
    errors: list[str] = []
    files = 0
    total_bytes = 0

    if not root.exists():
        return ScanResult(ok=False, errors=["root_missing"], files=0, bytes=0)
    if not root.is_dir():
        return ScanResult(ok=False, errors=["root_not_dir"], files=0, bytes=0)

    for path in root.rglob("*"):
        try:
            rel = path.relative_to(root)
        except Exception:
            errors.append("path_outside_root")
            continue

        parts = list(rel.parts)
        if any(part in {".git", "__pycache__"} for part in parts):
            errors.append("blocked_dir")
            continue

        if path.is_dir():
            continue

        if path.is_symlink():
            errors.append(f"symlink_blocked:{rel.as_posix()}")
            continue

        files += 1
        if files > int(max_files):
            errors.append("too_many_files")
            break

        base = path.name.lower()
        if base in BLOCKED_BASENAMES:
            errors.append(f"blocked_file:{rel.as_posix()}")
            continue

        ext = path.suffix.lower()
        if ext in BLOCKED_EXTENSIONS:
            errors.append(f"blocked_extension:{rel.as_posix()}")
            continue

        if ext and ext not in ALLOWED_EXTENSIONS:
            errors.append(f"extension_not_allowed:{rel.as_posix()}")
            continue
        if not ext and path.name not in ALLOWED_BASENAMES:
            errors.append(f"basename_not_allowed:{rel.as_posix()}")
            continue

        try:
            size = int(path.stat().st_size)
        except Exception:
            errors.append(f"stat_failed:{rel.as_posix()}")
            continue

        total_bytes += max(0, size)
        if total_bytes > int(max_total_bytes):
            errors.append("total_size_exceeded")
            break

        try:
            data = path.read_bytes()
        except Exception:
            errors.append(f"read_failed:{rel.as_posix()}")
            continue

        if _is_probably_binary(data):
            errors.append(f"binary_blocked:{rel.as_posix()}")
            continue

        if strict:
            try:
                text = data.decode("utf-8", errors="strict")
            except Exception:
                errors.append(f"utf8_required:{rel.as_posix()}")
                continue
            hits = _scan_text_for_danger(text)
            if hits:
                errors.append(f"dangerous_text:{rel.as_posix()}:{','.join(hits[:3])}")

    return ScanResult(ok=(not errors), errors=errors, files=int(files), bytes=int(total_bytes))


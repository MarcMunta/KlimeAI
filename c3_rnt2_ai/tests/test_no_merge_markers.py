from __future__ import annotations

from pathlib import Path


def test_no_merge_markers():
    base = Path(__file__).resolve().parents[1]
    roots = [base / "src", base / "config"]
    markers = ("<<<<<<<", "=======", ">>>>>>>")
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(marker in text for marker in markers):
                raise AssertionError(f"Merge marker found in {path}")

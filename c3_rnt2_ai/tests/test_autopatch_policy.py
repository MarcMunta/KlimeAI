from __future__ import annotations

from pathlib import Path

from c3rnt2.autopilot import validate_autopatch_diff


def test_autopatch_rejects_forbidden_paths(tmp_path: Path) -> None:
    diff_text = "\n".join(
        [
            "diff --git a/data/secret.txt b/data/secret.txt",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/data/secret.txt",
            "@@",
            "+secret",
        ]
    )
    ok, message, _paths = validate_autopatch_diff(tmp_path, {}, diff_text)
    assert ok is False
    assert "forbidden" in message

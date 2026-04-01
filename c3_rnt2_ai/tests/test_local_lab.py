from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.local_lab import (
    check_lesson,
    collect_local_lab_status,
    create_lesson,
    ensure_host_layout,
    load_progress,
    next_module,
    write_bootstrap_plan,
    write_rag_sources_manifest,
    write_roadmap,
)


def _settings(tmp_path: Path) -> dict:
    learning_root = tmp_path / "vault-learning"
    return {
        "core": {"backend": "external", "external_engine": "ollama", "external_base_url": "http://127.0.0.1:11434"},
        "local_lab": {
            "enabled": True,
            "track": "python_fastapi_react",
            "curriculum_path": "config/local_lab_curriculum.yaml",
            "progress_path": str(learning_root / "progress.json"),
            "lessons_path": str(learning_root / "lessons"),
            "workspaces_path": str(tmp_path / "workspaces"),
            "continue_config_path": str(tmp_path / ".continue" / "config.yaml"),
            "sandbox_root": str(tmp_path / "sandbox"),
            "guardrails_enabled": True,
            "lab_confirmation_token": "LAB_CONFIRMED",
            "host_paths": {
                "ollama": str(tmp_path / "AI" / "ollama"),
                "openwebui": str(tmp_path / "AI" / "openwebui"),
                "workspaces": str(tmp_path / "workspaces"),
                "vault_learning": str(learning_root),
                "cyber_range": str(tmp_path / "cyber-range"),
            },
        },
    }


def test_local_lab_create_and_check_lesson(tmp_path: Path) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    settings = _settings(tmp_path)

    init_result = ensure_host_layout(settings, base_dir)
    assert init_result["ok"] is True
    assert Path(init_result["progress_path"]).exists()
    assert Path(init_result["continue_config_path"]).exists()
    assert (tmp_path / "vault-learning" / "ROADMAP.md").exists()
    assert (tmp_path / "vault-learning" / "BOOTSTRAP_PLAN.md").exists()
    assert (tmp_path / "vault-learning" / "rag_sources.json").exists()

    lesson = create_lesson(settings, base_dir, module_id="python-basics")
    workspace = Path(lesson["workspace"])
    assert lesson["ok"] is True
    assert (workspace / "LESSON.md").exists()
    assert (workspace / "TASK.md").exists()

    (workspace / "src" / "challenge.py").write_text(
        "\n".join(
            [
                "def score_attempt(numbers: list[int]) -> int:",
                '    """Return the sum of the positive even numbers in the input."""',
                "",
                "    return sum(value for value in numbers if value > 0 and value % 2 == 0)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = check_lesson(settings, base_dir, workspace=workspace)
    assert result["ok"] is True
    assert (workspace / "REVIEW.md").exists()
    review_payload = json.loads((workspace / "review.json").read_text(encoding="utf-8"))
    assert review_payload["ok"] is True

    progress = load_progress(settings, base_dir)
    assert progress["lessons"]
    assert progress["lessons"][-1]["status"] == "passed"


def test_local_lab_planning_artifacts_and_next_module(tmp_path: Path) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    settings = _settings(tmp_path)
    ensure_host_layout(settings, base_dir)

    roadmap = write_roadmap(settings, base_dir)
    rag_sources = write_rag_sources_manifest(settings, base_dir)
    bootstrap = write_bootstrap_plan(settings, base_dir)
    nxt = next_module(settings, base_dir)

    assert roadmap["ok"] is True
    assert rag_sources["ok"] is True
    assert bootstrap["ok"] is True
    assert nxt["ok"] is True
    assert str((nxt.get("module") or {}).get("id")) == "python-basics"
    assert Path(roadmap["path"]).exists()
    assert Path(rag_sources["path"]).exists()
    assert Path(bootstrap["path"]).exists()


def test_collect_status_detects_ollama_fallback(monkeypatch, tmp_path: Path) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    settings = _settings(tmp_path)
    ensure_host_layout(settings, base_dir)

    def fake_run(cmd: list[str]) -> dict:
        joined = " ".join(cmd)
        if "docker info" in joined:
            return {"ok": True, "returncode": 0, "stdout": '"29.3.1"', "stderr": ""}
        if "wsl -l -v" in joined:
            return {"ok": True, "returncode": 0, "stdout": "Ubuntu", "stderr": ""}
        if "--version" in joined:
            return {"ok": True, "returncode": 0, "stdout": "ollama version is 0.18.3", "stderr": ""}
        return {"ok": False, "returncode": 1, "stdout": "", "stderr": "unexpected"}

    monkeypatch.setattr("c3rnt2.local_lab._detect_ollama_command", lambda: [r"C:\Users\demo\AppData\Local\Programs\Ollama\ollama.exe"])
    monkeypatch.setattr("c3rnt2.local_lab._run_command", fake_run)
    monkeypatch.setattr("c3rnt2.local_lab._check_port", lambda port: port in {11434, 3000})
    monkeypatch.setattr(
        "c3rnt2.local_lab._ollama_tags_payload",
        lambda: {
            "ok": True,
            "models": [
                "qwen2.5-coder:14b-instruct-q4_K_S",
                "qwen3:14b",
                "nomic-embed-text:latest",
            ],
        },
    )

    status = collect_local_lab_status(settings, base_dir)
    assert status["commands"]["ollama"]["ok"] is True
    assert status["commands"]["ollama"]["runtime"] == "windows_host"
    assert status["ollama_models"]["ready"] is True

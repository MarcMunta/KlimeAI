from __future__ import annotations

from pathlib import Path

import pytest


def _write_skill(root: Path, *, namespace: str, skill_id: str, prompt: str, keywords: list[str]) -> Path:
    skill_dir = root / namespace / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "skill.yaml").write_text(
        "\n".join(
            [
                f"id: {skill_id}",
                f"name: {skill_id}",
                "version: 1.0.0",
                "tags: [test]",
                "triggers:",
                f"  keywords: {keywords!r}",
                "  regex: []",
                "token_budget: 256",
                "priority: 0",
                "safety:",
                "  network: false",
                "  filesystem_write: false",
                "  shell: false",
                "requires_approval: false",
                "source: local:test",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    (skill_dir / "install.json").write_text(
        '{"source":"local:test","installed_at":"2026-02-05T00:00:00Z","approved_at":"2026-02-05T00:00:00Z"}\n',
        encoding="utf-8",
    )
    return skill_dir


def test_skill_yaml_schema_parses() -> None:
    from c3rnt2.skills.schema import parse_skill_yaml

    spec, errors = parse_skill_yaml(
        {
            "id": "hello-skill",
            "name": "Hello",
            "version": "1.0.0",
            "tags": ["t1"],
            "triggers": {"keywords": ["hello"], "regex": [r"\\bhello\\b"]},
            "token_budget": 200,
            "priority": 10,
            "safety": {"network": False, "filesystem_write": False, "shell": False},
            "requires_approval": False,
            "source": "local:test",
        }
    )
    assert errors == []
    assert spec is not None
    assert spec.id == "hello-skill"


def test_router_selection_is_deterministic(tmp_path: Path) -> None:
    from c3rnt2.skills.router import SkillsRouter
    from c3rnt2.skills.store import SkillStore

    skills_root = tmp_path / "skills"
    _write_skill(skills_root, namespace="vortex-core", skill_id="s1", prompt="P1", keywords=["alpha"])
    _write_skill(skills_root, namespace="vortex-core", skill_id="s2", prompt="P2", keywords=["alpha"])

    store = SkillStore(skills_root)
    store.refresh()
    router = SkillsRouter(store)
    messages = [{"role": "user", "content": "alpha"}]
    sel1 = router.select(messages, model="core", max_k=2, token_budget_total=600, strict=True)
    sel2 = router.select(messages, model="core", max_k=2, token_budget_total=600, strict=True)
    assert [s.ref for s in sel1] == [s.ref for s in sel2]


def test_scanner_blocks_binary_and_scripts(tmp_path: Path) -> None:
    from c3rnt2.skills.scanner import scan_tree

    root = tmp_path / "scan"
    root.mkdir()

    (root / "ok.md").write_text("hello", encoding="utf-8")
    (root / "evil.sh").write_text("rm -rf /", encoding="utf-8")
    (root / "bin.md").write_bytes(b"\x00\x01\x02")

    res = scan_tree(root, strict=True, max_files=20, max_total_bytes=1024 * 1024)
    assert res.ok is False
    assert any("blocked_extension" in e for e in res.errors)
    assert any("binary_blocked" in e for e in res.errors)


def test_stage_rejects_unallowed_url(tmp_path: Path) -> None:
    from c3rnt2.skills.installer import stage

    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    res = stage(skills_root, "http://example.com/skills.zip", strict=True)
    assert res.ok is False
    assert res.errors is not None
    assert "url_not_allowed" in res.errors


def test_chat_injects_skills_when_enabled(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    monkeypatch.setenv("KLIMEAI_SKILLS_ENABLED", "1")
    monkeypatch.setenv("KLIMEAI_SKILLS_STRICT", "1")

    marker = "SKILL_PROMPT_MARKER"
    skills_root = tmp_path / "skills"
    _write_skill(skills_root, namespace="vortex-core", skill_id="inject", prompt=marker, keywords=["inject"])

    class DummyModel:
        def __init__(self):
            self.tokenizer = None
            self.last_prompt = ""

        def generate(self, prompt: str, **_kwargs):
            self.last_prompt = prompt
            return "ok"

        def stream_generate(self, prompt: str, **_kwargs):
            self.last_prompt = prompt
            yield "ok"

    dummy = DummyModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    settings = {"core": {"backend": "vortex", "hf_system_prompt": "SYS"}, "rag": {"enabled": False}}
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={"model": "core", "messages": [{"role": "user", "content": "inject now"}]})
    assert resp.status_code == 200
    assert "SKILLS" in dummy.last_prompt
    assert marker in dummy.last_prompt


def test_list_skills_endpoint(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyModel:
        def __init__(self):
            self.tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return "ok"

        def stream_generate(self, _prompt: str, **_kwargs):
            yield "ok"

    dummy = DummyModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    skills_root = tmp_path / "skills"
    _write_skill(skills_root, namespace="vortex-core", skill_id="one", prompt="P", keywords=["x"])

    settings = {"core": {"backend": "vortex", "hf_system_prompt": "SYS"}, "rag": {"enabled": False}}
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.get("/v1/skills")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("object") == "list"
    ids = [item.get("id") for item in (data.get("data") or []) if isinstance(item, dict)]
    assert "vortex-core/one" in ids

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "skills_installed_total" in metrics.text

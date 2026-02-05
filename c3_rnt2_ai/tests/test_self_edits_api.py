from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


def _setup_client(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    monkeypatch.delenv("KLIMEAI_API_TOKEN", raising=False)
    monkeypatch.delenv("VORTEX_API_TOKEN", raising=False)
    monkeypatch.delenv("C3RNT2_API_TOKEN", raising=False)

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

    # Mimic repo layout: tmp_path is repo root, base_dir is c3_rnt2_ai/.
    (tmp_path / ".git").mkdir(parents=True, exist_ok=True)
    app_dir = tmp_path / "c3_rnt2_ai"
    app_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "_profile": "dev_small",
        "core": {"backend": "vortex", "hf_system_prompt": "SYS"},
        "rag": {"enabled": False},
    }
    app = server_mod.create_app(settings, base_dir=app_dir)
    return TestClient(app), app_dir


def test_self_edits_demo_list_accept_reject(tmp_path: Path, monkeypatch) -> None:
    client, _app_dir = _setup_client(tmp_path, monkeypatch)

    created = client.post("/v1/self-edits/proposals/demo", json={})
    assert created.status_code == 200
    payload = created.json()
    assert payload.get("ok") is True
    pid = payload.get("id")
    assert isinstance(pid, str) and pid

    listed = client.get("/v1/self-edits/proposals?status=pending")
    assert listed.status_code == 200
    data = listed.json().get("data") or []
    assert any(item.get("id") == pid for item in data if isinstance(item, dict))

    detail = client.get(f"/v1/self-edits/proposals/{pid}")
    assert detail.status_code == 200
    d = detail.json()
    assert d.get("id") == pid
    assert isinstance(d.get("diff"), str)
    assert isinstance(d.get("fileChanges"), list)

    accepted = client.post(f"/v1/self-edits/proposals/{pid}/accept", json={})
    assert accepted.status_code == 200
    assert accepted.json().get("status") == "accepted"

    listed2 = client.get("/v1/self-edits/proposals?status=pending")
    assert listed2.status_code == 200
    data2 = listed2.json().get("data") or []
    row = next((item for item in data2 if isinstance(item, dict) and item.get("id") == pid), None)
    assert row is not None
    assert row.get("status") == "accepted"

    rejected = client.post(f"/v1/self-edits/proposals/{pid}/reject", json={})
    assert rejected.status_code == 200
    assert rejected.json().get("status") == "rejected"

    listed3 = client.get("/v1/self-edits/proposals?status=pending")
    assert listed3.status_code == 200
    data3 = listed3.json().get("data") or []
    assert not any(item.get("id") == pid for item in data3 if isinstance(item, dict))


def test_self_edits_apply_fails_for_forbidden_path(tmp_path: Path, monkeypatch) -> None:
    client, app_dir = _setup_client(tmp_path, monkeypatch)

    pid = "forbidden1"
    pdir = app_dir / "skills" / "_proposals" / "self_edits" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "meta.json").write_text(
        json.dumps(
            {
                "id": pid,
                "created_at": time.time(),
                "title": "forbidden",
                "summary": "should fail",
                "author": "agent",
                "status": "accepted",
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (pdir / "patch.diff").write_text(
        "\n".join(
            [
                "--- a/.env",
                "+++ b/.env",
                "@@ -0,0 +1 @@",
                "+SECRET=1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    resp = client.post(f"/v1/self-edits/proposals/{pid}/apply", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is False
    assert str(data.get("error") or "").startswith("forbidden_path:")

    meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("status") == "failed"


from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.learning_loop.promoter import promote_latest


def test_promote_skips_when_eval_not_ok(tmp_path: Path) -> None:
    evals_path = tmp_path / "data" / "learning" / "evals.jsonl"
    evals_path.parent.mkdir(parents=True, exist_ok=True)
    evals_path.write_text(json.dumps({"adapter_path": "data/registry/hf_train/adapter", "improvement": 1.0, "eval_ok": False}) + "\n", encoding="utf-8")
    settings = {
        "learning": {"evals_path": str(evals_path), "promote_min_improvement": 0.0},
        "hf_train": {"registry_dir": str(tmp_path / "data" / "registry" / "hf_train")},
    }
    res = promote_latest(tmp_path, settings)
    assert res.promoted is False
    assert res.message == "eval_not_ok"

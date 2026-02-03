from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.learning_loop.promoter import promote_latest


def test_promote_latest_fail_closed_without_bench_ok(tmp_path: Path) -> None:
    evals = tmp_path / "data" / "learning" / "evals.jsonl"
    evals.parent.mkdir(parents=True, exist_ok=True)
    evals.write_text(json.dumps({"ts": 1.0, "adapter_path": "x", "improvement": 1.0, "eval_ok": True}) + "\n", encoding="utf-8")

    settings = {
        "learning": {"evals_path": str(evals), "require_bench_ok": True, "promote_min_improvement": 0.0},
        "hf_train": {"registry_dir": str(tmp_path / "data" / "registry" / "hf_train")},
    }
    res = promote_latest(tmp_path, settings, min_improvement=0.0)
    assert res.ok is True
    assert res.promoted is False
    assert "bench" in res.message


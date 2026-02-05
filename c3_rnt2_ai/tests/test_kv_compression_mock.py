from __future__ import annotations

from pathlib import Path

import torch

from c3rnt2.bench import BenchArgs, run_bench
from c3rnt2.model.lava_memory import LAVAMemory


def test_bench_mock_exports_kv_lowrank_fields(tmp_path: Path) -> None:
    settings = {
        "_profile": "p",
        "core": {"backend": "vortex", "hidden_size": 64},
        "vortex_model": {"latent_slots": 32},
        "runtime": {"kv_quant": "lowrank", "kv_lowrank_rank": 16},
    }
    out_path = tmp_path / "data" / "bench" / "last.json"
    report = run_bench(
        settings,
        base_dir=tmp_path,
        args=BenchArgs(
            profile="p",
            prompt="hello",
            prompt_file=None,
            ctx=None,
            max_new=4,
            warmup=0,
            repeat=1,
            seed=0,
            json_out=out_path,
            jsonl_out=None,
            mock=True,
            scenario="default",
        ),
    )
    assert report["ok"] is True
    assert report["kv_mode"] == "lowrank"
    assert report["kv_rank"] == 16
    assert "kv_bytes_est" in report


def test_lava_memory_lowrank_read_write_shapes() -> None:
    mem = LAVAMemory(hidden_size=16, latent_slots=8, top_k=2, kv_quant="lowrank", kv_lowrank_rank=4)
    x = torch.randn(2, 3, 16)

    out = mem.read_block(x)
    assert out.shape == x.shape

    mem.write_block(x)
    mem.set_kv_quant("int8")
    out2 = mem.read_block(x)
    assert out2.shape == x.shape


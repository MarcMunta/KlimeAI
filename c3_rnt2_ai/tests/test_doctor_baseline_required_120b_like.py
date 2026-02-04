from __future__ import annotations

from pathlib import Path

from c3rnt2.config import load_settings
from c3rnt2.doctor import _deep_check_120b_like_profile


class _DummyHFModel:
    is_hf = True

    def __init__(self):
        self.quant_fallback = False


class _DummyBenchModel:
    is_hf = True
    runtime_cfg = {"paged_lm_head": False}
    lm_head = None

    def encode_prompt(self, text: str):  # type: ignore[no-untyped-def]
        words = (text or "").split()
        ids = list(range(len(words)))
        return ids, len(ids)

    def decode_ids(self, ids: list[int], total_len: int | None = None) -> str:  # type: ignore[no-untyped-def]
        _ = total_len
        if not ids:
            return ""
        return ("x " * len(ids)).strip()

    def generate(self, _prompt: str, *, max_new_tokens: int):  # type: ignore[no-untyped-def]
        _ = max_new_tokens
        return "ok"


def test_doctor_deep_120b_like_requires_baseline_in_real_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    settings = load_settings("rtx4080_16gb_120b_like")

    import c3rnt2.doctor as doctor_mod
    import c3rnt2.bench as bench_mod
    import c3rnt2.model_loader as loader_mod

    # Make the check deterministic/offline: simulate PEFT present and avoid loading real weights.
    monkeypatch.setattr(doctor_mod.importlib.util, "find_spec", lambda name: object() if name == "peft" else None)
    monkeypatch.setattr(loader_mod, "load_inference_model", lambda _settings: _DummyHFModel())
    monkeypatch.setattr(bench_mod, "load_inference_model", lambda _settings: _DummyBenchModel())

    out = _deep_check_120b_like_profile(settings, tmp_path, mock=False)
    assert out["ok"] is False
    assert "bench_baseline_missing" in (out.get("errors") or [])
    info = out.get("info") or {}
    baseline = info.get("bench_baseline") or {}
    assert baseline.get("ok") is False
    assert "bench --profile" in str(baseline.get("hint") or "")

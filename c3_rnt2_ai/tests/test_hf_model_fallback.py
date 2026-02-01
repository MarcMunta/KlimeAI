from __future__ import annotations

import builtins
import types

import pytest

torch = pytest.importorskip("torch")

from c3rnt2.hf_model import load_hf_model


class _DummyTokenizer:
    pad_token_id = 0
    eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, _name, **_kwargs):
        return cls()

    def __call__(self, text, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, _ids, **_kwargs):
        return "ok"


class _DummyModel(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, _name, **kwargs):
        if kwargs.get("attn_implementation") == "flash_attention_2":
            raise RuntimeError("attn not supported")
        return cls()

    def generate(self, input_ids=None, **_kwargs):
        return torch.tensor([[1, 2, 3]])


def test_hf_load_fallback_attn_and_quant(monkeypatch):
    dummy = types.SimpleNamespace(AutoModelForCausalLM=_DummyModel, AutoTokenizer=_DummyTokenizer)
    monkeypatch.setitem(__import__("sys").modules, "transformers", dummy)

    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("no bnb")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    settings = {
        "core": {
            "backend": "hf",
            "hf_model": "dummy",
            "hf_attn_implementation": "flash_attention_2",
            "hf_load_in_4bit": True,
        }
    }
    model = load_hf_model(settings)
    assert model.quant_fallback is True

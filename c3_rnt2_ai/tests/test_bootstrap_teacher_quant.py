from __future__ import annotations

import sys
import types

from c3rnt2.continuous.bootstrap import _load_teacher


class FakeModel:
    def __init__(self):
        self.to_called = False
        self.hf_device_map = {"": "cuda:0"}

    def to(self, device):
        self.to_called = True
        return self

    def eval(self):
        return self


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return FakeModel()


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return object()


def test_load_teacher_quant_no_to(monkeypatch):
    fake_tf = types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModel, AutoTokenizer=FakeAutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.SimpleNamespace())

    model, _tok, info, input_device = _load_teacher("qwen", device="cuda", quant="4bit", max_memory=None)
    assert model.to_called is False
    assert input_device == "cuda:0"
    assert info.get("quant") == "4bit"

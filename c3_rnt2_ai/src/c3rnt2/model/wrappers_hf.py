from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HFWrapper:
    model_name: str

    def load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers not installed") from exc
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

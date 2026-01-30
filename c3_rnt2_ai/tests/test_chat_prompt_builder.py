from __future__ import annotations

from c3rnt2.prompting.chat_format import build_chat_prompt


class FakeTokenizer:
    def __init__(self):
        self.called = False
        self.last = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.called = True
        self.last = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return "CHAT_TEMPLATE"


def test_chat_prompt_uses_template():
    tok = FakeTokenizer()
    messages = [{"role": "user", "content": "Hello"}]
    out = build_chat_prompt(messages, backend="hf", tokenizer=tok, default_system="SYS")
    assert out == "CHAT_TEMPLATE"
    assert tok.called is True
    assert tok.last["add_generation_prompt"] is True

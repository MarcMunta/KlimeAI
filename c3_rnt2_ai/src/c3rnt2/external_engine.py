from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests


@dataclass(frozen=True)
class ExternalEngineConfig:
    engine: str
    base_url: str
    model: str | None = None
    api_key: str | None = None
    timeout_s: float = 60.0
    autostart: bool = False
    start_command: str | None = None
    start_workdir: str | None = None
    startup_wait_s: float = 2.0


def _extract_text(resp: dict) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message")
    if isinstance(msg, dict) and msg.get("content") is not None:
        return str(msg.get("content") or "")
    if first.get("text") is not None:
        return str(first.get("text") or "")
    return ""


def _extract_delta(evt: dict) -> str:
    choices = evt.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    delta = first.get("delta")
    if isinstance(delta, dict) and delta.get("content") is not None:
        return str(delta.get("content") or "")
    if first.get("text") is not None:
        return str(first.get("text") or "")
    return ""


class ExternalEngineModel:
    is_external = True
    is_hf = False
    is_llama_cpp = False

    def __init__(self, cfg: ExternalEngineConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self._proc: subprocess.Popen | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.cfg.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or os.environ.get("SGLANG_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _url(self, path: str) -> str:
        return self.cfg.base_url.rstrip("/") + str(path)

    def _ensure_started(self) -> None:
        if not bool(self.cfg.autostart):
            return
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = str(self.cfg.start_command or "").strip()
        if not cmd:
            raise RuntimeError("external_engine_autostart_enabled_but_no_start_command")
        workdir = str(self.cfg.start_workdir) if self.cfg.start_workdir else None
        self._proc = subprocess.Popen(cmd, shell=True, cwd=workdir)
        time.sleep(max(0.0, float(self.cfg.startup_wait_s)))

    def generate(self, prompt: str, *, max_new_tokens: int = 64, temperature: float = 1.0, top_p: float = 1.0, **_kwargs) -> str:
        self._ensure_started()
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": str(prompt)}],
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        if self.cfg.model:
            payload["model"] = str(self.cfg.model)
        url = self._url("/v1/chat/completions")
        resp = self.session.post(url, json=payload, headers=self._headers(), timeout=float(self.cfg.timeout_s))
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            return ""
        return _extract_text(data)

    def stream_generate(self, prompt: str, *, max_new_tokens: int = 64, temperature: float = 1.0, top_p: float = 1.0, **_kwargs) -> Iterable[str]:
        self._ensure_started()
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": str(prompt)}],
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": True,
        }
        if self.cfg.model:
            payload["model"] = str(self.cfg.model)
        url = self._url("/v1/chat/completions")
        with self.session.post(url, json=payload, headers=self._headers(), timeout=float(self.cfg.timeout_s), stream=True) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                if not isinstance(evt, dict):
                    continue
                delta = _extract_delta(evt)
                if delta:
                    yield delta


def load_external_engine_model(settings: dict) -> ExternalEngineModel:
    core = settings.get("core", {}) or {}
    backend = str(core.get("backend", "external") or "external").strip().lower()
    engine = str(core.get("external_engine") or core.get("engine") or "").strip().lower()
    if backend in {"vllm", "sglang"}:
        engine = backend
        backend = "external"
    if not engine:
        engine = "external"
    base_url = str(core.get("external_base_url") or core.get("external_url") or "http://127.0.0.1:30000").strip()
    model = core.get("external_model") or core.get("hf_model")
    api_key = core.get("external_api_key")
    timeout_s = float(core.get("external_timeout_s", 60.0) or 60.0)
    autostart = bool(core.get("external_autostart", False))
    start_command = core.get("external_start_command")
    workdir = core.get("external_start_workdir") or core.get("external_workdir")
    startup_wait_s = float(core.get("external_startup_wait_s", 2.0) or 2.0)
    cfg = ExternalEngineConfig(
        engine=str(engine),
        base_url=base_url,
        model=str(model) if model else None,
        api_key=str(api_key) if api_key else None,
        timeout_s=float(timeout_s),
        autostart=bool(autostart),
        start_command=str(start_command) if start_command else None,
        start_workdir=str(workdir) if workdir else None,
        startup_wait_s=float(startup_wait_s),
    )
    return ExternalEngineModel(cfg)

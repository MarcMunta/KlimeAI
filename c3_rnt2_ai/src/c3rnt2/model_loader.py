from __future__ import annotations

from typing import Any

from .model.core_transformer import CoreTransformer
from .hf_model import load_hf_model
from .tensorrt_backend import load_tensorrt_model


def _fallback_backend(core: dict, current: str) -> str | None:
    fallback = core.get("backend_fallback") or core.get("hf_fallback")
    if fallback and str(fallback).lower() != current:
        return str(fallback).lower()
    return None


def load_inference_model(settings: dict, backend_override: str | None = None) -> Any:
    core = settings.get("core", {}) or {}
    backend = str(backend_override or core.get("backend", "vortex")).lower()
    if backend == "hf":
        try:
            return load_hf_model(settings)
        except Exception:
            fb = _fallback_backend(core, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            raise
    if backend == "llama_cpp":
        try:
            from .llama_cpp_backend import load_llama_cpp_model

            return load_llama_cpp_model(settings)
        except Exception:
            fb = _fallback_backend(core, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            raise
    if backend == "tensorrt":
        try:
            return load_tensorrt_model(settings)
        except Exception:
            fb = _fallback_backend(core, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            return CoreTransformer.from_settings(settings)
    return CoreTransformer.from_settings(settings)

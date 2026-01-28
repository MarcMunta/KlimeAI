from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceInfo:
    device: str
    cuda_available: bool
    name: str | None
    vram_gb: float | None
    dtype: str


def detect_device(prefer_bf16: bool = True) -> DeviceInfo:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        logger.warning("PyTorch not available: %s", exc)
        return DeviceInfo(device="cpu", cuda_available=False, name=None, vram_gb=None, dtype="fp32")

    cuda = torch.cuda.is_available()
    if cuda:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vram_bytes = torch.cuda.get_device_properties(idx).total_memory
        vram_gb = round(vram_bytes / (1024**3), 2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            dtype = "bf16"
        else:
            dtype = "fp16"
        device = f"cuda:{idx}"
        logger.info("CUDA detected: %s, VRAM %.2f GB", name, vram_gb)
        return DeviceInfo(device=device, cuda_available=True, name=name, vram_gb=vram_gb, dtype=dtype)

    logger.info("CUDA not available, using CPU")
    return DeviceInfo(device="cpu", cuda_available=False, name=None, vram_gb=None, dtype="fp32")


class autocast_context:
    """Lightweight autocast wrapper used across modules."""

    def __init__(self, enabled: bool, dtype: Optional[str] = None):
        self.enabled = enabled
        self.dtype = dtype
        self._ctx = None

    def __enter__(self):
        if not self.enabled:
            return None
        try:
            import torch
        except Exception:
            return None
        dtype = torch.bfloat16 if self.dtype == "bf16" else torch.float16
        self._ctx = torch.autocast(device_type="cuda", dtype=dtype)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False

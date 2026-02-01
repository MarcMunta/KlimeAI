from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
<<<<<<< HEAD
=======
from dataclasses import dataclass
>>>>>>> 7ef3a231663391568cb83c4c686642e75f55c974
import time

import numpy as np

from ..compression.entropy_coder import decompress

try:
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

torch: Any = _torch


@dataclass
class DecompressStats:
    bytes_decompressed: int = 0
    ms_cpu_decompress: float = 0.0
    ms_h2d: float = 0.0
    ms_triton_copy: float = 0.0


<<<<<<< HEAD
def _maybe_stream(stream: object | None):
    if stream is None or torch is None:
        return None
    return torch.cuda.stream(stream)


def _to_tensor(tile: np.ndarray, device: str = "cpu", pin_memory: bool = False, non_blocking: bool = False, stream: object | None = None, stats: DecompressStats | None = None):
=======
def _to_tensor(tile: np.ndarray, device: str = "cpu", pin_memory: bool = False, non_blocking: bool = False):
>>>>>>> 7ef3a231663391568cb83c4c686642e75f55c974
    if torch is None:
        raise RuntimeError("PyTorch not available")
    if not tile.flags["C_CONTIGUOUS"]:
        tile = np.ascontiguousarray(tile)
    tensor = torch.from_numpy(tile)
    if pin_memory:
        tensor = tensor.pin_memory()
    if device.startswith("cuda"):
        start = time.perf_counter()
        if stream is not None:
            with torch.cuda.stream(stream):
                tensor = tensor.to(device, non_blocking=non_blocking)
        else:
            tensor = tensor.to(device, non_blocking=non_blocking)
        if stats is not None:
            stats.ms_h2d += (time.perf_counter() - start) * 1000.0
    else:
        tensor = tensor.to(device, non_blocking=non_blocking)
    return tensor


if triton is not None:  # pragma: no cover
    @triton.jit
    def _copy_kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(in_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x, mask=mask)


    def _triton_copy(inp: "torch.Tensor") -> "torch.Tensor":
        out = torch.empty_like(inp)
        n = inp.numel()
        grid = (triton.cdiv(n, 1024),)
        _copy_kernel[grid](inp, out, n, BLOCK=1024)
        return out
else:
    def _triton_copy(inp: "torch.Tensor") -> "torch.Tensor":
        return inp


def decompress_to_tensor(
    tile: Any,
    device: str = "cpu",
    codec: str | None = None,
    shape: Tuple[int, int] | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool | None = None,
    backend: str = "none",
    pinned: bool | None = None,
    stream: object | None = None,
<<<<<<< HEAD
    stats: DecompressStats | None = None,
):
    """Decompress tile payload if needed and move to device."""
    if pinned is not None:
        pin_memory = pinned
    if stats is None:
        stats = None

    if isinstance(tile, np.ndarray):
        tensor = _to_tensor(tile, device=device, pin_memory=pin_memory, non_blocking=non_blocking, stream=stream, stats=stats)
        if stats is not None:
            stats.bytes_decompressed += int(tensor.numel() * tensor.element_size())
=======
):
    """Decompress tile payload if needed and move to device."""
    if torch is None:
        raise RuntimeError("PyTorch not available")
    stats = DecompressStats()
    use_pin = pinned if pinned is not None else pin_memory
    if use_pin is None:
        use_pin = device.startswith("cuda")
    use_non_blocking = non_blocking if non_blocking is not None else device.startswith("cuda")

    if isinstance(tile, np.ndarray):
        arr = tile
>>>>>>> 7ef3a231663391568cb83c4c686642e75f55c974
    elif isinstance(tile, (bytes, bytearray)):
        if codec is None or shape is None:
            raise ValueError("codec and shape required for compressed tiles")
        start = time.perf_counter()
        raw = decompress(bytes(tile), codec=codec)
<<<<<<< HEAD
        if stats is not None:
            stats.ms_cpu_decompress += (time.perf_counter() - start) * 1000.0
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        tensor = _to_tensor(arr, device=device, pin_memory=pin_memory, non_blocking=non_blocking, stream=stream, stats=stats)
        if stats is not None:
            stats.bytes_decompressed += int(tensor.numel() * tensor.element_size())
=======
        stats.ms_cpu_decompress = (time.perf_counter() - start) * 1000.0
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
>>>>>>> 7ef3a231663391568cb83c4c686642e75f55c974
    else:
        raise TypeError("Unsupported tile type")

    stats.bytes_decompressed = int(arr.nbytes)
    tensor = _to_tensor(arr, device="cpu", pin_memory=bool(use_pin), non_blocking=False)

    if device != "cpu":
        start = time.perf_counter()
        if stream is not None and torch is not None and hasattr(torch.cuda, "stream"):
            try:
                with torch.cuda.stream(stream):
                    tensor = tensor.to(device, non_blocking=bool(use_non_blocking))
            except Exception:
                tensor = tensor.to(device, non_blocking=bool(use_non_blocking))
        else:
            tensor = tensor.to(device, non_blocking=bool(use_non_blocking))
        stats.ms_h2d = (time.perf_counter() - start) * 1000.0

    if backend == "triton" and torch is not None and triton is not None:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
            start = time.perf_counter()
<<<<<<< HEAD
            tensor = _triton_copy(tensor)
            if stats is not None:
                stats.ms_triton_copy += (time.perf_counter() - start) * 1000.0
    return tensor
=======
            out = _triton_copy(tensor)
            stats.ms_triton_copy = (time.perf_counter() - start) * 1000.0
            return out, stats
    return tensor, stats
>>>>>>> 7ef3a231663391568cb83c4c686642e75f55c974

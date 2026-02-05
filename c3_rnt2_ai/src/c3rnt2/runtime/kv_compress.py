from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFF_FFFF_FFFF_FFFF


def _dtype_nbytes(dtype: Any) -> int:
    if torch is None:
        return 0
    try:
        return int(torch.empty((), dtype=dtype).element_size())
    except Exception:
        return 0


@dataclass(frozen=True)
class LowRankProjector:
    hidden_size: int
    rank: int
    proj: "torch.Tensor"
    proj_t: "torch.Tensor"

    def estimate_bytes(self, *, slots: int | None = None) -> int | None:
        if torch is None:
            return None
        try:
            proj_bytes = int(self.proj.numel() * self.proj.element_size())
        except Exception:
            proj_bytes = 0
        if slots is None:
            return proj_bytes
        try:
            slots_i = max(0, int(slots))
        except Exception:
            slots_i = 0
        return proj_bytes + int(slots_i) * int(self.rank) * _dtype_nbytes(self.proj.dtype)


_PROJ_CACHE: dict[tuple[int, int, str, str], LowRankProjector] = {}


def get_lowrank_projector(
    hidden_size: int,
    rank: int,
    *,
    device: "torch.device | str",
    dtype: "torch.dtype",
    cache: bool = True,
) -> LowRankProjector:
    if torch is None:
        raise RuntimeError("torch not available")
    h = int(hidden_size)
    r = int(rank)
    if h <= 0:
        raise ValueError("hidden_size must be > 0")
    if r <= 0 or r >= h:
        raise ValueError("rank must be in (0, hidden_size)")

    dev = str(device)
    key = (h, r, dev, str(dtype))
    if cache and key in _PROJ_CACHE:
        proj = _PROJ_CACHE[key]
        if proj.proj.device == torch.device(dev) and proj.proj.dtype == dtype:
            return proj

    seed = _stable_seed("c3rnt2.lowrank_kv", h, r)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    mat = torch.randn(h, r, generator=gen, dtype=torch.float32)
    mat = mat / mat.norm(dim=0, keepdim=True).clamp_min(1e-6)
    proj = mat.to(device=device, dtype=dtype)
    proj_t = proj.transpose(0, 1).contiguous()
    out = LowRankProjector(hidden_size=h, rank=r, proj=proj, proj_t=proj_t)
    if cache:
        _PROJ_CACHE[key] = out
    return out


def project_down(x: "torch.Tensor", projector: LowRankProjector) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch not available")
    return x @ projector.proj


def project_up(x_lr: "torch.Tensor", projector: LowRankProjector) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch not available")
    return x_lr @ projector.proj_t


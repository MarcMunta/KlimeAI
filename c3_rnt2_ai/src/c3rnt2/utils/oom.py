from __future__ import annotations


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda oom" in msg:
        return True
    try:
        import torch
    except Exception:
        return False
    try:
        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except Exception:
        return False


def clear_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            return

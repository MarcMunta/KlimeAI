from __future__ import annotations

import argparse
import time

import torch

from c3rnt2.nn.paged_linear import PagedLinear


def _bench(module, x, iters: int = 50) -> float:
    # warmup
    for _ in range(5):
        _ = module(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = module(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return (iters / max(1e-6, elapsed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-features', type=int, default=4096)
    parser.add_argument('--out-features', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--tile-out', type=int, default=256)
    parser.add_argument('--tile-in', type=int, default=512)
    parser.add_argument('--dtype', type=str, default='bf16')
    parser.add_argument('--accum-fp32', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16

    linear = torch.nn.Linear(args.in_features, args.out_features, bias=True, device=device, dtype=dtype)
    paged = PagedLinear.from_linear(
        linear,
        tile_out=args.tile_out,
        tile_in=args.tile_in,
        cache_budget_bytes=1024 * 1024 * 1024,
        compression=None,
        device=device,
        prefetch_depth=2,
        pin_memory=True,
        accum_fp32=args.accum_fp32,
    )

    x = torch.randn(args.batch, args.in_features, device=device, dtype=dtype)

    base_tps = _bench(linear, x)
    paged_tps = _bench(paged, x)

    print({
        'device': device,
        'dtype': str(dtype),
        'base_tps': round(base_tps, 3),
        'paged_tps': round(paged_tps, 3),
        'speedup': round(paged_tps / max(1e-6, base_tps), 3),
    })


if __name__ == '__main__':
    main()

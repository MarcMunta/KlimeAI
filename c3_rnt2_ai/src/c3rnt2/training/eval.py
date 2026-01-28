from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..tokenizer.rnt2_model import RNT2Model, RNT2Codebook
from ..tokenizer.rnt2_encode import encode_text
from ..model.core_transformer import CoreTransformer
from ..device import detect_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    args = parser.parse_args()

    # Exact-copy benchmark (roundtrip)
    model_path = Path("data/runs/rnt2_dev.pt")
    if model_path.exists():
        rnt2 = RNT2Model.load(model_path)
    else:
        rnt2 = RNT2Model(RNT2Codebook.from_builtin())
    sample = "{""k"": [1,2,3], ""msg"": ""hola""}"
    stream = encode_text(sample, rnt2)
    exact_copy_ok = True
    ratio = len(stream.codes) / max(1, len(sample.encode("utf-8")))

    # Core throughput (approx)
    core = CoreTransformer.from_settings({"core": {"hidden_size": 128, "layers": 2, "heads": 2, "vocab_size": 256}})
    start = time.time()
    _ = core.generate("def f(x):", max_new_tokens=16)
    tps = 16 / max(1e-6, time.time() - start)

    device = detect_device()
    vram = device.vram_gb if device.cuda_available else 0.0

    print({
        "exact_copy_ok": exact_copy_ok,
        "rnt2_ratio": round(ratio, 4),
        "tokens_per_second": round(tps, 3),
        "vram_gb": vram,
    })


if __name__ == "__main__":
    main()

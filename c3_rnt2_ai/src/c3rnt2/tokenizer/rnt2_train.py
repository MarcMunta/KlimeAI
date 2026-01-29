from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from collections import Counter

from .rnt2_model import RNT2Codebook, RNT2Model
from .vortex_tok import VortexMacroCodebook, VortexTokModel


def iter_corpus_files(corpus_dir: Path) -> Iterable[Path]:
    for path in corpus_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".json", ".py"}:
            yield path


def iter_blocks(corpus_dir: Path, block_size: int) -> Iterable[bytes]:
    for path in iter_corpus_files(corpus_dir):
        data = path.read_bytes()
        for i in range(0, len(data), block_size):
            block = data[i : i + block_size]
            if block:
                yield block.ljust(block_size, b"\x00")


def _iter_patch_ids(corpus_dir: Path, block_size: int, codebook: RNT2Codebook) -> Iterable[int]:
    for path in iter_corpus_files(corpus_dir):
        data = path.read_bytes()
        for i in range(0, len(data), block_size):
            block = data[i : i + block_size]
            if not block:
                continue
            padded = block.ljust(block_size, b"\x00")
            code = codebook.find(padded)
            if code is not None:
                yield code


def build_macro_codebook(
    corpus_dir: Path,
    block_size: int,
    codebook: RNT2Codebook,
    macro_size: int,
    macro_min_len: int,
) -> VortexMacroCodebook:
    if macro_size <= 0:
        return VortexMacroCodebook(sequences=[])
    patch_ids = list(_iter_patch_ids(corpus_dir, block_size, codebook))
    if len(patch_ids) < macro_min_len:
        return VortexMacroCodebook(sequences=[])
    counts: Counter[tuple[int, ...]] = Counter()
    for i in range(0, len(patch_ids) - macro_min_len + 1):
        seq = tuple(patch_ids[i : i + macro_min_len])
        counts[seq] += 1
    sequences = [list(seq) for seq, _ in counts.most_common(macro_size)]
    return VortexMacroCodebook(sequences=sequences)


def train(
    codebook_size: int,
    block_size: int,
    corpus_dir: Path,
    output_path: Path,
    vortex_output: Path | None = None,
    macro_size: int = 0,
    macro_min_len: int = 2,
) -> None:
    blocks = list(iter_blocks(corpus_dir, block_size))
    if not blocks:
        codebook = RNT2Codebook.from_builtin(block_size=block_size, size=codebook_size)
    else:
        codebook = RNT2Codebook.from_corpus(blocks=blocks, block_size=block_size, size=codebook_size)
    model = RNT2Model(codebook=codebook)
    model.save(output_path)
    if vortex_output is not None:
        macro = build_macro_codebook(corpus_dir, block_size, codebook, macro_size, macro_min_len)
        vortex = VortexTokModel(patch_codebook=codebook, macro_codebook=macro)
        vortex.save(vortex_output)
    print({
        "blocks": len(blocks),
        "block_size": block_size,
        "codebook_size": codebook_size,
        "output": str(output_path),
        "vortex_output": str(vortex_output) if vortex_output else None,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/corpora"))
    parser.add_argument("--output", type=Path, default=Path("data/runs/rnt2_dev.pt"))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--vortex-output", type=Path, default=Path("data/runs/vortex_tok.pt"))
    parser.add_argument("--macro-size", type=int, default=256)
    parser.add_argument("--macro-min-len", type=int, default=2)
    args = parser.parse_args()
    train(
        args.codebook_size,
        args.block_size,
        args.corpus,
        args.output,
        vortex_output=args.vortex_output,
        macro_size=args.macro_size,
        macro_min_len=args.macro_min_len,
    )


if __name__ == "__main__":
    main()

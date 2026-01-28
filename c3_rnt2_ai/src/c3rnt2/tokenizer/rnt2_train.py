from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from .rnt2_model import RNT2Codebook, RNT2Model


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


def train(codebook_size: int, block_size: int, corpus_dir: Path, output_path: Path) -> None:
    blocks = list(iter_blocks(corpus_dir, block_size))
    if not blocks:
        codebook = RNT2Codebook.from_builtin(block_size=block_size, size=codebook_size)
    else:
        codebook = RNT2Codebook.from_corpus(blocks=blocks, block_size=block_size, size=codebook_size)
    model = RNT2Model(codebook=codebook)
    model.save(output_path)
    print({
        "blocks": len(blocks),
        "block_size": block_size,
        "codebook_size": codebook_size,
        "output": str(output_path),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/corpora"))
    parser.add_argument("--output", type=Path, default=Path("data/runs/rnt2_dev.pt"))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=1024)
    args = parser.parse_args()
    train(args.codebook_size, args.block_size, args.corpus, args.output)


if __name__ == "__main__":
    main()

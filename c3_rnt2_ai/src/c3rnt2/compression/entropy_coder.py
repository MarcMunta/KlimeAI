from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompressionResult:
    payload: bytes
    codec: str


def compress(data: bytes, codec: str = "zstd") -> CompressionResult:
    codec = codec.lower()
    if codec == "zstd":
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=3)
        return CompressionResult(payload=cctx.compress(data), codec=codec)
    if codec == "lz4":
        import lz4.frame

        return CompressionResult(payload=lz4.frame.compress(data), codec=codec)
    raise ValueError(f"Unknown codec: {codec}")


def decompress(data: bytes, codec: str) -> bytes:
    codec = codec.lower()
    if codec == "zstd":
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    if codec == "lz4":
        import lz4.frame

        return lz4.frame.decompress(data)
    raise ValueError(f"Unknown codec: {codec}")

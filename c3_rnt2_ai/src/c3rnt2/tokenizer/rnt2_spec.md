# RNT-2 Spec (MVP)

## Goals
- Exact reversible tokenization over UTF-8 bytes.
- Reduce token count vs byte-level baseline when blocks match codebook.
- Escape channel guarantees lossless decode.

## Parameters
- Block size (B): 64 bytes
- Codebooks (C): 1 (MVP)
- Codebook size (K): 1024 entries
- Codes per block (L): 1 (MVP)

## Stream format
- Header:
  - magic: RNT2
  - version: 1
  - block_size B
  - codebook_size K
  - num_blocks N
- Body:
  - codes: N uint16 values (0..K-1) per block
  - escapes: list of (block_index, raw_bytes_len, raw_bytes)

## Encoding
1) Split UTF-8 bytes into fixed blocks of size B. Pad final block with 0x00 and note original length.
2) For each block, check for exact match in codebook.
3) If match: emit code; no escape entry.
4) If no match: emit fallback code 0 and add escape entry containing the raw bytes and original length.

## Decoding
1) Read header and code list.
2) For each block:
   - If block index is in escapes: use raw bytes (trim to original length).
   - Else: use codebook[code] bytes (trim to block size).
3) Concatenate bytes and decode UTF-8.

## Notes
- MVP uses codebook built from frequent blocks in corpus.
- Future: multi-codebook and neural encoder with VQ.

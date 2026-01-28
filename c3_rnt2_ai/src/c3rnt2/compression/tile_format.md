# C3 Tile Format (MVP)

## Tile
- Shape: T x T (default 128x128)
- Stored as:
  - codebook (K x T) or shared external codebook
  - scale (per-row or per-tile)
  - residual (optional)

## Pack
- Header:
  - magic: C3PK
  - version: 1
  - tile_size
  - dtype
  - num_tiles
- Body:
  - tiles serialized as: [codebook_id][scale][payload]
  - payload compressed via zstd or lz4

## MVP simplification
- Per-tile codebook is not shared.
- Payload is raw FP16 (or int8) array compressed.
- Format is designed to be upgraded without breaking versioned header.

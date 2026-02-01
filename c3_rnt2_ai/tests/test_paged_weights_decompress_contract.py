import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from c3rnt2.runtime.cache_manager import CacheManager
from c3rnt2.runtime.paged_weights import PagedWeights


def test_paged_weights_decompress_contract_cpu():
    arr = np.arange(4, dtype=np.float16).reshape(2, 2)
    tile_store = {0: arr}
    cache = CacheManager(capacity_bytes=1024 * 1024)
    paged = PagedWeights(tile_store=tile_store, cache=cache, device="cpu", prefetch_depth=0, pin_memory=False)
    tiles = paged.request_tiles([0])
    assert len(tiles) == 1
    assert hasattr(tiles[0], "shape")
    assert paged.stats.bytes_decompressed == arr.nbytes

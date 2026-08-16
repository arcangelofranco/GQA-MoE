import torch

from src.model.kv_cache import KVCache


def test_initial_state_is_empty() -> None:
    """Checks that a freshly created KVCache starts empty with no k/v tensors."""
    cache = KVCache()
    print(f"[initial] length={cache.length} k={cache.k} v={cache.v}")
    assert cache.length == 0
    assert cache.k is None
    assert cache.v is None


def test_append_on_empty_assigns() -> None:
    """Checks that appending to an empty cache assigns k/v directly and returns them."""
    cache = KVCache()
    k = torch.randn(2, 3, 1, 8)
    v = torch.randn(2, 3, 1, 8)
    k_out, v_out = cache.append(k, v)
    print(
        f"[append-empty] length={cache.length} "
        f"k_equal={torch.equal(cache.k, k)} v_equal={torch.equal(cache.v, v)}"
    )
    assert cache.length == 1
    assert torch.equal(cache.k, k)
    assert torch.equal(cache.v, v)
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)


def test_append_grows_and_concatenates() -> None:
    """Checks that successive appends concatenate along the sequence dimension and grow length."""
    cache = KVCache()
    parts_k, parts_v = [], []
    for s in [1, 2, 4]:
        k = torch.randn(2, 3, s, 8)
        v = torch.randn(2, 3, s, 8)
        parts_k.append(k)
        parts_v.append(v)
        cache.append(k, v)

    expected_k = torch.cat(parts_k, dim=2)
    expected_v = torch.cat(parts_v, dim=2)
    print(f"[grow] length={cache.length} expected=7")
    assert cache.length == 7
    assert torch.equal(cache.k, expected_k)
    assert torch.equal(cache.v, expected_v)


def test_append_preserves_dtype() -> None:
    """Checks that appending bfloat16 tensors keeps the cache's k/v in bfloat16."""
    cache = KVCache()
    k = torch.randn(2, 2, 1, 8, dtype=torch.bfloat16)
    v = torch.randn(2, 2, 1, 8, dtype=torch.bfloat16)
    cache.append(k, v)
    cache.append(torch.randn(2, 2, 1, 8, dtype=torch.bfloat16),
                 torch.randn(2, 2, 1, 8, dtype=torch.bfloat16))
    print(f"[dtype] k.dtype={cache.k.dtype} v.dtype={cache.v.dtype}")
    assert cache.k.dtype == torch.bfloat16
    assert cache.v.dtype == torch.bfloat16


def test_length_tracks_total_appended() -> None:
    """Checks that cache.length always equals the running total of appended positions."""
    cache = KVCache()
    total = 0
    for s in [5, 8, 3]:
        cache.append(torch.randn(1, 2, s, 8), torch.randn(1, 2, s, 8))
        total += s
        assert cache.length == total
    print(f"[length] total={total}")
    assert total == 16
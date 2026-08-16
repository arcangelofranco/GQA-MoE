from types import SimpleNamespace

import pytest
import torch

from src.model.attention import GQAttention
from src.model.blocks import precompute_rope
from src.model.kv_cache import KVCache

NANO = SimpleNamespace(
    n_heads=8, n_kv_heads=2, head_dim=32, hidden_dim=256, rms_norm_eps=1e-6
)
SMALL = SimpleNamespace(
    n_heads=8, n_kv_heads=4, head_dim=64, hidden_dim=512, rms_norm_eps=1e-6
)
CONFIGS = [NANO, SMALL]
MAX_SEQ_LEN = 1024


def _make_attn(config: SimpleNamespace, name: str = "nano") -> GQAttention:
    """Builds a GQAttention module from a test config namespace.

    Args:
        config: Duck-typed model config (n_heads, n_kv_heads, head_dim, hidden_dim, rms_norm_eps).
        name: Unused; kept for call-site readability in parametrized tests.

    Returns:
        A freshly initialized `GQAttention`.
    """
    return GQAttention(config)


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
@pytest.mark.parametrize("s", [1, 16, 512])
@pytest.mark.parametrize("b", [1, 4])
def test_shape(b: int, s: int, config: SimpleNamespace) -> None:
    """Checks that GQAttention preserves (B, S, hidden_dim) and stays finite.

    Args:
        b: Batch size.
        s: Sequence length.
        config: Model config under test (nano or small).
    """
    attn = _make_attn(config)
    cos, sin = precompute_rope(config.head_dim, MAX_SEQ_LEN)
    x = torch.randn(b, s, config.hidden_dim)
    out, kv_cache = attn(x, cos, sin)
    print(
        f"[shape] B={b} S={s} n_heads={config.n_heads} hidden_dim={config.hidden_dim} "
        f"-> out.shape={tuple(out.shape)}"
    )
    assert out.shape == (b, s, config.hidden_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
def test_causality(config: SimpleNamespace) -> None:
    """Checks that changing a future token does not change earlier positions' output.

    Args:
        config: Model config under test (nano or small).
    """
    attn = _make_attn(config)
    cos, sin = precompute_rope(config.head_dim, MAX_SEQ_LEN)
    b, s, j = 2, 16, 7
    x0 = torch.randn(b, s, config.hidden_dim)
    out0, _ = attn(x0, cos, sin)

    x1 = x0.clone()
    x1[:, j] = torch.randn(b, config.hidden_dim)
    out1, _ = attn(x1, cos, sin)

    before = out0[:, :j, :]
    after = out1[:, :j, :]
    max_dev = (before - after).abs().max().item()
    print(
        f"[causality] hidden_dim={config.hidden_dim} j={j} -> "
        f"max|out(i<j) before-after|={max_dev:.2e}"
    )
    assert torch.allclose(before, after, atol=1e-5)


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("s", [16, 128])
def test_kv_cache_equivalence(config: SimpleNamespace, b: int, s: int) -> None:
    """Checks that step-by-step KV-cache decoding matches a full forward pass.

    Args:
        config: Model config under test (nano or small).
        b: Batch size.
        s: Sequence length.
    """
    attn = _make_attn(config)
    attn.eval()
    cos, sin = precompute_rope(config.head_dim, MAX_SEQ_LEN)
    x = torch.randn(b, s, config.hidden_dim)

    with torch.no_grad():
        full_out, _ = attn(x, cos, sin)

        cache = KVCache()
        cached_parts = []
        for t in range(s):
            step_out, cache = attn(x[:, t : t + 1], cos, sin, kv_cache=cache)
            cached_parts.append(step_out)
        cached_out = torch.cat(cached_parts, dim=1)

    max_dev = (full_out - cached_out).abs().max().item()
    mean_dev = (full_out - cached_out).abs().mean().item()
    print(
        f"[kv-cache] hidden_dim={config.hidden_dim} B={b} S={s} -> "
        f"full-vs-cached max|dev|={max_dev:.2e} mean|dev|={mean_dev:.2e}"
    )
    assert full_out.shape == cached_out.shape
    assert torch.allclose(full_out, cached_out, atol=1e-4)
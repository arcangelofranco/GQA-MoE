from types import SimpleNamespace

import pytest
import torch

from src.model.block import TransformerBlock
from src.model.blocks import precompute_rope
from src.model.kv_cache import KVCache

NANO = SimpleNamespace(
    n_heads=8, n_kv_heads=2, head_dim=32, hidden_dim=256, rms_norm_eps=1e-6,
    n_experts=4, top_k=2, expert_intermediate=256, shared_intermediate=688,
)
SMALL = SimpleNamespace(
    n_heads=8, n_kv_heads=4, head_dim=64, hidden_dim=512, rms_norm_eps=1e-6,
    n_experts=8, top_k=2, expert_intermediate=512, shared_intermediate=1376,
)
CONFIGS = [NANO, SMALL]
MAX_SEQ_LEN = 1024


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
@pytest.mark.parametrize("s", [1, 16, 512])
@pytest.mark.parametrize("b", [1, 4])
def test_shape(b: int, s: int, config: SimpleNamespace) -> None:
    """Checks that TransformerBlock preserves shape and stays finite.

    Args:
        b: Batch size.
        s: Sequence length.
        config: Model config under test (nano or small).
    """
    block = TransformerBlock(config)
    cos, sin = precompute_rope(config.head_dim, MAX_SEQ_LEN)
    x = torch.randn(b, s, config.hidden_dim)
    out, aux, _ = block(x, cos, sin)
    print(
        f"[shape] B={b} S={s} hidden_dim={config.hidden_dim} -> "
        f"out.shape={tuple(out.shape)}"
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
def test_aux_loss_propagated(config: SimpleNamespace) -> None:
    """Checks that the block's MoE aux loss is a finite, non-negative scalar.

    Args:
        config: Model config under test (nano or small).
    """
    block = TransformerBlock(config)
    block.train()
    cos, sin = precompute_rope(config.head_dim, MAX_SEQ_LEN)
    x = torch.randn(2, 16, config.hidden_dim)
    out, aux, _ = block(x, cos, sin)
    print(
        f"[aux] hidden_dim={config.hidden_dim} -> aux.shape={tuple(aux.shape)} "
        f"value={aux.item():.6f} finite={torch.isfinite(aux).item()}"
    )
    assert aux.dim() == 0
    assert torch.isfinite(aux)
    assert aux.item() >= 0.0


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
def test_kv_cache_grows(config: SimpleNamespace) -> None:
    """Checks that the KV cache grows by one position per incremental decoding step.

    Args:
        config: Model config under test (nano or small).
    """
    block = TransformerBlock(config)
    block.eval()
    cos, sin = precompute_rope(config.head_dim, MAX_SEQ_LEN)
    b, n_kv = 2, config.n_kv_heads
    cache = KVCache()
    with torch.no_grad():
        for step in range(4):
            x = torch.randn(b, 1, config.hidden_dim)
            out, aux, cache = block(x, cos, sin, kv_cache=cache)
            k_len = cache.k.shape[2]
            print(
                f"[cache-grow] hidden_dim={config.hidden_dim} step={step} -> "
                f"k.shape={tuple(cache.k.shape)}"
            )
            assert cache.k.shape == (b, n_kv, step + 1, config.head_dim)
            assert cache.v.shape == (b, n_kv, step + 1, config.head_dim)


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
@pytest.mark.parametrize("s", [16, 64])
def test_kv_cache_equivalence(config: SimpleNamespace, s: int) -> None:
    """Checks that step-by-step KV-cache decoding matches a full forward pass.

    Args:
        config: Model config under test (nano or small).
        s: Sequence length.
    """
    block = TransformerBlock(config)
    block.eval()
    cos, sin = precompute_rope(config.head_dim, MAX_SEQ_LEN)
    b = 2
    x = torch.randn(b, s, config.hidden_dim)

    with torch.no_grad():
        full_out, full_aux, _ = block(x, cos, sin)

        cache = KVCache()
        cached_parts = []
        for t in range(s):
            step_out, _, cache = block(x[:, t : t + 1], cos, sin, kv_cache=cache)
            cached_parts.append(step_out)
        cached_out = torch.cat(cached_parts, dim=1)

    max_dev = (full_out - cached_out).abs().max().item()
    mean_dev = (full_out - cached_out).abs().mean().item()
    print(
        f"[block kv-cache] hidden_dim={config.hidden_dim} B={b} S={s} -> "
        f"full-vs-cached max|dev|={max_dev:.2e} mean|dev|={mean_dev:.2e}"
    )
    assert full_out.shape == cached_out.shape
    assert torch.allclose(full_out, cached_out, atol=1e-4)
import pytest
import torch

from src.model.blocks import precompute_rope, rotate_half, apply_rope

S_LENS = [1, 16, 512, 1024]
HEAD_DIMS = [32, 64]
B = 2
N_HEADS = 4


@pytest.mark.parametrize("s", S_LENS)
@pytest.mark.parametrize("dim", HEAD_DIMS)
def test_shape(s: int, dim: int) -> None:
    """Checks that applying RoPE preserves the input tensor shape.

    Args:
        s: Sequence length.
        dim: Head dimension.
    """
    cos, sin = precompute_rope(dim, max_seq_len=max(s, 16))
    x = torch.randn(B, N_HEADS, s, dim)
    out = apply_rope(x, cos[:s], sin[:s])
    print(f"[shape] S={s} head_dim={dim} -> out.shape={tuple(out.shape)}")
    assert out.shape == x.shape


@pytest.mark.parametrize("s", S_LENS)
@pytest.mark.parametrize("dim", HEAD_DIMS)
def test_isometry(s: int, dim: int) -> None:
    """Checks that RoPE is an isometry: it preserves each vector's norm.

    Args:
        s: Sequence length.
        dim: Head dimension.
    """
    cos, sin = precompute_rope(dim, max_seq_len=max(s, 16))
    x = torch.randn(B, N_HEADS, s, dim)
    out = apply_rope(x, cos[:s], sin[:s])
    before = x.norm(dim=-1)
    after = out.norm(dim=-1)
    max_dev = (after - before).abs().max().item()
    print(
        f"[isometry] S={s} head_dim={dim} -> "
        f"norm before={before.mean().item():.8f} after={after.mean().item():.8f} "
        f"max|dev|={max_dev:.2e}"
    )
    assert torch.allclose(after, before, atol=1e-5)


@pytest.mark.parametrize("dim", HEAD_DIMS)
def test_precompute_idempotent(dim: int) -> None:
    """Checks that precompute_rope returns identical tensors across repeated calls.

    Args:
        dim: Head dimension.
    """
    cos1, sin1 = precompute_rope(dim, max_seq_len=1024)
    cos2, sin2 = precompute_rope(dim, max_seq_len=1024)
    print(
        f"[idempotent] head_dim={dim} -> "
        f"cos equal={torch.equal(cos1, cos2)} sin equal={torch.equal(sin1, sin2)}"
    )
    assert torch.equal(cos1, cos2)
    assert torch.equal(sin1, sin2)


@pytest.mark.parametrize("dim", HEAD_DIMS)
def test_positional_selectivity(dim: int) -> None:
    """Checks that rotating two copies of the same vector to different positions
    lowers their dot product relative to the unrotated pair.

    Args:
        dim: Head dimension.
    """
    cos, sin = precompute_rope(dim, max_seq_len=16)
    x0 = torch.randn(B, N_HEADS, 1, dim)
    x1 = x0.clone()
    unrotated_ip = (x0 * x1).sum(dim=-1)
    r0 = apply_rope(x0, cos[0:1], sin[0:1])
    r1 = apply_rope(x1, cos[1:2], sin[1:2])
    rotated_ip = (r0 * r1).sum(dim=-1)
    print(
        f"[selectivity] head_dim={dim} -> "
        f"mean <x0,x1>={unrotated_ip.mean().item():.6f} "
        f"mean <RoPE0,RoPE1>={rotated_ip.mean().item():.6f}"
    )
    assert (rotated_ip < unrotated_ip).all()


@pytest.mark.parametrize("dim", HEAD_DIMS)
def test_rotate_half_inverse(dim: int) -> None:
    """Checks that applying rotate_half twice negates the original tensor.

    Args:
        dim: Head dimension.
    """
    x = torch.randn(B, N_HEADS, 16, dim)
    inv = rotate_half(rotate_half(x))
    expected = -x
    print(
        f"[rotate_half] head_dim={dim} -> "
        f"max|rotate_half(rotate_half(x)) - (-x)|={(inv + x).abs().max().item():.2e}"
    )
    assert torch.equal(inv, expected)


@pytest.mark.parametrize("s", [16, 512])
@pytest.mark.parametrize("dim", HEAD_DIMS)
def test_absolute_offset(s: int, dim: int) -> None:
    """Checks that RoPE output depends on the absolute position offset, not just relative order.

    Args:
        s: Sequence length.
        dim: Head dimension.
    """
    cos, sin = precompute_rope(dim, max_seq_len=s + 16)
    x = torch.randn(B, N_HEADS, s, dim)
    off0 = apply_rope(x, cos[:s], sin[:s])
    off5 = apply_rope(x, cos[5 : 5 + s], sin[5 : 5 + s])
    max_diff = (off0 - off5).abs().max().item()
    print(
        f"[offset] S={s} head_dim={dim} -> offset 0 vs offset 5: max|diff|={max_diff:.4f}"
    )
    assert not torch.allclose(off0, off5)
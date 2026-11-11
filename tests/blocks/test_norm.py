import math

import pytest
import torch

from src.model.blocks import RMSNorm

B_SIZES = [1, 4]
S_LENS = [1, 16, 512]
HIDDEN_DIMS = [256, 512]


@pytest.mark.parametrize("b", B_SIZES)
@pytest.mark.parametrize("s", S_LENS)
@pytest.mark.parametrize("dim", HIDDEN_DIMS)
def test_shape(b: int, s: int, dim: int) -> None:
    """Verifies that RMSNorm preserves the input shape.

    Args:
        b: Batch size.
        s: Sequence length.
        dim: Hidden dimension.
    """
    norm = RMSNorm(dim)
    x = torch.randn(b, s, dim)
    out = norm(x)
    print(f"[shape] B={b} S={s} dim={dim} -> out.shape={tuple(out.shape)}")
    assert out.shape == x.shape


@pytest.mark.parametrize("dim", HIDDEN_DIMS)
@pytest.mark.parametrize("b", B_SIZES)
@pytest.mark.parametrize("s", S_LENS)
def test_l2_norm_is_sqrt_dim(b: int, s: int, dim: int) -> None:
    """Verifies that a unit weight rescales each vector's L2 norm to ``sqrt(dim)``.

    With default initialization the normalized output must have a fixed norm
    independent of the input scale, which is the key property of RMSNorm.

    Args:
        b: Batch size.
        s: Sequence length.
        dim: Hidden dimension.
    """
    norm = RMSNorm(dim)
    x = torch.randn(b, s, dim)
    out = norm(x)
    expected = math.sqrt(dim)
    norms = out.norm(dim=-1)
    max_dev = (norms - expected).abs().max().item()
    print(
        f"[l2] B={b} S={s} dim={dim} -> "
        f"mean norm={norms.mean().item():.8f} expected sqrt(dim)={expected:.8f} "
        f"max|dev|={max_dev:.2e}"
    )
    assert torch.allclose(norms, torch.tensor(expected), atol=1e-5)


def test_weight_participates() -> None:
    """Verifies the learnable weight scales the normalized output proportionally.

    Doubling ``norm.weight`` must double both the output and its norm,
    confirming the parameter is wired into the forward pass and not ignored.
    """
    norm = RMSNorm(256)
    x = torch.randn(3, 5, 256)
    base = norm(x)
    base_norm = base.norm(dim=-1).mean().item()
    with torch.no_grad():
        norm.weight.mul_(2.0)
    doubled = norm(x)
    doubled_norm = doubled.norm(dim=-1).mean().item()
    ratio = doubled_norm / base_norm
    print(
        f"[weight] with weight=1: mean norm={base_norm:.8f} | "
        f"with weight=2: mean norm={doubled_norm:.8f} | ratio={ratio:.6f}"
    )
    assert torch.allclose(doubled, 2.0 * base, atol=1e-5)
    assert torch.allclose(doubled.norm(dim=-1), 2.0 * base.norm(dim=-1), atol=1e-5)


@pytest.mark.parametrize("dim", HIDDEN_DIMS)
def test_stability_near_zero(dim: int) -> None:
    """Verifies RMSNorm stays finite and nonzero for near-zero inputs.

    The epsilon term in the denominator must prevent division by zero and
    NaN/Inf propagation on degenerate inputs.

    Args:
        dim: Hidden dimension.
    """
    norm = RMSNorm(dim)
    x = torch.full((2, 8, dim), 1e-10)
    out = norm(x)
    print(
        f"[zero] dim={dim} -> input=full(1e-10) | "
        f"nan={torch.isnan(out).sum().item()} inf={torch.isinf(out).sum().item()} "
        f"mean norm={out.norm(dim=-1).mean().item():.8f}"
    )
    assert torch.isfinite(out).all()
    assert out.norm(dim=-1).max() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 not supported on CPU")
@pytest.mark.parametrize("dim", HIDDEN_DIMS)
def test_bf16_dtype_invariance(dim: int) -> None:
    """Verifies RMSNorm preserves bf16 dtype and stays finite on CUDA.

    Skipped when CUDA is unavailable, since bf16 is not supported on CPU here.

    Args:
        dim: Hidden dimension.
    """
    norm = RMSNorm(dim).to("cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 4, dim, device="cuda", dtype=torch.bfloat16)
    out = norm(x)
    print(
        f"[dtype] dim={dim} -> input dtype={x.dtype} | output dtype={out.dtype} | "
        f"nan={torch.isnan(out).sum().item()} inf={torch.isinf(out).sum().item()}"
    )
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
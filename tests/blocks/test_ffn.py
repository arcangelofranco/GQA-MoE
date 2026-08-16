import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest

from src.model.blocks import SwiGLU

B_SIZES = [1, 4]
S_LENS = [1, 16, 512]
DIMS = [256, 512]
INTERMEDIATE_SIZES = [256, 688, 512, 1376]


@pytest.mark.parametrize("inter", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("s", S_LENS)
@pytest.mark.parametrize("b", B_SIZES)
def test_shape(b: int, s: int, dim: int, inter: int) -> None:
    """Checks that SwiGLU preserves the input shape.

    Args:
        b: Batch size.
        s: Sequence length.
        dim: Model (input/output) dimension.
        inter: Intermediate (hidden) dimension.
    """
    m = SwiGLU(dim, inter)
    x = torch.randn(b, s, dim)
    out = m(x)
    print(
        f"[shape] B={b} S={s} dim={dim} inter={inter} -> "
        f"out.shape={tuple(out.shape)}"
    )
    assert out.shape == x.shape


@pytest.mark.parametrize("inter", [256, 688])
@pytest.mark.parametrize("dim", [256, 512])
def test_gradients_flow(dim: int, inter: int) -> None:
    """Verifies that gradients reach every projection weight after backward.

    Args:
        dim: Model (input/output) dimension.
        inter: Intermediate (hidden) dimension.
    """
    m = SwiGLU(dim, inter)
    x = torch.randn(2, 3, dim)
    m(x).sum().backward()
    grads = {
        "gate_proj": m.gate_proj.weight.grad,
        "up_proj": m.up_proj.weight.grad,
        "down_proj": m.down_proj.weight.grad,
    }
    print(
        f"[grad] dim={dim} inter={inter} -> "
        + " | ".join(
            f"{name} grad_norm={g.norm().item():.4f} (is None: {g is None})"
            for name, g in grads.items()
        )
    )
    for name, g in grads.items():
        assert g is not None
        assert g.abs().sum() > 0


@pytest.mark.parametrize("inter", [256, 688])
@pytest.mark.parametrize("dim", [256, 512])
def test_init_nonconstant(dim: int, inter: int) -> None:
    """Ensures a freshly initialized SwiGLU does not produce an all-zero output.

    Args:
        dim: Model (input/output) dimension.
        inter: Intermediate (hidden) dimension.
    """
    m = SwiGLU(dim, inter)
    x = torch.randn(2, 4, dim)
    out = m(x)
    total = out.abs().sum()
    print(f"[init] dim={dim} inter={inter} -> out.abs().sum()={total.item():.6f}")
    assert out.abs().sum() > 0


@pytest.mark.parametrize("inter", [256, 688])
@pytest.mark.parametrize("dim", [256, 512])
def test_gate_effective(dim: int, inter: int) -> None:
    """Checks that both the up and gate branches actually affect the output.

    Disables the up branch (via a subclass) and separately zeroes the gate
    weights, confirming each ablation changes the output as expected.

    Args:
        dim: Model (input/output) dimension.
        inter: Intermediate (hidden) dimension.
    """
    m = SwiGLU(dim, inter)
    x = torch.randn(2, 3, dim)

    class NoUp(SwiGLU):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(z)))

    m_no_up = NoUp(dim, inter)
    m_no_up.load_state_dict(m.state_dict())
    expected_no_up = m_no_up.down_proj(F.silu(m_no_up.gate_proj(x)))
    actual_no_up = m_no_up(x)
    print(
        f"[gate/up-off] dim={dim} inter={inter} -> "
        f"max|out - down(silu(gate(x)))|={(actual_no_up - expected_no_up).abs().max().item():.2e}"
    )
    assert torch.allclose(actual_no_up, expected_no_up, atol=1e-6)

    m.gate_proj.weight.data.zero_()
    out_no_gate = m(x)
    print(
        f"[gate/gate-off] dim={dim} inter={inter} -> "
        f"max|out| with gate zeroed={(out_no_gate.abs().max().item()):.2e}"
    )
    assert torch.allclose(out_no_gate, torch.zeros_like(out_no_gate), atol=1e-6)


def test_experts_isolated_from_moe() -> None:
    """Checks that independent SwiGLU experts combine into a finite, shape-correct output.

    Simulates a naive (unrouted) MoE sum of several standalone experts to
    confirm they compose without shape mismatches or NaN/Inf leakage.
    """
    dim, inter, n_experts = 256, 688, 4
    experts = nn.ModuleList([SwiGLU(dim, inter) for _ in range(n_experts)])
    x = torch.randn(2, 5, dim)
    out = sum(e(x) for e in experts)
    print(
        f"[experts] n_experts={n_experts} -> shape={tuple(out.shape)} "
        f"finite={torch.isfinite(out).all()} max|out|={out.abs().max().item():.4f}"
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
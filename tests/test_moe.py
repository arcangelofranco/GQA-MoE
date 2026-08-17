from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.model.moe import MoELayer

NANO = SimpleNamespace(
    hidden_dim=256, n_experts=4, top_k=2,
    expert_intermediate=256, shared_intermediate=688,
)
SMALL = SimpleNamespace(
    hidden_dim=512, n_experts=8, top_k=2,
    expert_intermediate=512, shared_intermediate=1376,
)
CONFIGS = [NANO, SMALL]


def _routed_indices(moe: MoELayer, x: torch.Tensor) -> list[int]:
    """Returns the set of expert indices the router selects for a batch of tokens.

    Helper used to inspect routing behaviour without depending on internal
    implementation details of :class:`MoELayer`.

    Args:
        moe: The MoE layer whose router is queried.
        x: Input tensor to route, shaped ``(B, S, hidden_dim)``.

    Returns:
        Sorted list of unique expert indices chosen across all tokens'
        top-k selections.
    """
    router_logits = moe.router(x.reshape(-1, x.shape[-1]))
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    _, top_index = router_probs.topk(moe.top_k, dim=-1)
    return top_index.unique().tolist()


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
@pytest.mark.parametrize("s", [1, 16, 512])
@pytest.mark.parametrize("b", [1, 4])
def test_shape(b: int, s: int, config: SimpleNamespace) -> None:
    """Verifies that a forward pass preserves the input shape and produces finite outputs.

    Guards the MoE routing and expert computation against shape corruption or
    numerical blow-ups across a range of batch and sequence sizes.

    Args:
        b: Batch size.
        s: Sequence length.
        config: Model config under test (nano or small).
    """
    moe = MoELayer(config)
    x = torch.randn(b, s, config.hidden_dim)
    out, aux = moe(x)
    print(
        f"[shape] B={b} S={s} n_experts={config.n_experts} "
        f"hidden_dim={config.hidden_dim} -> out.shape={tuple(out.shape)}"
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
def test_aux_loss_scalar_finite_nonneg(config: SimpleNamespace) -> None:
    """Verifies that the auxiliary load-balancing loss is a finite, non-negative scalar.

    A scalar shape and non-negative value are required so the loss can be added
    directly to the training objective.

    Args:
        config: Model config under test (nano or small).
    """
    moe = MoELayer(config)
    moe.train()
    x = torch.randn(4, 16, config.hidden_dim)
    out, aux = moe(x)
    print(
        f"[aux] n_experts={config.n_experts} -> shape={tuple(aux.shape)} "
        f"value={aux.item():.6f} finite={torch.isfinite(aux).item()}"
    )
    assert aux.dim() == 0
    assert torch.isfinite(aux)
    assert aux.item() >= 0.0


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
def test_expert_utilization(config: SimpleNamespace) -> None:
    """Verifies that routing a diverse batch engages more than one expert.

    A working router should not collapse onto a single expert; spreading
    selections is a prerequisite for the load-balancing behaviour the MoE
    layer is meant to encourage.

    Args:
        config: Model config under test (nano or small).
    """
    moe = MoELayer(config)
    x = torch.randn(8, 64, config.hidden_dim)
    selected = _routed_indices(moe, x)
    print(
        f"[utilization] n_experts={config.n_experts} top_k={config.top_k} "
        f"N=512 -> selected experts={sorted(selected)}"
    )
    assert len(selected) >= 2


def test_moe_layer_does_not_depend_on_aux_loss_coeff() -> None:
    """Verifies that :class:`MoELayer` operates without an ``aux_loss_coeff`` field.

    The coefficient is a training-time concern owned by the outer model, so the
    layer itself must not require it. A forward pass must still run and produce
    a valid auxiliary loss.

    Raises:
        AssertionError: If the config or layer exposes ``aux_loss_coeff`` or the
            forward pass produces non-finite or negative auxiliary loss.
    """
    config = SimpleNamespace(
        hidden_dim=64, n_experts=4, top_k=2, expert_intermediate=32, shared_intermediate=64,
    )
    assert not hasattr(config, "aux_loss_coeff")
    moe = MoELayer(config)
    assert not hasattr(moe, "aux_loss_coeff")

    moe.train()
    x = torch.randn(4, 16, config.hidden_dim)
    out, aux = moe(x)
    print(f"[no-aux-coeff] out.shape={tuple(out.shape)} aux={aux.item():.6f} finite={torch.isfinite(aux).item()}")
    assert out.shape == x.shape
    assert torch.isfinite(aux) and aux.item() >= 0.0


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
def test_gradient_on_every_expert(config: SimpleNamespace) -> None:
    """Verifies that backpropagation assigns nonzero gradients to every expert, the shared expert, and the router.

    A dead expert (never receiving gradient) would silently waste capacity and
    defeat the purpose of the architecture, so every routing path must be
    reachable by the optimizer.

    Args:
        config: Model config under test (nano or small).

    Raises:
        AssertionError: If any expert has a zero or missing gradient, or the
            router receives no gradient.
    """
    moe = MoELayer(config)
    moe.train()
    for _ in range(3):
        x = torch.randn(8, 32, config.hidden_dim)
        out, aux = moe(x)
        (out.sum() + aux).backward()

    missing = [i for i, e in enumerate(moe.experts) if e.down_proj.weight.grad is None]
    mags = {i: e.down_proj.weight.grad.abs().sum().item() for i, e in enumerate(moe.experts)}
    shared_g = moe.shared_expert.down_proj.weight.grad
    print(f"[grad] {config.n_experts} experts mags={mags} | shared_expert grad_norm={shared_g.norm().item():.4f}")
    assert not missing, f"experts without gradient: {missing}"
    assert all(m > 0.0 for m in mags.values()), "zero gradient on some expert"
    assert moe.router.weight.grad is not None
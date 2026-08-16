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
    """Computes the set of expert indices selected by the router for a batch.

    Args:
        moe: MoE layer whose router to query.
        x: Input tensor to route.

    Returns:
        Sorted-by-value unique list of expert indices chosen across all
        tokens' top-k selections.
    """
    router_logits = moe.router(x.reshape(-1, x.shape[-1]))
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    _, top_index = router_probs.topk(moe.top_k, dim=-1)
    return top_index.unique().tolist()


@pytest.mark.parametrize("config", CONFIGS, ids=["nano", "small"])
@pytest.mark.parametrize("s", [1, 16, 512])
@pytest.mark.parametrize("b", [1, 4])
def test_shape(b: int, s: int, config: SimpleNamespace) -> None:
    """Checks that MoELayer preserves shape and stays finite.

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
    """Checks that the MoE aux loss is a finite, non-negative scalar.

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
    """Checks that routing over a large batch spreads selections across at least 2 experts.

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
    """Checks that MoELayer works without an aux_loss_coeff field on the config or the layer."""
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
    """Checks that every expert (and the shared expert and router) receives a nonzero gradient.

    Args:
        config: Model config under test (nano or small).
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
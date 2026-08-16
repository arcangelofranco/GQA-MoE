import math
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from src.model.transformer import ForwardOutput, Transformer
from src.model.kv_cache import KVCache


def _small_cfg(**overrides: object) -> SimpleNamespace:
    """Builds a small model config namespace, overridable per field.

    Args:
        **overrides: Field values overriding the small-model defaults.

    Returns:
        A `SimpleNamespace` with all fields `Transformer` expects,
        including the derived `hidden_dim = n_heads * head_dim`.
    """
    defaults = dict(
        vocab_size=100, n_layers=3, n_heads=4, n_kv_heads=2, head_dim=16,
        expert_intermediate=32, shared_intermediate=48, n_experts=4, top_k=2,
        max_seq_len=64, rope_theta=10000.0, rms_norm_eps=1e-6,
        aux_loss_coeff=0.01, tie_embeddings=True,
    )
    defaults.update(overrides)
    cfg = SimpleNamespace(**defaults)
    cfg.hidden_dim = cfg.n_heads * cfg.head_dim
    return cfg


def _nano_cfg() -> SimpleNamespace:
    """Builds a config matching the "nano" preset's architecture.

    Returns:
        The "nano"-sized config namespace.
    """
    return _small_cfg(
        vocab_size=8000, n_layers=12, n_heads=8, n_kv_heads=2, head_dim=32,
        expert_intermediate=256, shared_intermediate=688, max_seq_len=1024,
    )


def _small_preset_cfg() -> SimpleNamespace:
    """Builds a config matching the "small" preset's architecture.

    Returns:
        The "small"-sized config namespace.
    """
    return _small_cfg(
        vocab_size=16000, n_layers=16, n_heads=8, n_kv_heads=4, head_dim=64,
        expert_intermediate=512, shared_intermediate=1376, n_experts=8, top_k=2,
        max_seq_len=2048,
    )


@pytest.mark.parametrize("b", [1, 2, 3])
@pytest.mark.parametrize("s", [1, 8, 16])
def test_output_shape(b: int, s: int) -> None:
    """Checks that the Transformer's logits are shaped (B, S, vocab_size).

    Args:
        b: Batch size.
        s: Sequence length.
    """
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (b, s))
    logits = model(idx).logits
    print(f"[shape] B={b} S={s} -> logits.shape={tuple(logits.shape)}")
    assert logits.shape == (b, s, cfg.vocab_size)


def test_forward_output_fields_are_named_and_ordered() -> None:
    """Checks that ForwardOutput exposes (logits, aux_loss, kv_caches) as a named, ordered tuple."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(idx)
    assert isinstance(out, ForwardOutput)
    assert out._fields == ("logits", "aux_loss", "kv_caches")

    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.aux_loss.ndim == 0
    assert len(out.kv_caches) == cfg.n_layers

    assert isinstance(out, tuple) and len(out) == 3
    assert out[0] is out.logits


@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
def test_weight_tying(tie: bool) -> None:
    """Checks that tie_embeddings controls whether lm_head shares storage with the embeddings.

    Args:
        tie: Whether `tie_embeddings` is enabled for this config.
    """
    cfg = _small_cfg(tie_embeddings=tie)
    model = Transformer(cfg)
    same_ptr = model.lm_head.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()
    print(f"[tie] tie_embeddings={tie} -> shared data_ptr={same_ptr}")
    assert same_ptr is tie


PARAM_PRESETS = [
    (_nano_cfg, 19_811_328, "nano"),
    (_small_preset_cfg, 155_339_264, "small"),
]


@pytest.mark.parametrize("cfg_fn,expected,name", PARAM_PRESETS)
def test_param_count_matches_validated_presets(
    cfg_fn: Callable[[], SimpleNamespace], expected: int, name: str
) -> None:
    """Checks that a preset-equivalent config's parameter count matches the validated reference.

    Args:
        cfg_fn: Zero-arg factory building the config to instantiate.
        expected: Expected parameter count for that config.
        name: Preset name, used only for the diagnostic print.
    """
    model = Transformer(cfg_fn())
    n_params = model.num_params()
    print(f"[params] {name} num_params={n_params} expected={expected}")
    assert n_params == expected


@pytest.mark.parametrize("n_layers", [8], ids=["8layers"])
def test_differentiated_init_residual_stream(n_layers: int) -> None:
    """Checks that output projections are initialized with a tighter std than input projections.

    Args:
        n_layers: Number of transformer layers to build.
    """
    torch.manual_seed(0)
    cfg = _small_cfg(n_layers=n_layers)
    model = Transformer(cfg)
    wo_std = model.blocks[0].attention.wo.weight.std().item()
    wq_std = model.blocks[0].attention.wq.weight.std().item()
    expected_wo_std = 0.02 / math.sqrt(2 * cfg.n_layers)
    print(
        f"[init] wq_std={wq_std:.6f} expected 0.02 | "
        f"wo_std={wo_std:.6f} expected {expected_wo_std:.6f}"
    )
    assert wq_std == pytest.approx(0.02, rel=0.15)
    assert wo_std == pytest.approx(expected_wo_std, rel=0.15)
    assert wo_std < wq_std


@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("s", [32])
def test_no_nan_or_inf_in_logits(b: int, s: int) -> None:
    """Checks that both logits and aux_loss stay finite for a forward pass.

    Args:
        b: Batch size.
        s: Sequence length.
    """
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (b, s))
    out = model(idx)
    logits, aux_loss = out.logits, out.aux_loss
    print(
        f"[finite] logits finite={torch.isfinite(logits).all().item()} "
        f"aux finite={torch.isfinite(aux_loss).all().item()}"
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss).all()


def test_aux_loss_coeff_applied_exactly_once() -> None:
    """Checks that aux_loss_coeff scales the raw MoE aux loss exactly once, not cumulatively."""
    torch.manual_seed(0)
    cfg = _small_cfg(n_layers=1, aux_loss_coeff=0.37)
    model = Transformer(cfg)
    model.train()

    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    aux_loss_total = model(idx).aux_loss

    block = model.blocks[0]
    with torch.no_grad():
        x = model.tok_embeddings(idx)
        attn_out, _ = block.attention(block.attention_norm(x), model.cos, model.sin)
        h = x + attn_out
        _, raw_aux = block.moe(block.moe_norm(h))

    expected = cfg.aux_loss_coeff * raw_aux
    print(f"[aux-scale] raw={raw_aux.item():.6f} coeff={cfg.aux_loss_coeff} "
          f"expected={expected.item():.6f} actual={aux_loss_total.item():.6f}")
    assert aux_loss_total.item() == pytest.approx(expected.item(), rel=1e-5)
    assert raw_aux.item() != pytest.approx(aux_loss_total.item())


@pytest.mark.parametrize("n_layers", [3], ids=["3layers"])
def test_aux_loss_total_gradient_reaches_every_block_router(n_layers: int) -> None:
    """Checks that backpropagating through logits + aux_loss reaches every block's router.

    Args:
        n_layers: Number of transformer layers to build.
    """
    torch.manual_seed(0)
    cfg = _small_cfg(n_layers=n_layers)
    model = Transformer(cfg)
    model.train()
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    out = model(idx)
    logits, aux_loss_total = out.logits, out.aux_loss
    (logits.sum() + aux_loss_total).backward()
    for i, block in enumerate(model.blocks):
        grad = block.moe.router.weight.grad
        print(
            f"[router-grad] block {i} grad is None={grad is None} "
            f"norm={(grad.abs().sum().item() if grad is not None else 0.0):.4f}"
        )
        assert grad is not None and grad.abs().sum() > 0, f"block {i}'s router has no gradient"


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("s", [10, 20])
def test_kv_cache_equivalence_full_model(b: int, s: int) -> None:
    """Checks that step-by-step KV-cache decoding matches a full forward pass on the whole model.

    Args:
        b: Batch size.
        s: Sequence length.
    """
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = Transformer(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (b, s))

    logits_full = model(idx, kv_caches=None).logits

    kv_caches = [KVCache() for _ in range(cfg.n_layers)]
    outs = []
    for t in range(s):
        step_out = model(idx[:, t : t + 1], kv_caches=kv_caches)
        logits_t, kv_caches = step_out.logits, step_out.kv_caches
        outs.append(logits_t)
    logits_incremental = torch.cat(outs, dim=1)

    max_dev = (logits_full - logits_incremental).abs().max().item()
    mean_dev = (logits_full - logits_incremental).abs().mean().item()
    print(
        f"[full kv-cache] B={b} S={s} n_layers={cfg.n_layers} -> "
        f"max|dev|={max_dev:.2e} mean|dev|={mean_dev:.2e}"
    )
    assert torch.allclose(logits_full, logits_incremental, atol=1e-4)
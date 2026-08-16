import pytest
import torch

from src.runtime.sampler import SamplingPolicy


def _peaked_logits(batch: int = 3, vocab: int = 20, seed: int = 0) -> torch.Tensor:
    """Generates random logits for sampler tests.

    Args:
        batch: Batch size.
        vocab: Vocabulary size.
        seed: Seed for reproducibility.

    Returns:
        A (batch, vocab) tensor of random logits.
    """
    torch.manual_seed(seed)
    return torch.randn(batch, vocab)


def test_greedy_returns_argmax_and_ignores_top_k_top_p() -> None:
    """Checks that temperature=0.0 always returns the argmax, ignoring top_k/top_p."""
    logits = _peaked_logits()
    expected = logits.argmax(dim=-1, keepdim=True)

    out = SamplingPolicy(temperature=0.0, top_k=3, top_p=0.5).select(logits)
    print(f"[greedy] out.shape={tuple(out.shape)} equal_to_argmax={torch.equal(out, expected)}")
    assert out.shape == (logits.size(0), 1)
    assert torch.equal(out, expected)


def test_output_shape_and_dtype() -> None:
    """Checks that select() returns one int64 id per batch row."""
    logits = _peaked_logits(batch=5, vocab=37)
    out = SamplingPolicy(temperature=1.0).select(logits)
    print(f"[shape/dtype] out.shape={tuple(out.shape)} out.dtype={out.dtype}")
    assert out.shape == (5, 1)
    assert out.dtype in (torch.int64, torch.long)


def test_sampled_ids_always_within_vocab_range() -> None:
    """Checks that sampled ids stay within [0, vocab) across repeated draws."""
    logits = _peaked_logits(batch=8, vocab=15)
    policy = SamplingPolicy(temperature=1.3, top_k=5, top_p=0.9)
    for _ in range(20):
        out = policy.select(logits)
        assert (out >= 0).all() and (out < 15).all()
    print(f"[vocab-range] draws=20 -> min={out.min().item()} max={out.max().item()} vocab=15")


def test_top_k_one_is_deterministic_and_matches_argmax() -> None:
    """Checks that top_k=1 collapses sampling onto the argmax even with temperature=1.0.

    With a single admitted candidate, there is nothing left for randomness
    to act on.
    """
    torch.manual_seed(1)
    logits = torch.randn(4, 25)
    expected = logits.argmax(dim=-1, keepdim=True)
    policy = SamplingPolicy(temperature=1.0, top_k=1)
    for _ in range(10):
        out = policy.select(logits)
        assert torch.equal(out, expected)
    print(f"[top_k=1] draws=10 -> last_out={out.flatten().tolist()} matches_argmax={torch.equal(out, expected)}")


def test_top_p_near_zero_is_deterministic_and_matches_argmax() -> None:
    """Checks that a tiny top_p keeps only the most probable token, matching top_k=1.

    A tiny top_p still keeps the most probable token (the candidate list
    is never empty) and no other one: same practical effect as top_k=1.
    """
    torch.manual_seed(2)
    logits = torch.randn(4, 25)
    expected = logits.argmax(dim=-1, keepdim=True)
    policy = SamplingPolicy(temperature=1.0, top_p=1e-6)
    for _ in range(10):
        out = policy.select(logits)
        assert torch.equal(out, expected)
    print(f"[top_p~0] draws=10 -> last_out={out.flatten().tolist()} matches_argmax={torch.equal(out, expected)}")


def test_top_k_larger_than_vocab_size_does_not_crash() -> None:
    """Checks that top_k exceeding the vocab size is clamped instead of crashing."""
    logits = _peaked_logits(batch=2, vocab=6)
    out = SamplingPolicy(temperature=1.0, top_k=1000).select(logits)
    print(f"[top_k>vocab] top_k=1000 vocab=6 -> out.shape={tuple(out.shape)} out={out.flatten().tolist()}")
    assert out.shape == (2, 1)
    assert (out >= 0).all() and (out < 6).all()


def test_no_nan_or_inf_probabilities_with_combined_top_k_top_p() -> None:
    """Checks that combining top_k and top_p never yields NaN/Inf outputs."""
    logits = _peaked_logits(batch=4, vocab=50)
    policy = SamplingPolicy(temperature=0.7, top_k=10, top_p=0.9)
    for _ in range(20):
        out = policy.select(logits)
        assert torch.isfinite(out.float()).all()
    print(f"[finite] draws=20 -> all_finite={torch.isfinite(out.float()).all().item()}")


def test_is_greedy_only_at_zero_temperature() -> None:
    """Checks that is_greedy is true only for temperature=0.0, not for near-zero values."""
    zero_greedy = SamplingPolicy(temperature=0.0).is_greedy
    near_zero_greedy = SamplingPolicy(temperature=1e-3).is_greedy
    print(f"[is_greedy] temperature=0.0 -> is_greedy={zero_greedy} | temperature=1e-3 -> is_greedy={near_zero_greedy}")
    assert zero_greedy
    assert not near_zero_greedy


# Sampling parameters now have a single place where they get validated:
# previously any absurd value would reach torch, which failed much later
# and with a message disconnected from the actual cause.
@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_k": 0},
        {"top_p": 0.0},
        {"top_p": 1.5},
    ],
)
def test_rejects_out_of_range_settings(kwargs: dict[str, float | int]) -> None:
    """Checks that out-of-range sampling settings raise ValueError at construction.

    Args:
        kwargs: Single invalid keyword argument to pass to `SamplingPolicy`.
    """
    with pytest.raises(ValueError) as exc_info:
        SamplingPolicy(**kwargs)
    print(f"[reject] kwargs={kwargs} -> raised={exc_info.value!r}")

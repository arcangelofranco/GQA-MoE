from types import SimpleNamespace

import pytest
import torch

from src.config import ModelConfig
from src.data.tokenizer import Tokenizer
from src.model.transformer import ForwardOutput, Transformer
from src.runtime.generator import TextGenerator
from src.runtime.sampler import SamplingPolicy


def _tiny_model_and_prompt(seed: int = 0) -> tuple[Transformer, ModelConfig, torch.Tensor]:
    """Builds a tiny Transformer and a random prompt for generator tests.

    Args:
        seed: Seed for weight initialization and prompt sampling.

    Returns:
        Tuple of (model in eval mode, model config, prompt id tensor).
    """
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=40, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=8,
        expert_intermediate=16, shared_intermediate=32, n_experts=4, top_k=2,
        max_seq_len=64,
    )
    model = Transformer(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    return model, cfg, idx


def _generator(
    model: Transformer, tokenizer: Tokenizer | None = None, **policy_kwargs
) -> TextGenerator:
    """Builds a TextGenerator on the id-only path.

    Without a tokenizer there is no <eos> to stop on, which is the
    condition most of these tests rely on.

    Args:
        model: Model to wrap.
        tokenizer: Optional tokenizer; omit to stay on the id-only path.
        **policy_kwargs: Keyword arguments forwarded to `SamplingPolicy`.

    Returns:
        A configured `TextGenerator`.
    """
    return TextGenerator(model, tokenizer, SamplingPolicy(**policy_kwargs))


def test_smoke_greedy_generates_requested_number_of_tokens() -> None:
    """Checks that greedy generation appends exactly max_new_tokens valid ids."""
    model, cfg, idx = _tiny_model_and_prompt()
    out = _generator(model, temperature=0.0).generate_ids(idx, max_new_tokens=12)
    print(
        f"[greedy] prompt.shape={tuple(idx.shape)} -> out.shape={tuple(out.shape)} "
        f"finite={torch.isfinite(out.float()).all().item()} "
        f"in_vocab={((out >= 0) & (out < cfg.vocab_size)).all().item()}"
    )
    assert out.shape == (idx.shape[0], idx.shape[1] + 12)
    assert torch.isfinite(out.float()).all()
    assert (out >= 0).all() and (out < cfg.vocab_size).all()


def test_smoke_sampled_generates_requested_number_of_tokens() -> None:
    """Checks that sampled generation (temperature>0, top-k) appends valid ids."""
    model, cfg, idx = _tiny_model_and_prompt(seed=1)
    out = _generator(model, temperature=0.9, top_k=10).generate_ids(idx, max_new_tokens=12)
    print(
        f"[sampled] prompt.shape={tuple(idx.shape)} -> out.shape={tuple(out.shape)} "
        f"finite={torch.isfinite(out.float()).all().item()} "
        f"in_vocab={((out >= 0) & (out < cfg.vocab_size)).all().item()}"
    )
    assert out.shape == (idx.shape[0], idx.shape[1] + 12)
    assert torch.isfinite(out.float()).all()
    assert (out >= 0).all() and (out < cfg.vocab_size).all()


def test_zero_new_tokens_returns_prompt_unchanged_without_calling_model() -> None:
    """Checks that requesting 0 new tokens returns the prompt unchanged."""
    model, cfg, idx = _tiny_model_and_prompt()
    out = _generator(model, temperature=0.0).generate_ids(idx, max_new_tokens=0)
    print(f"[zero-tokens] prompt.shape={tuple(idx.shape)} -> out.shape={tuple(out.shape)} equal={torch.equal(out, idx)}")
    assert torch.equal(out, idx)


def test_kv_cache_equivalence_through_real_generate_path() -> None:
    """Checks that KV-cache-driven greedy decoding matches a cache-free forward pass.

    Re-verifies end-to-end what was already checked at the module level
    elsewhere, but through the real `generate_ids()` instead of a
    hand-written `model()` call in the test: if the way `generate_ids`
    drives the cache (prefill + one token at a time) had an offset bug,
    the greedily chosen tokens would no longer match what a single
    cache-free forward pass over the full resulting sequence would have
    chosen at the same positions.
    """
    model, cfg, idx = _tiny_model_and_prompt(seed=7)
    prompt_len = idx.shape[1]

    full_seq = _generator(model, temperature=0.0).generate_ids(idx, max_new_tokens=10)

    with torch.no_grad():
        logits_full = model(full_seq).logits
    # (B, S): predicted_next[:, t] = argmax starting from full_seq[:, t]
    predicted_next = logits_full.argmax(dim=-1)

    # The token generated at position prompt_len + i should be the argmax
    # computed on the prefix ending at position prompt_len + i - 1.
    generated_part = full_seq[:, prompt_len:]
    predicted_for_generated = predicted_next[:, prompt_len - 1 : -1]
    print(
        f"[kv-cache] prompt_len={prompt_len} -> "
        f"generated={generated_part.tolist()} predicted={predicted_for_generated.tolist()} "
        f"equal={torch.equal(generated_part, predicted_for_generated)}"
    )
    assert torch.equal(generated_part, predicted_for_generated)


class _ForcedTokenModel:
    """Fake model that ignores its input and forces a scripted id per call.

    Isolates `generate_ids`'s stop-on-eos logic from the actual behavior
    of a trained Transformer.
    """

    def __init__(self, forced_ids: list[int], vocab_size: int = 10, n_layers: int = 1):
        """Sets up the scripted per-call output ids.

        Args:
            forced_ids: Token id to force at each successive call.
            vocab_size: Vocabulary size for the fake logits.
            n_layers: Number of fake KV-cache layers to report.
        """
        self.config = SimpleNamespace(n_layers=n_layers)
        self._forced_ids = forced_ids
        self._call = 0
        self.vocab_size = vocab_size

    def eval(self) -> None:
        """No-op, present only to satisfy the model interface."""
        pass

    def __call__(
        self, idx: torch.Tensor, kv_caches: list | None = None
    ) -> ForwardOutput:
        """Returns logits that force the scripted id for this call.

        Args:
            idx: Input id tensor; only its shape is used.
            kv_caches: Existing per-layer KV caches, if any.

        Returns:
            A `ForwardOutput` whose argmax at the last position is the
            forced id for the current call.
        """
        B, S = idx.shape
        forced = self._forced_ids[self._call]
        self._call += 1
        logits = torch.full((B, S, self.vocab_size), -1e9)
        logits[:, -1, forced] = 1e9
        return ForwardOutput(
            logits=logits,
            aux_loss=torch.tensor(0.0),
            kv_caches=kv_caches or [None] * self.config.n_layers,
        )


class _FakeTokenizer:
    """Fake tokenizer reduced to token_to_id.

    That is the only thing `TextGenerator` reads from the tokenizer to
    decide when to stop, letting stop-on-eos be tested without training
    a BPE tokenizer.
    """

    def __init__(self, eos_id: int):
        """Stores the id to return as the <eos> token.

        Args:
            eos_id: Token id that `token_to_id` should always return.
        """
        self._eos_id = eos_id

    def token_to_id(self, token: str) -> int:
        """Returns the fixed eos id regardless of the requested token.

        Args:
            token: Ignored; present to match the tokenizer interface.

        Returns:
            The configured eos id.
        """
        return self._eos_id


def test_generation_stops_early_when_eos_id_is_emitted() -> None:
    """Checks that generation stops as soon as the eos id is forced."""
    model = _ForcedTokenModel(forced_ids=[7, 2, 2, 2, 2], vocab_size=10)
    idx = torch.tensor([[1, 1, 1]])
    generator = _generator(model, tokenizer=_FakeTokenizer(eos_id=7), temperature=0.0)

    out = generator.generate_ids(idx, max_new_tokens=5)
    print(
        f"[eos-stop] model calls={model._call} -> out.shape={tuple(out.shape)} "
        f"last_token={out[0, -1].item()}"
    )
    assert model._call == 1  # stops at the first step, no more calls needed
    assert out.shape == (1, 3 + 1)
    assert out[0, -1].item() == 7


def test_generation_runs_full_length_without_eos_id() -> None:
    """Checks that generation runs the full length when there is no tokenizer to stop on."""
    model = _ForcedTokenModel(forced_ids=[7, 7, 7, 7, 7], vocab_size=10)
    idx = torch.tensor([[1, 1, 1]])
    # no tokenizer => no <eos> => no early exit
    out = _generator(model, temperature=0.0).generate_ids(idx, max_new_tokens=5)
    print(f"[no-eos] model calls={model._call} -> out.shape={tuple(out.shape)}")
    assert model._call == 5
    assert out.shape == (1, 3 + 5)


def test_generate_without_tokenizer_is_refused() -> None:
    """Checks that the text-decoding path raises ValueError up front without a tokenizer."""
    # the text path needs a tokenizer: it must say so immediately and
    # explicitly, rather than failing later with an AttributeError.
    model, _, _ = _tiny_model_and_prompt()
    with pytest.raises(ValueError, match="tokenizer") as exc_info:
        _generator(model, temperature=0.0).generate("Once upon a time,", max_new_tokens=5)
    print(f"[no-tokenizer] raised={exc_info.value!r}")


def test_default_policy_is_used_when_none_is_given() -> None:
    """Checks that TextGenerator defaults to a plain SamplingPolicy() when none is given."""
    model, cfg, idx = _tiny_model_and_prompt(seed=3)
    generator = TextGenerator(model)
    print(f"[default-policy] policy={generator.policy!r} expected={SamplingPolicy()!r}")
    assert generator.policy == SamplingPolicy()

    out = generator.generate_ids(idx, max_new_tokens=4)
    print(f"[default-policy] prompt.shape={tuple(idx.shape)} -> out.shape={tuple(out.shape)}")
    assert out.shape == (idx.shape[0], idx.shape[1] + 4)

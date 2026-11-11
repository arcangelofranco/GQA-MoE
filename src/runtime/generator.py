import torch

from src.data.tokenizer import Tokenizer
from src.model.kv_cache import KVCache
from src.model.transformer import Transformer
from src.runtime.sampler import SamplingPolicy


class TextGenerator:
    """Autoregressive decoding for one model, tokenizer, and sampling policy.

    Owns the whole decoding path: prompt encoding, the incremental KV-cache
    loop, next-token sampling, the ``<eos>`` stop condition, and decoding back
    to text. Everything that stays fixed for a run (model, tokenizer, sampling
    policy, device) is stated once at construction and reused across
    :meth:`generate` calls.

    Two entry points are offered:

    - :meth:`generate_ids` works at the token-id level and can run without a
      tokenizer, never stopping early.
    - :meth:`generate` works on plain text and requires a tokenizer, stopping
      as soon as every sequence in the batch has emitted ``<eos>``.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer | None = None,
        policy: SamplingPolicy | None = None,
        device: str = "cpu",
    ):
        """Bind a model to the tokenizer, sampling policy, and device to decode with.

        Args:
            model: Transformer model to generate from.
            tokenizer: Tokenizer used to encode prompts, decode outputs, and
                resolve the ``<eos>`` id. May be ``None`` to use the id-level
                :meth:`generate_ids` path only, which never stops early.
            policy: How to pick each next token. Defaults to
                ``SamplingPolicy()``, i.e. plain temperature-1.0 sampling.
            device: Device to place encoded prompts on. The model is expected
                to already live on the same device.

        Raises:
            KeyError: If a tokenizer is given but has no ``<eos>`` token.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.policy = policy if policy is not None else SamplingPolicy()
        self.device = device
        self.eos_id = tokenizer.token_to_id("<eos>") if tokenizer is not None else None

    @torch.no_grad()
    def generate_ids(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively extend a prompt, decoding incrementally with a KV cache.

        Runs under ``torch.no_grad`` for efficiency. Fresh per-block KV caches
        are created for the run, then the loop feeds the model one step at a
        time: the newest sampled token becomes the next step's input, while the
        KV caches accumulate the full context so past tokens are never
        reprocessed. If an ``<eos>`` id is configured and every sequence in the
        batch emits it, generation stops early before exhausting
        ``max_new_tokens``.

        Args:
            idx: Prompt token ids of shape ``(B, S)``.
            max_new_tokens: Maximum number of new tokens to append. The actual
                output length may be shorter if the batch stops early on
                ``<eos>``.

        Returns:
            torch.Tensor: Token ids of shape ``(B, S + n_generated)``, the
            prompt followed by the generated continuation. Identical to ``idx``
            if no new token was produced.
        """
        self.model.eval()
        kv_caches = [KVCache() for _ in range(self.model.config.n_layers)]
        generated = []
        step_input = idx

        for _ in range(max_new_tokens):
            out = self.model(step_input, kv_caches=kv_caches)
            kv_caches = out.kv_caches
            next_id = self.policy.select(out.logits[:, -1, :])
            generated.append(next_id)
            if self.eos_id is not None and (next_id == self.eos_id).all():
                break
            step_input = next_id

        if not generated:
            return idx
        return torch.cat([idx] + generated, dim=1)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Generate a text continuation for a prompt.

        Encodes the prompt, delegates to :meth:`generate_ids` for the
        autoregressive loop, and decodes the resulting token ids back to a
        string.

        Args:
            prompt: Text prompt to continue.
            max_new_tokens: Maximum number of new tokens to append.

        Returns:
            str: The decoded prompt plus generated continuation.

        Raises:
            ValueError: If this generator was built without a tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError(
                "generate() needs a tokenizer: build TextGenerator with one, "
                "or use generate_ids() to work at the token-id level."
            )
        idx = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        out_ids = self.generate_ids(idx, max_new_tokens)
        return self.tokenizer.decode(out_ids[0].tolist())

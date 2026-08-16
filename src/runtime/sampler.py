from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamplingPolicy:
    """How the next token is chosen from a decoding step's logits.

    Groups the three sampling settings that always travel together, so a
    caller states them once instead of threading them through every call.

    Attributes:
        temperature: Softmax temperature. 0.0 means greedy (argmax)
            decoding, in which case top_k and top_p are ignored.
        top_k: If set, restrict sampling to the top_k most likely tokens.
        top_p: If set, restrict sampling to the smallest nucleus of tokens
            whose cumulative probability is >= top_p.
    """

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self):
        """Validate the policy.

        Raises:
            ValueError: If temperature is negative, top_k is set and < 1, or
                top_p is set and outside (0, 1].
        """
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature ({self.temperature}) must be >= 0.0 (0.0 = greedy decoding)."
            )
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k ({self.top_k}) must be >= 1 when set.")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p ({self.top_p}) must be in (0, 1] when set.")

    @property
    def is_greedy(self) -> bool:
        """Whether this policy decodes greedily, i.e. temperature is 0.0."""
        return self.temperature == 0.0

    def select(self, logits: torch.Tensor) -> torch.Tensor:
        """Choose the next token ids from a step's logits.

        Args:
            logits: Unnormalized log-probabilities of shape (..., vocab_size).

        Returns:
            Selected token ids of shape (..., 1).
        """
        if self.is_greedy:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / self.temperature

        if self.top_k is not None:
            logits = self._mask_below_top_k(logits)
        if self.top_p is not None:
            logits = self._mask_outside_nucleus(logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _mask_below_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask every token outside the top_k most likely ones.

        Args:
            logits: Temperature-scaled logits of shape (..., vocab_size).

        Returns:
            Logits with the excluded positions set to -inf.
        """
        # top_k may exceed the vocabulary: clamp it, torch.topk would raise.
        k = min(self.top_k, logits.size(-1))
        kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < kth_value, float("-inf"))

    def _mask_outside_nucleus(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask every token outside the smallest nucleus reaching top_p.

        Args:
            logits: Temperature-scaled logits of shape (..., vocab_size).

        Returns:
            Logits with the excluded positions set to -inf. The single most
            likely token is always kept, however small top_p is.
        """
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        remove_sorted = cumulative_probs > self.top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove_mask = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
        return logits.masked_fill(remove_mask, float("-inf"))

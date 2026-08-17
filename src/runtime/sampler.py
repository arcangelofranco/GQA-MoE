from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamplingPolicy:
    """How the next token is chosen from a decoding step's logits.

    Groups the three sampling settings that always travel together, so a
    caller states them once instead of threading them through every call. The
    selection order at decode time is: temperature scaling, top-k masking,
    then top-p (nucleus) masking, followed by a softmax and a multinomial draw.

    Attributes:
        temperature: Softmax temperature. ``0.0`` means greedy (argmax)
            decoding, in which case ``top_k`` and ``top_p`` are ignored.
        top_k: If set, restrict sampling to the ``top_k`` most likely tokens.
        top_p: If set, restrict sampling to the smallest nucleus of tokens
            whose cumulative probability is ``>= top_p``.
    """

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self):
        """Validate the policy.

        Enforces the invariants of the three settings so that invalid
        configurations fail fast at construction time rather than surfacing as
        obscure errors (or silently wrong distributions) during decoding.

        Raises:
            ValueError: If ``temperature`` is negative, ``top_k`` is set and
                ``< 1``, or ``top_p`` is set and outside ``(0, 1]``.
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
        """Whether this policy decodes greedily, i.e. temperature is 0.0.

        Returns:
            bool: ``True`` when ``temperature == 0.0``, in which case
            :meth:`select` returns the argmax token and ignores the top-k and
            top-p filters.
        """
        return self.temperature == 0.0

    def select(self, logits: torch.Tensor) -> torch.Tensor:
        """Choose the next token ids from a step's logits.

        Applies the configured sampling pipeline to the raw logits: greedy
        mode short-circuits to ``argmax``; otherwise the logits are scaled by
        the temperature, optionally truncated by the top-k and top-p filters,
        converted to a probability distribution, and drawn from once per row.
        The draw is agnostic to training/eval mode and never builds a graph,
        since sampling is not part of backprop.

        Args:
            logits: Unnormalized log-probabilities of shape
                ``(..., vocab_size)``.

        Returns:
            torch.Tensor: Selected token ids of shape ``(..., 1)``.
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

        Keeps only the ``top_k`` positions with the highest logits and sets
        every other position to ``-inf`` so that the subsequent softmax assigns
        them zero probability. The request is clamped to the vocabulary size,
        since ``torch.topk`` raises if ``k`` exceeds the last dimension.

        Args:
            logits: Temperature-scaled logits of shape ``(..., vocab_size)``.

        Returns:
            torch.Tensor: Logits with the excluded positions set to ``-inf``.
        """
        # top_k may exceed the vocabulary: clamp it, torch.topk would raise.
        k = min(self.top_k, logits.size(-1))
        kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < kth_value, float("-inf"))

    def _mask_outside_nucleus(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask every token outside the smallest nucleus reaching top_p.

        Sorts the logits descending, accumulates their softmax probabilities,
        and marks every position after the smallest prefix whose cumulative
        probability reaches ``top_p``. The single most likely token is always
        retained regardless of ``top_p``, guaranteeing a non-empty support even
        for very small nuclei. The mask is scattered back into the original
        index order before being applied.

        Args:
            logits: Temperature-scaled logits of shape ``(..., vocab_size)``.

        Returns:
            torch.Tensor: Logits with the excluded positions set to ``-inf``.
            The single most likely token is always kept, however small
            ``top_p`` is.
        """
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        remove_sorted = cumulative_probs > self.top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove_mask = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
        return logits.masked_fill(remove_mask, float("-inf"))

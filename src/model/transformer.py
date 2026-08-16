import math
from typing import NamedTuple

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.model.blocks.norm import RMSNorm
from src.model.blocks.rope import precompute_rope
from src.model.block import TransformerBlock
from src.model.kv_cache import KVCache


class ForwardOutput(NamedTuple):
    """Everything one Transformer forward pass produces.

    The three results serve different callers — generation reads the logits
    and the caches, training reads the logits and the aux loss — so they are
    named rather than positional: adding a field cannot silently shift what a
    caller reads. A NamedTuple rather than a dataclass because it stays a
    plain tuple at runtime, which is what torch's output handling expects.

    Attributes:
        logits: Next-token scores of shape (B, S, vocab_size).
        aux_loss: MoE load-balancing loss, summed over blocks and scaled by
            aux_loss_coeff exactly once. A 0.0 tensor outside training.
        kv_caches: One updated cache per block, to hand back to the next
            incremental call.
    """

    logits: torch.Tensor
    aux_loss: torch.Tensor
    kv_caches: list[KVCache | None]


class Transformer(nn.Module):
    """Decoder-only Transformer with GQA and MoE feed-forward blocks."""

    def __init__(self, config: ModelConfig):
        """Build the embedding, blocks, output head, and RoPE tables, then init weights.

        Args:
            config: Model configuration, forwarded to every sub-module.
        """
        super().__init__()

        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize Linear/Embedding weights with a normal distribution.

        Args:
            module: Sub-module being visited by `self.apply`.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        index: torch.Tensor,
        kv_caches: list[KVCache | None] | None = None,
    ) -> ForwardOutput:
        """Run a forward pass over the whole Transformer.

        Args:
            index: Input token ids of shape (B, S).
            kv_caches: One KV cache per block, or None during training/prefill
            without caching. If provided, must have one entry per block.

        Returns:
            A ForwardOutput carrying the logits, the scaled MoE auxiliary
            loss, and the updated per-block cache list.

        Raises:
            ValueError: If kv_caches is provided but its length doesn't match
            the number of blocks.
        """
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        elif len(kv_caches) != len(self.blocks):
            raise ValueError(
                f"kv_caches has {len(kv_caches)} entries, but the model has {len(self.blocks)} layers: one cache per block."
            )

        x = self.tok_embeddings(index)
        aux_loss_total = torch.tensor(0.0, device=index.device)
        new_kv_caches = []
        for block, cache in zip(self.blocks, kv_caches):
            x, aux_loss, cache = block(x, self.cos, self.sin, cache)
            aux_loss_total = aux_loss_total + aux_loss
            new_kv_caches.append(cache)
        aux_loss_total = aux_loss_total * self.config.aux_loss_coeff

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return ForwardOutput(logits, aux_loss_total, new_kv_caches)

    def num_params(self, non_embedding: bool = False) -> int:
        """Count the model's parameters.

        Args:
            non_embedding: If True, exclude the token embedding weights
            (useful when embeddings are tied with lm_head).

        Returns:
            Total number of parameters.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n
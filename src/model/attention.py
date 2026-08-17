import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.model.blocks.norm import RMSNorm
from src.model.blocks.rope import apply_rope
from src.model.kv_cache import KVCache

class GQAttention(nn.Module):
    """Grouped-Query Attention with QK-Norm, RoPE, and optional KV caching.

    Implements grouped-query attention (GQA), where a single key/value head is
    shared across a group of ``n_rep = n_heads // n_kv_heads`` query heads. This
    cuts the memory footprint of the KV cache and the KV compute cost by a
    factor of ``n_rep`` while retaining most of the quality of full MHA.

    Keys and queries are normalized with per-head RMSNorm (QK-Norm) before
    rotary position embeddings are applied; the values are left unnormalized.
    The module supports both the batched training path (causal masking, no
    cache) and incremental autoregressive decoding through an optional
    :class:`KVCache`.
    """

    def __init__(self, config: ModelConfig):
        """Build the Q/K/V/O projections and the QK-Norm layers.

        Args:
            config: Model configuration. Consumes ``hidden_dim``, ``n_heads``,
                ``n_kv_heads``, ``head_dim``, and ``rms_norm_eps``.

        Raises:
            AssertionError: If ``n_heads`` is not a multiple of ``n_kv_heads``,
                which is a structural requirement of grouped-query attention.
        """
        super().__init__()

        assert config.n_heads % config.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_heads // config.n_kv_heads

        self.wq = nn.Linear(config.hidden_dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.hidden_dim, bias=False)

        # QK-Norm: separate weights for Q and K, one per head_dim
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Run grouped-query self-attention over a (possibly incremental) sequence.

        Computes Q/K/V projections, applies QK-Norm and RoPE, appends to the
        KV cache when provided, then runs scaled dot-product attention. When a
        non-empty cache is present, the sequence is treated as a continuation
        at absolute positions ``[cache_len, cache_len + S)`` and a truncated
        causal mask lets each new token attend to itself and all cached tokens.
        Otherwise a full causal mask is applied natively by
        :func:`torch.nn.functional.scaled_dot_product_attention`.

        Args:
            x: Input tensor of shape ``(B, S, hidden_dim)``.
            cos: RoPE cosine table of shape ``(max_seq_len, head_dim)``.
            sin: RoPE sine table of shape ``(max_seq_len, head_dim)``.
            kv_cache: Optional :class:`KVCache`. When ``None`` (training or
                cache-free prefill) the position offset is zero and no
                append happens.

        Returns:
            tuple[torch.Tensor, KVCache | None]: A pair ``(output, kv_cache)``
            where ``output`` has shape ``(B, S, hidden_dim)`` and ``kv_cache``
            is the same object passed in (or ``None``), updated in place with
            the new keys and values.

        Raises:
            RuntimeError: If ``max_seq_len`` precomputed for the RoPE tables is
                shorter than ``cache_len + S``.
        """
        B, S, _ = x.shape

        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm before RoPE. V receives neither.
        q, k = self.q_norm(q), self.k_norm(k)

        # Absolute position offset = how many tokens are already in the cache.
        # With kv_cache = None (training) or an empty cache, the offset is 0.
        cache_len = 0 if kv_cache is None else kv_cache.length

        cos_slice = cos[cache_len: cache_len + S]
        sin_slice = sin[cache_len: cache_len + S]
        q = apply_rope(q, cos_slice, sin_slice)
        k = apply_rope(k, cos_slice, sin_slice)

        if kv_cache is not None:
            k, v = kv_cache.append(k, v)

        T = k.shape[2]  # total key length: cache + new tokens

        k_rep = k.repeat_interleave(self.n_rep, dim=1)
        v_rep = v.repeat_interleave(self.n_rep, dim=1)

        if S == T:
            # No prior cache (training or prefill with an empty cache):
            # standard causal mask, handled natively and efficiently by SDPA.
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        else:
            # Incremental decoding with a non-empty cache.
            mask = torch.ones(S, T, dtype=torch.bool, device=x.device).tril(diagonal=T - S)
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(attn_out), kv_cache
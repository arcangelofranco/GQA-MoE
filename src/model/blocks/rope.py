import torch

def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin rotation tables for RoPE.

    Builds the per-position, per-channel rotation angles for rotary position
    embeddings. The inverse frequencies follow a geometric progression scaled
    by ``theta`` (matching the original RoPE paper and the Llama family), and
    the same frequency vector is duplicated across the two halves of the head
    dimension so that the rotation can be applied via the complex-number
    formulation ``(x * cos + rotate_half(x) * sin)``.

    Args:
        head_dim: Dimension of each attention head. Must be even, since the
            head is split into two equal halves for the rotation.
        max_seq_len: Maximum sequence length to precompute positions for.
            Positions ``0`` through ``max_seq_len - 1`` are covered.
        theta: Base of the inverse-frequency geometric progression. Larger
            values yield longer effective context ranges.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A pair ``(cos, sin)``, each of
        shape ``(max_seq_len, head_dim)``, that can be passed directly to
        :func:`apply_rope`. Both tensors are plain CPU tensors and are shared
        (not recreated) across all attention heads and batch items.

    Raises:
        ValueError: If ``head_dim`` is odd.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}.")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq) # [S, D/2]
    emb = torch.cat((freqs, freqs), dim=-1) # [S, D]
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by swapping and negating its two halves.

    Implements the ``rotate_half`` primitive at the heart of RoPE: given the
    channel dimension split into two equal halves ``(x1, x2)``, it returns
    ``(-x2, x1)``. When combined with the cos/sin tables in
    :func:`apply_rope`, this produces the geometric rotation of the 2D
    position-dependent subspace.

    Args:
        x: Input tensor of shape ``(..., D)`` with ``D`` even.

    Returns:
        torch.Tensor: A tensor of the same shape as ``x``, with the two halves
        swapped and the new first half negated: ``(x1, x2) -> (-x2, x1)``.

    Raises:
        ValueError: If the last dimension of ``x`` is odd.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Expected an even last dimension, got {x.shape[-1]}.")
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to queries or keys.

    Rotates every head's ``(S, D)`` slice position-wise using the
    precomputed cos/sin tables, injecting relative-position information into
    the attention scores. The broadcast reshapes the tables to ``(1, 1, S, D)``
    so the same rotation is applied to every head and batch item.

    Args:
        x: Input tensor of shape ``(B, n_heads, S, D)`` containing queries or
            keys.
        cos: Cosine table of shape ``(S, D)`` as returned by
            :func:`precompute_rope`.
        sin: Sine table of shape ``(S, D)`` as returned by
            :func:`precompute_rope`.

    Returns:
        torch.Tensor: The rotated tensor, of the same shape as ``x``.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin
        
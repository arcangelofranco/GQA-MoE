import torch

def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin rotation tables for RoPE.

    Args:
        head_dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute positions for.
        theta: Base for the inverse frequency geometric progression.

    Returns:
        A tuple (cos, sin), each of shape (max_seq_len, head_dim).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq) # [S, D/2]
    emb = torch.cat((freqs, freqs), dim=-1) # [S, D]
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by swapping and negating its two halves.

    Args:
        x: Input tensor of shape (..., D).

    Returns:
        Tensor of the same shape, with (x1, x2) mapped to (-x2, x1).
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to queries or keys.

    Args:
        x: Input tensor of shape (B, n_heads, S, D).
        cos: Cosine table of shape (S, D), as returned by precompute_rope.
        sin: Sine table of shape (S, D), as returned by precompute_rope.

    Returns:
        Rotated tensor of the same shape as x.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin
        
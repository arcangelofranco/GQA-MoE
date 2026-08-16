import torch


class KVCache:
    """Per-layer cache of keys/values for incremental decoding.

    Stores tensors of shape (B, n_kv_heads, S, head_dim), growing along the
    sequence dimension (dim=2) as new tokens are appended.
    """

    def __init__(self):
        """Initialize an empty cache."""
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Return the number of cached tokens (0 if the cache is empty)."""
        return 0 if self._k is None else self._k.shape[2]

    @property
    def k(self) -> torch.Tensor | None:
        """Return the cached keys, or None if the cache is empty."""
        return self._k

    @property
    def v(self) -> torch.Tensor | None:
        """Return the cached values, or None if the cache is empty."""
        return self._v

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values to the cache and return the full cache.

        Args:
            k: New keys of shape (B, n_kv_heads, S_new, head_dim).
            v: New values of shape (B, n_kv_heads, S_new, head_dim).

        Returns:
            A tuple (k, v) with the full cached keys/values, including the
            newly appended ones.
        """
        if self._k is None:
            self._k, self._v = k, v
        else:
            self._k = torch.cat([self._k, k], dim=2)
            self._v = torch.cat([self._v, v], dim=2)
        return self._k, self._v
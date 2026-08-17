import torch


class KVCache:
    """Per-layer cache of keys/values for incremental decoding.

    Stores the key and value tensors for a single attention layer so that
    autoregressive decoding only recomputes the new token instead of re-encoding
    the whole prefix at every step. Tensors have shape
    ``(B, n_kv_heads, S, head_dim)`` and grow along the sequence dimension
    (``dim=2``) as new tokens are appended.

    The cache is lazy: it stays empty (``None``) until the first append, which
    avoids allocating memory for batch/layer combinations that are never used
    (e.g. when the block runs without caching during training).
    """

    def __init__(self):
        """Initialize an empty cache.

        Both internal tensors are ``None`` until the first :meth:`append`
        call; the effective sequence length reported by :attr:`length` is zero.
        """
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Return the number of cached tokens (0 if the cache is empty).

        Returns:
            int: The current sequence length held in the cache.
        """
        return 0 if self._k is None else self._k.shape[2]

    @property
    def k(self) -> torch.Tensor | None:
        """Return the cached keys, or None if the cache is empty.

        Returns:
            torch.Tensor | None: The full key tensor, or ``None`` when no keys
            have been appended yet.
        """
        return self._k

    @property
    def v(self) -> torch.Tensor | None:
        """Return the cached values, or None if the cache is empty.

        Returns:
            torch.Tensor | None: The full value tensor, or ``None`` when no
            values have been appended yet.
        """
        return self._v

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values to the cache and return the full cache.

        On the first call the cache is seeded with the incoming tensors; on
        subsequent calls the new keys/values are concatenated onto the sequence
        dimension of the existing cache. The full (concatenated) tensors are
        returned to the caller, which reads them back from here anyway, so the
        caller can pass the returned references directly to the attention
        computation without re-reading the properties.

        Args:
            k: New keys of shape ``(B, n_kv_heads, S_new, head_dim)``.
            v: New values of shape ``(B, n_kv_heads, S_new, head_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair ``(k, v)`` with the full
            cached keys/values, including the newly appended tokens.
        """
        if self._k is None:
            self._k, self._v = k, v
        else:
            self._k = torch.cat([self._k, k], dim=2)
            self._v = torch.cat([self._v, v], dim=2)
        return self._k, self._v
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm: ``x / sqrt(mean(x^2, dim=-1) + eps) * weight``.

    A cheaper alternative to standard LayerNorm that normalizes using the RMS
    of the features instead of the full mean/variance, saving the mean
    subtraction step. It is the normalization of choice in the Llama family and
    in most modern MoE transformers because it delivers comparable quality at a
    lower computational and memory cost. Unlike LayerNorm there is no bias
    term, only a learnable per-feature scale ``weight``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the learnable scale.

        Args:
            dim: Feature dimension to normalize over. The scale parameter has
                this shape.
            eps: Small constant added to the variance before the inverse
                square root, guarding against division by zero and reducing
                numerical instability at small magnitudes.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization, computed in float32 for stability.

        Computes the RMS statistic over the last dimension, normalizes the
        activations, and rescales them by the learnable per-feature weight. The
        whole computation is performed in float32 regardless of the input
        dtype, protecting against overflow/catastrophic cancellation in
        low-precision training (e.g. bf16), and the result is cast back to the
        input dtype before returning.

        Args:
            x: Input tensor of shape ``(..., dim)``. Any number of leading
                dimensions is supported.

        Returns:
            torch.Tensor: Normalized tensor with the same shape and dtype as
            ``x``.
        """
        x_f = x.to(torch.float32)
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        return (self.weight * x_norm).to(x.dtype)
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm: x / sqrt(mean(x^2, dim=-1) + eps) * weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the learnable scale.

        Args:
            dim: Feature dimension to normalize over.
            eps: Small constant added to the variance for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization, computed in float32 for stability.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of the same shape and dtype as x.
        """
        x_f = x.to(torch.float32)
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        return (self.weight * x_norm).to(x.dtype)
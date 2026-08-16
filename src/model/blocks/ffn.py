import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated feed-forward block: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, dim: int, intermediate_size: int):
        """Initialize the SwiGLU projections.

        Args:
            dim: Input and output feature dimension.
            intermediate_size: Hidden dimension of the gate/up projections.
        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU transformation.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Output tensor of shape (..., dim).
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
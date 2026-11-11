import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated feed-forward block: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    A SwiGLU (Swish-gated Linear Unit) feed-forward network used as the
    per-token transformation inside transformer layers. Unlike a plain two-layer
    MLP, it gates the ``up_proj`` expansion through the elementwise product
    with the Swish-activated ``gate_proj`` output, which empirically improves
    quality at the same parameter count. This implementation matches the Llama
    family design and omits biases on all three projections.
    """

    def __init__(self, dim: int, intermediate_size: int):
        """Initialize the SwiGLU projections.

        Args:
            dim: Input and output feature dimension.
            intermediate_size: Hidden dimension of the gate/up projections
                (typically ``~2.67 * dim`` in Llama-style models). The down
                projection maps back from this size to ``dim``.
        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU transformation.

        Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))`` in a single
        call. The gating term ``silu(gate_proj(x))`` modulates each hidden
        activation of ``up_proj(x)`` elementwise, allowing the network to
        selectively pass or suppress information at every position.

        Args:
            x: Input tensor of shape ``(..., dim)``. Any number of leading
                dimensions is supported.

        Returns:
            torch.Tensor: Output tensor of shape ``(..., dim)``.
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
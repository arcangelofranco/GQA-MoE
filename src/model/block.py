import torch
import torch.nn as nn

from src.config import ModelConfig
from src.model.blocks.norm import RMSNorm
from src.model.attention import GQAttention
from src.model.kv_cache import KVCache
from src.model.moe import MoELayer

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: GQA attention + MoE feed-forward, each with a residual connection."""

    def __init__(self, config: ModelConfig):
        """Build the attention and MoE sub-layers with their pre-norms.

        Args:
            config: Model configuration, forwarded to GQAttention and MoELayer.
        """
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attention = GQAttention(config)
        self.moe_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.moe = MoELayer(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, KVCache | None]:
        """Run one Transformer block over the input.

        Args:
            x: Input tensor of shape (B, S, hidden_dim).
            cos: RoPE cosine table, forwarded to the attention layer.
            sin: RoPE sine table, forwarded to the attention layer.
            kv_cache: This block's own KV cache (each layer has an
                independent one), or None during training.

        Returns:
            A tuple (output, aux_loss, kv_cache): output has shape
            (B, S, hidden_dim); aux_loss is the MoE load-balancing loss;
            kv_cache is the same object passed in (or None), updated in place.
        """
        attention_out, kv_cache = self.attention(self.attention_norm(x), cos, sin, kv_cache)
        h = x + attention_out
        moe_out, aux_loss = self.moe(self.moe_norm(h))
        return h + moe_out, aux_loss, kv_cache
import torch
import torch.nn as nn

from src.config import ModelConfig
from src.model.blocks.norm import RMSNorm
from src.model.attention import GQAttention
from src.model.kv_cache import KVCache
from src.model.moe import MoELayer

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: GQA attention + MoE feed-forward, each with a residual connection.

    A single transformer layer following the pre-normalization (pre-norm)
    design used in the Llama family: the input is normalized before each
    sub-layer, and the sub-layer output is added back to the input via a
    residual connection. The two sub-layers are grouped-query self-attention
    and a mixture-of-experts feed-forward module.
    """

    def __init__(self, config: ModelConfig):
        """Build the attention and MoE sub-layers with their pre-norms.

        Args:
            config: Model configuration, forwarded unchanged to
                :class:`GQAttention` and :class:`MoELayer`.
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

        Applies the pre-norm attention sub-layer, adds its output to the
        residual stream, then applies the pre-norm MoE sub-layer and adds its
        output as well. The MoE's load-balancing auxiliary loss is forwarded to
        the caller so it can be accumulated into the training loss.

        Args:
            x: Input tensor of shape ``(B, S, hidden_dim)``.
            cos: RoPE cosine table, forwarded unchanged to the attention layer.
            sin: RoPE sine table, forwarded unchanged to the attention layer.
            kv_cache: This block's own KV cache. Each layer has an independent
                cache, so callers must keep one per block; ``None`` during
                training or cache-free inference.

        Returns:
            tuple[torch.Tensor, torch.Tensor, KVCache | None]: A tuple
            ``(output, aux_loss, kv_cache)`` where ``output`` has shape
            ``(B, S, hidden_dim)``, ``aux_loss`` is the MoE load-balancing
            loss (a scalar tensor), and ``kv_cache`` is the same object passed
            in (or ``None``), updated in place.
        """
        attention_out, kv_cache = self.attention(self.attention_norm(x), cos, sin, kv_cache)
        h = x + attention_out
        moe_out, aux_loss = self.moe(self.moe_norm(h))
        return h + moe_out, aux_loss, kv_cache
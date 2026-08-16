import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.model.blocks.ffn import SwiGLU

class MoELayer(nn.Module):
    """Sparse mixture-of-experts feed-forward layer with a shared expert.

    Each token is routed to its top_k highest-scoring experts (out of
    n_experts), plus a shared expert that always processes every token.
    """

    def __init__(self, config: ModelConfig):
        """Build the router, the routed experts, and the shared expert.

        Args:
            config: Model configuration; uses hidden_dim, n_experts, top_k,
                expert_intermediate, and shared_intermediate.
        """
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k

        self.router = nn.Linear(config.hidden_dim, config.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(config.hidden_dim, config.expert_intermediate) for _ in range(config.n_experts)]
        )

        self.shared_expert = SwiGLU(config.hidden_dim, config.shared_intermediate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts and combine with the shared expert's output.

        Args:
            x: Input tensor of shape (B, S, hidden_dim).

        Returns:
            A tuple (output, aux_loss): output has shape (B, S, hidden_dim);
            aux_loss is the load-balancing loss (0.0 tensor in eval mode).
        """
        B, S, H = x.shape
        x_flat = x.reshape(-1, H) # (N, H), N = B*S

        router_logits = self.router(x_flat) # (N, n_experts)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        top_weights, top_index = router_probs.topk(self.top_k, dim=-1) # (N, top_k)
        top_weights = (top_weights / top_weights.sum(-1, keepdim=True)).type_as(x_flat)

        routed_sum = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            
            sel = top_index == i
            token_index = sel.any(dim=-1).nonzero(as_tuple=True)[0]

            if token_index.numel() == 0:
                continue

            weight = (top_weights * sel).sum(-1, keepdim=True)[token_index]
            contribution = weight * expert(x_flat[token_index])

            routed_sum = routed_sum.index_add(0, token_index, contribution)

        if self.training:
            aux_loss = self._load_balancing_loss(router_probs, top_index, x_flat.device)
        else:
            aux_loss = torch.tensor(0.0, device=x_flat.device)

        output = routed_sum + self.shared_expert(x_flat)
        return output.view(B, S, H), aux_loss

    def _load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        top_index: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the Switch Transformer-style auxiliary load-balancing loss.

        Args:
            router_probs: Softmax router probabilities of shape (N, n_experts).
            top_index: Indices of the top_k selected experts, shape (N, top_k).
            device: Device to allocate intermediate tensors on.

        Returns:
            Scalar auxiliary loss tensor.
        """
        with torch.no_grad():
            tokens_per_expert = torch.zeros(self.n_experts, device=device)
            tokens_per_expert.scatter_add_(
                0, top_index.reshape(-1),
                torch.ones_like(top_index.reshape(-1), dtype=torch.float32)
            )
            frac_tokens = tokens_per_expert / tokens_per_expert.sum()

        frac_probs = router_probs.mean(dim=0)
        return self.n_experts * (frac_tokens * frac_probs).sum()
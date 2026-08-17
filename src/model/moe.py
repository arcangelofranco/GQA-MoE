import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.model.blocks.ffn import SwiGLU

class MoELayer(nn.Module):
    """Sparse mixture-of-experts feed-forward layer with a shared expert.

    Implements a sparse MoE following the Switch/DeepSeek-style design: a small
    learned router scores every token against all ``n_experts``, and each token
    is dispatched to its ``top_k`` highest-scoring experts only. A dedicated
    shared expert additionally processes every token unconditionally, which
    acts as a dense "anchor" pathway that preserves stable gradients and
    prevents the router from over-specializing.

    The per-token expert outputs are weighted by the normalized top-k routing
    probabilities, so the module behaves like a mixture while only paying for
    ``top_k + 1`` experts per token.
    """

    def __init__(self, config: ModelConfig):
        """Build the router, the routed experts, and the shared expert.

        Args:
            config: Model configuration. Consumes ``hidden_dim``,
                ``n_experts``, ``top_k``, ``expert_intermediate``, and
                ``shared_intermediate``.
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

        Flattens the sequence into per-token vectors, computes softmax routing
        probabilities, selects the ``top_k`` experts per token, renormalizes
        the selected weights, and accumulates each expert's gated contribution
        into the token's output. The shared expert's output is then added to
        every token. The load-balancing auxiliary loss is only computed and
        returned during training; in eval mode it is a constant zero so that
        inference returns a valid but inert scalar.

        Args:
            x: Input tensor of shape ``(B, S, hidden_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair ``(output, aux_loss)``
            where ``output`` has shape ``(B, S, hidden_dim)`` and ``aux_loss``
            is the load-balancing loss (a scalar tensor, ``0.0`` in eval mode).
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

        Penalizes routing skew: if the fraction of tokens dispatched to an
        expert (measured from the actual discrete assignments) diverges from
        the fraction of routing probability mass that expert receives, the loss
        grows. Minimizing it encourages uniform expert utilization and prevents
        the router from collapsing onto a few experts.

        The token-count term is computed under ``no_grad`` because it depends
        only on the discrete assignments, which are non-differentiable; only
        the probability term contributes to the gradient.

        Args:
            router_probs: Softmax router probabilities of shape
                ``(N, n_experts)``, one distribution per token.
            top_index: Indices of the ``top_k`` selected experts, shape
                ``(N, top_k)``.
            device: Device to allocate the intermediate count tensors on.

        Returns:
            torch.Tensor: A scalar auxiliary loss tensor (zero-dim).
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
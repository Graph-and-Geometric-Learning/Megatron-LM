# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz Router for hyperbolic MoE (HELM-MiCE).
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz Router for Mixture of Curvature Experts.

Routes tokens to experts based on space-like dimensions in hyperbolic space.
Key differences from standard router:
1. Gate operates on space-like dimensions only (removes time coordinate)
2. Supports learnable bias for load balancing
3. Optional curvature-aware scoring

Based on:
    - HELM-MiCE: Hyperbolic LLM with Mixture of Curvature Experts
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...manifolds import Lorentz


class LorentzRouter(nn.Module):
    """
    Router for Lorentz MoE (Mixture of Curvature Experts).

    Computes routing probabilities based on space-like dimensions.
    Supports top-k routing with optional load balancing.

    Args:
        manifold: Lorentz manifold instance
        hidden_size: Model hidden size (space-like dimension)
        num_experts: Number of routed experts
        topk: Number of experts to route each token to
        score_func: Scoring function ('softmax' or 'sigmoid')
        bias_update_speed: Speed of load balancing bias update
        use_bias: Whether to use learnable bias
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        num_experts: int,
        topk: int = 2,
        score_func: str = 'softmax',
        bias_update_speed: float = 0.005,
        use_bias: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.topk = topk
        self.score_func = score_func
        self.bias_update_speed = bias_update_speed

        # Gate weight: projects space-like dims to expert scores
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # Learnable bias for load balancing
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.register_buffer('bias', torch.zeros(num_experts))

        self._logged_first_call = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor, shape (num_tokens, hidden_size+1)
                           Full Lorentz vectors with time coordinate
            padding_mask: Optional mask for padding tokens

        Returns:
            weights: Routing weights, shape (num_tokens, topk)
            indices: Expert indices, shape (num_tokens, topk)
            scores: Raw scores for all experts, shape (num_tokens, num_experts)
        """
        # Log first call
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzRouter forward (experts={self.num_experts}, topk={self.topk}, c={self.manifold.c.item():.4f})")
            self._logged_first_call = True

        # Extract space-like dimensions (remove time coordinate)
        x_space = hidden_states[..., 1:]  # (num_tokens, hidden_size)

        # Compute routing scores
        # scores[i, j] = <x_space[i], weight[j]>
        scores = F.linear(x_space.float(), self.weight.float())  # (num_tokens, num_experts)

        # Apply scoring function
        if self.score_func == 'softmax':
            scores = F.softmax(scores, dim=-1)
        elif self.score_func == 'sigmoid':
            scores = torch.sigmoid(scores)

        # Store original scores for auxiliary loss
        original_scores = scores.clone()

        # Add bias for load balancing
        scores = scores + self.bias

        # Apply padding mask if provided
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))

        # Top-k selection
        topk_scores, indices = torch.topk(scores, self.topk, dim=-1)

        # Get weights from original scores (before bias)
        weights = original_scores.gather(dim=-1, index=indices)

        # Normalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return weights.type_as(hidden_states), indices, original_scores

    @torch.no_grad()
    def update_bias(self, indices: torch.Tensor):
        """
        Update bias for load balancing.

        Called during training to balance expert utilization.

        Args:
            indices: Expert indices from routing, shape (num_tokens, topk)
        """
        # Count expert utilization
        flat_indices = indices.flatten()
        counts = torch.bincount(flat_indices, minlength=self.num_experts).float()

        # Compute mean utilization
        mean_count = counts.mean()

        # Update bias: increase for underutilized, decrease for overutilized
        self.bias.add_(self.bias_update_speed * (mean_count - counts))


class LorentzTopKRouter(LorentzRouter):
    """
    Top-K router compatible with Megatron's MoE infrastructure.

    Provides the same interface as Megatron's TopKRouter but operates
    in Lorentz space.
    """

    def __init__(
        self,
        manifold: Lorentz,
        config,
        **kwargs,
    ):
        """
        Initialize from Megatron config.

        Args:
            manifold: Lorentz manifold
            config: TransformerConfig with MoE settings
        """
        super().__init__(
            manifold=manifold,
            hidden_size=config.hidden_size,
            num_experts=config.num_moe_experts,
            topk=config.moe_router_topk,
            score_func='softmax',
            bias_update_speed=0.005,
            use_bias=True,
        )
        self.config = config

    def routing(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Megatron-compatible routing interface.

        Args:
            logits: Router logits (not used, we compute from input)

        Returns:
            probs: Routing probabilities
            routing_map: Binary routing map
        """
        # Note: This method expects pre-computed logits
        # In practice, use forward() which computes from hidden states
        scores = F.softmax(logits, dim=-1)
        topk_scores, indices = torch.topk(scores, self.topk, dim=-1)

        # Create routing map
        routing_map = torch.zeros_like(scores)
        routing_map.scatter_(dim=-1, index=indices, value=1.0)

        return scores, routing_map


class LorentzAuxLossRouter(LorentzRouter):
    """
    Lorentz router with auxiliary loss for load balancing.

    Uses the standard MoE auxiliary loss instead of bias updates.
    More compatible with Megatron's distributed training.
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        num_experts: int,
        topk: int = 2,
        aux_loss_coeff: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            manifold=manifold,
            hidden_size=hidden_size,
            num_experts=num_experts,
            topk=topk,
            use_bias=False,  # Use aux loss instead
            **kwargs,
        )
        self.aux_loss_coeff = aux_loss_coeff

    def compute_aux_loss(
        self,
        scores: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.

        Args:
            scores: Router scores, shape (num_tokens, num_experts)
            indices: Selected expert indices, shape (num_tokens, topk)

        Returns:
            Auxiliary loss scalar
        """
        num_tokens = scores.shape[0]

        # Fraction of tokens routed to each expert
        # f_i = (1/N) * sum_j 1[j routes to i]
        routing_counts = torch.zeros(self.num_experts, device=scores.device)
        for k in range(self.topk):
            routing_counts.scatter_add_(
                dim=0,
                index=indices[:, k],
                src=torch.ones(num_tokens, device=scores.device),
            )
        f = routing_counts / (num_tokens * self.topk)

        # Mean probability for each expert
        # P_i = (1/N) * sum_j p_ij
        P = scores.mean(dim=0)

        # Auxiliary loss: encourage uniform distribution
        # L_aux = num_experts * sum_i f_i * P_i
        aux_loss = self.num_experts * (f * P).sum()

        return self.aux_loss_coeff * aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with auxiliary loss.

        Returns:
            weights: Routing weights
            indices: Expert indices
            scores: Raw scores
            aux_loss: Auxiliary loss for load balancing
        """
        weights, indices, scores = super().forward(hidden_states, padding_mask)
        aux_loss = self.compute_aux_loss(scores, indices)
        return weights, indices, scores, aux_loss

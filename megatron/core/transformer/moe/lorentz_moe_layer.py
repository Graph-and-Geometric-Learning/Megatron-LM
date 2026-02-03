# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz MoE Layer for hyperbolic transformers (HELM-MiCE).
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz Mixture of Experts Layer.

Combines Lorentz router, curvature experts, and shared experts
into a complete MoE layer for hyperbolic transformers.

Architecture:
    input -> router -> dispatch to experts -> combine -> output
           -> shared expert ---------------------^

Key features:
1. Variable curvature per routed expert
2. Optional shared expert (always active)
3. Lorentz residual for combining outputs
4. Load balancing via auxiliary loss or bias updates

Based on:
    - HELM-MiCE: Hyperbolic LLM with Mixture of Curvature Experts
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn

from ...manifolds import Lorentz
from ..lorentz_residual import LorentzResidual
from .lorentz_router import LorentzRouter, LorentzAuxLossRouter
from .lorentz_experts import (
    LorentzExpertGroup,
    LorentzGroupedExperts,
    LorentzSharedExpert,
)


@dataclass
class LorentzMoEConfig:
    """Configuration for Lorentz MoE layer."""

    # Expert configuration
    num_routed_experts: int = 8
    num_shared_experts: int = 1
    num_activated_experts: int = 2  # Top-k
    ffn_hidden_size: int = 3072
    expert_ffn_hidden_size: Optional[int] = None  # If None, use ffn_hidden_size // num_routed

    # Curvature configuration
    curvature_range: Tuple[float, float] = (0.1, 2.0)
    learnable_curvature: bool = True

    # Router configuration
    score_func: str = 'softmax'
    use_aux_loss: bool = True
    aux_loss_coeff: float = 0.01
    bias_update_speed: float = 0.005

    # Load balancing
    use_load_balancing: bool = True


class LorentzMoE(nn.Module):
    """
    Lorentz Mixture of Experts layer.

    Combines:
    - Lorentz router for token-to-expert assignment
    - Multiple curvature experts for diverse representations
    - Optional shared expert for baseline computation
    - Lorentz residual for combining outputs

    Args:
        manifold: Lorentz manifold for the layer
        hidden_size: Model hidden size (space-like dimension)
        config: LorentzMoEConfig with expert settings
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        config: LorentzMoEConfig,
    ):
        super().__init__()
        self.manifold = manifold
        self.hidden_size = hidden_size
        self.config = config

        # Compute expert FFN size
        if config.expert_ffn_hidden_size is not None:
            expert_ffn_size = config.expert_ffn_hidden_size
        else:
            expert_ffn_size = config.ffn_hidden_size // config.num_routed_experts

        # Router
        if config.use_aux_loss:
            self.router = LorentzAuxLossRouter(
                manifold=manifold,
                hidden_size=hidden_size,
                num_experts=config.num_routed_experts,
                topk=config.num_activated_experts,
                score_func=config.score_func,
                aux_loss_coeff=config.aux_loss_coeff,
            )
        else:
            self.router = LorentzRouter(
                manifold=manifold,
                hidden_size=hidden_size,
                num_experts=config.num_routed_experts,
                topk=config.num_activated_experts,
                score_func=config.score_func,
                bias_update_speed=config.bias_update_speed,
            )

        # Routed experts with different curvatures
        self.experts = LorentzGroupedExperts(
            input_manifold=manifold,
            hidden_size=hidden_size,
            ffn_hidden_size=expert_ffn_size,
            num_experts=config.num_routed_experts,
            curvature_range=config.curvature_range,
            learnable_curvature=config.learnable_curvature,
        )

        # Shared experts (if any)
        if config.num_shared_experts > 0:
            shared_ffn_size = config.ffn_hidden_size  # Shared gets full size
            self.shared_experts = LorentzSharedExpert(
                manifold=manifold,
                hidden_size=hidden_size,
                ffn_hidden_size=shared_ffn_size,
            )
        else:
            self.shared_experts = None

        # For combining routed expert outputs with input
        self.combine_residual = LorentzResidual(manifold)

        self._logged_first_call = False

    def _permute_tokens(
        self,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permute tokens based on routing indices.

        Organizes tokens so all tokens for expert i are contiguous.

        Args:
            hidden_states: Input tensor, shape (num_tokens, hidden+1)
            indices: Expert indices, shape (num_tokens, topk)
            weights: Routing weights, shape (num_tokens, topk)

        Returns:
            permuted_states: Permuted tokens
            tokens_per_expert: Count of tokens per expert
            permuted_weights: Permuted routing weights
        """
        num_tokens = hidden_states.shape[0]
        topk = indices.shape[1]
        num_experts = self.config.num_routed_experts

        # Flatten indices and weights for easier processing
        flat_indices = indices.flatten()  # (num_tokens * topk,)
        flat_weights = weights.flatten()  # (num_tokens * topk,)

        # Expand hidden states for each topk selection
        expanded_states = hidden_states.unsqueeze(1).expand(-1, topk, -1)
        flat_states = expanded_states.reshape(-1, hidden_states.shape[-1])

        # Sort by expert index
        sorted_indices, sort_order = flat_indices.sort()
        permuted_states = flat_states[sort_order]
        permuted_weights = flat_weights[sort_order]

        # Count tokens per expert
        tokens_per_expert = torch.bincount(
            sorted_indices,
            minlength=num_experts
        )

        return permuted_states, tokens_per_expert, permuted_weights

    def _unpermute_tokens(
        self,
        permuted_output: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Unpermute and combine expert outputs.

        Args:
            permuted_output: Output from experts, shape (total_tokens, hidden+1)
            indices: Original routing indices
            weights: Original routing weights
            original_shape: Original input shape

        Returns:
            Combined output, shape (num_tokens, hidden+1)
        """
        num_tokens = indices.shape[0]
        topk = indices.shape[1]
        hidden_dim = permuted_output.shape[-1]

        # Get sort order (same as in permute)
        flat_indices = indices.flatten()
        _, sort_order = flat_indices.sort()

        # Inverse sort order
        inverse_order = torch.empty_like(sort_order)
        inverse_order[sort_order] = torch.arange(len(sort_order), device=sort_order.device)

        # Unpermute
        unpermuted_output = permuted_output[inverse_order]

        # Reshape to (num_tokens, topk, hidden)
        unpermuted_output = unpermuted_output.view(num_tokens, topk, hidden_dim)

        # Weight by routing probabilities and sum
        weighted_output = unpermuted_output * weights.unsqueeze(-1)
        output = weighted_output.sum(dim=1)

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor, shape (seq, batch, hidden+1) or (tokens, hidden+1)

        Returns:
            output: MoE output, same shape as input
            aux_loss: Optional auxiliary loss for load balancing
            expert_indices: Optional expert assignments for analysis
        """
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzMoE forward (experts={self.config.num_routed_experts}, topk={self.config.num_activated_experts})")
            self._logged_first_call = True

        # Handle different input shapes
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            # (seq, batch, hidden) -> (seq*batch, hidden)
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        num_tokens = hidden_states.shape[0]

        # Route tokens to experts
        if self.config.use_aux_loss:
            weights, indices, scores, aux_loss = self.router(hidden_states)
        else:
            weights, indices, scores = self.router(hidden_states)
            aux_loss = None

            # Update bias for load balancing during training
            if self.training and self.config.use_load_balancing:
                self.router.update_bias(indices)

        # Permute tokens by expert assignment
        permuted_states, tokens_per_expert, permuted_weights = self._permute_tokens(
            hidden_states, indices, weights
        )

        # Process through routed experts
        expert_output, _ = self.experts(
            permuted_states,
            tokens_per_expert,
            permuted_weights,  # (num_tokens * topk,)
        )

        # Unpermute and combine
        routed_output = self._unpermute_tokens(
            expert_output, indices, weights, original_shape
        )

        # Process through shared expert (if any)
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            # Combine shared and routed outputs using Lorentz residual
            output = self.combine_residual(shared_output, routed_output)
        else:
            output = routed_output

        # Restore original shape
        if len(original_shape) == 3:
            output = output.view(original_shape)

        return output, aux_loss, indices


class LorentzMoEBlock(nn.Module):
    """
    Complete Lorentz MoE block with normalization and residual.

    Structure:
        input -> norm -> MoE -> residual(input, output) -> output

    Args:
        manifold: Lorentz manifold
        hidden_size: Model hidden size
        config: LorentzMoEConfig
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        config: LorentzMoEConfig,
    ):
        super().__init__()
        self.manifold = manifold

        from ..lorentz_norm import LorentzRMSNorm

        self.norm = LorentzRMSNorm(manifold, hidden_size)
        self.moe = LorentzMoE(manifold, hidden_size, config)
        self.residual = LorentzResidual(manifold)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor

        Returns:
            output: Block output
            aux_loss: Optional auxiliary loss
            indices: Optional expert indices
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        moe_output, aux_loss, indices = self.moe(hidden_states)
        output = self.residual(residual, moe_output)
        return output, aux_loss, indices

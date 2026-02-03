# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz Experts for hyperbolic MoE (HELM-MiCE).
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz Experts for Mixture of Curvature Experts.

Each expert operates in its own curvature space, enabling the model
to learn different geometric representations for different aspects of the data.

Key features:
1. Variable curvature per expert
2. Curvature transfer when routing between manifolds
3. SwiGLU activation in hyperbolic space

Based on:
    - HELM-MiCE: Hyperbolic LLM with Mixture of Curvature Experts
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...manifolds import Lorentz, project_space_to_lorentz
from ..lorentz_mlp import LorentzMLP


class LorentzExpert(nn.Module):
    """
    Single Lorentz expert with its own curvature.

    Each expert operates in a potentially different curvature space,
    enabling diverse geometric representations.

    Args:
        input_manifold: Input Lorentz manifold
        hidden_size: Model hidden size (space-like dimension)
        ffn_hidden_size: FFN intermediate size
        expert_curvature: Curvature for this expert (None = same as input)
        learnable_curvature: Whether expert curvature is learnable
    """

    def __init__(
        self,
        input_manifold: Lorentz,
        hidden_size: int,
        ffn_hidden_size: int,
        expert_curvature: Optional[float] = None,
        learnable_curvature: bool = False,
    ):
        super().__init__()
        self.input_manifold = input_manifold
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size

        # Expert operates in its own curvature space
        if expert_curvature is not None:
            self.expert_manifold = Lorentz(c=expert_curvature, learnable=learnable_curvature)
        else:
            self.expert_manifold = input_manifold

        # Expert MLP in expert's curvature space
        self.mlp = LorentzMLP(
            manifold=self.expert_manifold,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=False,
        )

        self._logged_first_call = False

    def _curvature_transfer_in(self, x: torch.Tensor) -> torch.Tensor:
        """Transfer input to expert's curvature space."""
        if self.expert_manifold is self.input_manifold:
            return x

        # Scale by sqrt(c_expert / c_input)
        scale = (self.expert_manifold.c / self.input_manifold.c).sqrt()
        return x * scale

    def _curvature_transfer_out(self, x: torch.Tensor) -> torch.Tensor:
        """Transfer output back to input's curvature space."""
        if self.expert_manifold is self.input_manifold:
            return x

        # Scale by sqrt(c_input / c_expert)
        scale = (self.input_manifold.c / self.expert_manifold.c).sqrt()
        return x * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (..., hidden_size+1) - full Lorentz vector

        Returns:
            Output tensor, shape (..., hidden_size+1) - full Lorentz vector
        """
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzExpert forward (c_in={self.input_manifold.c.item():.4f}, c_expert={self.expert_manifold.c.item():.4f})")
            self._logged_first_call = True

        # Transfer to expert curvature
        x = self._curvature_transfer_in(x)

        # Apply MLP in expert space
        x = self.mlp(x)

        # Transfer back to input curvature
        x = self._curvature_transfer_out(x)

        return x


class LorentzExpertGroup(nn.Module):
    """
    Group of Lorentz experts with different curvatures.

    Creates experts with curvatures distributed across a range,
    enabling the model to learn at multiple scales.

    Args:
        input_manifold: Input Lorentz manifold
        hidden_size: Model hidden size
        ffn_hidden_size: FFN intermediate size
        num_experts: Number of experts
        curvature_range: (min_c, max_c) range for expert curvatures
        learnable_curvature: Whether curvatures are learnable
    """

    def __init__(
        self,
        input_manifold: Lorentz,
        hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        curvature_range: Tuple[float, float] = (0.1, 2.0),
        learnable_curvature: bool = True,
    ):
        super().__init__()
        self.input_manifold = input_manifold
        self.num_experts = num_experts

        # Distribute curvatures across range
        curvatures = np.linspace(
            curvature_range[0],
            curvature_range[1],
            num_experts
        ).tolist()

        # Create experts with different curvatures
        self.experts = nn.ModuleList([
            LorentzExpert(
                input_manifold=input_manifold,
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                expert_curvature=c,
                learnable_curvature=learnable_curvature,
            )
            for c in curvatures
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        """
        Forward pass through a specific expert.

        Args:
            hidden_states: Input tensor
            expert_idx: Index of expert to use

        Returns:
            Expert output
        """
        return self.experts[expert_idx](hidden_states)

    def forward_batch(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched forward pass for all experts.

        Args:
            hidden_states: Permuted input tensor, shape (total_tokens, hidden+1)
            tokens_per_expert: Number of tokens for each expert

        Returns:
            Expert outputs, shape (total_tokens, hidden+1)
        """
        outputs = []
        offset = 0

        for i, expert in enumerate(self.experts):
            num_tokens = tokens_per_expert[i].item()
            if num_tokens > 0:
                expert_input = hidden_states[offset:offset + num_tokens]
                expert_output = expert(expert_input)
                outputs.append(expert_output)
            offset += num_tokens

        if outputs:
            return torch.cat(outputs, dim=0)
        else:
            return hidden_states.new_empty(0, hidden_states.shape[-1])


class LorentzSharedExpert(nn.Module):
    """
    Shared expert that processes all tokens.

    Unlike routed experts, the shared expert sees all tokens
    and provides a baseline computation.

    Args:
        manifold: Lorentz manifold
        hidden_size: Model hidden size
        ffn_hidden_size: FFN intermediate size (often larger for shared)
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        ffn_hidden_size: int,
    ):
        super().__init__()
        self.manifold = manifold
        self.mlp = LorentzMLP(
            manifold=manifold,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=False,
        )
        self._logged_first_call = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (..., hidden_size+1)

        Returns:
            Output tensor, shape (..., hidden_size+1)
        """
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzSharedExpert forward (c={self.manifold.c.item():.4f})")
            self._logged_first_call = True
        return self.mlp(x)


class LorentzGroupedExperts(nn.Module):
    """
    Grouped experts for efficient batched computation.

    Processes tokens for all experts in a single batched operation
    when possible, similar to Megatron's GroupedMLP.

    Args:
        input_manifold: Input Lorentz manifold
        hidden_size: Model hidden size
        ffn_hidden_size: FFN intermediate size per expert
        num_experts: Number of experts
        curvature_range: Range for expert curvatures
        learnable_curvature: Whether curvatures are learnable
    """

    def __init__(
        self,
        input_manifold: Lorentz,
        hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        curvature_range: Tuple[float, float] = (0.1, 2.0),
        learnable_curvature: bool = True,
    ):
        super().__init__()
        self.input_manifold = input_manifold
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts = num_experts

        # Create expert group
        self.expert_group = LorentzExpertGroup(
            input_manifold=input_manifold,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_experts=num_experts,
            curvature_range=curvature_range,
            learnable_curvature=learnable_curvature,
        )

    def forward(
        self,
        permuted_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for grouped experts.

        Interface compatible with Megatron's GroupedMLP.

        Args:
            permuted_hidden_states: Permuted input, shape (total_tokens, hidden+1)
            tokens_per_expert: Number of tokens per expert
            permuted_probs: Optional routing probabilities

        Returns:
            output: Expert outputs
            bias: Always None for Lorentz experts
        """
        output = self.expert_group.forward_batch(
            permuted_hidden_states,
            tokens_per_expert,
        )

        # Apply routing probabilities if provided
        # Note: permuted_probs should be (num_tokens,) or (num_tokens, 1)
        if permuted_probs is not None:
            if permuted_probs.dim() == 1:
                output = output * permuted_probs.unsqueeze(-1)
            else:
                output = output * permuted_probs

        return output, None

    def get_expert_curvatures(self) -> List[float]:
        """Get list of expert curvatures."""
        return [
            expert.expert_manifold.c.item()
            for expert in self.expert_group.experts
        ]

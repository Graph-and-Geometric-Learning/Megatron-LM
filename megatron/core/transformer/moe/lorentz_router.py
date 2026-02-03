# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz Router for hyperbolic MoE (HELM-MiCE).
# Routes tokens using hyperbolic distance in Lorentz space.

"""
Lorentz Router for Mixture of Curvature Experts.

Routes tokens to experts based on hyperbolic distance in Lorentz space.
Computes routing scores using Lorentz inner product / hyperbolic distance
rather than standard Euclidean dot product.

Key features:
1. INHERITS from TopKRouter to get ALL load balancing features:
   - aux_loss (micro-batch, sequence-level, global)
   - z_loss (numerical stability)
   - expert_bias (dynamic load balancing)
   - input_jitter (regularization)
   - token_dropping (capacity-based)
   - Process group coordination (TP, CP, EP)
2. Only OVERRIDES gating() to use Lorentz inner product for routing

Architecture:
- Input: Euclidean hidden states from transformer
- gating(): Project to Lorentz space, compute hyperbolic routing scores
- routing(): Inherited from TopKRouter (applies z_loss, aux_loss, etc.)
- forward(): Inherited from TopKRouter (orchestrates gating → routing)

Based on:
    - HELM-MiCE: Hyperbolic LLM with Mixture of Curvature Experts
    - Megatron-LM MoE: TopKRouter implementation
"""

from typing import Optional
import torch
import torch.nn as nn

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.router import TopKRouter

# Import Lorentz manifold and operations
from megatron.core.manifolds import Lorentz, project_space_to_lorentz

# ProcessGroupCollection is optional for testing
try:
    from megatron.core.process_groups_config import ProcessGroupCollection
except ImportError:
    ProcessGroupCollection = None


class LorentzTopKRouter(TopKRouter):
    """
    Lorentz-aware TopKRouter that routes using hyperbolic distance.

    INHERITS from TopKRouter to get all load balancing features for free:
    - aux_loss (micro-batch, sequence-level, global)
    - z_loss (numerical stability)
    - expert_bias (dynamic load balancing)
    - input_jitter (regularization)
    - token_dropping (capacity-based)
    - Process group coordination (TP, CP, EP)

    Only OVERRIDES gating() to compute logits using Lorentz inner product.

    The routing score for token x and expert e is:
        score(x, e) = 2c + 2*<x_L, w_e>_L
    where x_L is the token in Lorentz space, w_e is the expert embedding,
    and <.,.>_L is the Lorentz (Minkowski) inner product.

    Args:
        config: TransformerConfig with MoE settings
        pg_collection: ProcessGroupCollection for distributed training
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        # Initialize parent TopKRouter - gets ALL infrastructure
        super().__init__(config, pg_collection)

        # Get curvature from config
        self.global_curvature = getattr(config, 'hyperbolic_curvature', 1.0)
        learnable = getattr(config, 'learnable_curvature', False)

        # Create Lorentz manifold
        self.manifold = Lorentz(c=self.global_curvature, learnable=learnable)

        # Expert embeddings in Lorentz space (hidden_size + 1 dimensions)
        # These are learnable points on the Lorentz manifold
        # Initialize with space dimensions random, time computed from constraint
        expert_space = torch.randn(self.num_experts, config.hidden_size) * 0.01
        expert_lorentz = project_space_to_lorentz(expert_space, self.manifold.c)
        self.expert_embeddings = nn.Parameter(expert_lorentz)

        self._logged_first_call = False
        self._use_lorentz_routing = True  # Flag to enable/disable Lorentz routing

    def _lorentz_inner_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute batched Lorentz inner product for attention-like scoring.

        <x, y>_L = -x_0*y_0 + x_1*y_1 + ... + x_d*y_d

        Args:
            x: Query tensor, shape (..., d+1)
            y: Key tensor, shape (num_keys, d+1)

        Returns:
            Inner product scores, shape (..., num_keys)
        """
        # Negate time coordinate for Minkowski signature
        x_signed = x.clone()
        x_signed[..., 0] = -x_signed[..., 0]
        return x_signed @ y.transpose(-1, -2)

    def _project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure expert embeddings lie on the Lorentz manifold.

        Recomputes time coordinate from space coordinates to satisfy:
        -t^2 + ||x||^2 = -c  =>  t = sqrt(c + ||x||^2)
        """
        space = x[..., 1:]
        c = self.manifold.c
        time = torch.sqrt(c + (space * space).sum(dim=-1, keepdim=True))
        return torch.cat([time, space], dim=-1)

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        """
        OVERRIDE: Compute logits using Lorentz inner product.

        This is the ONLY method we override from TopKRouter.
        The parent class handles everything else:
        - routing(): applies z_loss, aux_loss, expert_bias, token_dropping
        - forward(): orchestrates gating → routing

        Args:
            input (torch.Tensor): Input tensor of shape (num_tokens, hidden_size).

        Returns:
            torch.Tensor: Logits tensor of shape (num_tokens, num_experts).
        """
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzTopKRouter.gating() "
                  f"(experts={self.num_experts}, topk={self.topk}, "
                  f"c={self.global_curvature}, lorentz_routing={self._use_lorentz_routing})")
            self._logged_first_call = True

        if not self._use_lorentz_routing:
            # Fall back to Euclidean routing using parent's gating method
            return super().gating(input)

        # Project input tokens to Lorentz space
        # input shape: (num_tokens, hidden_size) -> (num_tokens, hidden_size+1)
        input_lorentz = project_space_to_lorentz(input, self.manifold.c)

        # Project expert embeddings to ensure they're on the manifold
        expert_lorentz = self._project_to_manifold(self.expert_embeddings)

        # Compute Lorentz inner product scores
        # score = 2c + 2*<input, expert>_L
        # Higher inner product (less negative) = closer in hyperbolic space
        inner_products = self._lorentz_inner_batch(input_lorentz, expert_lorentz)
        logits = 2 * self.manifold.c + 2 * inner_products

        return logits

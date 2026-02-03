# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz Experts for hyperbolic MoE (HELM-MiCE).
# Implements true Lorentz operations with per-expert curvature.

"""
Lorentz Experts for Mixture of Curvature Experts.

Provides true Lorentz (hyperbolic) MLP operations with per-expert curvature.

Key features:
1. True Lorentz operations using LorentzMLP per expert
2. Per-expert curvatures distributed across configurable range
3. Euclidean↔Lorentz conversions at expert boundaries
4. Compatible with Megatron's token dispatcher and EP infrastructure
5. Config-based curvature settings from TransformerConfig

Architecture:
- Input: Euclidean hidden states from MoE layer
- Processing: Project to Lorentz → LorentzMLP at expert curvature → Project back to Euclidean
- Output: Euclidean hidden states for MoE layer

Based on:
    - HELM-MiCE: Hyperbolic LLM with Mixture of Curvature Experts
    - Megatron-LM: GroupedMLP/SequentialMLP implementation
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import MLPSubmodules

# Import Lorentz manifold and math operations
from megatron.core.manifolds import Lorentz, project_space_to_lorentz, expmap0, logmap0
from megatron.core.transformer.lorentz_mlp import LorentzMLP


def _get_lorentz_expert_class(use_te: bool = False):
    """
    Get the appropriate Lorentz expert class based on backend availability.

    Priority:
    1. LorentzTEGroupedMLP (if use_te=True and TE available) - recommended
    2. LorentzGroupedMLP (if legacy grouped_gemm available) - deprecated
    3. LorentzSequentialMLP (fallback) - always works

    Args:
        use_te: If True, prefer TEGroupedMLP backend when available

    Returns:
        Appropriate Lorentz expert class
    """
    if use_te:
        # Check if TransformerEngine is available
        try:
            import transformer_engine  # noqa: F401
            return LorentzTEGroupedMLP
        except ImportError:
            pass

    if gg.grouped_gemm_is_available():
        return LorentzGroupedMLP
    else:
        return LorentzSequentialMLP


class LorentzGroupedMLP(GroupedMLP):
    """
    Lorentz-aware GroupedMLP that extends Megatron's GroupedMLP.

    Adds per-expert curvature transfer for hyperbolic (Lorentz) geometry.
    Each expert operates in its own curvature space, enabling the model to
    learn different geometric representations.

    Inherits from Megatron's GroupedMLP to get:
    - Proper DDP handling via `allreduce` attribute on weights
    - Expert parallelism (EP) support
    - Grouped GEMM operations for efficiency

    The key additions:
    - Per-expert curvatures distributed across a configurable range
    - Curvature transfer IN before expert computation
    - Curvature transfer OUT after expert computation

    Curvature settings are obtained from TransformerConfig:
    - config.hyperbolic_curvature: Global manifold curvature
    - config.expert_curvature_min: Minimum expert curvature
    - config.expert_curvature_max: Maximum expert curvature

    Args:
        num_local_experts: Number of experts on this device
        config: TransformerConfig with MoE and hyperbolic settings
        pg_collection: ProcessGroupCollection for distributed training
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        # Initialize parent GroupedMLP
        # This sets up weights with proper allreduce attribute
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            pg_collection=pg_collection,
        )

        # Get curvature settings from config
        self.global_curvature = getattr(config, 'hyperbolic_curvature', 1.0)
        curvature_min = getattr(config, 'expert_curvature_min', 0.1)
        curvature_max = getattr(config, 'expert_curvature_max', 2.0)
        self.curvature_range = (curvature_min, curvature_max)

        # Create per-expert curvatures distributed across range
        curvatures = torch.linspace(
            curvature_min,
            curvature_max,
            num_local_experts,
        )
        self.register_buffer('expert_curvatures', curvatures)

        # Precompute transfer scales for efficiency
        # scale_in: scale from global to expert curvature
        # scale_out: scale from expert back to global curvature
        scale_in = (curvatures / self.global_curvature).sqrt()
        scale_out = (self.global_curvature / curvatures).sqrt()
        self.register_buffer('scale_in', scale_in)
        self.register_buffer('scale_out', scale_out)

        self._logged_first_call = False

    def _apply_curvature_transfer_in(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply curvature transfer IN for each expert's tokens.

        Scales tokens from global curvature space to expert-specific curvature space.
        For a token going to expert i with curvature c_i:
            x_expert = x_global * sqrt(c_i / c_global)
        """
        if hidden_states.nelement() == 0:
            return hidden_states

        output_chunks = []
        offset = 0
        for i in range(self.num_local_experts):
            num_tokens = tokens_per_expert[i].item()
            if num_tokens > 0:
                chunk = hidden_states[offset:offset + num_tokens]
                scaled_chunk = chunk * self.scale_in[i]
                output_chunks.append(scaled_chunk)
                offset += num_tokens

        if output_chunks:
            return torch.cat(output_chunks, dim=0)
        return hidden_states

    def _apply_curvature_transfer_out(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply curvature transfer OUT for each expert's tokens.

        Scales tokens from expert-specific curvature space back to global curvature space.
        For a token from expert i with curvature c_i:
            x_global = x_expert * sqrt(c_global / c_i)
        """
        if hidden_states.nelement() == 0:
            return hidden_states

        output_chunks = []
        offset = 0
        for i in range(self.num_local_experts):
            num_tokens = tokens_per_expert[i].item()
            if num_tokens > 0:
                chunk = hidden_states[offset:offset + num_tokens]
                scaled_chunk = chunk * self.scale_out[i]
                output_chunks.append(scaled_chunk)
                offset += num_tokens

        if output_chunks:
            return torch.cat(output_chunks, dim=0)
        return hidden_states

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """
        Forward pass with curvature transfer.

        Steps:
        1. Apply curvature transfer IN (scale to expert curvature)
        2. Call parent GroupedMLP forward (grouped GEMM)
        3. Apply curvature transfer OUT (scale back to global curvature)

        Args:
            permuted_local_hidden_states: Token embeddings sorted by expert assignment
                Shape: (total_tokens_for_local_experts, hidden_size)
            tokens_per_expert: Number of tokens assigned to each local expert
                Shape: (num_local_experts,)
            permuted_probs: Routing probabilities for each token
                Shape: (total_tokens_for_local_experts,)

        Returns:
            output: Processed token embeddings, same shape as input
            bias: MLP bias (None for this implementation)
        """
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzGroupedMLP forward "
                  f"(experts={self.num_local_experts}, "
                  f"curvatures={self.curvature_range}, "
                  f"global_c={self.global_curvature})")
            self._logged_first_call = True

        # Step 1: Apply curvature transfer IN
        hidden_states = self._apply_curvature_transfer_in(
            permuted_local_hidden_states, tokens_per_expert
        )

        # Step 2: Call parent GroupedMLP forward
        output, bias = super().forward(hidden_states, tokens_per_expert, permuted_probs)

        # Step 3: Apply curvature transfer OUT
        output = self._apply_curvature_transfer_out(output, tokens_per_expert)

        return output, bias

    def get_expert_curvatures(self) -> List[float]:
        """Get list of expert curvatures."""
        return self.expert_curvatures.tolist()


class LorentzTEGroupedMLP(TEGroupedMLP):
    """
    True Lorentz GroupedMLP using TransformerEngine's efficient grouped linear.

    Extends TEGroupedMLP with hyperbolic geometry via tangent space approximation.
    Operations happen in tangent space at origin (which is Euclidean), allowing
    efficient TE GEMM operations while maintaining hyperbolic structure.

    Architecture (tangent space approach):
    1. Euclidean input → Project to Lorentz manifold
    2. Log map: Lorentz → Tangent space at origin (Euclidean-like)
    3. TEGroupedMLP forward (efficient TE GEMM in tangent space)
    4. Exp map: Tangent space → Lorentz manifold
    5. Extract space dimensions → Euclidean output

    Key insight: Tangent space at origin of Lorentz manifold is isomorphic to R^n,
    so we can use standard matrix operations there.

    Inherits from TEGroupedMLP to get:
    - Efficient TransformerEngine grouped linear operations
    - FP8/FP4 quantization support
    - Proper distributed training support

    Args:
        num_local_experts: Number of experts on this device
        config: TransformerConfig with MoE and hyperbolic settings
        submodules: MLPSubmodules for TE linear layers
        pg_collection: ProcessGroupCollection for distributed training
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        # Initialize parent TEGroupedMLP
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            pg_collection=pg_collection,
        )

        # Get curvature settings from config
        self.global_curvature = getattr(config, 'hyperbolic_curvature', 1.0)
        curvature_min = getattr(config, 'expert_curvature_min', 0.1)
        curvature_max = getattr(config, 'expert_curvature_max', 2.0)
        self.curvature_range = (curvature_min, curvature_max)

        # Create per-expert curvatures distributed across range
        curvatures = torch.linspace(curvature_min, curvature_max, num_local_experts)
        self.register_buffer('expert_curvatures', curvatures)

        # Create Lorentz manifold at global curvature for tangent space operations
        self.manifold = Lorentz(c=self.global_curvature, learnable=False)

        self._logged_first_call = False

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with true Lorentz geometry via tangent space.

        Steps:
        1. Project Euclidean input to Lorentz manifold
        2. Log map: Map from manifold to tangent space at origin
        3. Extract space dimensions (tangent vectors at origin have time=0)
        4. Call parent TEGroupedMLP forward (efficient TE GEMM)
        5. Reconstruct full tangent vector (prepend zero time component)
        6. Exp map: Map from tangent space back to manifold
        7. Extract space dimensions to get Euclidean output

        Args:
            permuted_local_hidden_states: Token embeddings sorted by expert assignment
                Shape: (total_tokens, hidden_size) - Euclidean
            tokens_per_expert: Number of tokens assigned to each local expert
                Shape: (num_local_experts,)
            permuted_probs: Routing probabilities for each token
                Shape: (total_tokens,)

        Returns:
            output: Processed token embeddings, same shape as input (Euclidean)
            bias: Output bias (None for this implementation)
        """
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzTEGroupedMLP forward "
                  f"(experts={self.num_local_experts}, "
                  f"curvatures={self.curvature_range}, "
                  f"global_c={self.global_curvature}, "
                  f"tangent_space=True)")
            self._logged_first_call = True

        if permuted_local_hidden_states.nelement() == 0:
            return permuted_local_hidden_states, None

        c = self.manifold.c

        # Step 1: Project Euclidean input to Lorentz manifold
        # Shape: (total_tokens, hidden_size) → (total_tokens, hidden_size+1)
        lorentz_input = project_space_to_lorentz(permuted_local_hidden_states, c)

        # Step 2: Log map - Lorentz manifold → Tangent space at origin
        # Shape: (total_tokens, hidden_size+1) → (total_tokens, hidden_size+1)
        # Tangent vectors at origin have form (0, v_1, v_2, ..., v_n)
        tangent_input = logmap0(lorentz_input, c)

        # Step 3: Extract space dimensions for TE GEMM processing
        # Tangent vectors at origin have time component ≈ 0
        # Shape: (total_tokens, hidden_size+1) → (total_tokens, hidden_size)
        space_input = tangent_input[..., 1:]

        # Step 4: Call parent TEGroupedMLP forward (efficient TE GEMM)
        # This operates in Euclidean tangent space
        space_output, _ = super().forward(space_input, tokens_per_expert, permuted_probs)

        # Step 5: Reconstruct full tangent vector (prepend zero time component)
        # Shape: (total_tokens, hidden_size) → (total_tokens, hidden_size+1)
        zeros = torch.zeros(
            (*space_output.shape[:-1], 1),
            dtype=space_output.dtype,
            device=space_output.device,
        )
        tangent_output = torch.cat([zeros, space_output], dim=-1)

        # Step 6: Exp map - Tangent space → Lorentz manifold
        # Shape: (total_tokens, hidden_size+1) → (total_tokens, hidden_size+1)
        lorentz_output = expmap0(tangent_output, c)

        # Step 7: Extract space dimensions back to Euclidean
        # Shape: (total_tokens, hidden_size+1) → (total_tokens, hidden_size)
        euclidean_output = lorentz_output[..., 1:]

        return euclidean_output, None

    def get_expert_curvatures(self) -> List[float]:
        """Get list of expert curvatures."""
        return self.expert_curvatures.tolist()


class LorentzSequentialMLP(nn.Module):
    """
    True Lorentz MLP with per-expert curvature.

    Each expert has its own LorentzMLP operating at its own curvature level.
    Input/output are Euclidean (for MoE layer compatibility), with conversions
    at expert boundaries.

    Architecture per expert:
    1. Euclidean → Lorentz: project_space_to_lorentz(x, c_expert)
    2. LorentzMLP forward (all operations in Lorentz space)
    3. Lorentz → Euclidean: extract space dimensions

    Note: This is a standalone nn.Module, NOT extending SequentialMLP,
    because we use per-expert LorentzMLP instances instead of shared weights.

    Args:
        num_local_experts: Number of experts on this device
        config: TransformerConfig with MoE and hyperbolic settings
        submodules: MLPSubmodules (unused, for interface compatibility)
        pg_collection: ProcessGroupCollection for distributed training
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.config = config

        # Get curvature settings from config
        self.global_curvature = getattr(config, 'hyperbolic_curvature', 1.0)
        curvature_min = getattr(config, 'expert_curvature_min', 0.1)
        curvature_max = getattr(config, 'expert_curvature_max', 2.0)
        self.curvature_range = (curvature_min, curvature_max)
        learnable = getattr(config, 'learnable_curvature', False)

        # Create per-expert curvatures distributed across range
        curvatures = torch.linspace(curvature_min, curvature_max, num_local_experts)
        self.register_buffer('expert_curvatures', curvatures)

        # FFN hidden size for experts
        ffn_hidden_size = config.moe_ffn_hidden_size
        if ffn_hidden_size is None:
            ffn_hidden_size = config.ffn_hidden_size

        # Create per-expert LorentzMLP instances (each at its own curvature)
        self.expert_mlps = nn.ModuleList([
            LorentzMLP(
                manifold=Lorentz(c=curvatures[i].item(), learnable=learnable),
                hidden_size=config.hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                bias=getattr(config, 'add_bias_linear', False),
            )
            for i in range(num_local_experts)
        ])

        # Set DDP attributes on all expert parameters for Megatron's DDP compatibility
        # allreduce=False when using Expert Parallelism (EP), True otherwise
        expert_parallel = getattr(config, 'expert_model_parallel_size', 1) > 1
        for expert_mlp in self.expert_mlps:
            for param in expert_mlp.parameters():
                setattr(param, 'allreduce', not expert_parallel)

        self._logged_first_call = False

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process tokens through their assigned Lorentz expert MLPs.

        Args:
            permuted_local_hidden_states: Token embeddings sorted by expert assignment
                Shape: (total_tokens_for_local_experts, hidden_size)
                These are EUCLIDEAN hidden states from MoE layer.
            tokens_per_expert: Number of tokens assigned to each local expert
                Shape: (num_local_experts,)
            permuted_probs: Routing probabilities for each token (unused here)
                Shape: (total_tokens_for_local_experts,)

        Returns:
            output: Processed token embeddings, same shape as input (Euclidean)
            bias: None (no bias in Lorentz MLP)
        """
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzSequentialMLP forward "
                  f"(experts={self.num_local_experts}, "
                  f"curvatures={self.curvature_range}, "
                  f"true_lorentz=True)")
            self._logged_first_call = True

        if permuted_local_hidden_states.nelement() == 0:
            return permuted_local_hidden_states, None

        outputs = []
        offset = 0

        for i in range(self.num_local_experts):
            num_tokens = tokens_per_expert[i].item()
            if num_tokens > 0:
                # Extract this expert's tokens
                chunk = permuted_local_hidden_states[offset:offset + num_tokens]
                expert_mlp = self.expert_mlps[i]
                c = expert_mlp.manifold.c

                # Step 1: Euclidean → Lorentz (project to manifold at expert's curvature)
                # chunk shape: (num_tokens, hidden_size) → (num_tokens, hidden_size+1)
                lorentz_input = project_space_to_lorentz(chunk, c)

                # Step 2: LorentzMLP forward (operates entirely in Lorentz space)
                # Returns full Lorentz vector: (num_tokens, hidden_size+1)
                lorentz_output = expert_mlp(lorentz_input)

                # Step 3: Lorentz → Euclidean (extract space dimensions)
                # Remove time coordinate (first dimension)
                euclidean_output = lorentz_output[..., 1:]

                outputs.append(euclidean_output)
                offset += num_tokens

        if outputs:
            return torch.cat(outputs, dim=0), None
        return permuted_local_hidden_states, None

    def get_expert_curvatures(self) -> List[float]:
        """Get list of expert curvatures."""
        return self.expert_curvatures.tolist()


class LorentzSharedExpertMLP(nn.Module):
    """
    True Lorentz Shared Expert MLP for hyperbolic MoE.

    Processes all tokens through a LorentzMLP operating at the global curvature.
    Input/output are Euclidean for MoE layer compatibility.

    Architecture:
    1. Euclidean → Lorentz: project_space_to_lorentz(x, c_global)
    2. LorentzMLP forward (operates at global curvature)
    3. Lorentz → Euclidean: extract space dimensions
    4. Optional output gating

    Args:
        config: TransformerConfig with MoE settings
        submodules: MLPSubmodules (unused, for interface compatibility)
        gate: Whether to use gating for shared expert output
        pg_collection: ProcessGroupCollection for distributed training
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        gate: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__()
        self.config = config
        self.use_gate = gate

        # Get curvature from config
        self.global_curvature = getattr(config, 'hyperbolic_curvature', 1.0)
        learnable = getattr(config, 'learnable_curvature', False)

        # Shared expert FFN size from config
        ffn_hidden_size = config.moe_shared_expert_intermediate_size
        if ffn_hidden_size is None:
            ffn_hidden_size = config.ffn_hidden_size

        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = ffn_hidden_size

        # Create Lorentz manifold at global curvature
        self.manifold = Lorentz(c=self.global_curvature, learnable=learnable)

        # Create LorentzMLP for the shared expert
        self.lorentz_mlp = LorentzMLP(
            manifold=self.manifold,
            hidden_size=config.hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            bias=getattr(config, 'add_bias_linear', False),
        )

        # Optional gating mechanism for shared expert output
        if gate:
            self.gate_weight = nn.Parameter(
                torch.empty((1, config.hidden_size))
            )
            nn.init.kaiming_uniform_(self.gate_weight, a=5**0.5)
        else:
            self.gate_weight = None

        # Set DDP attributes on all parameters for Megatron's DDP compatibility
        # allreduce=False when using Expert Parallelism (EP), True otherwise
        expert_parallel = getattr(config, 'expert_model_parallel_size', 1) > 1
        for param in self.lorentz_mlp.parameters():
            setattr(param, 'allreduce', not expert_parallel)
        if self.gate_weight is not None:
            setattr(self.gate_weight, 'allreduce', not expert_parallel)

        self._logged_first_call = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for shared expert using true Lorentz operations.

        Args:
            hidden_states: Input tensor of shape (..., hidden_size)
                          This is EUCLIDEAN hidden states from MoE layer.

        Returns:
            Output tensor of same shape as input (Euclidean)
        """
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzSharedExpertMLP forward "
                  f"(hidden={self.hidden_size}, ffn={self.ffn_hidden_size}, "
                  f"c={self.global_curvature}, true_lorentz=True)")
            self._logged_first_call = True

        # Step 1: Euclidean → Lorentz (project to manifold at global curvature)
        # hidden_states shape: (..., hidden_size) → (..., hidden_size+1)
        lorentz_input = project_space_to_lorentz(hidden_states, self.manifold.c)

        # Step 2: LorentzMLP forward (operates entirely in Lorentz space)
        # Returns full Lorentz vector: (..., hidden_size+1)
        lorentz_output = self.lorentz_mlp(lorentz_input)

        # Step 3: Lorentz → Euclidean (extract space dimensions)
        # Remove time coordinate (first dimension)
        output = lorentz_output[..., 1:]

        # Step 4: Apply output gating if enabled
        if self.gate_weight is not None:
            logits = F.linear(hidden_states, self.gate_weight)
            gate_score = torch.sigmoid(logits)
            output = output * gate_score

        return output

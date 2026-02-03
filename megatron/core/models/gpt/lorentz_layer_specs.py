# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz layer specifications for hyperbolic GPT models.
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz Layer Specifications.

Provides ModuleSpec factories for building hyperbolic GPT layers.

Usage:
    from megatron.core.models.gpt.lorentz_layer_specs import (
        get_lorentz_gpt_layer_spec,
        LorentzHyperbolicConfig,
    )

    # Configure hyperbolic settings
    hyperbolic_config = LorentzHyperbolicConfig(
        use_hyperbolic=True,
        curvature=1.0,
        model_type='dense',  # or 'mice'
    )

    # Get layer spec
    layer_spec = get_lorentz_gpt_layer_spec(config, hyperbolic_config)
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.identity_op import IdentityOp

from megatron.core.manifolds import Lorentz
from megatron.core.transformer.lorentz_norm import LorentzRMSNorm
from megatron.core.transformer.lorentz_residual import LorentzResidual, LorentzBiasDropoutAdd
from megatron.core.transformer.lorentz_attention import LorentzDotProductAttention
from megatron.core.transformer.lorentz_mlp import LorentzMLP
from megatron.core.tensor_parallel.lorentz_layers import (
    LorentzColumnParallelLinear,
    LorentzRowParallelLinear,
)


@dataclass
class LorentzHyperbolicConfig:
    """
    Configuration for hyperbolic (Lorentz) geometry in transformers.

    This config is separate from TransformerConfig to avoid modifying
    core Megatron files, making it easier to maintain and merge.
    """

    use_hyperbolic: bool = False
    """Enable hyperbolic (Lorentz) geometry."""

    curvature: float = 1.0
    """Curvature parameter c. Negative curvature = -1/c."""

    learnable_curvature: bool = False
    """Make curvature a learnable parameter."""

    model_type: Literal['dense', 'mice'] = 'dense'
    """Model type: 'dense' for HELM-D, 'mice' for HELM-MiCE."""

    # Residual connection settings
    use_residual_scale: bool = True
    """Use scaling in Lorentz residual connections."""

    residual_scale: Optional[float] = None
    """Fixed scale for residual (None = learnable)."""

    # MLA settings (for HELM-MiCE)
    q_lora_rank: int = 0
    """Low-rank dimension for query projection. 0 = no compression."""

    kv_lora_rank: int = 512
    """Low-rank dimension for KV projection."""

    # MoE settings (for HELM-MiCE)
    n_routed_experts: int = 4
    """Number of routed experts."""

    n_shared_experts: int = 1
    """Number of shared experts (always active)."""

    n_activated_experts: int = 2
    """Top-k experts activated per token."""

    moe_balance_loss_weight: float = 0.01
    """Weight for load balancing loss."""

    def __post_init__(self):
        """Validate config."""
        if self.curvature <= 0:
            raise ValueError(f"Curvature must be positive, got {self.curvature}")
        if self.model_type not in ('dense', 'mice'):
            raise ValueError(f"model_type must be 'dense' or 'mice', got {self.model_type}")


def get_lorentz_gpt_layer_spec(
    config,
    hyperbolic_config: LorentzHyperbolicConfig,
) -> ModuleSpec:
    """
    Get layer spec for hyperbolic GPT model.

    Automatically selects between dense (HELM-D) and MoE (HELM-MiCE)
    based on hyperbolic_config.model_type.

    Args:
        config: TransformerConfig
        hyperbolic_config: LorentzHyperbolicConfig

    Returns:
        ModuleSpec for the layer
    """
    if hyperbolic_config.model_type == 'mice':
        return get_lorentz_gpt_layer_spec_mice(config, hyperbolic_config)
    else:
        return get_lorentz_gpt_layer_spec_dense(config, hyperbolic_config)


def get_lorentz_gpt_layer_spec_dense(
    config,
    hyperbolic_config: LorentzHyperbolicConfig,
) -> ModuleSpec:
    """
    Get layer spec for dense hyperbolic GPT (HELM-D).

    Architecture:
        - LorentzRMSNorm -> LorentzAttention -> LorentzResidual
        - LorentzRMSNorm -> LorentzMLP -> LorentzResidual

    Args:
        config: TransformerConfig
        hyperbolic_config: LorentzHyperbolicConfig

    Returns:
        ModuleSpec for dense hyperbolic layer
    """
    # Create manifold instance (will be shared across layers)
    manifold = Lorentz(
        c=hyperbolic_config.curvature,
        learnable=hyperbolic_config.learnable_curvature,
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            # Input LayerNorm (Lorentz)
            input_layernorm=ModuleSpec(
                module=LorentzRMSNorm,
                params={
                    "manifold": manifold,
                    "dim": config.hidden_size,  # Space-like dimension
                },
            ),
            # Self Attention (Lorentz)
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": config.attn_mask_type},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ModuleSpec(
                        module=LorentzColumnParallelLinear,
                        params={
                            "manifold": manifold,
                            "input_size": config.hidden_size + 1,  # +1 for time
                            "output_size": 3 * config.hidden_size,  # QKV
                            "gather_output": False,
                        },
                    ),
                    core_attention=ModuleSpec(
                        module=LorentzDotProductAttention,
                        params={
                            "manifold": manifold,
                            "num_attention_heads": config.num_attention_heads,
                            "hidden_size_per_head": config.kv_channels,
                            "attention_dropout": config.attention_dropout,
                        },
                    ),
                    linear_proj=ModuleSpec(
                        module=LorentzRowParallelLinear,
                        params={
                            "manifold": manifold,
                            "input_size": config.hidden_size,
                            "output_size": config.hidden_size,
                            "input_is_parallel": True,
                        },
                    ),
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            # Attention residual (Lorentz)
            self_attn_bda=ModuleSpec(
                module=LorentzBiasDropoutAdd,
                params={
                    "manifold": manifold,
                    "dropout": config.hidden_dropout,
                    "use_scale": hyperbolic_config.use_residual_scale,
                    "scale": hyperbolic_config.residual_scale,
                },
            ),
            # Pre-MLP LayerNorm (Lorentz)
            pre_mlp_layernorm=ModuleSpec(
                module=LorentzRMSNorm,
                params={
                    "manifold": manifold,
                    "dim": config.hidden_size,
                },
            ),
            # MLP (Lorentz)
            mlp=ModuleSpec(
                module=LorentzMLP,
                params={
                    "manifold": manifold,
                    "hidden_size": config.hidden_size,
                    "ffn_hidden_size": config.ffn_hidden_size,
                },
            ),
            # MLP residual (Lorentz)
            mlp_bda=ModuleSpec(
                module=LorentzBiasDropoutAdd,
                params={
                    "manifold": manifold,
                    "dropout": config.hidden_dropout,
                    "use_scale": hyperbolic_config.use_residual_scale,
                    "scale": hyperbolic_config.residual_scale,
                },
            ),
        ),
    )


def get_lorentz_gpt_layer_spec_mice(
    config,
    hyperbolic_config: LorentzHyperbolicConfig,
) -> ModuleSpec:
    """
    Get layer spec for MoE hyperbolic GPT (HELM-MiCE).

    Architecture:
        - LorentzRMSNorm -> LorentzMLA -> LorentzResidual
        - LorentzRMSNorm -> LorentzMoE -> LorentzResidual

    Note: This is a placeholder. Full MoE implementation requires
    additional components (LorentzMLA, LorentzMoE, etc.).

    Args:
        config: TransformerConfig
        hyperbolic_config: LorentzHyperbolicConfig

    Returns:
        ModuleSpec for MoE hyperbolic layer
    """
    # TODO: Implement HELM-MiCE layer spec
    # This requires additional components:
    # - LorentzMLA (Multi-head Latent Attention)
    # - LorentzMoE (Mixture of Curvature Experts)
    # - LorentzRouter (Curvature-aware routing)

    raise NotImplementedError(
        "HELM-MiCE layer spec not yet implemented. "
        "Use model_type='dense' for HELM-D."
    )


class LorentzLayerSpecProvider:
    """
    Provider for Lorentz layer specifications.

    Similar to LocalSpecProvider/TESpecProvider but for hyperbolic layers.
    """

    def __init__(
        self,
        manifold: Lorentz,
        config,
    ):
        self.manifold = manifold
        self.config = config

    def column_parallel_linear(self):
        """Return Lorentz column parallel linear spec."""
        return ModuleSpec(
            module=LorentzColumnParallelLinear,
            params={"manifold": self.manifold},
        )

    def row_parallel_linear(self):
        """Return Lorentz row parallel linear spec."""
        return ModuleSpec(
            module=LorentzRowParallelLinear,
            params={"manifold": self.manifold},
        )

    def layer_norm(self):
        """Return Lorentz RMS norm spec."""
        return ModuleSpec(
            module=LorentzRMSNorm,
            params={"manifold": self.manifold, "dim": self.config.hidden_size},
        )

    def core_attention(self):
        """Return Lorentz dot product attention spec."""
        return ModuleSpec(
            module=LorentzDotProductAttention,
            params={
                "manifold": self.manifold,
                "num_attention_heads": self.config.num_attention_heads,
                "hidden_size_per_head": self.config.kv_channels,
            },
        )

    def mlp(self):
        """Return Lorentz MLP spec."""
        return ModuleSpec(
            module=LorentzMLP,
            params={
                "manifold": self.manifold,
                "hidden_size": self.config.hidden_size,
                "ffn_hidden_size": self.config.ffn_hidden_size,
            },
        )

    def bias_dropout_add(self):
        """Return Lorentz bias-dropout-add spec."""
        return ModuleSpec(
            module=LorentzBiasDropoutAdd,
            params={
                "manifold": self.manifold,
                "dropout": self.config.hidden_dropout,
            },
        )

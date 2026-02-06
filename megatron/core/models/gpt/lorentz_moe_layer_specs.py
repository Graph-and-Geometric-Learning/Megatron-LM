# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz MoE layer specifications for hyperbolic GPT models.
# Uses Megatron's MoELayer with custom Lorentz components.

"""
Lorentz MoE Layer Specifications.

Provides ModuleSpec factories for building hyperbolic MoE GPT layers that integrate
with Megatron's full pretrain infrastructure.

Key Design:
- Uses Megatron's MoELayer as the base (NOT a custom implementation)
- Plugs in Lorentz components: LorentzTopKRouter, LorentzGroupedMLP, LorentzSharedExpertMLP
- Maintains compatibility with Megatron's DDP, optimizer, and gradient handling

Usage:
    from megatron.core.models.gpt.lorentz_moe_layer_specs import (
        get_lorentz_moe_module_spec,
        get_lorentz_gpt_decoder_block_spec,
    )

    # Get MoE module spec with Lorentz components
    moe_spec = get_lorentz_moe_module_spec(config)

    # Get full decoder block spec
    block_spec = get_lorentz_gpt_decoder_block_spec(config)
"""

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

# Import Lorentz components
from megatron.core.transformer.moe.lorentz_router import LorentzTopKRouter
from megatron.core.transformer.moe.lorentz_experts import (
    LorentzGroupedMLP,
    LorentzSequentialMLP,
    LorentzSharedExpertMLP,
    LorentzTEGroupedMLP,
)
from megatron.core.transformer.moe import grouped_gemm_util as gg

try:
    from megatron.core.extensions.transformer_engine import TENorm

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


def get_lorentz_moe_module_spec(
    config: TransformerConfig,
    use_te: bool = False,
) -> ModuleSpec:
    """
    Get MoE module spec with Lorentz components.

    Uses Megatron's MoELayer as the container, with:
    - LorentzTopKRouter: Routes on space dimensions of Lorentz vectors
    - LorentzTEGroupedMLP: True Lorentz with TE backend (recommended)
    - LorentzGroupedMLP: Per-expert curvature transfer with legacy GroupedMLP
    - LorentzSequentialMLP: Fallback when no grouped GEMM available
    - LorentzSharedExpertMLP: Shared expert in Lorentz space

    Expert priority:
    1. LorentzTEGroupedMLP (if use_te=True and TE available) - recommended
    2. LorentzGroupedMLP (if legacy grouped_gemm available) - deprecated
    3. LorentzSequentialMLP (fallback) - always works

    Args:
        config: TransformerConfig with MoE and hyperbolic settings
        use_te: Whether to use Transformer Engine layers (recommended)

    Returns:
        ModuleSpec for Lorentz MoE layer
    """
    assert config.num_moe_experts is not None, "num_moe_experts must be set for MoE"

    # Backend for standard linear layers (used in shared experts and TE)
    backend: BackendSpecProvider = LocalSpecProvider()

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp_submodules = MLPSubmodules(
        linear_fc1=linear_fc1,
        linear_fc2=linear_fc2,
        activation_func=activation_func,
    )

    # Choose expert module based on backend availability
    # Priority: TE > legacy grouped_gemm > sequential
    if use_te and HAVE_TE:
        # Use LorentzTEGroupedMLP with TransformerEngine (recommended)
        # Get TE-specific submodules for grouped linear
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
        te_backend = TESpecProvider()
        # Get the submodules for TEGroupedMLP (moe_grouped_gemm=True, legacy=False)
        _, te_mlp_submodules = te_backend.grouped_mlp_modules(
            moe_use_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=False,
        )
        if te_mlp_submodules is not None:
            te_mlp_submodules.activation_func = te_backend.activation_func()
        experts_spec = ModuleSpec(
            module=LorentzTEGroupedMLP,
            submodules=te_mlp_submodules,
        )
    elif gg.grouped_gemm_is_available():
        # Fallback to legacy LorentzGroupedMLP
        experts_spec = ModuleSpec(
            module=LorentzGroupedMLP,
            submodules=None,  # LorentzGroupedMLP handles its own submodules
        )
    else:
        # Final fallback to LorentzSequentialMLP
        experts_spec = ModuleSpec(
            module=LorentzSequentialMLP,
            submodules=mlp_submodules,
        )

    # Use LorentzSharedExpertMLP if shared experts are configured
    if config.moe_shared_expert_intermediate_size is not None:
        shared_experts_spec = ModuleSpec(
            module=LorentzSharedExpertMLP,
            submodules=mlp_submodules,
        )
    else:
        shared_experts_spec = None

    # MoE module spec using Megatron's MoELayer with Lorentz components
    moe_module_spec = ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            experts=experts_spec,
            shared_experts=shared_experts_spec,
            router=LorentzTopKRouter,  # Router is passed as a class/callable
        ),
        metainfo={"fuse_pre_mlp_layernorm": False},
    )

    return moe_module_spec


def get_lorentz_gpt_layer_spec(
    config: TransformerConfig,
    use_te: bool = False,
    is_moe_layer: bool = True,
) -> ModuleSpec:
    """
    Get transformer layer spec for Lorentz GPT.

    Args:
        config: TransformerConfig
        use_te: Whether to use Transformer Engine
        is_moe_layer: Whether this layer uses MoE (vs dense MLP)

    Returns:
        ModuleSpec for transformer layer
    """
    # Use TE backend for attention/layernorm when available
    # Lorentz geometry only affects MoE components (router, experts)
    # LayerNorm and Attention are standard and benefit from TE optimizations
    if use_te and HAVE_TE:
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend: BackendSpecProvider = LocalSpecProvider()

    # Layer norm - use RMSNorm if configured
    if config.normalization == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)

    # MLP spec - either MoE or dense
    if is_moe_layer and config.num_moe_experts is not None:
        mlp_spec = get_lorentz_moe_module_spec(config, use_te)
    else:
        # Dense MLP
        mlp_spec = ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(),
                linear_fc2=backend.row_parallel_linear(),
                activation_func=backend.activation_func(),
            ),
        )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=layer_norm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                    k_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=layer_norm,
            mlp=mlp_spec,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )


def get_lorentz_gpt_decoder_layer_specs(
    config: TransformerConfig,
    use_te: bool = False,
) -> list:
    """
    Get list of layer specs for Lorentz GPT decoder.

    Handles the moe_layer_freq pattern to determine which layers use MoE.

    Args:
        config: TransformerConfig with moe_layer_freq
        use_te: Whether to use Transformer Engine

    Returns:
        List of ModuleSpec for each layer
    """
    # Parse moe_layer_freq to determine MoE pattern
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0
            for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"moe_layer_freq list length ({len(moe_layer_pattern)}) must match "
            f"num_layers ({config.num_layers})"
        )
    else:
        raise ValueError(f"Invalid moe_layer_freq: {config.moe_layer_freq}")

    # Create dense and MoE layer specs
    dense_layer_spec = get_lorentz_gpt_layer_spec(config, use_te, is_moe_layer=False)
    moe_layer_spec = get_lorentz_gpt_layer_spec(config, use_te, is_moe_layer=True)

    # Build layer specs list
    layer_specs = []
    for layer_idx in range(config.num_layers):
        if moe_layer_pattern[layer_idx] == 1:
            layer_specs.append(moe_layer_spec)
        else:
            layer_specs.append(dense_layer_spec)

    return layer_specs


def get_lorentz_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_te: bool = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """
    Get full decoder block spec for Lorentz GPT.

    This is the main entry point for building Lorentz MoE models with Megatron.

    Args:
        config: TransformerConfig with MoE and hyperbolic settings
        use_te: Whether to use Transformer Engine
        vp_stage: Virtual pipeline stage (for pipeline parallelism)
        pp_rank: Pipeline parallel rank

    Returns:
        TransformerBlockSubmodules for the decoder block
    """
    # Get all layer specs
    layer_specs = get_lorentz_gpt_decoder_layer_specs(config, use_te)

    # Slice for pipeline parallelism
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Layer norm implementation
    # Use TENorm when TE is available - it supports sequence parallelism
    # Note: FusedLayerNorm (Apex) doesn't support RMSNorm, so fallback to WrappedTorchNorm
    if use_te and HAVE_TE:
        # TENorm supports sequence parallelism and RMSNorm
        layer_norm_impl = TENorm
    elif HAVE_APEX and config.normalization != "RMSNorm":
        # FusedLayerNorm supports sequence parallelism but not RMSNorm
        layer_norm_impl = FusedLayerNorm
    else:
        # WrappedTorchNorm does NOT support sequence parallelism
        from megatron.core.transformer.torch_norm import WrappedTorchNorm
        layer_norm_impl = WrappedTorchNorm

    # Build block spec
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        layer_norm=layer_norm_impl,
    )

    return block_spec

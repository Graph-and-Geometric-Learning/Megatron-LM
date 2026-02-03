# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz GPT model builders for hyperbolic models.

"""
Lorentz GPT Model Builders.

Provides model builder functions for hyperbolic GPT models that integrate with
Megatron's full pretrain infrastructure.

Usage:
    from lorentz_gpt_builders import gpt_builder_lorentz_moe

    # In pretrain script:
    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder_lorentz_moe),
        ModelType.encoder_or_decoder,
        forward_step,
    )
"""

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.lorentz_moe_layer_specs import (
    get_lorentz_gpt_decoder_block_spec,
)
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args


def gpt_builder_lorentz_moe(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    """
    Build GPT model with Lorentz MoE layers.

    This builder uses Megatron's GPTModel with custom Lorentz MoE layer specs,
    ensuring full compatibility with:
    - Megatron's DistributedDataParallel
    - Megatron's distributed optimizer
    - Expert parallelism
    - Pipeline parallelism

    Args:
        args: Training arguments from get_args()
        pre_process: Whether to include embedding layer
        post_process: Whether to include output layer
        vp_stage: Virtual pipeline stage
        config: Optional TransformerConfig (created from args if None)
        pg_collection: ProcessGroupCollection for distributed training

    Returns:
        GPTModel with Lorentz MoE transformer layers
    """
    print_rank_0('Building Lorentz MoE GPT model ...')

    if config is None:
        config = core_transformer_config_from_args(args)

    # Validate hyperbolic settings
    if hasattr(args, 'use_lorentz_moe') and args.use_lorentz_moe:
        print_rank_0(f'  Hyperbolic curvature: {getattr(args, "hyperbolic_curvature", 1.0)}')
        print_rank_0(f'  Expert curvature range: [{getattr(args, "expert_curvature_min", 0.1)}, '
                     f'{getattr(args, "expert_curvature_max", 2.0)}]')

    # Determine use_te for experts
    # Use TE for experts if:
    # 1. transformer_impl is 'transformer_engine', OR
    # 2. --moe-grouped-gemm is set (without --moe-use-legacy-grouped-gemm)
    use_te_from_impl = getattr(args, 'transformer_impl', 'local') == 'transformer_engine'
    use_te_from_moe = (
        getattr(args, 'moe_grouped_gemm', False) and
        not getattr(args, 'moe_use_legacy_grouped_gemm', False)
    )
    use_te = use_te_from_impl or use_te_from_moe

    if use_te:
        print_rank_0(f'  Using TE backend for experts (moe_grouped_gemm={use_te_from_moe})')

    # Get Lorentz MoE transformer layer spec
    transformer_layer_spec = get_lorentz_gpt_decoder_block_spec(
        config=config,
        use_te=use_te,
        vp_stage=vp_stage,
    )

    # Build GPTModel with Lorentz MoE layers
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=getattr(args, 'fp16_lm_cross_entropy', False),
        parallel_output=True,
        share_embeddings_and_output_weights=not getattr(args, 'untie_embeddings_and_output_weights', False),
        position_embedding_type=getattr(args, 'position_embedding_type', 'learned_absolute'),
        rotary_percent=getattr(args, 'rotary_percent', 1.0),
        rotary_base=getattr(args, 'rotary_base', 10000),
        rope_scaling=getattr(args, 'use_rope_scaling', False),
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )

    return model


def gpt_builder_lorentz_dense(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    """
    Build dense GPT model with Lorentz (hyperbolic) geometry.

    This is for HELM-D style dense models without MoE.
    Uses standard Megatron layer specs with potential Lorentz modifications.

    Args:
        args: Training arguments from get_args()
        pre_process: Whether to include embedding layer
        post_process: Whether to include output layer
        vp_stage: Virtual pipeline stage
        config: Optional TransformerConfig
        pg_collection: ProcessGroupCollection

    Returns:
        GPTModel with Lorentz dense transformer layers
    """
    print_rank_0('Building Lorentz Dense GPT model ...')

    if config is None:
        config = core_transformer_config_from_args(args)

    # For dense Lorentz, we can use the existing lorentz_layer_specs
    from megatron.core.models.gpt.lorentz_layer_specs import (
        get_lorentz_gpt_layer_spec,
        LorentzHyperbolicConfig,
    )

    # Create hyperbolic config from args
    hyperbolic_config = LorentzHyperbolicConfig(
        use_hyperbolic=True,
        curvature=getattr(args, 'hyperbolic_curvature', 1.0),
        learnable_curvature=getattr(args, 'learnable_curvature', False),
        model_type='dense',
    )

    # Get layer spec
    transformer_layer_spec = get_lorentz_gpt_layer_spec(config, hyperbolic_config)

    # Build model
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=getattr(args, 'fp16_lm_cross_entropy', False),
        parallel_output=True,
        share_embeddings_and_output_weights=not getattr(args, 'untie_embeddings_and_output_weights', False),
        position_embedding_type=getattr(args, 'position_embedding_type', 'learned_absolute'),
        rotary_percent=getattr(args, 'rotary_percent', 1.0),
        rotary_base=getattr(args, 'rotary_base', 10000),
        rope_scaling=getattr(args, 'use_rope_scaling', False),
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )

    return model

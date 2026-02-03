# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Pretrain Lorentz MoE GPT using Megatron's full infrastructure.

"""
Pretrain Lorentz MoE GPT Model (HELM-MiCE).

Training script for hyperbolic MoE GPT using Megatron's full pretrain framework.

Key Features:
- Uses megatron.training.pretrain() as the entry point
- Megatron's DistributedDataParallel (NOT PyTorch DDP)
- Megatron's distributed optimizer
- Expert parallelism support via Megatron's MoE infrastructure
- Per-expert curvature transfer (Mixture of Curvature Experts)

Components:
- LorentzTopKRouter: Routes on space dimensions of Lorentz vectors
- LorentzGroupedMLP: Per-expert curvature transfer with Megatron's GroupedMLP
- LorentzSharedExpertMLP: Shared expert in Lorentz space

Usage:
    torchrun --nproc_per_node=8 pretrain_lorentz_moe_gpt.py \\
        --num-experts 8 \\
        --moe-router-topk 2 \\
        --use-lorentz-moe \\
        --hyperbolic-curvature 1.0 \\
        --expert-curvature-min 0.1 \\
        --expert-curvature-max 2.0 \\
        --tensor-model-parallel-size 1 \\
        --pipeline-model-parallel-size 1 \\
        ...

IMPORTANT: This script does NOT use:
- torch.nn.parallel.DistributedDataParallel
- Custom training loops
- torch.optim.AdamW directly
- Manual gradient synchronization

All of these are handled by Megatron's pretrain() infrastructure.
"""

# Capture the true program start time BEFORE any heavy imports.
import time
_PROGRAM_START_TIME = time.time()

import os
import warnings

# Suppress warnings on non-rank-0
rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import Optional

import torch

from lorentz_gpt_builders import gpt_builder_lorentz_moe
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch for training."""
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None, None

    # Get batches based on the TP rank
    batch = get_batch_on_this_tp_rank(data_iterator)

    # Pop packed sequence params (not supported in Lorentz MoE yet)
    cu_seqlens = batch.pop('cu_seqlens', None)
    batch.pop('cu_seqlens_padded', None)
    batch.pop('max_seqlen', None)
    local_cp_size = batch.pop('local_cp_size', None)

    if cu_seqlens is None and local_cp_size is None:
        # Slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)
        packed_seq_params = None
    else:
        # Packed sequence format not fully supported for Lorentz MoE yet
        raise NotImplementedError("Packed sequence format not yet supported for Lorentz MoE")

    return (*batch.values(), packed_seq_params)


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    model: Optional[GPTModel] = None,  # noqa: ARG001 - kept for interface compatibility
):
    """
    Loss function for Lorentz MoE GPT.

    Args:
        loss_mask: Mask for valid tokens
        output_tensor: Model output (losses)
        model: The model (optional, for accessing aux losses)

    Returns:
        loss: Scalar loss for this micro-batch
        num_tokens: Number of valid tokens
        report: Dict with metrics for logging
    """
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    # Build report dict
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check for NaN/Inf in loss
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel):
    """
    Forward training step.

    Args:
        data_iterator: Input data iterator
        model: The GPT Model

    Returns:
        output_tensor: Model output
        loss_func_partial: Partial loss function with loss_mask bound
    """
    timers = get_timers()

    # Get the batch
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(
            data_iterator, vp_stage
        )
    timers('batch-generator').stop()

    # Forward pass
    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )

    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    """Check if dataset should be built on this rank."""
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        and parallel_state.get_tensor_model_parallel_rank() == 0
    )


def core_gpt_dataset_config_from_args(args):
    """Build GPT dataset config from args."""
    tokenizer = get_tokenizer()

    # Get data blend configuration
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build train, validation, and test datasets."""
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for Lorentz MoE GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage),
        config,
    ).build()

    print_rank_0("> finished creating Lorentz MoE GPT datasets ...")

    return train_ds, valid_ds, test_ds


def add_lorentz_moe_args(parser):
    """Add Lorentz MoE specific arguments.

    Note: Most hyperbolic args (use_lorentz_moe, hyperbolic_curvature, etc.)
    are defined in TransformerConfig and auto-generated as CLI args.
    This function is kept for any additional args not in the config.
    """
    # All hyperbolic arguments are now defined in TransformerConfig
    # and automatically available as CLI arguments:
    #   --use-lorentz-moe
    #   --use-hyperbolic
    #   --hyperbolic-curvature
    #   --learnable-curvature
    #   --expert-curvature-min
    #   --expert-curvature-max
    return parser


if __name__ == "__main__":
    # Record timestamp for startup tracking
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=time.time())

    # Use Megatron's pretrain() - handles EVERYTHING:
    # - Model creation with Megatron's DDP wrapping
    # - Megatron's distributed optimizer
    # - Training loop
    # - Checkpointing
    # - Logging
    # - Expert parallelism
    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder_lorentz_moe),
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_lorentz_moe_args,
        args_defaults={
            'tokenizer_type': 'GPT2BPETokenizer',
            'transformer_impl': 'local',  # Use local implementation for Lorentz components
        },
    )

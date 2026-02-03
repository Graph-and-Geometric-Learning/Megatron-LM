# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Tensor parallel layers for Lorentz (hyperbolic) geometry.
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz Tensor Parallel Layers.

These layers extend Megatron's tensor parallel linear layers with hyperbolic geometry support.

Key Design Principles:
1. Time coordinate (x₀) is NEVER partitioned across TP ranks
2. Only space-like dimensions (x₁...xₐ) participate in tensor parallelism
3. After TP communication ops, project back to manifold (reconstruct time)

Usage:
    from megatron.core.tensor_parallel.lorentz_layers import (
        LorentzColumnParallelLinear,
        LorentzRowParallelLinear,
    )

    # Column parallel: splits output space-like dims across ranks
    fc1 = LorentzColumnParallelLinear(
        manifold, input_size=dim+1, output_size=4*dim, config=config, ...
    )

    # Row parallel: splits input space-like dims, reduces output
    fc2 = LorentzRowParallelLinear(
        manifold, input_size=4*dim+1, output_size=dim, config=config, ...
    )
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.utils import (
    divide,
    get_pg_rank,
    get_pg_size,
    get_tensor_model_parallel_group_if_none,
)

from .layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    _initialize_affine_weight_gpu,
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
)
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)

from ..manifolds import Lorentz, project_space_to_lorentz


class LorentzLinear(nn.Module):
    """
    Simple Lorentz linear layer (non-parallel).

    Applies linear transformation to Lorentz vectors.
    Input can be full Lorentz vector (d+1) or space-only (d).
    Output can be full Lorentz vector or space-only.

    Args:
        manifold: Lorentz manifold instance
        in_features: Input dimension (includes time if full Lorentz)
        out_features: Output space-like dimension
        bias: Whether to include bias
        input_includes_time: If True, input is (d+1), else (d)
        return_space: If True, return space-like only (d), else (d+1)
    """

    def __init__(
        self,
        manifold: Lorentz,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_includes_time: bool = True,
        return_space: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.input_includes_time = input_includes_time
        self.return_space = return_space

        # Compute actual input dimension for linear
        # If input includes time, we apply linear to full input
        # Weight shape: (out_features, in_features)
        self.weight = Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_space: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            return_space: Override instance setting for return_space

        Returns:
            Output tensor (full Lorentz or space-only)
        """
        if return_space is None:
            return_space = self.return_space

        # Apply linear transformation
        x_space = torch.nn.functional.linear(x, self.weight, self.bias)

        if return_space:
            return x_space

        # Reconstruct time coordinate and return full Lorentz vector
        return project_space_to_lorentz(x_space, self.manifold.c)


class LorentzColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer for Lorentz geometry.

    Partitions output space-like dimensions across TP ranks.
    Time coordinate is reconstructed locally after optional gather.

    The transformation is: Y = XW^T + b
    where W is partitioned along its first dimension (output).

    Input: (seq, batch, in_dim) - can be full Lorentz or space-only
    Output: (seq, batch, out_dim+1) if full Lorentz, else (seq, batch, out_dim/tp)

    Args:
        manifold: Lorentz manifold instance
        input_size: Input dimension
        output_size: Output space-like dimension (before TP partitioning)
        config: ModelParallelConfig
        init_method: Weight initialization method
        bias: Include bias
        gather_output: If True, gather output across TP ranks
        input_includes_time: If True, input is full Lorentz (d+1)
        skip_bias_add: If True, return bias separately
        is_expert: If True, treat as MoE expert layer
    """

    def __init__(
        self,
        manifold: Lorentz,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool = True,
        gather_output: bool = False,
        input_includes_time: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()

        self.manifold = manifold
        self.input_size = input_size
        self.output_size = output_size  # Space-like output dimension
        self.gather_output = gather_output
        self.input_includes_time = input_includes_time
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.config = config

        # Setup TP group
        self.tp_group = get_tensor_model_parallel_group_if_none(
            tp_group, is_expert=is_expert
        )
        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)

        # Partition output space-like dimension
        self.output_size_per_partition = divide(output_size, world_size)

        # Weight: (output_per_partition, input_size)
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.output_size_per_partition,
                    0,  # partition_dim
                    init_method,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=0,
                    is_expert=is_expert,
                )

        # Bias (partitioned)
        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input_: torch.Tensor,
        return_space: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_: Input tensor (seq, batch, dim)
            return_space: If True, return space-like dimensions only

        Returns:
            output: Output tensor
            output_bias: Bias if skip_bias_add, else None
        """
        # Copy input to TP region (no-op if already there)
        input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

        # Linear transformation
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = torch.nn.functional.linear(input_parallel, self.weight, bias)

        if self.gather_output:
            # Gather space-like dimensions across TP ranks
            output_space = gather_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            output_space = output_parallel

        # Return space-like or full Lorentz
        if return_space:
            output = output_space
        else:
            # Reconstruct time coordinate
            output = project_space_to_lorentz(output_space, self.manifold.c)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def __repr__(self):
        tp = self.output_size // self.output_size_per_partition
        return (
            f"LorentzColumnParallelLinear(in={self.input_size}, "
            f"out_space={self.output_size}, TP={tp})"
        )


class LorentzRowParallelLinear(nn.Module):
    """
    Row-parallel linear layer for Lorentz geometry.

    Partitions input space-like dimensions across TP ranks.
    Performs all-reduce on output, then reconstructs time coordinate.

    The transformation is: Y = XW^T + b
    where W is partitioned along its second dimension (input).

    Input: (seq, batch, in_dim/tp) - partitioned space-like
    Output: (seq, batch, out_dim+1) - full Lorentz after reduce

    Args:
        manifold: Lorentz manifold instance
        input_size: Input space-like dimension (before TP partitioning)
        output_size: Output space-like dimension
        config: ModelParallelConfig
        init_method: Weight initialization method
        bias: Include bias (not partitioned)
        input_is_parallel: If True, input is already partitioned
        skip_bias_add: If True, return bias separately
        is_expert: If True, treat as MoE expert layer
    """

    def __init__(
        self,
        manifold: Lorentz,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()

        self.manifold = manifold
        self.input_size = input_size  # Space-like input dimension
        self.output_size = output_size  # Space-like output dimension
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.config = config

        # Setup TP group
        self.tp_group = get_tensor_model_parallel_group_if_none(
            tp_group, is_expert=is_expert
        )
        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)

        # Partition input space-like dimension
        self.input_size_per_partition = divide(input_size, world_size)

        # Weight: (output_size, input_per_partition)
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,  # partition_dim
                    init_method,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    is_expert=is_expert,
                )

        # Bias (NOT partitioned - same on all ranks)
        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input_: torch.Tensor,
        return_space: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_: Input tensor (seq, batch, dim/tp) - partitioned space-like
            return_space: If True, return space-like dimensions only

        Returns:
            output: Output tensor (full Lorentz or space-like)
            output_bias: Bias if skip_bias_add, else None
        """
        # Handle input parallelism
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(
                input_, group=self.tp_group
            )

        # Linear transformation (each rank computes partial output)
        output_parallel = torch.nn.functional.linear(input_parallel, self.weight)

        # All-reduce to get full output (sum of partial outputs)
        output_space = reduce_from_tensor_model_parallel_region(
            output_parallel, group=self.tp_group
        )

        # Add bias (after reduce, same on all ranks)
        if not self.skip_bias_add and self.bias is not None:
            output_space = output_space + self.bias

        # Return space-like or full Lorentz
        if return_space:
            output = output_space
        else:
            # Reconstruct time coordinate
            output = project_space_to_lorentz(output_space, self.manifold.c)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def __repr__(self):
        tp = self.input_size // self.input_size_per_partition
        return (
            f"LorentzRowParallelLinear(in_space={self.input_size}, "
            f"out_space={self.output_size}, TP={tp})"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def lorentz_gather_from_tp_region(
    x_space: torch.Tensor,
    manifold: Lorentz,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Gather space-like dimensions from TP region and reconstruct time.

    Args:
        x_space: Partitioned space-like tensor
        manifold: Lorentz manifold
        group: TP process group

    Returns:
        Full Lorentz tensor
    """
    gathered = gather_from_tensor_model_parallel_region(x_space, group=group)
    return project_space_to_lorentz(gathered, manifold.c)


def lorentz_reduce_from_tp_region(
    x: torch.Tensor,
    manifold: Lorentz,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Reduce from TP region and project to manifold.

    Used after operations that may violate manifold constraint.

    Args:
        x: Tensor to reduce (full Lorentz)
        manifold: Lorentz manifold
        group: TP process group

    Returns:
        Reduced tensor on manifold
    """
    reduced = reduce_from_tensor_model_parallel_region(x, group=group)
    # Re-project to ensure manifold constraint
    return manifold.projx(reduced)

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz MLP for hyperbolic transformers.
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz MLP (Feed-Forward Network).

Implements SwiGLU-style MLP in Lorentz geometry.
Operations are performed on space-like dimensions, with time reconstructed as needed.

Architecture:
    gate = silu(W1(x))      # Gate path
    up = W3(x)              # Up projection
    x_space = gate * up     # Element-wise in space
    out = W2(project(x_space))  # Down projection

Based on:
    - Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..manifolds import Lorentz, project_space_to_lorentz
from ..tensor_parallel.lorentz_layers import (
    LorentzLinear,
    LorentzColumnParallelLinear,
    LorentzRowParallelLinear,
)


class LorentzMLP(nn.Module):
    """
    Lorentz MLP with SwiGLU activation.

    Standard MLP: y = W2(activation(W1(x)))
    SwiGLU MLP: y = W2(silu(W1(x)) * W3(x))

    In Lorentz space:
    - W1, W3: Project to intermediate dimension (space-like)
    - Gate activation applied in space-like dims
    - Element-wise multiply in space-like dims
    - Reconstruct time coordinate
    - W2: Project back to hidden size

    Args:
        manifold: Lorentz manifold instance
        hidden_size: Model hidden size (space-like dimension)
        ffn_hidden_size: Intermediate hidden size (space-like dimension)
        bias: Whether to use bias
        activation: Activation function (default: SiLU)
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        ffn_hidden_size: int,
        bias: bool = True,
        activation=F.silu,
    ):
        super().__init__()
        self.manifold = manifold
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.activation = activation

        # Gate projection: (hidden+1) -> ffn_hidden (space-like)
        self.w1 = LorentzLinear(
            manifold,
            in_features=hidden_size + 1,  # Full Lorentz input
            out_features=ffn_hidden_size,
            bias=bias,
            input_includes_time=True,
            return_space=True,  # Output space-like only
        )

        # Up projection: (hidden+1) -> ffn_hidden (space-like)
        self.w3 = LorentzLinear(
            manifold,
            in_features=hidden_size + 1,
            out_features=ffn_hidden_size,
            bias=bias,
            input_includes_time=True,
            return_space=True,
        )

        # Down projection: (ffn_hidden+1) -> hidden (space-like)
        self.w2 = LorentzLinear(
            manifold,
            in_features=ffn_hidden_size + 1,  # After time reconstruction
            out_features=hidden_size,
            bias=bias,
            input_includes_time=True,
            return_space=False,  # Output full Lorentz
        )
        self._logged_first_call = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (..., hidden_size+1) - full Lorentz vector

        Returns:
            Output tensor, shape (..., hidden_size+1) - full Lorentz vector
        """
        # Log first call to verify Lorentz MLP is being used
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzMLP forward (hidden={self.hidden_size}, ffn={self.ffn_hidden_size}, c={self.manifold.c.item():.4f})")
            self._logged_first_call = True

        # Gate path: SiLU activation on space-like
        gate_space = self.activation(self.w1(x))  # (..., ffn_hidden)

        # Up projection (space-like)
        up_space = self.w3(x)  # (..., ffn_hidden)

        # Element-wise multiply in space-like dimensions
        x_space = gate_space * up_space  # (..., ffn_hidden)

        # Reconstruct time coordinate
        x = project_space_to_lorentz(x_space, self.manifold.c)

        # Down projection
        return self.w2(x)


class LorentzFeedForward(nn.Module):
    """
    Simple Lorentz feed-forward network (non-gated).

    y = W2(activation(W1(x)))

    Args:
        manifold: Lorentz manifold instance
        hidden_size: Model hidden size (space-like)
        ffn_hidden_size: Intermediate hidden size (space-like)
        bias: Whether to use bias
        activation: Activation function
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        ffn_hidden_size: int,
        bias: bool = True,
        activation=F.gelu,
    ):
        super().__init__()
        self.manifold = manifold
        self.activation = activation

        # Up projection
        self.fc1 = LorentzLinear(
            manifold,
            in_features=hidden_size + 1,
            out_features=ffn_hidden_size,
            bias=bias,
            input_includes_time=True,
            return_space=True,
        )

        # Down projection
        self.fc2 = LorentzLinear(
            manifold,
            in_features=ffn_hidden_size + 1,
            out_features=hidden_size,
            bias=bias,
            input_includes_time=True,
            return_space=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Up projection + activation
        x_space = self.activation(self.fc1(x))

        # Reconstruct time
        x = project_space_to_lorentz(x_space, self.manifold.c)

        # Down projection
        return self.fc2(x)


class LorentzParallelMLP(nn.Module):
    """
    Tensor-parallel Lorentz MLP with SwiGLU.

    Uses LorentzColumnParallelLinear for W1, W3 (split FFN dimension)
    and LorentzRowParallelLinear for W2 (all-reduce output).

    This is the version to use for distributed training.

    Args:
        manifold: Lorentz manifold instance
        hidden_size: Model hidden size (space-like)
        ffn_hidden_size: Intermediate hidden size (space-like)
        config: ModelParallelConfig for TP settings
        init_method: Weight initialization method
        output_layer_init_method: Output layer initialization
        bias: Whether to use bias
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        ffn_hidden_size: int,
        config,
        init_method,
        output_layer_init_method=None,
        bias: bool = True,
    ):
        super().__init__()
        self.manifold = manifold

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Gate projection (column parallel - split output)
        self.w1 = LorentzColumnParallelLinear(
            manifold,
            input_size=hidden_size + 1,
            output_size=ffn_hidden_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=False,  # Keep partitioned
        )

        # Up projection (column parallel - split output)
        self.w3 = LorentzColumnParallelLinear(
            manifold,
            input_size=hidden_size + 1,
            output_size=ffn_hidden_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=False,  # Keep partitioned
        )

        # Down projection (row parallel - all-reduce output)
        self.w2 = LorentzRowParallelLinear(
            manifold,
            input_size=ffn_hidden_size,  # Input is partitioned
            output_size=hidden_size,
            config=config,
            init_method=output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (seq, batch, hidden_size+1)

        Returns:
            output: Output tensor
            output_bias: Optional bias (if skip_bias_add)
        """
        # Gate path: SiLU activation (return space-like, partitioned)
        gate_space, _ = self.w1(x, return_space=True)
        gate_space = F.silu(gate_space)

        # Up projection (space-like, partitioned)
        up_space, _ = self.w3(x, return_space=True)

        # Element-wise multiply (still partitioned)
        x_space = gate_space * up_space

        # Reconstruct time for partitioned tensor
        x = project_space_to_lorentz(x_space, self.manifold.c)

        # Down projection (all-reduce, then project to manifold)
        output, output_bias = self.w2(x, return_space=False)

        return output, output_bias

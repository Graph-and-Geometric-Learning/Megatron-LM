# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz residual connection for hyperbolic transformers.
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz Residual Connection (LResNet).

In hyperbolic geometry, we cannot simply add vectors as in Euclidean space.
The residual connection must project the sum back to the manifold.

Standard residual: y = x + f(x)
Lorentz residual: y = √c * (x + w*f(x)) / √|⟨x + w*f(x), x + w*f(x)⟩ₗ|

Based on:
    - Lorentzian Residual Neural Networks (https://arxiv.org/abs/2412.14695)
"""

import math
from typing import Optional, Union
import torch
import torch.nn as nn

from ..manifolds import Lorentz, project_space_to_lorentz


class LorentzResidual(nn.Module):
    """
    Residual connection in Lorentz space.

    Computes weighted sum of two Lorentz vectors and normalizes to manifold.

    μ = √c * (x + w*y) / √|⟨x + w*y, x + w*y⟩ₗ|

    Optionally applies scaling to the space-like dimensions.

    Args:
        manifold: Lorentz manifold instance
        weight: Initial weight for residual (default: 1.0)
        use_scale: If True, apply scaling to space dimensions
        scale: Initial scale value (or learnable if learn_scale=True)
        learn_scale: If True, scale is a learnable parameter
        learn_weight: If True, weight is a learnable parameter
    """

    def __init__(
        self,
        manifold: Lorentz,
        weight: float = 1.0,
        use_scale: bool = False,
        scale: Optional[float] = None,
        learn_scale: bool = False,
        learn_weight: bool = False,
    ):
        super().__init__()
        self.manifold = manifold

        # Residual weight
        if learn_weight:
            self.w_y = nn.Parameter(torch.tensor(weight))
        else:
            self.register_buffer('w_y', torch.tensor(weight))

        # Optional scaling
        self.use_scale = use_scale
        if use_scale:
            if scale is not None:
                if learn_scale:
                    # Store log(scale) for numerical stability
                    self.scale = nn.Parameter(torch.tensor(math.log(scale)))
                    self.learned_scale = True
                else:
                    self.register_buffer('scale', torch.tensor(scale))
                    self.learned_scale = False
            else:
                # Default learnable scale
                self.scale = nn.Parameter(torch.tensor(math.log(4.0)))
                self.learned_scale = True
        else:
            self.scale = None
            self.learned_scale = False

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: First Lorentz vector (typically the "identity" path)
            y: Second Lorentz vector (typically the "residual" path)
            weight: Optional override for residual weight

        Returns:
            Combined Lorentz vector on manifold
        """
        # Use provided weight or learned weight
        w = weight if weight is not None else self.w_y

        # Weighted sum
        ave = x + y * w

        # Normalize to manifold using Lorentzian inner product
        # ⟨x,x⟩ₗ should be -c for points on manifold, but sum may violate this
        inner = self.manifold.l_inner(ave, ave, keepdim=True, dim=-1)
        denom = torch.abs(inner).clamp_min(1e-6).sqrt()
        result = torch.sqrt(self.manifold.c) * ave / denom

        # Optional scaling
        if self.use_scale and self.scale is not None:
            if self.learned_scale:
                s = self.scale.exp()
            else:
                s = self.scale

            # Scale space-like dimensions
            x_space = s * result[..., 1:]

            # Reconstruct time
            result = project_space_to_lorentz(x_space, self.manifold.c)

        return result

    def __repr__(self):
        w = self.w_y.item() if hasattr(self.w_y, 'item') else self.w_y
        scale_info = ""
        if self.use_scale:
            if self.learned_scale:
                scale_info = f", scale=learnable({self.scale.exp().item():.2f})"
            else:
                scale_info = f", scale={self.scale.item():.2f}"
        return f"LorentzResidual(weight={w}{scale_info})"


class LorentzBiasDropoutAdd(nn.Module):
    """
    Lorentz version of bias-dropout-add fusion.

    Combines:
    1. Optional bias addition (in space-like dims)
    2. Dropout (in space-like dims)
    3. Residual addition (via Lorentz residual)

    Args:
        manifold: Lorentz manifold instance
        dropout: Dropout probability
        use_scale: Whether to use scaling in residual
        scale: Scale value for residual
    """

    def __init__(
        self,
        manifold: Lorentz,
        dropout: float = 0.0,
        use_scale: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.manifold = manifold
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.residual = LorentzResidual(
            manifold,
            use_scale=use_scale,
            scale=scale,
            learn_scale=scale is None,
        )

    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (output from attention/MLP)
            bias: Optional bias to add
            residual: Residual tensor to add

        Returns:
            Combined tensor on manifold
        """
        # Add bias to space-like dimensions
        if bias is not None:
            x_space = x[..., 1:] + bias
            x = project_space_to_lorentz(x_space, self.manifold.c)

        # Apply dropout (training only)
        if self.dropout is not None and self.training:
            x_space = self.dropout(x[..., 1:])
            x = project_space_to_lorentz(x_space, self.manifold.c)

        # Residual connection
        return self.residual(residual, x)

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz normalization layers for hyperbolic transformers.
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz Normalization Layers.

These layers apply normalization to the space-like dimensions of Lorentz vectors,
then reconstruct the time coordinate to satisfy the manifold constraint.

Key insight: Normalization only affects space-like dims (x₁...xₐ).
Time coordinate is always: x₀ = √(c + ||x_{1:d}||²)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..manifolds import Lorentz, project_space_to_lorentz


class LorentzRMSNorm(nn.Module):
    """
    RMS Normalization in Lorentz geometry.

    Applies RMSNorm to space-like dimensions, then reconstructs time coordinate.

    RMSNorm formula: x_norm = x / RMS(x) * weight
    where RMS(x) = √(mean(x²))

    Args:
        manifold: Lorentz manifold instance
        dim: Space-like dimension (not including time)
        eps: Numerical stability epsilon
    """

    def __init__(
        self,
        manifold: Lorentz,
        dim: int,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(
        self,
        x: torch.Tensor,
        space_only: bool = False,
        return_space: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Shape (..., dim+1) if space_only=False, else (..., dim)
            space_only: If True, input contains only space-like dims
            return_space: If True, return only space-like dims

        Returns:
            Normalized tensor on manifold or space-like only
        """
        # Extract space-like dimensions
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]

        # Apply RMS normalization
        normed_space = F.rms_norm(x_space, (self.dim,), self.weight, self.eps)

        if return_space:
            return normed_space

        # Reconstruct time coordinate
        return project_space_to_lorentz(normed_space, self.manifold.c)

    def __repr__(self):
        return f"LorentzRMSNorm(dim={self.dim}, eps={self.eps})"


class LorentzLayerNorm(nn.Module):
    """
    Layer Normalization in Lorentz geometry.

    Applies standard LayerNorm to space-like dimensions, then reconstructs time.

    Args:
        manifold: Lorentz manifold instance
        dim: Space-like dimension (not including time)
        eps: Numerical stability epsilon
    """

    def __init__(
        self,
        manifold: Lorentz,
        dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.layer = nn.LayerNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        space_only: bool = False,
        return_space: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Shape (..., dim+1) if space_only=False, else (..., dim)
            space_only: If True, input contains only space-like dims
            return_space: If True, return only space-like dims

        Returns:
            Normalized tensor on manifold or space-like only
        """
        # Extract space-like dimensions
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]

        # Apply layer normalization
        normed_space = self.layer(x_space)

        if return_space:
            return normed_space

        # Reconstruct time coordinate
        return project_space_to_lorentz(normed_space, self.manifold.c)

    def __repr__(self):
        return f"LorentzLayerNorm(dim={self.dim})"


class LorentzNormalization(nn.Module):
    """
    Unit normalization in Lorentz geometry.

    Normalizes space-like dimensions to unit norm, then reconstructs time.
    Useful for normalizing Q, K before attention computation.

    Args:
        manifold: Lorentz manifold instance
    """

    def __init__(self, manifold: Lorentz):
        super().__init__()
        self.manifold = manifold

    def forward(
        self,
        x: torch.Tensor,
        norm_factor: Optional[torch.Tensor] = None,
        space_only: bool = False,
        return_space: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            norm_factor: Optional precomputed normalization factor
            space_only: If True, input contains only space-like dims
            return_space: If True, return only space-like dims

        Returns:
            Normalized tensor
        """
        # Extract space-like dimensions
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]

        # Normalize
        if norm_factor is not None:
            x_space = x_space * norm_factor
        else:
            x_space = x_space / x_space.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        if return_space:
            return x_space

        # Reconstruct time coordinate
        return project_space_to_lorentz(x_space, self.manifold.c)


class LorentzActivation(nn.Module):
    """
    Activation function in Lorentz geometry.

    Applies activation to space-like dimensions, then reconstructs time.

    Args:
        manifold: Lorentz manifold instance
        activation: Activation function (e.g., F.silu, F.relu)
    """

    def __init__(self, manifold: Lorentz, activation=F.silu):
        super().__init__()
        self.manifold = manifold
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        space_only: bool = False,
        return_space: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            space_only: If True, input contains only space-like dims
            return_space: If True, return only space-like dims

        Returns:
            Activated tensor
        """
        # Extract space-like dimensions
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]

        # Apply activation
        x_space = self.activation(x_space)

        if return_space:
            return x_space

        # Reconstruct time coordinate
        return project_space_to_lorentz(x_space, self.manifold.c)


class LorentzDropout(nn.Module):
    """
    Dropout in Lorentz geometry.

    Applies dropout to space-like dimensions, then reconstructs time.

    Args:
        manifold: Lorentz manifold instance
        p: Dropout probability
    """

    def __init__(self, manifold: Lorentz, p: float = 0.1):
        super().__init__()
        self.manifold = manifold
        self.dropout = nn.Dropout(p)

    def forward(
        self,
        x: torch.Tensor,
        space_only: bool = False,
        return_space: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            space_only: If True, input contains only space-like dims
            return_space: If True, return only space-like dims

        Returns:
            Tensor with dropout applied
        """
        if not self.training:
            return x

        # Extract space-like dimensions
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]

        # Apply dropout
        x_space = self.dropout(x_space)

        if return_space:
            return x_space

        # Reconstruct time coordinate
        return project_space_to_lorentz(x_space, self.manifold.c)

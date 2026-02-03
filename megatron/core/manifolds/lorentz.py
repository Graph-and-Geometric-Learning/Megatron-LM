# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz manifold class for hyperbolic geometry.
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz (Hyperboloid) Manifold.

The Lorentz manifold is defined in (d+1)-dimensional Minkowski space:
    -x₀² + x₁² + x₂² + ... + xₐ² = -c,  where x₀ > 0, c > 0

Terminology:
    - x₀: Time-like coordinate (always positive, computed from space coordinates)
    - x₁...xₐ: Space-like coordinates (the actual learned features)
    - c: Curvature parameter (negative curvature = -1/c)

This manifold is well-suited for representing hierarchical structures common in language.

Usage:
    manifold = Lorentz(c=1.0, learnable=False)

    # Project space-like features to manifold
    x_lorentz = manifold.project_space(x_space)

    # Compute hyperbolic attention scores
    scores = manifold.attention_scores(query, key)

    # Aggregate values using Lorentzian centroid
    output = manifold.lorentzian_centroid(values, attention_weights)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from . import lorentz_math as lmath


class Lorentz(nn.Module):
    """
    Lorentz (Hyperboloid) Manifold for hyperbolic neural networks.

    The manifold constraint is: -x₀² + ||x_{1:d}||² = -c

    All operations preserve this constraint by:
    1. Operating on space-like dimensions (x₁...xₐ)
    2. Reconstructing time coordinate: x₀ = √(c + ||x_{1:d}||²)

    Args:
        c: Curvature parameter (default: 1.0). Negative curvature is -1/c.
        learnable: If True, curvature is a learnable parameter.

    Attributes:
        c: Curvature parameter (tensor)
        eps: Dictionary of epsilon values for numerical stability by dtype
        max_norm: Maximum norm for clamping (numerical stability)
        min_norm: Minimum norm for clamping (numerical stability)
    """

    def __init__(self, c: float = 1.0, learnable: bool = False):
        super().__init__()

        if learnable:
            # Learnable curvature (must be positive)
            self._c = nn.Parameter(torch.tensor([c], dtype=torch.float32))
        else:
            self.register_buffer('_c', torch.tensor([c], dtype=torch.float32))

        self.max_norm = 50.0
        self.min_norm = 1e-6
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8, torch.float16: 1e-4, torch.bfloat16: 1e-4}

    @property
    def c(self) -> torch.Tensor:
        """Curvature parameter (always positive)."""
        return self._c.abs().clamp_min(1e-6)

    def get_eps(self, dtype: torch.dtype) -> float:
        """Get epsilon for numerical stability."""
        return self.eps.get(dtype, 1e-6)

    # =========================================================================
    # Core Operations
    # =========================================================================

    def l_inner(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        keepdim: bool = False,
        dim: int = -1
    ) -> torch.Tensor:
        """
        Minkowski (Lorentzian) inner product.

        ⟨x,y⟩ₗ = -x₀y₀ + x₁y₁ + ... + xₐyₐ

        Args:
            x: First tensor, shape (..., d+1)
            y: Second tensor, shape (..., d+1)
            keepdim: Keep the reduced dimension
            dim: Dimension to reduce

        Returns:
            Inner product value(s)
        """
        return lmath.lorentz_inner(x, y, keepdim=keepdim, dim=dim)

    def cinner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Batched Minkowski inner product for attention.

        Computes ⟨xᵢ, yⱼ⟩ₗ for all pairs.

        Args:
            x: Query tensor, shape (..., seq_q, d+1)
            y: Key tensor, shape (..., seq_k, d+1)

        Returns:
            Inner product matrix, shape (..., seq_q, seq_k)
        """
        return lmath.lorentz_inner_batch(x, y)

    def lorentzian_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Squared Lorentzian distance.

        d²(x,y) = -2(c + ⟨x,y⟩ₗ)

        Args:
            x: First point(s) on manifold
            y: Second point(s) on manifold

        Returns:
            Squared distance
        """
        return lmath.lorentz_distance_squared(x, y, self.c)

    def induced_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Geodesic (induced) distance.

        d(x,y) = √c * arcosh(-⟨x,y⟩ₗ / c)

        Args:
            x: First point(s) on manifold
            y: Second point(s) on manifold

        Returns:
            Geodesic distance
        """
        return lmath.induced_distance(x, y, self.c)

    # =========================================================================
    # Projection Operations
    # =========================================================================

    def projx(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Project point to manifold.

        Recomputes time coordinate to satisfy manifold constraint.

        Args:
            x: Point with potentially invalid time coordinate
            dim: Feature dimension

        Returns:
            Point on manifold
        """
        return lmath.project_to_lorentz(x, self.c, dim=dim)

    def project_space(self, x_space: torch.Tensor) -> torch.Tensor:
        """
        Project space-like coordinates to full Lorentz vector.

        Given space coordinates (d dimensions), adds time coordinate
        to create (d+1) dimensional Lorentz vector.

        Args:
            x_space: Space-like coordinates, shape (..., d)

        Returns:
            Full Lorentz vector, shape (..., d+1)
        """
        return lmath.project_space_to_lorentz(x_space, self.c)

    def proju(self, x: torch.Tensor, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Project vector onto tangent space at x.

        Args:
            x: Point on manifold
            v: Vector to project
            dim: Feature dimension

        Returns:
            Projected vector in tangent space
        """
        return lmath.project_to_tangent(x, v, self.c, dim=dim)

    # =========================================================================
    # Exponential and Logarithmic Maps
    # =========================================================================

    def expmap0(self, u: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Exponential map from origin.

        Maps tangent vector at origin to point on manifold.

        Args:
            u: Tangent vector at origin
            dim: Feature dimension

        Returns:
            Point on manifold
        """
        return lmath.expmap0(u, self.c, dim=dim)

    def logmap0(self, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Logarithmic map to origin.

        Maps point on manifold to tangent vector at origin.

        Args:
            y: Point on manifold
            dim: Feature dimension

        Returns:
            Tangent vector at origin
        """
        return lmath.logmap0(y, self.c, dim=dim)

    def expmap(self, x: torch.Tensor, u: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Exponential map from point x.

        Args:
            x: Base point on manifold
            u: Tangent vector at x
            dim: Feature dimension

        Returns:
            Point on manifold
        """
        return lmath.expmap(x, u, self.c, dim=dim)

    def logmap(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Logarithmic map from x to y.

        Args:
            x: Source point on manifold
            y: Target point on manifold
            dim: Feature dimension

        Returns:
            Tangent vector at x pointing toward y
        """
        return lmath.logmap(x, y, self.c, dim=dim)

    # =========================================================================
    # Aggregation (for Attention)
    # =========================================================================

    def lorentzian_centroid(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: int = -2
    ) -> torch.Tensor:
        """
        Lorentzian centroid (weighted Fréchet mean).

        Used for attention value aggregation instead of standard weighted sum.
        μ = √c * (Σ wᵢxᵢ) / √|⟨Σ wᵢxᵢ, Σ wᵢxᵢ⟩ₗ|

        Args:
            x: Points on manifold, shape (..., n, d+1)
            weights: Optional weights for attention, shape (..., m, n)
            dim: Dimension to aggregate (ignored when weights provided)

        Returns:
            Centroid point(s) on manifold
        """
        return lmath.lorentzian_centroid(x, weights, self.c, dim=dim)

    def attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute hyperbolic attention scores.

        scores = (2c + 2*cinner(Q, K)) / scale + bias

        Args:
            query: Query tensor, shape (..., seq_q, d+1)
            key: Key tensor, shape (..., seq_k, d+1)
            scale: Optional scaling factor
            bias: Optional bias

        Returns:
            Attention scores, shape (..., seq_q, seq_k)
        """
        # Squared Lorentzian distance as attention score
        scores = 2 * self.c + 2 * self.cinner(query, key)

        if scale is not None:
            scores = scores / scale

        if bias is not None:
            scores = scores + bias

        return scores

    # =========================================================================
    # Parallel Transport
    # =========================================================================

    def ptransp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Parallel transport from x to y.

        Args:
            x: Source point
            y: Target point
            v: Tangent vector at x
            dim: Feature dimension

        Returns:
            Transported vector at y
        """
        return lmath.parallel_transport(x, y, v, self.c, dim=dim)

    def ptransp0(self, y: torch.Tensor, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Parallel transport from origin to y.

        Args:
            y: Target point
            v: Tangent vector at origin
            dim: Feature dimension

        Returns:
            Transported vector at y
        """
        return lmath.parallel_transport0(y, v, self.c, dim=dim)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def origin(
        self,
        *size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Create origin point(s) on manifold.

        The origin is (√c, 0, 0, ..., 0).

        Args:
            size: Shape of output tensor
            dtype: Output dtype
            device: Output device

        Returns:
            Origin point(s)
        """
        if dtype is None:
            dtype = self.c.dtype
        if device is None:
            device = self.c.device

        return lmath.origin(*size, c=self.c, dtype=dtype, device=device)

    def random_normal(
        self,
        *size: int,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Sample random point on manifold.

        Samples from normal distribution in tangent space at origin,
        then maps to manifold via exponential map.

        Args:
            size: Shape of output tensor
            mean: Mean of normal distribution
            std: Std of normal distribution
            dtype: Output dtype
            device: Output device

        Returns:
            Random point(s) on manifold
        """
        if dtype is None:
            dtype = self.c.dtype
        if device is None:
            device = self.c.device

        # Sample in tangent space
        v = torch.randn(*size, dtype=dtype, device=device) * std + mean
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Map to manifold
        return self.expmap0(v)

    def check_point_on_manifold(
        self,
        x: torch.Tensor,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        dim: int = -1
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if point lies on manifold.

        Args:
            x: Point to check
            atol: Absolute tolerance
            rtol: Relative tolerance
            dim: Feature dimension

        Returns:
            (is_on_manifold, error_message)
        """
        return lmath.check_on_manifold(x, self.c, atol=atol, rtol=rtol, dim=dim)

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def lorentz_to_poincare(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Convert from Lorentz to Poincaré ball model.

        Args:
            x: Point on Lorentz manifold
            dim: Feature dimension

        Returns:
            Point on Poincaré ball
        """
        return lmath.lorentz_to_poincare(x, self.c, dim=dim)

    def poincare_to_lorentz(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Convert from Poincaré ball to Lorentz model.

        Args:
            x: Point on Poincaré ball
            dim: Feature dimension

        Returns:
            Point on Lorentz manifold
        """
        return lmath.poincare_to_lorentz(x, self.c, dim=dim)

    # =========================================================================
    # Mobius Operations (for compatibility)
    # =========================================================================

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in Lorentz space.

        Args:
            x: First point
            y: Second point

        Returns:
            x ⊕ y
        """
        u = self.logmap0(y)
        v = self.ptransp0(x, u)
        return self.expmap(x, v)

    def mobius_matvec(self, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Möbius matrix-vector multiplication.

        Args:
            m: Matrix
            x: Point on manifold

        Returns:
            M ⊗ x
        """
        u = self.logmap0(x)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu)

    def __repr__(self) -> str:
        c_val = self.c.item() if self.c.numel() == 1 else self.c.tolist()
        learnable = isinstance(self._c, nn.Parameter)
        return f"Lorentz(c={c_val}, learnable={learnable})"

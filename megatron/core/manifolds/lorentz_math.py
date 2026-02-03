# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Hyperbolic math operations for Lorentz manifold.
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentzian math operations for hyperbolic geometry.

The Lorentz manifold (hyperboloid) is defined in (d+1)-dimensional Minkowski space:
    -x₀² + x₁² + x₂² + ... + xₐ² = -c,  where x₀ > 0, c > 0

Key operations:
- Minkowski inner product: ⟨u,v⟩ₗ = -u₀v₀ + Σᵢ₌₁ᵈ uᵢvᵢ
- Distance: d(x,y)² = -2(c + ⟨x,y⟩ₗ)
- Projection: time coordinate t = √(c + ||space||²)
"""

from typing import Optional, Tuple, Any
import torch
import torch.nn.functional as F


# Numerical stability constants
EPS = {torch.float32: 1e-6, torch.float64: 1e-8, torch.float16: 1e-4, torch.bfloat16: 1e-4}
MAX_NORM = 50.0
MIN_NORM = 1e-6


def get_eps(dtype: torch.dtype) -> float:
    """Get epsilon value for numerical stability based on dtype."""
    return EPS.get(dtype, 1e-6)


# =============================================================================
# Numerically Stable Functions
# =============================================================================

class LeakyClamp(torch.autograd.Function):
    """Clamp with leaky gradients outside bounds for numerical stability."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        with torch.no_grad():
            ctx.save_for_backward(x.ge(min_val) & x.le(max_val))
            return torch.clamp(x, min=min_val, max=max_val)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        eps = 1e-8
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None


def clamp(x: torch.Tensor, min_val: float = float("-inf"), max_val: float = float("+inf")) -> torch.Tensor:
    """Numerically stable clamp with leaky gradients."""
    return LeakyClamp.apply(x, min_val, max_val)


def safe_sqrt(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Numerically stable square root."""
    return torch.sqrt(clamp(x, min_val=eps))


class Arcosh(torch.autograd.Function):
    """Numerically stable arccosh that never returns NaNs."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            eps = get_eps(x.dtype)
            x = clamp(x, min_val=1.0 + eps)
            z = safe_sqrt(x * x - 1.0)
            ctx.save_for_backward(z)
            return torch.log(x + z)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        z, = ctx.saved_tensors
        return grad_output / z.clamp_min(1e-8)


def arcosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable arccosh: log(x + sqrt(x² - 1))."""
    return Arcosh.apply(x)


def arsinh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable arcsinh: log(x + sqrt(x² + 1))."""
    return torch.log(x + safe_sqrt(x * x + 1.0))


class Artanh(torch.autograd.Function):
    """Numerically stable arctanh that never returns NaNs."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        eps = get_eps(x.dtype)
        x = clamp(x, min_val=-1.0 + 4 * eps, max_val=1.0 - 4 * eps)
        ctx.save_for_backward(x)
        return 0.5 * (torch.log(1 + x) - torch.log(1 - x))

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        return grad_output / (1 - x ** 2)


def artanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable arctanh: 0.5 * log((1+x)/(1-x))."""
    return Artanh.apply(x)


# =============================================================================
# Core Lorentzian Operations
# =============================================================================

def lorentz_inner(
    u: torch.Tensor,
    v: torch.Tensor,
    keepdim: bool = False,
    dim: int = -1
) -> torch.Tensor:
    """
    Minkowski inner product.

    ⟨u,v⟩ₗ = -u₀v₀ + u₁v₁ + ... + uₐvₐ

    Args:
        u: Tensor of shape (..., d+1)
        v: Tensor of shape (..., d+1)
        keepdim: Whether to keep the reduced dimension
        dim: Dimension along which to compute inner product

    Returns:
        Minkowski inner product ⟨u,v⟩ₗ
    """
    d = u.size(dim) - 1
    uv = u * v

    if keepdim:
        return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=True)
    else:
        return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(dim, 1, d).sum(dim=dim)


def lorentz_inner_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Batched Minkowski inner product for attention computation.

    Computes ⟨xᵢ, yⱼ⟩ₗ for all pairs (i, j).
    Used for attention score computation: 2c + 2*cinner(Q, K)

    Args:
        x: Tensor of shape (..., seq_q, d+1)
        y: Tensor of shape (..., seq_k, d+1)

    Returns:
        Inner product matrix of shape (..., seq_q, seq_k)
    """
    # Negate time coordinate for Minkowski signature
    x_signed = x.clone()
    x_signed.narrow(-1, 0, 1).mul_(-1)
    return x_signed @ y.transpose(-1, -2)


def lorentz_norm(u: torch.Tensor, keepdim: bool = False, dim: int = -1) -> torch.Tensor:
    """
    Lorentzian norm: ||u||ₗ = √⟨u,u⟩ₗ

    Args:
        u: Tangent vector
        keepdim: Whether to keep the reduced dimension
        dim: Reduction dimension

    Returns:
        Lorentzian norm
    """
    inner = lorentz_inner(u, u, keepdim=keepdim, dim=dim)
    return safe_sqrt(inner)


# =============================================================================
# Projection Operations
# =============================================================================

@torch.jit.script
def project_to_lorentz(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Project a point to the Lorentz manifold.

    Given space-like coordinates x₁...xₐ, computes time coordinate:
        x₀ = √(c + ||x_{1:d}||²)

    Args:
        x: Tensor of shape (..., d+1) where x[..., 0] will be overwritten
        c: Curvature parameter
        dim: Feature dimension

    Returns:
        Point on Lorentz manifold satisfying ⟨x,x⟩ₗ = -c
    """
    dn = x.size(dim) - 1
    space = x.narrow(dim, 1, dn)
    time = torch.sqrt(c + (space * space).sum(dim=dim, keepdim=True))
    return torch.cat((time, space), dim=dim)


def project_space_to_lorentz(x_space: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Project space-like coordinates to full Lorentz vector.

    Args:
        x_space: Space-like coordinates of shape (..., d)
        c: Curvature parameter

    Returns:
        Full Lorentz vector of shape (..., d+1)
    """
    x_time = torch.sqrt(c + (x_space * x_space).sum(dim=-1, keepdim=True))
    return torch.cat((x_time, x_space), dim=-1)


def project_to_tangent(
    x: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Project vector v onto tangent space at point x.

    Πₓ(v) = v + ⟨x,v⟩ₗ * x / c

    Args:
        x: Point on manifold
        v: Vector to project
        c: Curvature parameter
        dim: Feature dimension

    Returns:
        Projected vector in tangent space at x
    """
    inner = lorentz_inner(x, v, keepdim=True, dim=dim)
    return v + (inner / c) * x


# =============================================================================
# Exponential and Logarithmic Maps
# =============================================================================

def expmap0(u: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Exponential map from origin.

    Maps tangent vector at origin to point on manifold.

    Args:
        u: Tangent vector at origin
        c: Curvature parameter
        dim: Feature dimension

    Returns:
        Point on manifold
    """
    norm = lorentz_norm(u, keepdim=True, dim=dim).clamp_max(MAX_NORM)
    sqrt_c = torch.sqrt(c)
    denom = norm / sqrt_c

    dn = u.size(dim) - 1
    time_part = torch.cosh(denom) * sqrt_c + torch.sinh(denom) * u.narrow(dim, 0, 1) * sqrt_c / norm
    space_part = sqrt_c * torch.sinh(denom) * u.narrow(dim, 1, dn) / norm

    return torch.cat((time_part, space_part), dim=dim)


def logmap0(y: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Logarithmic map to origin.

    Maps point on manifold to tangent vector at origin.

    Args:
        y: Point on manifold
        c: Curvature parameter
        dim: Feature dimension

    Returns:
        Tangent vector at origin
    """
    eps = get_eps(y.dtype)
    sqrt_c = torch.sqrt(c)

    # Distance from origin
    y_time = y.narrow(dim, 0, 1)
    dist = sqrt_c * arcosh((y_time / sqrt_c).clamp_min(1 + eps))

    # Direction
    dn = y.size(dim) - 1
    inner0 = -y_time * sqrt_c
    nomin = torch.cat((inner0 / c * sqrt_c + y.narrow(dim, 0, 1), y.narrow(dim, 1, dn)), dim=dim)
    denom = lorentz_norm(nomin, keepdim=True, dim=dim).clamp_min(eps)

    return dist * nomin / denom


def expmap(x: torch.Tensor, u: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Exponential map from point x.

    exp_x(v) = cosh(||v||ₗ/√c) * x + √c * sinh(||v||ₗ/√c) * v/||v||ₗ

    Args:
        x: Base point on manifold
        u: Tangent vector at x
        c: Curvature parameter
        dim: Feature dimension

    Returns:
        Point on manifold
    """
    norm = lorentz_norm(u, keepdim=True, dim=dim).clamp_max(MAX_NORM)
    u_normalized = u / norm.clamp_min(MIN_NORM)
    sqrt_c = torch.sqrt(c)

    return torch.cosh(norm / sqrt_c) * x + torch.sinh(norm / sqrt_c) * u_normalized * sqrt_c


def logmap(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Logarithmic map from point x to point y.

    Args:
        x: Source point on manifold
        y: Target point on manifold
        c: Curvature parameter
        dim: Feature dimension

    Returns:
        Tangent vector at x pointing toward y
    """
    eps = get_eps(x.dtype)

    # Distance
    inner = lorentz_inner(x, y, keepdim=True, dim=dim)
    dist = torch.sqrt(c) * arcosh((-inner / c).clamp_min(1 + eps))

    # Direction
    nomin = y + inner * x / c
    denom = lorentz_norm(nomin, keepdim=True, dim=dim).clamp_min(eps)

    return dist * nomin / denom


# =============================================================================
# Distance Functions
# =============================================================================

def lorentz_distance_squared(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Squared Lorentzian distance.

    d²(x,y) = -2(c + ⟨x,y⟩ₗ)

    This is used for attention scores: scores = 2c + 2*cinner(Q, K)

    Args:
        x: Point on manifold
        y: Point on manifold
        c: Curvature parameter

    Returns:
        Squared distance
    """
    inner = lorentz_inner(x, y)
    return -2 * (c + inner)


def induced_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Induced (geodesic) distance on manifold.

    d(x,y) = √c * arcosh(-⟨x,y⟩ₗ / c)

    Args:
        x: Point on manifold
        y: Point on manifold
        c: Curvature parameter

    Returns:
        Geodesic distance
    """
    eps = get_eps(x.dtype)
    inner = lorentz_inner(x, y)
    return torch.sqrt(c) * arcosh((-inner / c).clamp_min(1 + eps))


# =============================================================================
# Aggregation Operations
# =============================================================================

def lorentzian_centroid(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    c: torch.Tensor = torch.tensor(1.0),
    dim: int = -2
) -> torch.Tensor:
    """
    Lorentzian centroid (weighted Fréchet mean).

    Used for attention value aggregation instead of standard weighted sum.

    μ = √c * (Σ wᵢxᵢ) / √|⟨Σ wᵢxᵢ, Σ wᵢxᵢ⟩ₗ|

    Args:
        x: Points on manifold, shape (..., n, d+1)
        weights: Optional weights, shape (..., n) or (..., m, n) for batched attention
        c: Curvature parameter
        dim: Dimension to aggregate over

    Returns:
        Centroid point(s) on manifold
    """
    eps = get_eps(x.dtype)

    if weights is not None:
        # For attention: weights @ x
        # weights: (..., seq_q, seq_k), x: (..., seq_k, d+1)
        ave = weights @ x
    else:
        ave = x.mean(dim=dim)

    # Normalize to manifold
    inner = lorentz_inner(ave, ave, keepdim=True, dim=-1)
    denom = safe_sqrt(torch.abs(inner).clamp_min(eps))

    return torch.sqrt(c) * ave / denom


# =============================================================================
# Parallel Transport
# =============================================================================

def parallel_transport(
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Parallel transport vector v from tangent space at x to tangent space at y.

    Args:
        x: Source point
        y: Target point
        v: Tangent vector at x
        c: Curvature parameter
        dim: Feature dimension

    Returns:
        Transported vector at y
    """
    inner_yv = lorentz_inner(y, v, keepdim=True, dim=dim)
    inner_xy = lorentz_inner(x, y, keepdim=True, dim=dim)
    denom = (c - inner_xy).clamp_min(1e-7)
    return v + (inner_yv / denom) * (x + y)


def parallel_transport0(
    y: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Parallel transport from origin to point y.

    Args:
        y: Target point
        v: Tangent vector at origin
        c: Curvature parameter
        dim: Feature dimension

    Returns:
        Transported vector at y
    """
    sqrt_c = torch.sqrt(c)
    inner_yv = lorentz_inner(y, v, keepdim=True, dim=dim)

    # Inner product with origin: -y₀ * √c
    inner_y0 = -y.narrow(dim, 0, 1) * sqrt_c
    denom = (c - inner_y0).clamp_min(1e-7)

    # Origin point
    zero_point = torch.zeros_like(y)
    zero_point[..., 0] = sqrt_c

    return v + (inner_yv / denom) * (y + zero_point)


# =============================================================================
# Conversion Functions
# =============================================================================

def lorentz_to_poincare(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Convert from Lorentz model to Poincaré ball.

    Args:
        x: Point on Lorentz manifold
        c: Curvature parameter
        dim: Feature dimension

    Returns:
        Point on Poincaré ball
    """
    dn = x.size(dim) - 1
    sqrt_c_inv = c.reciprocal().sqrt()
    x_space = x.narrow(dim, 1, dn) * sqrt_c_inv
    return x_space / (x.narrow(dim, 0, 1) + sqrt_c_inv)


def poincare_to_lorentz(x: torch.Tensor, c: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert from Poincaré ball to Lorentz model.

    Args:
        x: Point on Poincaré ball
        c: Curvature parameter
        dim: Feature dimension
        eps: Numerical stability epsilon

    Returns:
        Point on Lorentz manifold
    """
    x_norm_sq = (x * x).sum(dim=dim, keepdim=True)
    c_inv = c.reciprocal()
    res = torch.cat((c_inv + x_norm_sq, 2 * c_inv * x), dim=dim) / (c_inv - x_norm_sq + eps)
    return c_inv.sqrt() * res


# =============================================================================
# Utility Functions
# =============================================================================

def check_on_manifold(
    x: torch.Tensor,
    c: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    dim: int = -1
) -> Tuple[bool, Optional[str]]:
    """
    Check if point lies on the Lorentz manifold.

    Verifies: -x₀² + x₁² + ... + xₐ² = -c

    Args:
        x: Point to check
        c: Curvature parameter
        atol: Absolute tolerance
        rtol: Relative tolerance
        dim: Feature dimension

    Returns:
        (is_on_manifold, error_message)
    """
    dn = x.size(dim) - 1
    x_sq = x ** 2
    quad_form = -x_sq.narrow(dim, 0, 1) + x_sq.narrow(dim, 1, dn).sum(dim=dim, keepdim=True)
    ok = torch.allclose(quad_form, -c.expand_as(quad_form), atol=atol, rtol=rtol)
    reason = None if ok else f"Minkowski quadratic form is not equal to {-c.item()}"
    return ok, reason


def origin(
    *size: int,
    c: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create origin point on manifold.

    The origin is (√c, 0, 0, ..., 0).

    Args:
        size: Shape of output tensor
        c: Curvature parameter
        dtype: Output dtype
        device: Output device

    Returns:
        Origin point(s) on manifold
    """
    if dtype is None:
        dtype = c.dtype
    if device is None:
        device = c.device

    zero_point = torch.zeros(*size, dtype=dtype, device=device)
    zero_point[..., 0] = torch.sqrt(c)
    return zero_point
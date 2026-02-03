# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Manifold classes for hyperbolic geometry in neural networks.

"""
Manifolds module for hyperbolic geometry support in Megatron-LM.

This module provides implementations of hyperbolic manifolds for use in
neural networks, particularly for HELM (Hyperbolic Large Language Models).

Main Classes:
    Lorentz: The Lorentz (hyperboloid) manifold, preferred for deep learning
             due to numerical stability.

Math Functions (in lorentz_math):
    lorentz_inner: Minkowski inner product
    lorentz_inner_batch: Batched inner product for attention
    lorentzian_centroid: Weighted Fréchet mean for value aggregation
    project_to_lorentz: Project point to manifold
    project_space_to_lorentz: Add time coordinate to space-like features
    arcosh, arsinh, artanh: Numerically stable inverse hyperbolic functions

Usage Example:
    from megatron.core.manifolds import Lorentz

    # Create manifold with fixed curvature
    manifold = Lorentz(c=1.0)

    # Or with learnable curvature
    manifold = Lorentz(c=1.0, learnable=True)

    # Project space-like features to Lorentz space
    x_space = torch.randn(batch, seq, dim)
    x_lorentz = manifold.project_space(x_space)  # shape: (batch, seq, dim+1)

    # Compute hyperbolic attention
    scores = manifold.attention_scores(query, key)  # 2c + 2*cinner(Q, K)
    output = manifold.lorentzian_centroid(values, attention_weights)
"""

from .lorentz import Lorentz
from .lorentz_math import (
    # Core operations
    lorentz_inner,
    lorentz_inner_batch,
    lorentz_norm,
    lorentz_distance_squared,
    induced_distance,

    # Projection
    project_to_lorentz,
    project_space_to_lorentz,
    project_to_tangent,

    # Maps
    expmap,
    expmap0,
    logmap,
    logmap0,

    # Aggregation
    lorentzian_centroid,

    # Parallel transport
    parallel_transport,
    parallel_transport0,

    # Numerically stable functions
    arcosh,
    arsinh,
    artanh,
    safe_sqrt,
    clamp,

    # Conversion
    lorentz_to_poincare,
    poincare_to_lorentz,

    # Utilities
    check_on_manifold,
    origin,
    get_eps,
)

__all__ = [
    # Classes
    'Lorentz',

    # Core operations
    'lorentz_inner',
    'lorentz_inner_batch',
    'lorentz_norm',
    'lorentz_distance_squared',
    'induced_distance',

    # Projection
    'project_to_lorentz',
    'project_space_to_lorentz',
    'project_to_tangent',

    # Maps
    'expmap',
    'expmap0',
    'logmap',
    'logmap0',

    # Aggregation
    'lorentzian_centroid',

    # Parallel transport
    'parallel_transport',
    'parallel_transport0',

    # Numerically stable functions
    'arcosh',
    'arsinh',
    'artanh',
    'safe_sqrt',
    'clamp',

    # Conversion
    'lorentz_to_poincare',
    'poincare_to_lorentz',

    # Utilities
    'check_on_manifold',
    'origin',
    'get_eps',
]

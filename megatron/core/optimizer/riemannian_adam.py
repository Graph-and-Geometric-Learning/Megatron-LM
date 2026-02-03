# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Riemannian Adam optimizer for hyperbolic (Lorentz) parameters.
# Adapted from HELM and geoopt.

"""
Riemannian Adam Optimizer.

Implements Adam optimization on Riemannian manifolds, specifically the Lorentz
(hyperboloid) manifold used in HELM models.

IMPORTANT - When to use Riemannian vs Standard Optimization:

For most HELM models, STANDARD AdamW is correct and recommended because:
- Model weights (linear layers, embeddings) are Euclidean parameters
- The manifold constraint on activations is maintained by forward pass
  operations (projection, exponential map, centroid computation)
- Weights transform vectors but are not themselves manifold points

Use RiemannianAdam ONLY when you have:
- Learnable anchor points that must stay on the manifold
- Hyperbolic embeddings where each embedding vector is a manifold point
- Other parameters that are actual points on the Lorentz manifold

Key differences from Euclidean Adam (when used):
1. Gradients are projected to tangent space at current point
2. Updates use exponential map instead of addition
3. Momentum is parallel transported between iterations

For mixed models (Euclidean + Manifold parameters):
- Use DualOptimizer wrapper with separate optimizers
- Mark manifold parameters with: param.manifold_point = True

Based on:
    - Riemannian Adaptive Optimization Methods (Bécigneul & Ganea, 2019)
    - geoopt library (https://github.com/geoopt/geoopt)
"""

import math
from typing import Dict, Iterable, Optional, Tuple, Union
import torch
from torch.optim.optimizer import Optimizer

from ..manifolds import Lorentz


class RiemannianAdam(Optimizer):
    """
    Riemannian Adam optimizer for Lorentz manifold parameters.

    Performs optimization using the exponential map and parallel transport
    to properly handle the curved geometry of hyperbolic space.

    Args:
        params: Iterable of parameters to optimize
        manifold: Lorentz manifold instance
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        stabilize: Stabilization frequency - project to manifold every N steps (default: None)
    """

    def __init__(
        self,
        params: Iterable,
        manifold: Lorentz,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        stabilize: Optional[int] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.manifold = manifold
        self.stabilize = stabilize
        self._step_count = 0

    def _is_lorentz_param(self, x: torch.Tensor) -> bool:
        """Check if parameter looks like a Lorentz vector (at least 2D with valid shape)."""
        if x.dim() < 1:
            return False
        # Lorentz vectors should have last dim > 1 (time + space)
        return x.shape[-1] > 1

    def _project_to_tangent(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Project gradient to tangent space at point x.

        For Lorentz manifold, the tangent space at x consists of vectors v
        such that ⟨x, v⟩_L = 0.

        Projection: v_tan = v + ⟨x, v⟩_L * x / c
        """
        if not self._is_lorentz_param(x):
            return grad  # Return as-is for non-Lorentz params
        inner = self.manifold.l_inner(x, grad, keepdim=True, dim=-1)
        return grad + inner * x / self.manifold.c

    def _expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map: move from x in direction v on the manifold.

        For Lorentz manifold:
            exp_x(v) = cosh(||v||_L / sqrt(c)) * x + sqrt(c) * sinh(||v||_L / sqrt(c)) * v / ||v||_L

        where ||v||_L = sqrt(⟨v, v⟩_L) is the Lorentz norm of tangent vector.

        For non-Lorentz parameters, falls back to Euclidean addition.
        """
        if not self._is_lorentz_param(x):
            return x + v  # Euclidean fallback

        c = self.manifold.c

        # Compute Lorentz norm of tangent vector
        # For tangent vectors, the Lorentz inner product is positive
        v_norm_sq = self.manifold.l_inner(v, v, keepdim=True, dim=-1)
        v_norm = torch.sqrt(v_norm_sq.clamp_min(1e-15))

        # Scaled norm
        v_norm_scaled = v_norm / torch.sqrt(c)

        # Exponential map
        cosh_term = torch.cosh(v_norm_scaled)
        sinh_term = torch.sinh(v_norm_scaled)

        # Avoid division by zero for small norms
        v_normalized = v / v_norm.clamp_min(1e-15)

        result = cosh_term * x + torch.sqrt(c) * sinh_term * v_normalized

        return result

    def _parallel_transport(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel transport vector v from tangent space at x to tangent space at y.

        For Lorentz manifold:
            PT_{x→y}(v) = v + ⟨y, v⟩_L / (c - ⟨x, y⟩_L) * (x + y)

        For non-Lorentz parameters, returns v unchanged.
        """
        if not self._is_lorentz_param(x):
            return v  # No transport needed for Euclidean params

        c = self.manifold.c

        inner_yv = self.manifold.l_inner(y, v, keepdim=True, dim=-1)
        inner_xy = self.manifold.l_inner(x, y, keepdim=True, dim=-1)

        denom = (c - inner_xy).clamp_min(1e-15)

        return v + (inner_yv / denom) * (x + y)

    def _project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point to Lorentz manifold.

        Ensures the point satisfies -x_0^2 + x_1^2 + ... + x_d^2 = -c

        For non-Lorentz parameters, returns x unchanged.
        """
        if not self._is_lorentz_param(x):
            return x  # No projection for Euclidean params

        # Extract space-like components
        x_space = x[..., 1:]

        # Recompute time component
        space_norm_sq = (x_space * x_space).sum(dim=-1, keepdim=True)
        x_time = torch.sqrt(self.manifold.c + space_norm_sq)

        return torch.cat([x_time, x_space], dim=-1)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (Riemannian version)
                if weight_decay != 0:
                    # Weight decay in tangent space
                    grad = grad + weight_decay * self._project_to_tangent(p, p)

                # Project gradient to tangent space
                grad = self._project_to_tangent(p, grad)

                # Get state
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                # For Riemannian params, use squared Lorentz norm; for Euclidean, use squared L2 norm
                if self._is_lorentz_param(p):
                    grad_sq = self.manifold.l_inner(grad, grad, keepdim=True, dim=-1)
                    exp_avg_sq.mul_(beta2).add_(grad_sq.expand_as(exp_avg_sq), alpha=1 - beta2)
                else:
                    # Euclidean: element-wise squared gradient
                    grad_sq = grad * grad
                    exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step size
                step_size = lr / bias_correction1

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Compute update direction (in tangent space)
                direction = -step_size * exp_avg / denom

                # Store old point for parallel transport
                old_p = p.clone()

                # Apply exponential map to get new point
                p.copy_(self._expmap(p, direction))

                # Parallel transport momentum to new tangent space
                exp_avg.copy_(self._parallel_transport(old_p, p, exp_avg))

        # Stabilize (project to manifold) periodically
        self._step_count += 1
        if self.stabilize is not None and self._step_count % self.stabilize == 0:
            self._stabilize_params()

        return loss

    def _stabilize_params(self):
        """Project all parameters to the manifold."""
        for group in self.param_groups:
            for p in group['params']:
                p.copy_(self._project_to_manifold(p))


class RiemannianSGD(Optimizer):
    """
    Riemannian SGD optimizer for Lorentz manifold parameters.

    Simpler than RiemannianAdam - just uses gradient descent with
    exponential map.

    Args:
        params: Iterable of parameters to optimize
        manifold: Lorentz manifold instance
        lr: Learning rate (default: 1e-2)
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (default: 0)
    """

    def __init__(
        self,
        params: Iterable,
        manifold: Lorentz,
        lr: float = 1e-2,
        momentum: float = 0,
        weight_decay: float = 0,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.manifold = manifold

    def _project_to_tangent(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Project gradient to tangent space."""
        inner = self.manifold.l_inner(x, grad, keepdim=True, dim=-1)
        return grad + inner * x / self.manifold.c

    def _expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map."""
        c = self.manifold.c
        v_norm_sq = self.manifold.l_inner(v, v, keepdim=True, dim=-1)
        v_norm = torch.sqrt(v_norm_sq.clamp_min(1e-15))
        v_norm_scaled = v_norm / torch.sqrt(c)
        v_normalized = v / v_norm.clamp_min(1e-15)
        return torch.cosh(v_norm_scaled) * x + torch.sqrt(c) * torch.sinh(v_norm_scaled) * v_normalized

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    grad = grad + weight_decay * p

                # Project to tangent space
                grad = self._project_to_tangent(p, grad)

                # Apply momentum if used
                state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                # Update using exponential map
                p.copy_(self._expmap(p, -lr * grad))

        return loss


class DualOptimizer:
    """
    Wrapper for using separate optimizers for Euclidean and Riemannian parameters.

    Useful for models that mix standard layers (embeddings, output) with
    hyperbolic layers (attention, MLP in Lorentz space).

    Args:
        euclidean_optimizer: Optimizer for Euclidean parameters (e.g., AdamW)
        riemannian_optimizer: Optimizer for Riemannian parameters (e.g., RiemannianAdam)
    """

    def __init__(
        self,
        euclidean_optimizer: Optimizer,
        riemannian_optimizer: Optimizer,
    ):
        self.euclidean_optimizer = euclidean_optimizer
        self.riemannian_optimizer = riemannian_optimizer

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for both optimizers."""
        self.euclidean_optimizer.zero_grad(set_to_none=set_to_none)
        self.riemannian_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """Step both optimizers."""
        loss = None
        if closure is not None:
            loss = self.euclidean_optimizer.step(closure)
            self.riemannian_optimizer.step()
        else:
            self.euclidean_optimizer.step()
            self.riemannian_optimizer.step()
        return loss

    def state_dict(self) -> Dict:
        """Return state dict for both optimizers."""
        return {
            'euclidean': self.euclidean_optimizer.state_dict(),
            'riemannian': self.riemannian_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict for both optimizers."""
        self.euclidean_optimizer.load_state_dict(state_dict['euclidean'])
        self.riemannian_optimizer.load_state_dict(state_dict['riemannian'])


def create_optimizer_for_lorentz_model(
    model: torch.nn.Module,
    manifold: Lorentz,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    use_riemannian: bool = False,
) -> Union[Optimizer, DualOptimizer]:
    """
    Create optimizer(s) for a Lorentz model.

    IMPORTANT: For most HELM models, standard AdamW works correctly because:
    - Weights (linear layers, embeddings) are Euclidean matrices/vectors
    - The manifold constraint is maintained by forward pass operations
      (projection, exponential map, centroid computation)

    Riemannian optimization is only needed for specialized cases where
    you have learnable anchor points that must stay on the manifold.

    Args:
        model: The model to optimize
        manifold: Lorentz manifold instance
        lr: Learning rate
        weight_decay: Weight decay
        betas: Adam beta parameters
        use_riemannian: Whether to use Riemannian optimizer for manifold params.
            Default False - use standard AdamW for everything.
            Set True only if model has learnable manifold anchor points.

    Returns:
        Optimizer or DualOptimizer
    """
    # For most cases, standard AdamW is correct and stable
    if not use_riemannian:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    # Separate parameters for Riemannian optimization
    # Only parameters with 'manifold_point' attribute set to True use Riemannian opt
    euclidean_params = []
    riemannian_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check for explicit manifold point marker
        if getattr(param, 'manifold_point', False):
            riemannian_params.append(param)
        else:
            euclidean_params.append(param)

    if not riemannian_params:
        # No manifold points found, use standard AdamW
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    # Create dual optimizer
    euclidean_opt = torch.optim.AdamW(
        euclidean_params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    ) if euclidean_params else None

    riemannian_opt = RiemannianAdam(
        riemannian_params,
        manifold=manifold,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    ) if riemannian_params else None

    if euclidean_opt is None:
        return riemannian_opt
    if riemannian_opt is None:
        return euclidean_opt

    return DualOptimizer(euclidean_opt, riemannian_opt)

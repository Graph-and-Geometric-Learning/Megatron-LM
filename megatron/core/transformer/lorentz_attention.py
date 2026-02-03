# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz attention for hyperbolic transformers.
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz Dot Product Attention.

In hyperbolic geometry, attention scores are computed using Lorentzian distance
instead of dot product, and value aggregation uses Lorentzian centroid.

Key differences from standard attention:
1. Attention scores: 2c + 2*cinner(Q, K)  (squared Lorentzian distance)
2. Value aggregation: Lorentzian centroid instead of weighted sum

Based on:
    - Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space
    - Fully Hyperbolic Neural Networks
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..manifolds import Lorentz, lorentzian_centroid, project_space_to_lorentz


class LorentzDotProductAttention(nn.Module):
    """
    Lorentz attention using hyperbolic distance for scoring.

    Attention scores: scores = 2c + 2*cinner(Q, K)
    This is the squared Lorentzian distance, suitable for attention.

    Value aggregation: output = lorentzian_centroid(V, softmax(scores))

    Args:
        manifold: Lorentz manifold instance
        config: Transformer config (for num_heads, hidden_size, etc.)
        layer_number: Layer index
        num_attention_heads: Number of attention heads
        attention_dropout: Dropout probability for attention weights
    """

    def __init__(
        self,
        manifold: Lorentz,
        num_attention_heads: int,
        hidden_size_per_head: int,
        attention_dropout: float = 0.0,
        layer_number: int = 1,
    ):
        super().__init__()
        self.manifold = manifold
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_head = hidden_size_per_head
        self.layer_number = layer_number

        # Learnable scale and bias for attention scores
        # Scale: similar to 1/sqrt(d) in standard attention
        self.scale = nn.Parameter(
            torch.tensor([math.sqrt(num_attention_heads * hidden_size_per_head)])
        )
        self.bias = nn.Parameter(torch.zeros(()))

        # Attention dropout
        self.attention_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else None
        self._logged_first_call = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: Query tensor, shape (batch, heads, seq_q, dim_per_head+1)
            key: Key tensor, shape (batch, heads, seq_k, dim_per_head+1)
            value: Value tensor, shape (batch, heads, seq_k, dim_per_head+1)
            attention_mask: Optional mask, shape (batch, 1, seq_q, seq_k) or similar
            output_attentions: If True, return attention weights

        Returns:
            output: Attention output, shape (batch, heads, seq_q, dim_per_head+1)
            attention_weights: Optional attention weights
        """
        # Log first call to verify Lorentz attention is being used
        if not self._logged_first_call:
            print(f"[LORENTZ] LorentzDotProductAttention forward (layer={self.layer_number}, heads={self.num_attention_heads}, c={self.manifold.c.item():.4f})")
            self._logged_first_call = True

        # Query, key, value should be full Lorentz vectors (with time coordinate)
        # Compute hyperbolic attention scores using Lorentzian inner product
        # cinner computes: -q₀k₀ + Σqᵢkᵢ for all pairs

        # scores[..., i, j] = 2c + 2*⟨Qᵢ, Kⱼ⟩ₗ = -d²(Qᵢ, Kⱼ)
        # This is the negative squared Lorentzian distance
        scores = self.manifold.attention_scores(
            query, key, scale=self.scale, bias=self.bias
        )

        # Apply attention mask (causal, padding, etc.)
        if attention_mask is not None:
            # Mask should be True where we want to mask out (set to -inf)
            scores = scores.masked_fill(attention_mask, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        if self.attention_dropout is not None and self.training:
            attention_weights = self.attention_dropout(attention_weights)

        # Aggregate values using Lorentzian centroid
        # This is the key difference from standard attention
        # Standard: output = attention_weights @ value
        # Hyperbolic: output = lorentzian_centroid(value, attention_weights)
        output = self.manifold.lorentzian_centroid(value, attention_weights)

        if output_attentions:
            return output, attention_weights
        return output, None


class LorentzCoreAttention(nn.Module):
    """
    Core attention computation for Lorentz geometry.

    This is a more complete attention module that handles:
    - QKV projection (with Lorentz linear layers)
    - Multi-head attention computation
    - Output projection

    Args:
        manifold: Lorentz manifold instance
        hidden_size: Model hidden size
        num_attention_heads: Number of attention heads
        attention_dropout: Dropout probability
        layer_number: Layer index
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        layer_number: int = 1,
    ):
        super().__init__()
        self.manifold = manifold
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_head = hidden_size // num_attention_heads

        # Core attention computation
        self.core_attention = LorentzDotProductAttention(
            manifold=manifold,
            num_attention_heads=num_attention_heads,
            hidden_size_per_head=self.hidden_size_per_head,
            attention_dropout=attention_dropout,
            layer_number=layer_number,
        )

        # QKV projections (space-like only, time reconstructed)
        # Note: These should use LorentzLinear or LorentzColumnParallelLinear
        # For simplicity, using standard linear here (user can replace with TP versions)
        self.query_proj = nn.Linear(hidden_size, num_attention_heads * self.hidden_size_per_head)
        self.key_proj = nn.Linear(hidden_size, num_attention_heads * self.hidden_size_per_head)
        self.value_proj = nn.Linear(hidden_size, num_attention_heads * self.hidden_size_per_head)

        # Output projection
        self.output_proj = nn.Linear(
            num_attention_heads * (self.hidden_size_per_head + 1),  # +1 for time
            hidden_size
        )

    def _project_to_lorentz(self, x: torch.Tensor) -> torch.Tensor:
        """Project space-like tensor to Lorentz manifold."""
        return project_space_to_lorentz(x, self.manifold.c)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            hidden_states: Input tensor, shape (batch, seq, hidden_size)
            attention_mask: Optional attention mask
            output_attentions: If True, return attention weights

        Returns:
            output: Attention output
            attention_weights: Optional attention weights
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V (space-like)
        query_space = self.query_proj(hidden_states)
        key_space = self.key_proj(hidden_states)
        value_space = self.value_proj(hidden_states)

        # Reshape to multi-head: (batch, seq, heads, dim_per_head)
        query_space = query_space.view(batch_size, seq_length, self.num_attention_heads, -1)
        key_space = key_space.view(batch_size, seq_length, self.num_attention_heads, -1)
        value_space = value_space.view(batch_size, seq_length, self.num_attention_heads, -1)

        # Transpose to (batch, heads, seq, dim_per_head)
        query_space = query_space.transpose(1, 2)
        key_space = key_space.transpose(1, 2)
        value_space = value_space.transpose(1, 2)

        # Project to Lorentz manifold (add time coordinate)
        query = self._project_to_lorentz(query_space)
        key = self._project_to_lorentz(key_space)
        value = self._project_to_lorentz(value_space)

        # Core attention computation
        context, attention_weights = self.core_attention(
            query, key, value, attention_mask, output_attentions
        )

        # Reshape back: (batch, heads, seq, dim+1) -> (batch, seq, heads*(dim+1))
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, -1)

        # Output projection (back to hidden_size)
        output = self.output_proj(context)

        return output, attention_weights


def apply_rotary_embeddings(
    x: torch.Tensor,
    freqs_complex: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to tensor.

    Works on space-like dimensions only.

    Args:
        x: Input tensor, shape (..., dim)
        freqs_complex: Complex frequencies for RoPE

    Returns:
        Tensor with rotary embeddings applied
    """
    # Reshape to complex pairs
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )

    # Apply rotation
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex

    # Convert back to real
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x)

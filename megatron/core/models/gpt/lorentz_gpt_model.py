# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Lorentz GPT model for hyperbolic transformers.
# Adapted from HELM (Hyperbolic Large Language Models).

"""
Lorentz GPT Model.

Extends GPTModel with hyperbolic geometry support.

Key differences from standard GPT:
1. Embeddings project to Lorentz manifold (d -> d+1 dimensions)
2. All hidden states are Lorentz vectors
3. Final logits computed from space-like dimensions only

Usage:
    from megatron.core.models.gpt.lorentz_gpt_model import LorentzGPTModel
    from megatron.core.models.gpt.lorentz_layer_specs import (
        LorentzHyperbolicConfig,
        get_lorentz_gpt_layer_spec,
    )

    hyperbolic_config = LorentzHyperbolicConfig(
        use_hyperbolic=True,
        curvature=1.0,
    )

    model = LorentzGPTModel(
        config=transformer_config,
        hyperbolic_config=hyperbolic_config,
        transformer_layer_spec=get_lorentz_gpt_layer_spec(config, hyperbolic_config),
        ...
    )
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.manifolds import Lorentz, project_space_to_lorentz
from megatron.core.transformer.lorentz_norm import LorentzRMSNorm

from .lorentz_layer_specs import LorentzHyperbolicConfig


class LorentzEmbedding(nn.Module):
    """
    Embedding layer that projects to Lorentz manifold.

    Standard embedding produces d-dimensional vectors.
    Lorentz embedding adds time coordinate to get (d+1)-dimensional Lorentz vectors.

    Args:
        manifold: Lorentz manifold instance
        num_embeddings: Vocabulary size
        embedding_dim: Space-like embedding dimension (output will be dim+1)
        padding_idx: Optional padding index
    """

    def __init__(
        self,
        manifold: Lorentz,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.manifold = manifold
        self.embedding_dim = embedding_dim

        # Standard embedding (produces space-like dimensions)
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs, shape (batch, seq)

        Returns:
            Lorentz embeddings, shape (batch, seq, embedding_dim+1)
        """
        # Get space-like embeddings
        x_space = self.embedding(input_ids)

        # Project to Lorentz manifold (add time coordinate)
        return project_space_to_lorentz(x_space, self.manifold.c)


class LorentzOutputLayer(nn.Module):
    """
    Output layer for Lorentz GPT.

    Projects from Lorentz space back to vocabulary logits.
    Uses only space-like dimensions for logit computation.

    Args:
        manifold: Lorentz manifold instance
        hidden_size: Space-like hidden dimension
        vocab_size: Vocabulary size
        bias: Whether to include bias
    """

    def __init__(
        self,
        manifold: Lorentz,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.manifold = manifold
        self.hidden_size = hidden_size

        # Linear projection from space-like dims to vocab
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Lorentz hidden states, shape (..., hidden_size+1)

        Returns:
            Logits, shape (..., vocab_size)
        """
        # Use only space-like dimensions for logits
        x_space = hidden_states[..., 1:]  # Remove time coordinate
        return self.output_layer(x_space)


class LorentzGPTModelMixin:
    """
    Mixin class to add hyperbolic support to GPT models.

    This mixin provides methods for:
    - Creating Lorentz manifold
    - Wrapping embeddings
    - Wrapping output layer
    - Final normalization

    Usage:
        class MyLorentzGPT(LorentzGPTModelMixin, GPTModel):
            def __init__(self, config, hyperbolic_config, ...):
                self.init_hyperbolic(hyperbolic_config)
                super().__init__(config, ...)
    """

    def init_hyperbolic(self, hyperbolic_config: LorentzHyperbolicConfig):
        """Initialize hyperbolic components."""
        self.hyperbolic_config = hyperbolic_config

        # Create manifold
        self.manifold = Lorentz(
            c=hyperbolic_config.curvature,
            learnable=hyperbolic_config.learnable_curvature,
        )

    def wrap_embedding(
        self,
        embedding: nn.Module,
        vocab_size: int,
        hidden_size: int,
    ) -> LorentzEmbedding:
        """
        Wrap standard embedding with Lorentz projection.

        Can either replace the embedding entirely or wrap it.
        """
        return LorentzEmbedding(
            manifold=self.manifold,
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )

    def wrap_output_layer(
        self,
        output_layer: nn.Module,
        hidden_size: int,
        vocab_size: int,
    ) -> LorentzOutputLayer:
        """Wrap standard output layer with Lorentz projection."""
        return LorentzOutputLayer(
            manifold=self.manifold,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

    def create_final_norm(self, hidden_size: int) -> LorentzRMSNorm:
        """Create final normalization layer."""
        return LorentzRMSNorm(
            manifold=self.manifold,
            dim=hidden_size,
        )


class SimpleLorentzGPT(nn.Module):
    """
    Simple Lorentz GPT model for testing.

    A minimal implementation that demonstrates the full pipeline:
    - Lorentz embedding
    - Lorentz transformer layers
    - Lorentz output layer

    For production use, extend the full GPTModel class instead.

    Args:
        config: TransformerConfig
        hyperbolic_config: LorentzHyperbolicConfig
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
    """

    def __init__(
        self,
        config,
        hyperbolic_config: LorentzHyperbolicConfig,
        vocab_size: int,
        max_seq_length: int = 2048,
    ):
        super().__init__()

        self.config = config
        self.hyperbolic_config = hyperbolic_config
        self.vocab_size = vocab_size
        self.hidden_size = config.hidden_size

        # Create manifold
        self.manifold = Lorentz(
            c=hyperbolic_config.curvature,
            learnable=hyperbolic_config.learnable_curvature,
        )

        # Embedding
        self.embedding = LorentzEmbedding(
            manifold=self.manifold,
            num_embeddings=vocab_size,
            embedding_dim=config.hidden_size,
        )

        # Transformer layers would go here
        # For a complete implementation, use get_lorentz_gpt_layer_spec()
        # and build_layers() from the transformer block

        # Final norm
        self.final_norm = LorentzRMSNorm(
            manifold=self.manifold,
            dim=config.hidden_size,
        )

        # Output projection
        self.output_layer = LorentzOutputLayer(
            manifold=self.manifold,
            hidden_size=config.hidden_size,
            vocab_size=vocab_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs

        Returns:
            Logits
        """
        # Embed to Lorentz space
        hidden_states = self.embedding(input_ids)

        # TODO: Pass through transformer layers
        # hidden_states = self.transformer(hidden_states, attention_mask)

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Output logits
        logits = self.output_layer(hidden_states)

        return logits


def create_lorentz_gpt(
    config,
    hyperbolic_config: LorentzHyperbolicConfig,
    vocab_size: int,
    max_seq_length: int = 2048,
) -> SimpleLorentzGPT:
    """
    Factory function to create a Lorentz GPT model.

    Args:
        config: TransformerConfig
        hyperbolic_config: LorentzHyperbolicConfig
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length

    Returns:
        LorentzGPT model
    """
    return SimpleLorentzGPT(
        config=config,
        hyperbolic_config=hyperbolic_config,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
    )

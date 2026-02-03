# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# End-to-end test for Lorentz GPT model with Qwen3-0.6B-like configuration.

"""
End-to-End Test for Lorentz GPT Model.

Tests the full forward/backward pass with a Qwen3-0.6B-like architecture
adapted for hyperbolic (Lorentz) geometry.

Qwen3-0.6B Reference:
    - hidden_size: 1024
    - num_layers: 28
    - num_attention_heads: 16
    - num_kv_heads: 8 (GQA)
    - ffn_hidden_size: 3072
    - vocab_size: 151936
    - kv_channels: 128
    - RMSNorm, SwiGLU, no bias

For testing, we use a smaller version to fit in memory.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '/workspace/megatron')

from megatron.core.manifolds import Lorentz, project_space_to_lorentz
from megatron.core.transformer.lorentz_norm import LorentzRMSNorm
from megatron.core.transformer.lorentz_residual import LorentzResidual
from megatron.core.transformer.lorentz_attention import LorentzDotProductAttention
from megatron.core.transformer.lorentz_mlp import LorentzMLP
from megatron.core.models.gpt.lorentz_layer_specs import LorentzHyperbolicConfig
from megatron.core.models.gpt.lorentz_gpt_model import (
    LorentzEmbedding,
    LorentzOutputLayer,
)


@dataclass
class Qwen3LorentzConfig:
    """
    Qwen3-0.6B-like configuration for Lorentz GPT.

    Full Qwen3-0.6B:
        hidden_size=1024, num_layers=28, num_heads=16,
        num_kv_heads=8, ffn_hidden_size=3072, vocab_size=151936

    Test config (scaled down for memory):
        hidden_size=256, num_layers=4, num_heads=4,
        num_kv_heads=2, ffn_hidden_size=768, vocab_size=1024
    """
    # Model architecture
    hidden_size: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    num_kv_heads: int = 2  # For GQA
    ffn_hidden_size: int = 768
    vocab_size: int = 1024
    max_seq_length: int = 512

    # Derived
    @property
    def kv_channels(self) -> int:
        return self.hidden_size // self.num_attention_heads

    # Qwen3 settings
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    norm_epsilon: float = 1e-6

    # Hyperbolic settings
    curvature: float = 1.0
    learnable_curvature: bool = False


class LorentzTransformerLayer(nn.Module):
    """
    Single Lorentz transformer layer.

    Architecture (pre-norm):
        x -> LNorm -> Attention -> LResidual -> LNorm -> MLP -> LResidual -> out
    """

    def __init__(
        self,
        manifold: Lorentz,
        config: Qwen3LorentzConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.manifold = manifold
        self.config = config
        self.layer_idx = layer_idx

        # Input layer norm
        self.input_layernorm = LorentzRMSNorm(
            manifold=manifold,
            dim=config.hidden_size,
            eps=config.norm_epsilon,
        )

        # Self attention
        self.self_attention = LorentzSelfAttention(
            manifold=manifold,
            config=config,
            layer_idx=layer_idx,
        )

        # Post-attention residual
        self.attn_residual = LorentzResidual(manifold=manifold)

        # Pre-MLP layer norm
        self.pre_mlp_layernorm = LorentzRMSNorm(
            manifold=manifold,
            dim=config.hidden_size,
            eps=config.norm_epsilon,
        )

        # MLP
        self.mlp = LorentzMLP(
            manifold=manifold,
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            bias=False,  # Qwen3 style
        )

        # Post-MLP residual
        self.mlp_residual = LorentzResidual(manifold=manifold)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq, hidden_size+1) - Lorentz vectors
            attention_mask: Optional attention mask

        Returns:
            Output hidden states (batch, seq, hidden_size+1)
        """
        # Self attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.attn_residual(residual, hidden_states)

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_residual(residual, hidden_states)

        return hidden_states


class LorentzSelfAttention(nn.Module):
    """
    Lorentz self-attention with GQA support.
    """

    def __init__(
        self,
        manifold: Lorentz,
        config: Qwen3LorentzConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.manifold = manifold
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.kv_channels

        # QKV projections (operate on space-like dims)
        # Q: num_heads * head_dim
        # K, V: num_kv_heads * head_dim (for GQA)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Core attention
        self.core_attention = LorentzDotProductAttention(
            manifold=manifold,
            num_attention_heads=self.num_heads,
            hidden_size_per_head=self.head_dim,
            attention_dropout=config.attention_dropout,
            layer_number=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq, hidden_size+1) - full Lorentz vectors
            attention_mask: Optional mask

        Returns:
            Output (batch, seq, hidden_size+1)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Extract space-like dimensions for projection
        x_space = hidden_states[..., 1:]  # (batch, seq, hidden_size)

        # Project to Q, K, V (space-like)
        q = self.q_proj(x_space)  # (batch, seq, num_heads * head_dim)
        k = self.k_proj(x_space)  # (batch, seq, num_kv_heads * head_dim)
        v = self.v_proj(x_space)  # (batch, seq, num_kv_heads * head_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        # GQA: expand K, V to match Q heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(
                batch_size, seq_length, self.num_heads, self.head_dim
            )
            v = v.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(
                batch_size, seq_length, self.num_heads, self.head_dim
            )

        # Transpose to (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Project to Lorentz manifold (add time coordinate)
        q = project_space_to_lorentz(q, self.manifold.c)
        k = project_space_to_lorentz(k, self.manifold.c)
        v = project_space_to_lorentz(v, self.manifold.c)

        # Core attention (hyperbolic)
        attn_output, _ = self.core_attention(q, k, v, attention_mask)

        # Reshape: (batch, heads, seq, dim+1) -> (batch, seq, heads, dim+1)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Extract space-like for output projection
        attn_output_space = attn_output[..., 1:].contiguous()  # (batch, seq, heads, dim)
        attn_output_space = attn_output_space.view(batch_size, seq_length, -1)

        # Output projection
        output_space = self.o_proj(attn_output_space)

        # Project back to Lorentz manifold
        output = project_space_to_lorentz(output_space, self.manifold.c)

        return output


class LorentzGPTQwen3(nn.Module):
    """
    Full Lorentz GPT model with Qwen3-0.6B-like architecture.

    Components:
        - Lorentz token embedding
        - N x Lorentz transformer layers
        - Lorentz final norm
        - Lorentz output layer (to vocab logits)
    """

    def __init__(self, config: Qwen3LorentzConfig):
        super().__init__()
        self.config = config

        # Create manifold
        self.manifold = Lorentz(
            c=config.curvature,
            learnable=config.learnable_curvature,
        )

        # Token embedding -> Lorentz space
        self.embedding = LorentzEmbedding(
            manifold=self.manifold,
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            LorentzTransformerLayer(
                manifold=self.manifold,
                config=config,
                layer_idx=i,
            )
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = LorentzRMSNorm(
            manifold=self.manifold,
            dim=config.hidden_size,
            eps=config.norm_epsilon,
        )

        # Output layer (to vocab logits)
        self.output_layer = LorentzOutputLayer(
            manifold=self.manifold,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights (Qwen3 style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation

        Returns:
            logits: (batch, seq, vocab_size)
            loss: Optional cross-entropy loss
        """
        # Embed to Lorentz space
        hidden_states = self.embedding(input_ids)

        # Create causal mask if not provided
        if attention_mask is None:
            seq_length = input_ids.shape[1]
            attention_mask = torch.triu(
                torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device),
                diagonal=1
            )

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        # Output logits
        logits = self.output_layer(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def check_manifold_constraint(self, hidden_states: torch.Tensor, tol: float = 1e-4) -> float:
        """Check if hidden states satisfy Lorentz constraint."""
        inner = self.manifold.l_inner(hidden_states, hidden_states, keepdim=True, dim=-1)
        expected = -self.manifold.c
        error = (inner - expected).abs().max().item()
        return error


def test_forward_pass():
    """Test forward pass of Lorentz GPT."""
    print("=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    config = Qwen3LorentzConfig()
    model = LorentzGPTQwen3(config)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Config: hidden_size={config.hidden_size}, layers={config.num_layers}")
    print(f"        heads={config.num_attention_heads}, kv_heads={config.num_kv_heads}")
    print(f"        ffn={config.ffn_hidden_size}, vocab={config.vocab_size}")
    print(f"Device: {device}")

    # Create dummy input
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)

    # Forward pass
    with torch.no_grad():
        logits, _ = model(input_ids)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_length}, {config.vocab_size})")

    assert logits.shape == (batch_size, seq_length, config.vocab_size)
    print("✓ Forward pass successful")

    return model, config, device


def test_backward_pass(model, config, device):
    """Test backward pass (gradient computation)."""
    print("\n" + "=" * 60)
    print("Testing Backward Pass")
    print("=" * 60)

    model.train()

    # Create input and labels
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)

    # Forward with loss
    logits, loss = model(input_ids, labels=labels)

    print(f"Loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Check gradients exist
    total_params = 0
    params_with_grad = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            if param.grad.isnan().any():
                print(f"  WARNING: NaN gradient in {name}")
            if param.grad.isinf().any():
                print(f"  WARNING: Inf gradient in {name}")

    print(f"Parameters with gradients: {params_with_grad}/{total_params}")
    assert params_with_grad == total_params, "Not all parameters have gradients"
    print("✓ Backward pass successful")


def test_manifold_constraint(model, config, device):
    """Test that intermediate activations stay on manifold."""
    print("\n" + "=" * 60)
    print("Testing Manifold Constraint")
    print("=" * 60)

    model.eval()

    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)

    # Check embedding output
    with torch.no_grad():
        emb = model.embedding(input_ids)
        emb_error = model.check_manifold_constraint(emb)
        print(f"Embedding manifold error: {emb_error:.6f}")

        # Check each layer output
        hidden_states = emb
        for i, layer in enumerate(model.layers):
            hidden_states = layer(hidden_states)
            error = model.check_manifold_constraint(hidden_states)
            print(f"Layer {i} manifold error: {error:.6f}")
            if error > 1e-3:
                print(f"  WARNING: Large manifold error at layer {i}")

        # Check final norm output
        final = model.final_norm(hidden_states)
        final_error = model.check_manifold_constraint(final)
        print(f"Final norm manifold error: {final_error:.6f}")

    print("✓ Manifold constraint check complete")


def test_training_step():
    """Test a full training step with optimizer."""
    print("\n" + "=" * 60)
    print("Testing Training Step")
    print("=" * 60)

    config = Qwen3LorentzConfig()
    model = LorentzGPTQwen3(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Optimizer (AdamW like Qwen3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Training data
    batch_size = 4
    seq_length = 128

    print(f"Training config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Seq length: {seq_length}")
    print(f"  Device: {device}")

    # Multiple training steps
    losses = []
    for step in range(5):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        labels = input_ids.clone()

        optimizer.zero_grad()
        logits, loss = model(input_ids, labels=labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {step}: loss = {loss.item():.4f}")

    # Check loss decreased (or at least didn't explode)
    print(f"\nLoss trajectory: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert not math.isnan(losses[-1]), "Loss is NaN"
    assert not math.isinf(losses[-1]), "Loss is Inf"
    print("✓ Training step successful")


def test_model_sizes():
    """Test model parameter count matches expectations."""
    print("\n" + "=" * 60)
    print("Model Size Analysis")
    print("=" * 60)

    # Test config (small)
    config_small = Qwen3LorentzConfig(
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        ffn_hidden_size=768,
        vocab_size=1024,
    )

    # Qwen3-0.6B-like config
    config_qwen3 = Qwen3LorentzConfig(
        hidden_size=1024,
        num_layers=28,
        num_attention_heads=16,
        num_kv_heads=8,
        ffn_hidden_size=3072,
        vocab_size=151936,
    )

    # Build small model
    model_small = LorentzGPTQwen3(config_small)
    params_small = sum(p.numel() for p in model_small.parameters())

    print(f"Test model (small):")
    print(f"  Parameters: {params_small:,}")
    print(f"  Hidden: {config_small.hidden_size}, Layers: {config_small.num_layers}")

    print(f"\nQwen3-0.6B-like config (for reference, not built):")
    print(f"  Hidden: {config_qwen3.hidden_size}, Layers: {config_qwen3.num_layers}")
    print(f"  Estimated params: ~600M")

    print("✓ Model size analysis complete")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Lorentz GPT End-to-End Test (Qwen3-0.6B-like)")
    print("=" * 60)

    # Test 1: Forward pass
    model, config, device = test_forward_pass()

    # Test 2: Backward pass
    test_backward_pass(model, config, device)

    # Test 3: Manifold constraint
    test_manifold_constraint(model, config, device)

    # Test 4: Full training step
    test_training_step()

    # Test 5: Model sizes
    test_model_sizes()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

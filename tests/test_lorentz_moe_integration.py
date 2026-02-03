#!/usr/bin/env python3
"""
Test script for Lorentz MoE integration with Megatron backend.

Tests:
1. LorentzTopKRouter - Routes using hyperbolic distance in Lorentz space
   - Inherits from TopKRouter (gets aux_loss, z_loss, expert_bias for free)
2. LorentzSequentialMLP - True Lorentz MLP per expert with curvature
   - Has DDP 'allreduce' attribute on parameters
3. LorentzSharedExpertMLP - True Lorentz MLP at global curvature
   - Has DDP 'allreduce' attribute on parameters

Key insight: All components now use TRUE Lorentz operations:
- Inputs are projected from Euclidean to Lorentz space
- Operations are performed using LorentzMLP with per-expert curvature
- Outputs are projected back to Euclidean for MoE layer compatibility

Run with:
    python tests/test_lorentz_moe_integration.py
"""

import os
import sys
from unittest.mock import MagicMock

# Add megatron to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def create_mock_transformer_config(
    hidden_size=64,
    num_experts=4,
    topk=2,
    ffn_hidden_size=128,
    hyperbolic_curvature=1.0,
    expert_curvature_min=0.1,
    expert_curvature_max=2.0,
    expert_model_parallel_size=1,
    moe_aux_loss_coeff=0.01,
    moe_z_loss_coeff=None,
):
    """Create a mock TransformerConfig for testing."""
    config = MagicMock()
    config.hidden_size = hidden_size
    config.ffn_hidden_size = ffn_hidden_size
    config.num_moe_experts = num_experts
    config.moe_router_topk = topk
    config.moe_ffn_hidden_size = ffn_hidden_size
    config.moe_router_load_balancing_type = "aux_loss"
    config.moe_aux_loss_coeff = moe_aux_loss_coeff
    config.moe_router_score_function = "softmax"
    config.moe_router_pre_softmax = False
    config.moe_router_topk_scaling_factor = None
    config.moe_router_enable_expert_bias = False
    config.moe_router_bias_update_rate = 0.001
    config.moe_router_dtype = None
    config.moe_router_num_groups = None
    config.moe_router_group_topk = None
    config.moe_input_jitter_eps = None
    config.moe_token_dropping = False
    config.moe_expert_capacity_factor = None
    config.add_bias_linear = False
    config.init_method = lambda x: torch.nn.init.normal_(x, std=0.02)
    config.perform_initialization = True
    config.tensor_model_parallel_size = 1
    config.sequence_parallel = False
    config.expert_model_parallel_size = expert_model_parallel_size
    config.moe_grouped_gemm = False
    config.moe_shared_expert_intermediate_size = ffn_hidden_size * 2
    config.moe_z_loss_coeff = moe_z_loss_coeff
    config.moe_router_fusion = False
    config.moe_router_force_load_balancing = False
    config.moe_enable_routing_replay = False
    config.moe_token_drop_policy = None
    config.moe_pad_expert_input_to_capacity = False
    config.calculate_per_token_loss = False
    config.num_layers = 12
    config.mtp_num_layers = None
    config.params_dtype = torch.float32
    config.gated_linear_unit = True
    config.activation_func = torch.nn.functional.silu
    config.bias_activation_fusion = False
    config.activation_func_fp8_input_store = False
    config.apply_residual_connection_post_layernorm = False
    config.normalization = 'RMSNorm'
    config.fp8 = False
    config.fp4 = False
    # Hyperbolic config
    config.hyperbolic_curvature = hyperbolic_curvature
    config.expert_curvature_min = expert_curvature_min
    config.expert_curvature_max = expert_curvature_max
    config.learnable_curvature = False
    return config


def create_mock_pg_collection():
    """Create a mock ProcessGroupCollection for testing distributed features."""
    pg_collection = MagicMock()
    # Create mock process groups that behave like torch.distributed groups
    mock_pg = MagicMock()
    mock_pg.size.return_value = 1
    mock_pg.rank.return_value = 0
    pg_collection.tp = mock_pg
    pg_collection.cp = mock_pg
    pg_collection.tp_cp = mock_pg
    pg_collection.tp_dp_cp = mock_pg
    return pg_collection


# =============================================================================
# Test 1: Import tests
# =============================================================================
print("=" * 60)
print("Test 1: Import tests")
print("=" * 60)

try:
    from megatron.core.transformer.moe.lorentz_router import LorentzTopKRouter
    from megatron.core.transformer.moe.router import TopKRouter
    print("  LorentzTopKRouter imported successfully")
    print("  TopKRouter imported successfully")
except Exception as e:
    print(f"  Failed to import routers: {e}")
    sys.exit(1)

try:
    from megatron.core.transformer.moe.lorentz_experts import (
        LorentzGroupedMLP,
        LorentzSequentialMLP,
        LorentzSharedExpertMLP,
    )
    print("  LorentzGroupedMLP imported successfully")
    print("  LorentzSequentialMLP imported successfully")
    print("  LorentzSharedExpertMLP imported successfully")
except Exception as e:
    print(f"  Failed to import Lorentz experts: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from megatron.core.manifolds import Lorentz, project_space_to_lorentz
    from megatron.core.transformer.lorentz_mlp import LorentzMLP
    print("  Lorentz manifold and LorentzMLP imported successfully")
except Exception as e:
    print(f"  Failed to import Lorentz components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("  All imports successful")


# =============================================================================
# Test 2: LorentzTopKRouter inherits from TopKRouter
# =============================================================================
print("\n" + "=" * 60)
print("Test 2: LorentzTopKRouter inherits from TopKRouter")
print("=" * 60)

try:
    # Verify inheritance
    assert issubclass(LorentzTopKRouter, TopKRouter), \
        "LorentzTopKRouter should inherit from TopKRouter"
    print("  ✓ LorentzTopKRouter is subclass of TopKRouter")

    # Create config and mock pg_collection for instantiation
    config = create_mock_transformer_config()
    pg_collection = create_mock_pg_collection()

    # Create router
    router = LorentzTopKRouter(config=config, pg_collection=pg_collection)

    # Verify it's an instance of both
    assert isinstance(router, LorentzTopKRouter), "Should be LorentzTopKRouter instance"
    assert isinstance(router, TopKRouter), "Should also be TopKRouter instance"
    print("  ✓ Router is instance of both LorentzTopKRouter and TopKRouter")

    # Verify inherited attributes exist
    assert hasattr(router, 'topk'), "Should have 'topk' from TopKRouter"
    assert hasattr(router, 'routing_type'), "Should have 'routing_type' from TopKRouter"
    assert hasattr(router, 'score_function'), "Should have 'score_function' from TopKRouter"
    print("  ✓ Router has inherited attributes: topk, routing_type, score_function")

    # Verify Lorentz-specific attributes
    assert hasattr(router, 'manifold'), "Should have 'manifold' for Lorentz ops"
    assert hasattr(router, 'expert_embeddings'), "Should have 'expert_embeddings'"
    print("  ✓ Router has Lorentz-specific attributes: manifold, expert_embeddings")

    # Verify inherited methods exist
    assert hasattr(router, 'routing'), "Should have 'routing' method from TopKRouter"
    assert hasattr(router, 'apply_z_loss'), "Should have 'apply_z_loss' from TopKRouter"
    assert hasattr(router, '_apply_aux_loss'), "Should have '_apply_aux_loss' from TopKRouter"
    assert hasattr(router, '_apply_expert_bias'), "Should have '_apply_expert_bias' from TopKRouter"
    print("  ✓ Router has inherited methods: routing, apply_z_loss, _apply_aux_loss, _apply_expert_bias")

    print("  LorentzTopKRouter inheritance test passed")

except Exception as e:
    print(f"  LorentzTopKRouter inheritance test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Test 3: Lorentz manifold operations
# =============================================================================
print("\n" + "=" * 60)
print("Test 3: Lorentz manifold operations")
print("=" * 60)

try:
    hidden_size = 64
    batch_size = 8
    c = 1.0

    # Create Lorentz manifold
    manifold = Lorentz(c=c, learnable=False)
    print(f"  Created manifold: {manifold}")

    # Test projection from Euclidean to Lorentz
    euclidean_input = torch.randn(batch_size, hidden_size)
    lorentz_output = project_space_to_lorentz(euclidean_input, manifold.c)

    print(f"  Euclidean input shape: {euclidean_input.shape}")
    print(f"  Lorentz output shape: {lorentz_output.shape}")

    assert lorentz_output.shape == (batch_size, hidden_size + 1), \
        f"Expected shape (batch, hidden+1), got {lorentz_output.shape}"

    # Verify manifold constraint: -t^2 + ||x||^2 = -c
    time = lorentz_output[:, 0]
    space = lorentz_output[:, 1:]
    constraint = -time**2 + (space**2).sum(dim=-1)
    expected = -c * torch.ones(batch_size)
    assert torch.allclose(constraint, expected, atol=1e-5), \
        f"Manifold constraint violated: {constraint} != {expected}"

    print(f"  Manifold constraint verified: -t^2 + ||x||^2 = -{c}")
    print("  Lorentz projection test passed")

except Exception as e:
    print(f"  Lorentz manifold test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Test 4: LorentzMLP forward and backward
# =============================================================================
print("\n" + "=" * 60)
print("Test 4: LorentzMLP forward and backward")
print("=" * 60)

try:
    hidden_size = 64
    ffn_hidden_size = 256
    batch_size = 8
    c = 1.0

    # Create LorentzMLP
    manifold = Lorentz(c=c, learnable=False)
    lorentz_mlp = LorentzMLP(
        manifold=manifold,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        bias=False,
    )

    print(f"  Created LorentzMLP (hidden={hidden_size}, ffn={ffn_hidden_size}, c={c})")

    # Create input in Lorentz space
    euclidean_input = torch.randn(batch_size, hidden_size, requires_grad=True)
    lorentz_input = project_space_to_lorentz(euclidean_input, manifold.c)

    print(f"  Input shape (Lorentz): {lorentz_input.shape}")

    # Forward pass
    lorentz_output = lorentz_mlp(lorentz_input)

    print(f"  Output shape (Lorentz): {lorentz_output.shape}")

    assert lorentz_output.shape == lorentz_input.shape, \
        f"Expected shape {lorentz_input.shape}, got {lorentz_output.shape}"

    # Verify output is on manifold
    time = lorentz_output[:, 0]
    space = lorentz_output[:, 1:]
    constraint = -time**2 + (space**2).sum(dim=-1)
    assert torch.allclose(constraint, -c * torch.ones(batch_size), atol=1e-4), \
        "Output not on manifold"
    print("  Output verified on manifold")

    # Backward pass
    loss = lorentz_output.sum()
    loss.backward()

    assert euclidean_input.grad is not None, "Gradients not computed"
    print(f"  Gradients computed: {euclidean_input.grad.shape}")
    print("  LorentzMLP forward/backward test passed")

except Exception as e:
    print(f"  LorentzMLP test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Test 5: LorentzSequentialMLP (True Lorentz per expert) + DDP attributes
# =============================================================================
print("\n" + "=" * 60)
print("Test 5: LorentzSequentialMLP (True Lorentz per expert) + DDP attributes")
print("=" * 60)

try:
    hidden_size = 64
    ffn_hidden_size = 128
    num_experts = 4
    batch_size = 8

    # Create config with expert_model_parallel_size=1 (no EP)
    config = create_mock_transformer_config(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_experts=num_experts,
        hyperbolic_curvature=1.0,
        expert_curvature_min=0.1,
        expert_curvature_max=2.0,
        expert_model_parallel_size=1,  # No EP, should have allreduce=True
    )

    # Create LorentzSequentialMLP
    expert_mlp = LorentzSequentialMLP(
        num_local_experts=num_experts,
        config=config,
        submodules=None,
        pg_collection=None,
    )

    print(f"  Created LorentzSequentialMLP (experts={num_experts})")
    print(f"  Expert curvatures: {expert_mlp.get_expert_curvatures()}")

    # Verify each expert has its own LorentzMLP
    assert len(expert_mlp.expert_mlps) == num_experts, \
        f"Expected {num_experts} expert MLPs, got {len(expert_mlp.expert_mlps)}"

    for i, mlp in enumerate(expert_mlp.expert_mlps):
        assert isinstance(mlp, LorentzMLP), f"Expert {i} is not LorentzMLP"
        print(f"    Expert {i}: c={mlp.manifold.c.item():.3f}")

    # Verify DDP 'allreduce' attribute on expert parameters
    params_with_allreduce = 0
    for expert_mlp_inner in expert_mlp.expert_mlps:
        for param in expert_mlp_inner.parameters():
            assert hasattr(param, 'allreduce'), \
                f"Parameter should have 'allreduce' attribute"
            assert param.allreduce == True, \
                f"With EP=1, allreduce should be True, got {param.allreduce}"
            params_with_allreduce += 1

    print(f"  ✓ All {params_with_allreduce} parameters have allreduce=True (EP=1)")

    # Test with expert_model_parallel_size > 1 (with EP)
    config_ep = create_mock_transformer_config(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_experts=num_experts,
        expert_model_parallel_size=2,  # EP enabled, should have allreduce=False
    )
    expert_mlp_ep = LorentzSequentialMLP(
        num_local_experts=num_experts,
        config=config_ep,
        submodules=None,
        pg_collection=None,
    )

    for expert_mlp_inner in expert_mlp_ep.expert_mlps:
        for param in expert_mlp_inner.parameters():
            assert param.allreduce == False, \
                f"With EP>1, allreduce should be False, got {param.allreduce}"

    print(f"  ✓ With EP>1, all parameters have allreduce=False")

    # Test forward pass with Euclidean input
    tokens_per_expert = torch.tensor([2, 2, 2, 2])
    total_tokens = tokens_per_expert.sum().item()
    euclidean_input = torch.randn(total_tokens, hidden_size, requires_grad=True)
    permuted_probs = torch.ones(total_tokens) / num_experts

    print(f"  Input shape (Euclidean): {euclidean_input.shape}")
    print(f"  Tokens per expert: {tokens_per_expert.tolist()}")

    # Forward pass
    output, bias = expert_mlp(euclidean_input, tokens_per_expert, permuted_probs)

    print(f"  Output shape (Euclidean): {output.shape}")

    assert output.shape == euclidean_input.shape, \
        f"Expected {euclidean_input.shape}, got {output.shape}"
    assert bias is None, "Expected bias to be None"

    # Backward pass
    loss = output.sum()
    loss.backward()

    assert euclidean_input.grad is not None, "Gradients not computed"
    print(f"  Gradients computed: {euclidean_input.grad.shape}")
    print("  LorentzSequentialMLP test passed")

except Exception as e:
    print(f"  LorentzSequentialMLP test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Test 6: LorentzSharedExpertMLP (True Lorentz at global curvature) + DDP
# =============================================================================
print("\n" + "=" * 60)
print("Test 6: LorentzSharedExpertMLP (True Lorentz at global curvature) + DDP")
print("=" * 60)

try:
    hidden_size = 64
    ffn_hidden_size = 256

    # Create config
    config = create_mock_transformer_config(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        hyperbolic_curvature=1.0,
        expert_model_parallel_size=1,  # No EP
    )

    # Create shared expert
    shared_expert = LorentzSharedExpertMLP(
        config=config,
        submodules=None,
        gate=False,
        pg_collection=None,
    )

    print(f"  Created LorentzSharedExpertMLP (c={shared_expert.global_curvature})")
    assert hasattr(shared_expert, 'lorentz_mlp'), "Shared expert should have lorentz_mlp"
    assert isinstance(shared_expert.lorentz_mlp, LorentzMLP), \
        "Shared expert should use LorentzMLP"

    # Verify DDP 'allreduce' attribute
    params_with_allreduce = 0
    for param in shared_expert.lorentz_mlp.parameters():
        assert hasattr(param, 'allreduce'), \
            "Parameter should have 'allreduce' attribute"
        assert param.allreduce == True, \
            f"With EP=1, allreduce should be True"
        params_with_allreduce += 1

    print(f"  ✓ All {params_with_allreduce} parameters have allreduce=True (EP=1)")

    # Test with EUCLIDEAN input
    batch_size = 4
    seq_len = 8
    euclidean_input = torch.randn(batch_size * seq_len, hidden_size, requires_grad=True)

    print(f"  Input shape (Euclidean): {euclidean_input.shape}")

    # Forward pass
    output = shared_expert(euclidean_input)

    print(f"  Output shape (Euclidean): {output.shape}")
    assert output.shape == euclidean_input.shape, \
        f"Expected {euclidean_input.shape}, got {output.shape}"

    # Backward pass
    loss = output.sum()
    loss.backward()

    assert euclidean_input.grad is not None, "Gradients not computed"
    print(f"  Gradients computed: {euclidean_input.grad.shape}")
    print("  LorentzSharedExpertMLP test passed")

except Exception as e:
    print(f"  LorentzSharedExpertMLP test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Test 7: LorentzSharedExpertMLP with gating + DDP
# =============================================================================
print("\n" + "=" * 60)
print("Test 7: LorentzSharedExpertMLP with gating + DDP")
print("=" * 60)

try:
    hidden_size = 64
    ffn_hidden_size = 256

    config = create_mock_transformer_config(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        hyperbolic_curvature=1.0,
        expert_model_parallel_size=1,
    )

    # Create shared expert WITH gating
    shared_expert_gated = LorentzSharedExpertMLP(
        config=config,
        submodules=None,
        gate=True,  # Enable gating
        pg_collection=None,
    )

    assert shared_expert_gated.gate_weight is not None, "Gate weight should exist"
    print(f"  Gate weight shape: {shared_expert_gated.gate_weight.shape}")

    # Verify gate_weight also has allreduce attribute
    assert hasattr(shared_expert_gated.gate_weight, 'allreduce'), \
        "Gate weight should have 'allreduce' attribute"
    print(f"  ✓ Gate weight has allreduce={shared_expert_gated.gate_weight.allreduce}")

    # Test with Euclidean input
    batch_size = 4
    seq_len = 8
    euclidean_input = torch.randn(batch_size * seq_len, hidden_size, requires_grad=True)

    # Forward pass
    output = shared_expert_gated(euclidean_input)

    assert output.shape == euclidean_input.shape
    print("  Forward pass with gating successful")

    # Backward pass
    loss = output.sum()
    loss.backward()

    assert euclidean_input.grad is not None
    print("  Backward pass with gating successful")
    print("  LorentzSharedExpertMLP with gating test passed")

except Exception as e:
    print(f"  LorentzSharedExpertMLP with gating test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Test 8: LorentzTopKRouter gating() with mock pg_collection
# =============================================================================
print("\n" + "=" * 60)
print("Test 8: LorentzTopKRouter gating() method")
print("=" * 60)

try:
    hidden_size = 64
    num_experts = 4
    topk = 2

    config = create_mock_transformer_config(
        hidden_size=hidden_size,
        num_experts=num_experts,
        topk=topk,
        hyperbolic_curvature=1.0,
    )
    pg_collection = create_mock_pg_collection()

    # Create router
    router = LorentzTopKRouter(
        config=config,
        pg_collection=pg_collection,
    )

    print(f"  Created LorentzTopKRouter (experts={num_experts}, topk={topk})")
    assert hasattr(router, 'manifold'), "Router should have manifold"
    assert hasattr(router, 'expert_embeddings'), "Router should have expert embeddings"
    print(f"  Expert embeddings shape: {router.expert_embeddings.shape}")
    assert router.expert_embeddings.shape == (num_experts, hidden_size + 1), \
        f"Expected ({num_experts}, {hidden_size+1}), got {router.expert_embeddings.shape}"

    # Test gating() directly (input already flattened)
    num_tokens = 32
    flat_input = torch.randn(num_tokens, hidden_size)

    print(f"  Input shape (flattened): {flat_input.shape}")

    # Call gating() directly
    logits = router.gating(flat_input)

    print(f"  Logits shape: {logits.shape}")
    assert logits.shape == (num_tokens, num_experts), \
        f"Expected ({num_tokens}, {num_experts}), got {logits.shape}"

    print("  ✓ gating() returns logits with correct shape")

    # Verify logits can be used for softmax
    probs = torch.softmax(logits, dim=-1)
    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(dim=-1), torch.ones(num_tokens), atol=1e-5)
    print("  ✓ Logits produce valid softmax probabilities")

    print("  LorentzTopKRouter gating() test passed")

except Exception as e:
    print(f"  LorentzTopKRouter gating() test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Test 9: Full MoE-like forward simulation
# =============================================================================
print("\n" + "=" * 60)
print("Test 9: Full MoE-like forward simulation")
print("=" * 60)

try:
    hidden_size = 64
    ffn_hidden_size = 256
    num_experts = 4
    topk = 2

    config = create_mock_transformer_config(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_experts=num_experts,
        topk=topk,
        hyperbolic_curvature=1.0,
        expert_curvature_min=0.1,
        expert_curvature_max=2.0,
    )

    # Simulate what MoE layer does:
    # 1. Router selects experts (using hyperbolic routing)
    # 2. Token dispatcher routes tokens
    # 3. Experts process tokens (using true Lorentz MLP)
    # 4. Token dispatcher combines outputs

    batch_size = 4
    seq_len = 8
    num_tokens = batch_size * seq_len

    # Input hidden states (Euclidean)
    hidden_states = torch.randn(num_tokens, hidden_size, requires_grad=True)

    print(f"  Hidden states shape: {hidden_states.shape}")

    # Create true Lorentz SequentialMLP
    expert_mlp = LorentzSequentialMLP(
        num_local_experts=num_experts,
        config=config,
        submodules=None,
        pg_collection=None,
    )

    # Simulate routing (simplified - random assignment)
    tokens_per_expert = torch.tensor([num_tokens // num_experts] * num_experts)
    permuted_probs = torch.ones(num_tokens) / topk

    print(f"  Tokens per expert: {tokens_per_expert.tolist()}")

    # Expert forward pass (true Lorentz operations)
    expert_output, _ = expert_mlp(hidden_states, tokens_per_expert, permuted_probs)

    print(f"  Expert output shape: {expert_output.shape}")

    # Shared expert forward pass (true Lorentz operations)
    shared_expert = LorentzSharedExpertMLP(
        config=config,
        submodules=None,
        gate=False,
        pg_collection=None,
    )

    shared_output = shared_expert(hidden_states)
    print(f"  Shared expert output shape: {shared_output.shape}")

    # Combine outputs
    final_output = expert_output + shared_output
    print(f"  Final output shape: {final_output.shape}")

    # Backward pass
    loss = final_output.sum()
    loss.backward()

    assert hidden_states.grad is not None
    print(f"  Gradients computed: {hidden_states.grad.shape}")

    # Check gradient norms are reasonable
    grad_norm = hidden_states.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0 and grad_norm < 1e6, \
        f"Gradient norm {grad_norm} is suspicious"

    print("  Full MoE simulation test passed")

except Exception as e:
    print(f"  Full MoE simulation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Test 10: Gradient flow through Lorentz projections
# =============================================================================
print("\n" + "=" * 60)
print("Test 10: Gradient flow through Lorentz projections")
print("=" * 60)

try:
    hidden_size = 64
    batch_size = 8
    c = 1.0

    manifold = Lorentz(c=c, learnable=False)

    # Create input with gradients
    euclidean_input = torch.randn(batch_size, hidden_size, requires_grad=True)

    # Project to Lorentz
    lorentz = project_space_to_lorentz(euclidean_input, manifold.c)

    # Compute some loss on Lorentz vector
    loss = lorentz.sum()
    loss.backward()

    assert euclidean_input.grad is not None, "Gradient should flow through projection"
    grad_norm = euclidean_input.grad.norm().item()
    print(f"  Gradient norm after projection: {grad_norm:.4f}")
    assert grad_norm > 0, "Gradient norm should be positive"

    # Test that extracting space dimensions preserves gradients
    euclidean_input2 = torch.randn(batch_size, hidden_size, requires_grad=True)
    lorentz2 = project_space_to_lorentz(euclidean_input2, manifold.c)
    space_back = lorentz2[..., 1:]  # Extract space dimensions
    loss2 = space_back.sum()
    loss2.backward()

    assert euclidean_input2.grad is not None, "Gradient should flow through round-trip"
    print(f"  Gradient after round-trip: {euclidean_input2.grad.norm().item():.4f}")

    print("  Gradient flow test passed")

except Exception as e:
    print(f"  Gradient flow test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
print("\nKey findings:")
print("  - LorentzTopKRouter INHERITS from TopKRouter")
print("    → Gets aux_loss, z_loss, expert_bias, input_jitter for FREE")
print("    → Only overrides gating() to use Lorentz inner product")
print("  - LorentzSequentialMLP uses TRUE Lorentz operations per expert")
print("    → Each expert has its own LorentzMLP with distinct curvature")
print("    → All parameters have 'allreduce' attribute for DDP")
print("  - LorentzSharedExpertMLP uses LorentzMLP at global curvature")
print("    → Parameters have 'allreduce' attribute for DDP")
print("  - All components preserve Euclidean I/O for MoE layer compatibility")
print("  - Gradients flow correctly through Lorentz projections")
print("  - Forward and backward passes work correctly")

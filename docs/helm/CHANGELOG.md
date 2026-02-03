# HELM Integration Changelog

Track progress of HELM (Hyperbolic Large Language Models) integration into Megatron-LM.

## [Unreleased]

### Phase 1: Core Manifold Infrastructure
- [x] Create `megatron/core/manifolds/` directory
- [x] Port `lorentz.py` - Lorentz manifold class
- [x] Port `lorentz_math.py` - Hyperbolic math operations
- [x] Unit tests for manifold operations (Docker verified)

### Phase 2: Tensor Parallel Lorentz Layers
- [x] Create `lorentz_layers.py` with TP support
- [x] `LorentzLinear` - Non-parallel Lorentz linear layer
- [x] `LorentzColumnParallelLinear` - Partitions output space-like dims
- [x] `LorentzRowParallelLinear` - Partitions input space-like dims, reduces output

### Phase 3: Transformer Components
- [x] `lorentz_norm.py` - LorentzRMSNorm, LorentzLayerNorm, LorentzActivation, LorentzDropout
- [x] `lorentz_residual.py` - LorentzResidual (LResNet), LorentzBiasDropoutAdd
- [x] `lorentz_attention.py` - LorentzDotProductAttention, LorentzCoreAttention
- [x] `lorentz_mlp.py` - LorentzMLP (SwiGLU), LorentzFeedForward, LorentzParallelMLP

### Phase 4: HELM-D Integration
- [x] Create `LorentzHyperbolicConfig` dataclass (separate from TransformerConfig)
- [x] Create `lorentz_layer_specs.py` - Layer specifications for hyperbolic GPT
- [x] Create `lorentz_gpt_model.py` - Lorentz embeddings, output layer, model wrapper
- [x] End-to-end training test (Qwen3-0.6B-like architecture)

### Phase 5: HELM-MiCE Integration
- [ ] `lorentz_mla.py` - Multi-head Latent Attention (future)
- [x] `lorentz_moe_layer.py` - Mixture of Curvature Experts
- [x] `lorentz_router.py` - Curvature-aware router
- [x] `lorentz_experts.py` - Variable curvature experts

### Phase 6: Optimizers & Training (Future)
- [ ] `riemannian_adam.py` - Riemannian optimizer
- [ ] Training script integration

---

## Progress Log

### 2026-02-03
- Created feature branch `feature/helm-hyperbolic-integration`
- Set up documentation structure in `docs/helm/`
- Created integration plan and task tracking
- **Phase 1 Complete**: Ported Lorentz manifold infrastructure
  - `megatron/core/manifolds/__init__.py` - Module exports
  - `megatron/core/manifolds/lorentz.py` - Lorentz manifold class (~300 lines)
  - `megatron/core/manifolds/lorentz_math.py` - Hyperbolic math ops (~450 lines)
  - Verified: projection, inner product, attention scores, centroid all working
- **Phase 2 Complete**: Created TP-compatible Lorentz linear layers
  - `megatron/core/tensor_parallel/lorentz_layers.py` (~400 lines)
  - `LorentzLinear`: Non-parallel base layer
  - `LorentzColumnParallelLinear`: Output partitioned across TP ranks
  - `LorentzRowParallelLinear`: Input partitioned, output reduced
  - Key design: Time coord never partitioned, reconstructed after comm ops
- **Phase 3 Complete**: Ported transformer components
  - `megatron/core/transformer/lorentz_norm.py` - Normalization layers
  - `megatron/core/transformer/lorentz_residual.py` - Residual connections
  - `megatron/core/transformer/lorentz_attention.py` - Hyperbolic attention
  - `megatron/core/transformer/lorentz_mlp.py` - MLP with SwiGLU
  - All outputs verified on manifold (⟨x,x⟩ₗ = -c)
- **Phase 4 In Progress**: HELM-D integration
  - `megatron/core/models/gpt/lorentz_layer_specs.py` - Layer specifications
    - `LorentzHyperbolicConfig`: Dataclass for hyperbolic settings (curvature, model_type, etc.)
    - `get_lorentz_gpt_layer_spec()`: Factory for dense/MiCE layer specs
    - `get_lorentz_gpt_layer_spec_dense()`: Full dense hyperbolic layer spec
    - `LorentzLayerSpecProvider`: Provider class for modular spec creation
  - `megatron/core/models/gpt/lorentz_gpt_model.py` - Model components
    - `LorentzEmbedding`: Projects token embeddings to Lorentz manifold
    - `LorentzOutputLayer`: Projects from Lorentz space to vocabulary logits
    - `LorentzGPTModelMixin`: Mixin for adding hyperbolic support to GPT models
    - `SimpleLorentzGPT`: Minimal test model demonstrating full pipeline
  - Verified: All components work, embeddings produce valid Lorentz vectors
- **Phase 4 Complete**: End-to-end test with Qwen3-0.6B-like architecture
  - `tests/unit_tests/models/test_lorentz_gpt_e2e.py` - Full E2E test suite
  - `LorentzGPTQwen3`: Complete model with embedding, transformer layers, output
  - `LorentzSelfAttention`: Self-attention with GQA support
  - `LorentzTransformerLayer`: Full transformer layer (pre-norm style)
  - Test results:
    - Forward pass: ✓ (correct output shape)
    - Backward pass: ✓ (all 47 params have gradients, no NaN/Inf)
    - Manifold constraint: ✓ (max error < 1e-4 at all layers)
    - Training step: ✓ (5 steps, loss stable ~6.98)
  - Bug fix: Added `.contiguous()` before tensor view in attention (see DEBUG_AND_FIXES.md)
- **Phase 5 In Progress**: HELM-MiCE (Mixture of Curvature Experts)
  - `megatron/core/transformer/moe/lorentz_router.py` - Curvature-aware routing
    - `LorentzRouter`: Routes on space-like dimensions, top-k selection
    - `LorentzAuxLossRouter`: With auxiliary loss for load balancing
  - `megatron/core/transformer/moe/lorentz_experts.py` - Variable curvature experts
    - `LorentzExpert`: Single expert with curvature transfer
    - `LorentzExpertGroup`: Group with curvatures distributed across range
    - `LorentzGroupedExperts`: Batched expert computation
    - `LorentzSharedExpert`: Always-active shared expert
  - `megatron/core/transformer/moe/lorentz_moe_layer.py` - Full MoE layer
    - `LorentzMoEConfig`: Configuration dataclass
    - `LorentzMoE`: Complete layer with router, experts, shared experts
    - `LorentzMoEBlock`: MoE block with norm and residual
  - Key features:
    - Each expert operates in its own curvature (0.1 to 2.0 range)
    - Curvature transfer: `x * sqrt(c_expert / c_input)`
    - Load balancing via auxiliary loss or bias updates
  - Verified: Forward, backward pass work; outputs on manifold

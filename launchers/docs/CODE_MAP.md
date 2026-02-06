# HELM Hyperbolic Megatron-LM: Code Mapping Guide

This document maps hyperbolic/Lorentz mathematical concepts to their corresponding code locations in the Megatron-LM fork.

## Quick Reference Table

| Concept | File | Key Functions/Classes |
|---------|------|----------------------|
| Lorentz Manifold | `megatron/core/manifolds/lorentz.py` | `Lorentz` class |
| Hyperbolic Math Ops | `megatron/core/manifolds/lorentz_math.py` | All low-level operations |
| Hyperbolic Attention | `megatron/core/transformer/lorentz_attention.py` | `LorentzCoreAttention` |
| Hyperbolic MLP | `megatron/core/transformer/lorentz_mlp.py` | `LorentzMLP`, `LorentzParallelMLP` |
| Hyperbolic Normalization | `megatron/core/transformer/lorentz_norm.py` | `LorentzRMSNorm`, `LorentzLayerNorm` |
| Hyperbolic Residual | `megatron/core/transformer/lorentz_residual.py` | `LorentzResidual` |
| MoE Router | `megatron/core/transformer/moe/lorentz_router.py` | `LorentzTopKRouter` |
| MoE Experts | `megatron/core/transformer/moe/lorentz_experts.py` | `LorentzGroupedMLP`, `LorentzTEGroupedMLP` |
| Tensor Parallel Layers | `megatron/core/tensor_parallel/lorentz_layers.py` | `LorentzLinear`, `LorentzColumnParallelLinear` |
| Riemannian Optimizer | `megatron/core/optimizer/riemannian_adam.py` | `RiemannianAdam` |
| GPT Model | `megatron/core/models/gpt/lorentz_gpt_model.py` | `LorentzGPTModel` |
| Layer Specs (Dense) | `megatron/core/models/gpt/lorentz_layer_specs.py` | `get_lorentz_gpt_layer_spec()` |
| Layer Specs (MoE) | `megatron/core/models/gpt/lorentz_moe_layer_specs.py` | `get_lorentz_moe_layer_spec()` |
| Training Script | `pretrain_lorentz_gpt.py` | Main entry point |

---

## 1. Manifold Mathematics

### Core Lorentz Operations
**File:** `megatron/core/manifolds/lorentz_math.py`

| Mathematical Operation | Function | Line |
|----------------------|----------|------|
| Minkowski inner product: `⟨u,v⟩ₗ = -u₀v₀ + Σuᵢvᵢ` | `lorentz_inner()` | 118-144 |
| Batched inner product (for attention) | `lorentz_inner_batch()` | 147-164 |
| Lorentzian norm: `\|\|u\|\|ₗ = √⟨u,u⟩ₗ` | `lorentz_norm()` | 167-180 |
| Project to manifold: `x₀ = √(c + \|\|x_{1:d}\|\|²)` | `project_to_lorentz()` | 187-206 |
| Project space coords to full vector | `project_space_to_lorentz()` | 209-221 |
| Project to tangent space: `Πₓ(v) = v + ⟨x,v⟩ₗx/c` | `project_to_tangent()` | 224-245 |
| Exponential map from origin | `expmap0()` | 252-274 |
| Logarithmic map to origin | `logmap0()` | 277-304 |
| Exponential map from point x | `expmap()` | 307-326 |
| Logarithmic map from x to y | `logmap()` | 329-352 |
| Squared distance: `d²(x,y) = -2(c + ⟨x,y⟩ₗ)` | `lorentz_distance_squared()` | 359-376 |
| Geodesic distance: `d(x,y) = √c·arcosh(-⟨x,y⟩ₗ/c)` | `induced_distance()` | 379-395 |
| Lorentzian centroid (Fréchet mean) | `lorentzian_centroid()` | 402-437 |
| Parallel transport x→y | `parallel_transport()` | 444-467 |
| Parallel transport origin→y | `parallel_transport0()` | 470-499 |
| Lorentz → Poincaré conversion | `lorentz_to_poincare()` | 506-521 |
| Poincaré → Lorentz conversion | `poincare_to_lorentz()` | 524-540 |

### Numerical Stability
**File:** `megatron/core/manifolds/lorentz_math.py`

| Concept | Function/Constant | Line |
|---------|------------------|------|
| Epsilon per dtype | `EPS`, `get_eps()` | 24, 29-31 |
| Max/min norm clamping | `MAX_NORM`, `MIN_NORM` | 25-26 |
| Leaky gradient clamp | `LeakyClamp` | 38-57 |
| Safe sqrt | `safe_sqrt()` | 60-62 |
| Stable arcosh | `Arcosh`, `arcosh()` | 65-85 |
| Stable arctanh | `Artanh`, `artanh()` | 93-111 |

### Manifold Class (High-Level API)
**File:** `megatron/core/manifolds/lorentz.py`

| Method | Purpose | Line |
|--------|---------|------|
| `l_inner()` | Minkowski inner product | 86-107 |
| `cinner()` | Batched inner product | 109-122 |
| `lorentzian_distance()` | Squared distance | 124-137 |
| `induced_distance()` | Geodesic distance | 139-152 |
| `projx()` | Project point to manifold | 158-171 |
| `project_space()` | Space coords → full Lorentz | 173-186 |
| `proju()` | Project to tangent space | 188-200 |
| `expmap0()` / `expmap()` | Exponential maps | 206-248 |
| `logmap0()` / `logmap()` | Logarithmic maps | 221-262 |
| `lorentzian_centroid()` | Attention aggregation | 268-288 |
| `attention_scores()` | Hyperbolic attention scores | 290-320 |
| `ptransp()` / `ptransp0()` | Parallel transport | 326-353 |
| `mobius_add()` | Möbius addition | 476-489 |
| `mobius_matvec()` | Möbius matrix-vector | 491-504 |

---

## 2. Transformer Components

### Hyperbolic Attention
**File:** `megatron/core/transformer/lorentz_attention.py`

| Component | Class/Function | Line | Notes |
|-----------|---------------|------|-------|
| Core attention mechanism | `LorentzCoreAttention` | ~50+ | Main attention class |
| Attention score computation | Uses `manifold.attention_scores()` | | `2c + 2⟨Q,K⟩ₗ` |
| Value aggregation | Uses `manifold.lorentzian_centroid()` | | Fréchet mean, not sum |
| QKV projections | Lorentz-aware linear layers | | Projects to d+1 dims |

### Hyperbolic MLP (Feed-Forward)
**File:** `megatron/core/transformer/lorentz_mlp.py`

| Component | Class | Line | Notes |
|-----------|-------|------|-------|
| Single-GPU MLP | `LorentzMLP` | 35-130 | SwiGLU in hyperbolic space |
| Simple FFN (non-gated) | `LorentzFeedForward` | 133-188 | GELU activation |
| Tensor-parallel MLP | `LorentzParallelMLP` | 191-286 | For distributed training |

**MLP Data Flow:**
```
Input (d+1) → W1 gate (space) → SiLU → ×
            → W3 up (space)   ────────→ Element-wise multiply
                                        ↓
                              Reconstruct time (project_space_to_lorentz)
                                        ↓
                              W2 down → Output (d+1)
```

### Hyperbolic Normalization
**File:** `megatron/core/transformer/lorentz_norm.py`

| Component | Class | Line | Notes |
|-----------|-------|------|-------|
| RMS Normalization | `LorentzRMSNorm` | ~74-87 | Normalizes space dims only |
| Layer Normalization | `LorentzLayerNorm` | ~133-146 | Normalizes space dims only |
| Unit Normalization | `LorentzNormalization` | ~167-202 | Projects to unit sphere in space |

**Normalization Strategy:** All normalizations operate on space-like dimensions (x₁...xₐ), then reconstruct time coordinate via `project_space_to_lorentz()`.

### Hyperbolic Residual Connection
**File:** `megatron/core/transformer/lorentz_residual.py`

| Component | Class | Line | Notes |
|-----------|-------|------|-------|
| Basic residual | `LorentzResidual` | 27-135 | `y = normalize(x + w*f(x))` |
| Bias-dropout-add fusion | `LorentzBiasDropoutAdd` | 138-199 | Combined operation |

**Current Implementation:** Uses Euclidean addition + projection (see known issues).

---

## 3. Mixture of Experts (MoE)

### MoE Router
**File:** `megatron/core/transformer/moe/lorentz_router.py`

| Component | Class/Method | Line | Notes |
|-----------|-------------|------|-------|
| Hyperbolic router | `LorentzTopKRouter` | 51-172 | Inherits from `TopKRouter` |
| Expert embeddings | `self.expert_embeddings` | 93-95 | Learnable Lorentz points |
| Lorentz inner product | `_lorentz_inner_batch()` | 100-120 | For routing scores |
| Routing scores | `gating()` | 134-172 | `2c + 2⟨input, expert⟩ₗ` |

### MoE Experts
**File:** `megatron/core/transformer/moe/lorentz_experts.py`

| Component | Class | Line | Notes |
|-----------|-------|------|-------|
| Grouped MLP (basic) | `LorentzGroupedMLP` | 73-247 | Per-expert curvatures |
| TE-accelerated experts | `LorentzTEGroupedMLP` | 249-391 | Uses Transformer Engine |
| Sequential experts | `LorentzSequentialMLP` | 394-528 | Fallback implementation |

**Expert Implementations:**
- `LorentzGroupedMLP`: Curvature scaling only (incomplete hyperbolic geometry)
- `LorentzTEGroupedMLP`: Full tangent space operations (recommended)
- `LorentzSequentialMLP`: Per-expert `LorentzMLP` instances (slow but correct)

---

## 4. Tensor Parallelism

### Lorentz-Aware Parallel Layers
**File:** `megatron/core/tensor_parallel/lorentz_layers.py`

| Component | Class | Line | Notes |
|-----------|-------|------|-------|
| Basic linear | `LorentzLinear` | ~70+ | Single-GPU Lorentz linear |
| Column parallel | `LorentzColumnParallelLinear` | ~200+ | Splits output dimension |
| Row parallel | `LorentzRowParallelLinear` | ~400+ | Splits input dimension |

**Key Design:** Time coordinate (x₀) is NEVER partitioned. Only space dimensions are distributed.

---

## 5. Optimizer

### Riemannian Adam
**File:** `megatron/core/optimizer/riemannian_adam.py`

| Component | Class/Function | Line | Notes |
|-----------|---------------|------|-------|
| Riemannian Adam optimizer | `RiemannianAdam` | 47-290 | For manifold parameters |
| Tangent space projection | Within `step()` | 104-116 | `grad_tan = grad + ⟨x,grad⟩ₗx/c` |
| Manifold exponential map | Within `step()` | 118-151 | Update via `expmap` |
| Optimizer factory | `create_optimizer_for_lorentz_model()` | ~420+ | Creates appropriate optimizer |

**Usage:** Parameters with `param.manifold_point = True` get Riemannian updates.

---

## 6. Model Architecture

### Lorentz GPT Model
**File:** `megatron/core/models/gpt/lorentz_gpt_model.py`

| Component | Class/Method | Line | Notes |
|-----------|-------------|------|-------|
| Main model class | `LorentzGPTModel` | ~40+ | Wraps transformer layers |
| Embedding projection | `forward()` | ~91 | Projects embeddings to Lorentz |
| Output layer | `forward()` | ~133 | Extracts space dims for logits |

### Layer Specifications
**Dense Model:** `megatron/core/models/gpt/lorentz_layer_specs.py`
**MoE Model:** `megatron/core/models/gpt/lorentz_moe_layer_specs.py`

| Function | File | Notes |
|----------|------|-------|
| `get_lorentz_gpt_layer_spec()` | lorentz_layer_specs.py | Dense transformer layer |
| `get_lorentz_moe_layer_spec()` | lorentz_moe_layer_specs.py | MoE transformer layer |

---

## 7. Training & Configuration

### Training Script
**File:** `pretrain_lorentz_gpt.py`

Main entry point for training Lorentz GPT models.

### Configuration Options
**File:** `megatron/core/transformer/transformer_config.py`

| Config Option | Purpose | Line |
|--------------|---------|------|
| `hyperbolic_curvature` | Manifold curvature c | Added |
| `learnable_curvature` | Make curvature learnable | Added |
| `use_lorentz_attention` | Enable hyperbolic attention | Added |
| `use_lorentz_mlp` | Enable hyperbolic MLP | Added |

---

## 8. Tests

| Test File | Purpose |
|-----------|---------|
| `tests/test_lorentz_moe_integration.py` | MoE integration tests |
| `tests/unit_tests/models/test_lorentz_gpt_e2e.py` | End-to-end GPT tests |
| `tests/run_lorentz_moe_test.sh` | Test runner script |

---

## 9. Directory Structure

```
megatron/core/
├── manifolds/
│   ├── __init__.py          # Exports Lorentz, project_space_to_lorentz
│   ├── lorentz.py           # Lorentz manifold class
│   └── lorentz_math.py      # Low-level math operations
├── transformer/
│   ├── lorentz_attention.py # Hyperbolic attention
│   ├── lorentz_mlp.py       # Hyperbolic MLP
│   ├── lorentz_norm.py      # Hyperbolic normalization
│   ├── lorentz_residual.py  # Hyperbolic residual connections
│   └── moe/
│       ├── lorentz_router.py   # Hyperbolic MoE router
│       └── lorentz_experts.py  # Hyperbolic MoE experts
├── tensor_parallel/
│   └── lorentz_layers.py    # Tensor-parallel Lorentz layers
├── optimizer/
│   └── riemannian_adam.py   # Riemannian optimizer
└── models/gpt/
    ├── lorentz_gpt_model.py      # Lorentz GPT model
    ├── lorentz_layer_specs.py    # Dense layer specs
    └── lorentz_moe_layer_specs.py # MoE layer specs

launchers/training/
├── lorentz/                 # Lorentz model training scripts
│   ├── base/               # Base training templates
│   ├── multi-node/         # Multi-node training
│   └── single-node/        # Single-node training
└── standard/               # Standard (non-hyperbolic) MoE scripts

docs/helm/
├── README.md               # Overview
├── CODE_MAP.md            # This file
├── CHANGELOG.md           # Change history
└── DEBUG_AND_FIXES.md     # Known issues and fixes
```

---

## 10. Common Modification Scenarios

### "I want to change how attention scores are computed"
→ `megatron/core/manifolds/lorentz.py:290-320` (`attention_scores()`)
→ `megatron/core/transformer/lorentz_attention.py`

### "I want to change the distance metric"
→ `megatron/core/manifolds/lorentz_math.py:359-395`

### "I want to modify how values are aggregated in attention"
→ `megatron/core/manifolds/lorentz_math.py:402-437` (`lorentzian_centroid()`)

### "I want to change the MoE routing mechanism"
→ `megatron/core/transformer/moe/lorentz_router.py:134-172` (`gating()`)

### "I want to add a new expert type"
→ `megatron/core/transformer/moe/lorentz_experts.py`

### "I want to change curvature handling"
→ `megatron/core/manifolds/lorentz.py:73-76` (curvature property)
→ Expert-specific: `megatron/core/transformer/moe/lorentz_experts.py`

### "I want to fix numerical stability issues"
→ `megatron/core/manifolds/lorentz_math.py:24-111` (stability functions)

### "I want to change how residual connections work"
→ `megatron/core/transformer/lorentz_residual.py:83-125`

### "I want to modify the optimizer for manifold parameters"
→ `megatron/core/optimizer/riemannian_adam.py`

---

## 11. Known Issues (See DEBUG_AND_FIXES.md)

| Issue | Location | Severity |
|-------|----------|----------|
| Euclidean addition in residual | `lorentz_residual.py:104` | CRITICAL |
| `poincare_to_lorentz` formula | `lorentz_math.py:539` | CRITICAL |
| Missing `manifold_point=True` | `lorentz_router.py:95` | HIGH |
| `logmap0` numerator | `lorentz_math.py:301` | HIGH |
| TP time reconstruction | `lorentz_mlp.py:281` | MEDIUM |
| Centroid sign handling | `lorentz_math.py:437` | MEDIUM |

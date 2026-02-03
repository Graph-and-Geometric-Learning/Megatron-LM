# HELM Integration - Debug and Fixes Log

This document tracks bugs encountered during HELM integration and their fixes.

---

## Issue #1: Tensor View Incompatibility in Attention Output

**Date:** 2026-02-03
**File:** `tests/unit_tests/models/test_lorentz_gpt_e2e.py`
**Severity:** Runtime Error

### Symptom

```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

### Root Cause

In the `LorentzSelfAttention.forward()` method, after extracting space-like dimensions from the attention output, the tensor was non-contiguous due to slicing:

```python
# This creates a non-contiguous view
attn_output_space = attn_output[..., 1:]  # Slicing makes it non-contiguous
attn_output_space = attn_output_space.view(batch_size, seq_length, -1)  # FAILS
```

When you slice a tensor along the last dimension (`[..., 1:]`), PyTorch creates a view that is not contiguous in memory. The `view()` operation requires the tensor to be contiguous.

### Fix

Add `.contiguous()` before calling `view()`:

```python
attn_output_space = attn_output[..., 1:].contiguous()  # Make contiguous first
attn_output_space = attn_output_space.view(batch_size, seq_length, -1)  # Now works
```

### Lesson Learned

When working with Lorentz vectors where we frequently slice off the time coordinate (`x[..., 1:]`), always ensure contiguity before reshaping. This is a common pattern in our codebase:

```python
# Pattern: Extract space-like dims and reshape
x_space = lorentz_tensor[..., 1:].contiguous()
x_reshaped = x_space.view(new_shape)
```

---

## Test Results Summary (2026-02-03)

After fixing Issue #1, all end-to-end tests pass:

| Test | Status | Notes |
|------|--------|-------|
| Forward Pass | ✓ PASS | Output shape correct (batch, seq, vocab) |
| Backward Pass | ✓ PASS | All 47 parameters have gradients, no NaN/Inf |
| Manifold Constraint | ✓ PASS | Max error < 1e-4 at all layers |
| Training Step | ✓ PASS | 5 steps completed, loss stable (~6.98) |
| Model Size | ✓ PASS | 3.68M params for test config |

### Manifold Constraint Details

All intermediate activations remain on the Lorentz manifold (⟨x,x⟩ₗ = -c):

| Component | Max Error |
|-----------|-----------|
| Embedding | 0.000000 |
| Layer 0 | 0.000002 |
| Layer 1 | 0.000001 |
| Layer 2 | 0.000002 |
| Layer 3 | 0.000001 |
| Final Norm | 0.000061 |

The slightly higher error at final norm is expected due to accumulated numerical precision loss, but still well within acceptable tolerance (< 1e-4).

---

## Issue #2: Double Unsqueeze in MoE Expert Weighting

**Date:** 2026-02-03
**Files:** `lorentz_experts.py`, `lorentz_moe_layer.py`
**Severity:** Runtime Error

### Symptom

```
RuntimeError: shape '[32, 2, 65]' is invalid for input of size 266240
```

### Root Cause

In `LorentzMoE.forward()`, routing weights were passed with `.unsqueeze(-1)`:
```python
expert_output, _ = self.experts(
    permuted_states,
    tokens_per_expert,
    permuted_weights.unsqueeze(-1),  # Shape: (64, 1)
)
```

Then in `LorentzGroupedExperts.forward()`, another unsqueeze was applied:
```python
if permuted_probs is not None:
    output = output * permuted_probs.unsqueeze(-1)  # Double unsqueeze!
```

This caused broadcasting from `(64, 65) * (64, 1, 1)` → `(64, 64, 65)` instead of element-wise multiplication.

### Fix

1. Remove extra unsqueeze in call site:
```python
expert_output, _ = self.experts(
    permuted_states,
    tokens_per_expert,
    permuted_weights,  # Shape: (64,) - no unsqueeze
)
```

2. Handle both 1D and 2D inputs in `LorentzGroupedExperts.forward()`:
```python
if permuted_probs is not None:
    if permuted_probs.dim() == 1:
        output = output * permuted_probs.unsqueeze(-1)
    else:
        output = output * permuted_probs
```

### Lesson Learned

When passing tensors through multiple layers, be careful about dimension manipulation. Use explicit shape comments and assertions to catch broadcasting errors early.

---

## Issue #3: Riemannian Optimizer NaN with Regular Weights

**Date:** 2026-02-03
**File:** `megatron/core/optimizer/riemannian_adam.py`
**Severity:** Design Issue (Fixed)

### Symptom

When applying `RiemannianAdam` to all model parameters:
```
Step 1: loss = 7.07
Step 2: loss = nan
Step 3: loss = nan
```

### Root Cause

The initial implementation applied Riemannian optimization (exponential map, parallel transport) to ALL parameters, including linear layer weights. However:

1. **Linear layer weights are NOT Lorentz vectors** - they are Euclidean matrices that transform vectors
2. **Only activations flow on the manifold** - the forward pass maintains the constraint
3. **Applying Lorentz inner product to weight matrices is meaningless** and causes numerical instability

### Understanding

In hyperbolic neural networks:
- **Weights** (W_q, W_k, W_v, FFN weights): Euclidean matrices optimized with standard Adam
- **Embeddings**: Can be Euclidean (projected to manifold) or manifold points
- **Activations**: Flow on the manifold, maintained by projection operations
- **Manifold constraint**: Enforced by forward pass (exp_map, projection, centroid)

### Fix

1. **Default to standard AdamW** in `create_optimizer_for_lorentz_model()`:
   ```python
   def create_optimizer_for_lorentz_model(..., use_riemannian=False):
   ```

2. **Use Riemannian optimization only for marked parameters**:
   ```python
   # Mark a parameter as a manifold point
   param.manifold_point = True
   ```

3. **Updated `_is_lorentz_param`** to handle mixed parameter types gracefully

### Lesson Learned

**Standard AdamW is correct for HELM models**. Riemannian optimization is only needed for specialized use cases with learnable anchor points on the manifold. The manifold constraint on activations is maintained by forward pass operations, not by the optimizer.

---

## Known Issues / TODOs

### TODO #1: FlashAttention Compatibility

**Status:** Not Started
**Priority:** Medium

The current `LorentzDotProductAttention` uses standard attention computation. FlashAttention cannot be directly used because:
1. Hyperbolic attention scores use `2c + 2*cinner(Q,K)` instead of `Q @ K.T / sqrt(d)`
2. Value aggregation uses Lorentzian centroid instead of weighted sum

**Potential Solutions:**
- Custom CUDA kernel for hyperbolic attention
- Approximate with standard attention for large sequences (with accuracy tradeoff)

### TODO #2: Gradient Clipping in Hyperbolic Space

**Status:** Not Started
**Priority:** Low

Current implementation uses standard Euclidean gradient clipping. For true Riemannian optimization, gradients should be clipped in tangent space.

### TODO #3: Mixed Precision (BF16) Validation

**Status:** Not Tested
**Priority:** High

Need to validate numerical stability with BF16 precision. Hyperbolic operations (especially `arcosh`, `sqrt` for time reconstruction) may need careful handling to avoid precision loss.

---

## Testing Commands

Run end-to-end test:
```bash
docker run --rm --gpus all \
  --ipc=host \
  -v "$(pwd):/workspace/megatron" \
  -w /workspace/megatron \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  python tests/unit_tests/models/test_lorentz_gpt_e2e.py
```

Run individual component tests:
```bash
# Manifold tests
docker run --rm --gpus all \
  -v "$(pwd):/workspace/megatron" \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  python -c "
import sys; sys.path.insert(0, '/workspace/megatron')
from megatron.core.manifolds import Lorentz
# ... test code
"
```

# HELM: Hyperbolic Large Language Models for Megatron-LM

This directory contains documentation for the HELM integration into Megatron-LM.

## Overview

HELM implements fully hyperbolic LLMs using the **Lorentz model** (hyperboloid) of hyperbolic geometry. It has two variants:

| Variant | Description | Use Case |
|---------|-------------|----------|
| **HELM-D** | Dense hyperbolic LLM | Standard dense models |
| **HELM-MiCE** | Mixture of Curvature Experts | Sparse expert models |

## Key Concepts

### Lorentz Model (Hyperboloid)

Points in Lorentz space satisfy:
```
-x₀² + x₁² + x₂² + ... + xₐ² = -c,  where x₀ > 0, c > 0
```

- `x₀`: Time-like coordinate (computed from space coordinates)
- `x₁...xₐ`: Space-like coordinates (learned features)
- `c`: Curvature parameter

### Tensor Dimensions

All Lorentz tensors have dimension `d+1` instead of `d`:
```python
# Standard LLM
x.shape = (batch, seq, dim)

# HELM
x.shape = (batch, seq, dim + 1)  # First coordinate is time
```

## Files

- [CHANGELOG.md](CHANGELOG.md) - Progress tracking
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical design (TBD)

## Source Reference

Original HELM implementation: `/fsx/ubuntu/users/aosong/ft/hyper/helm/`

## Quick Start

(To be added after implementation is complete)

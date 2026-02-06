# Training Launchers

Training scripts for hyperbolic transformers (**HELM-D** and **HELM-MiCE**) and their Euclidean baselines.

> For a full walkthrough including environment setup and data preparation, see [../docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md).

## Directory Structure

Scripts are organized into two parallel hierarchies — `lorentz/` (hyperbolic) and `standard/` (Euclidean baseline). Each hierarchy uses a two-layer pattern: **config scripts** define hyperparameters; **base scripts** handle container launch and `torchrun`.

```
training/
├── README.md
│
├── lorentz/                              # ── Hyperbolic models ──
│   ├── base/                             #     (sourced by config scripts)
│   │   ├── _train_dense_base_docker.sh   #     Dense + Docker
│   │   ├── _train_moe_base_docker.sh     #     MoE + Docker
│   │   ├── _train_moe_base_apptainer.sh  #     MoE + Apptainer (single-node)
│   │   └── _train_moe_base_slurm.sh      #     MoE + Slurm multi-node
│   ├── single-node/
│   │   ├── train_4b_redpajama-small.sh
│   │   ├── train_moe_80M_te_redpajama-small.sh
│   │   └── train_moe_80M_te_redpajama-small_apptainer.sh
│   └── multi-node/
│       ├── train_8b_redpajama-small_2node.sh
│       ├── train_moe_80M_te_redpajama-small_2node.sh
│       └── train_moe_30b-a3b_redpajama-small_2node.sh
│
└── standard/                             # ── Euclidean baselines ──
    ├── base/
    │   ├── _train_moe_base_docker.sh
    │   ├── _train_moe_base_apptainer.sh
    │   └── _train_moe_base_slurm.sh
    ├── single-node/
    │   ├── train_4b_redpajama-small.sh
    │   ├── train_moe_80M_te_redpajama-small.sh
    │   └── train_moe_80M_sequential_redpajama-small.sh
    └── multi-node/
        ├── train_8b_redpajama-small_2node.sh
        ├── train_moe_80M_te_redpajama-small_2node.sh
        ├── train_moe_30b-a3b_redpajama-small_2node.sh
        └── train_mixtral_8x7b_redpajama-small_2node.sh
```

## How It Works

Each config script exports environment variables, then sources a base script:

```bash
# Example: lorentz/single-node/train_moe_80M_te_redpajama-small.sh
export MODEL_ARGS=(--hidden-size 256 --num-layers 4 ...)
export HYPERBOLIC_ARGS=(--use-lorentz-moe --hyperbolic-curvature 1.0 ...)
export MOE_ARGS=(--num-experts 4 --moe-router-topk 2 ...)
export BATCH_SIZE=4
export LR=3e-4
...
source "${SCRIPT_DIR}/../base/_train_moe_base_docker.sh"
```

The base script then:
1. Launches the container (Docker or Apptainer)
2. Configures `torchrun` with the correct number of nodes/GPUs
3. Runs the training entry point with all exported arguments

| Geometry | Entry Point |
|----------|-------------|
| Lorentz (hyperbolic) | `pretrain_lorentz_gpt.py` |
| Standard (Euclidean) | `pretrain_gpt.py` |

## Quick Start

### Single-Node (Docker)

```bash
# Lorentz MoE 80M — quick test
./launchers/training/lorentz/single-node/train_moe_80M_te_redpajama-small.sh

# Standard MoE 80M — Euclidean baseline
./launchers/training/standard/single-node/train_moe_80M_te_redpajama-small.sh

# Lorentz Dense 4B — requires 8 GPUs (TP=8)
./launchers/training/lorentz/single-node/train_4b_redpajama-small.sh
```

### Multi-Node (Slurm)

```bash
# Lorentz MoE 80M — 2 nodes, 16 GPUs
sbatch launchers/training/lorentz/multi-node/train_moe_80M_te_redpajama-small_2node.sh

# Lorentz MoE 30B-A3B — 2 nodes, 128 experts, production-scale
sbatch launchers/training/lorentz/multi-node/train_moe_30b-a3b_redpajama-small_2node.sh

# Lorentz Dense 8B — 2 nodes
sbatch launchers/training/lorentz/multi-node/train_8b_redpajama-small_2node.sh

# Monitor
squeue -u $USER
```

## Available Models

### Dense (HELM-D)

| Model | Layers | Hidden | Heads (Q/KV) | TP | Nodes | Geometry |
|-------|--------|--------|-------------|-----|-------|----------|
| 4B | 36 | 2560 | 32/8 | 8 | 1 | Lorentz / Standard |
| 8B | 36 | 4096 | 32/8 | 8 | 2 | Lorentz / Standard |

### MoE (HELM-MiCE)

| Model | Layers | Hidden | Experts | Top-K | TP | EP | Nodes | Geometry |
|-------|--------|--------|---------|-------|-----|-----|-------|----------|
| 80M (TE) | 4 | 256 | 4+1 shared | 2 | 1 | 1 | 1 | Lorentz / Standard |
| 80M (Sequential) | 4 | 256 | 4 | 2 | 1 | 1 | 1 | Standard only |
| 80M 2-node | 4 | 256 | 4+1 shared | 2 | 2 | 2 | 2 | Lorentz / Standard |
| 30B-A3B | 48 | 2048 | 128 | 8 | 2 | 8 | 2 | Lorentz / Standard |
| Mixtral 8x7B | 8 | 512 | 8 | 2 | 1 | 8 | 2 | Standard only |

## Model Types

### HELM-D (Dense)

Hyperbolic transformer with Lorentz geometry:
- All layers use hyperbolic operations
- Lorentz attention with centroid aggregation
- SwiGLU MLP with manifold projection

### HELM-MiCE (MoE)

Mixture of Curvature Experts:
- Routed experts operate at different curvatures (range configurable via `--expert-curvature-min` / `--expert-curvature-max`)
- Optional shared expert at base curvature
- Load balancing via auxiliary loss
- MoE layers at configurable frequency

Expert backends (Lorentz):
- **`LorentzTEGroupedMLP`** — TransformerEngine accelerated, full tangent-space ops (recommended)
- **`LorentzSequentialMLP`** — Per-expert LorentzMLP instances (slower, correct fallback)
- **`LorentzGroupedMLP`** — Curvature scaling only (incomplete hyperbolic geometry)

## Container Backends

| Backend | Base Script | Use Case |
|---------|-------------|----------|
| Docker | `_train_*_base_docker.sh` | Single-node dev/testing |
| Apptainer | `_train_*_base_apptainer.sh` | Single-node on Slurm clusters |
| Slurm + Apptainer | `_train_*_base_slurm.sh` | Multi-node distributed training |

## Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GPUS_PER_NODE` | GPUs per node | Auto-detect |
| `DATA_PATH` | Path to tokenized `.bin`/`.idx` data | (dummy data) |
| `CHECKPOINT_DIR` | Checkpoint output | `./checkpoints/<model>` |
| `APPTAINER_IMAGE` | Apptainer `.sif` image path | `~/images/lorentz-moe_25.04.sif` |
| `MASTER_ADDR` | Multi-node master IP | localhost (auto in Slurm) |
| `MASTER_PORT` | Distributed port | 6000–29500 |
| `TP` / `PP` / `EP` | Tensor / Pipeline / Expert parallelism | Varies per script |

## Outputs

| Artifact | Location |
|----------|----------|
| Checkpoints | `./checkpoints/<model-name>/` |
| TensorBoard | `./tensorboard/<model-name>/` |

```bash
tensorboard --logdir ./tensorboard/<model-name>/
```

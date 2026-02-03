# Lorentz GPT Training Launchers

Training scripts for hyperbolic transformers (HELM-D and HELM-MiCE).

## Directory Structure

```
training/
├── configs/              # Model and training configurations
│   ├── lorentz_dense_*.sh   # Dense model configs
│   └── lorentz_moe_*.sh     # MoE model configs
├── single_node/          # Single node training (1-8 GPUs)
│   └── lorentz/
│       ├── train_dense.sh   # HELM-D training
│       └── train_moe.sh     # HELM-MiCE training
├── multi_node/           # Multi-node distributed training
│   ├── lorentz/
│   │   ├── train_dense.sh
│   │   └── train_moe.sh
│   └── slurm_submit.sh      # SLURM job submission
└── docker/               # Container-based training
    └── run_training.sh
```

## Quick Start

### Single Node Training

```bash
# Dense model (HELM-D) with test config
cd launchers/training/single_node/lorentz
./train_dense.sh test

# MoE model (HELM-MiCE) with test config
./train_moe.sh test

# Production config (Qwen3-0.6B-like)
./train_dense.sh qwen3_0.6b
./train_moe.sh qwen3_moe_0.6b
```

### Multi-Node Training

```bash
# On each node, set environment and run:
export MASTER_ADDR=<rank0_node_ip>
export NUM_NODES=4
export NODE_RANK=<0,1,2,3>

cd launchers/training/multi_node/lorentz
./train_dense.sh qwen3_0.6b
```

### SLURM Cluster

```bash
# Submit job
cd launchers/training/multi_node
sbatch slurm_submit.sh dense qwen3_0.6b   # Dense model
sbatch slurm_submit.sh moe qwen3_moe_0.6b # MoE model
```

### Docker Training

```bash
cd launchers/training/docker
./run_training.sh dense test     # Dense, test config
./run_training.sh moe test       # MoE, test config
```

## Configuration Files

### Available Configs

| Config | Model Type | Description |
|--------|-----------|-------------|
| `test` | dense/moe | Small model for debugging |
| `qwen3_0.6b` | dense | Qwen3-0.6B architecture |
| `qwen3_moe_0.6b` | moe | Qwen3-MoE-0.6B architecture |

### Custom Configuration

Create a new config file in `configs/`:

```bash
# configs/lorentz_dense_custom.sh
export MODEL_TYPE="dense"
export MODEL_CONFIG="qwen3_0.6b"  # Python config name
export MODEL_NAME="my-custom-model"

export MAX_STEPS=100000
export BATCH_SIZE=512
export MICRO_BATCH_SIZE=4
export LR=2e-4
# ... other settings
```

Then run:
```bash
./train_dense.sh custom
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GPUS_PER_NODE` | GPUs per node | Auto-detect |
| `DATA_PATH` | Path to tokenized data | (dummy data) |
| `CHECKPOINT_DIR` | Checkpoint output | `./checkpoints/<model>` |
| `MASTER_ADDR` | Multi-node master IP | localhost |
| `MASTER_PORT` | Distributed port | 29500 |
| `NUM_NODES` | Number of nodes | 1 |
| `NODE_RANK` | This node's rank | 0 |

## Model Types

### HELM-D (Dense)

Standard hyperbolic transformer with Lorentz geometry:
- All layers use hyperbolic operations
- Lorentz attention with centroid aggregation
- SwiGLU MLP with manifold projection

### HELM-MiCE (MoE)

Mixture of Curvature Experts:
- Routed experts operate at different curvatures
- Shared expert at base curvature
- Load balancing via auxiliary loss
- MoE layers at configurable frequency

## Python Training Scripts

The shell launchers call these Python scripts:

- `pretrain_lorentz_gpt.py` - Dense model training
- `pretrain_lorentz_moe_gpt.py` - MoE model training

Direct Python usage:
```bash
# Single GPU
python pretrain_lorentz_gpt.py --config test

# Distributed
torchrun --nproc_per_node=4 pretrain_lorentz_gpt.py --config qwen3_0.6b
```

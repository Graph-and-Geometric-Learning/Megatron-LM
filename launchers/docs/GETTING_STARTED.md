# HELM Pretraining: Getting Started from Scratch

> **Repo:** [Graph-and-Geometric-Learning/Megatron-LM](https://github.com/Graph-and-Geometric-Learning/Megatron-LM)
>
> This guide walks through the full pipeline — environment setup, data preparation, and training — for pretraining hyperbolic (Lorentz) language models with Megatron-LM.
> For a detailed map of all code changes and hyperbolic components, see [CODE_MAP.md](./CODE_MAP.md).

---

## Table of Contents

1. [Environment Preparation](#1-environment-preparation)
   - [1.1 Container-Based Setup (Recommended)](#11-container-based-setup-recommended)
   - [1.2 Conda/UV Setup (Placeholder)](#12-conduv-setup-placeholder)
2. [Data Preparation](#2-data-preparation)
   - [2.1 Raw Data Downloading](#21-raw-data-downloading)
   - [2.2 Data Processing for Megatron Format](#22-data-processing-for-megatron-format)
3. [Training](#3-training)
   - [3.1 Script Architecture](#31-script-architecture)
   - [3.2 Model Inventory](#32-model-inventory)
   - [3.3 Lorentz (Hyperbolic) Training](#33-lorentz-hyperbolic-training)
   - [3.4 Standard (Euclidean Baseline) Training](#34-standard-euclidean-baseline-training)
   - [3.5 Key Environment Variables](#35-key-environment-variables)
   - [3.6 Outputs](#36-outputs)

---

## 1. Environment Preparation

We provide two environment options:

| Option | Status | Notes |
|--------|--------|-------|
| **Container-based** (Docker + Apptainer) | Implemented | Recommended |
| **Conda / UV** (native environment) | Placeholder | Not yet implemented |

### 1.1 Container-Based Setup (Recommended)

The container workflow uses **Docker** for local development and **Apptainer** for multi-node Slurm clusters.

```
Docker (local dev/testing)
    |  convert
    v
Apptainer .sif (multi-node Slurm)
```

**Base image:** NGC PyTorch `25.04-py3` with TransformerEngine pre-installed.

#### Step 1 — Build Docker Image

Build the Lorentz MoE image with all dependencies:

```bash
./launchers/setup/build_lorentz_moe_image.sh
```

This runs `docker build` using `docker/Dockerfile.lorentz-moe`, which installs:

- PyTorch + CUDA (from NGC base)
- TransformerEngine (from NGC base)
- `transformers`, `wandb`, `tensorboard`

#### Step 2 — Convert to Apptainer (for Slurm)

The same script automatically converts the Docker image to Apptainer format:

```bash
# Output: ~/images/lorentz-moe_25.04.sif
```

For multi-node training, set the image path:

```bash
export APPTAINER_IMAGE=~/images/lorentz-moe_25.04.sif
```

#### Alternative — Pull NGC Base Image Directly

If you only need the base NGC container without custom dependencies:

```bash
./launchers/setup/pull_ngc_to_apptainer.sh 25.04-py3
# Output: ~/users/images/pytorch_25.04-py3.sif
```

#### Verify Installation

```bash
# Docker
docker run --rm --gpus all lorentz-moe:25.04 \
    python -c "import torch; print(torch.cuda.is_available())"

# Apptainer
apptainer exec --nv ~/images/lorentz-moe_25.04.sif \
    python -c "import torch; print(torch.cuda.is_available())"
```

### 1.2 Conda/UV Setup (Placeholder)

Not yet implemented.

---

## 2. Data Preparation

> **Tip:** Store all preprocessed data on `/fsx` so it can be shared across nodes. Raw data can be archived to S3 (via `hsm_release`) when not actively needed.

### 2.1 Raw Data Downloading

Example process for downloading RedPajama-1T (~4.4 TB total):

```bash
mkdir -p /path/to/redpajama-1t
cd /path/to/redpajama-1t
```

Create a file called `run_download.sh`:

```bash
#!/usr/bin/env bash
wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
while read line; do
    dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
    mkdir -p $(dirname "$dload_loc")
    wget "$line" -O "$dload_loc"
done < urls.txt
```

Run it in the background:

```bash
nohup bash run_download.sh &
```

### 2.2 Data Processing for Megatron Format

Before training, raw text data must be converted into Megatron's binary format (`.bin` and `.idx` files). This step tokenizes the text and creates efficient binary files for high-performance data loading.

**Location:** `launchers/data_processing/`

#### Prerequisites

1. RedPajama dataset downloaded to `/workspace/dataset/redpajama-1t/` with this structure:

   ```
   /workspace/dataset/redpajama-1t/
   ├── arxiv/*.jsonl
   ├── c4/*.jsonl
   ├── github/*.jsonl
   ├── stackexchange/*.jsonl
   ├── wikipedia/*.jsonl
   └── common_crawl/
       ├── 2019-30/*.zst
       ├── 2020-05/*.zst
       └── ...
   ```

2. Python dependencies: `zstandard` (for `.zst` files, auto-installed), HuggingFace `transformers` (for tokenizer)

#### Available Scripts

| Script | Sources | Use Case |
|--------|---------|----------|
| `preprocess_redpajama.sh` | All 10 sources | Production training |
| `preprocess_redpajama_small.sh` | Wikipedia + StackExchange | Development / testing |
| `preprocess_redpajama_tiny.sh` | 10k samples from Wikipedia | Quick debugging |

#### Usage

**Quick test (tiny):**

```bash
cd launchers/data_processing
bash preprocess_redpajama_tiny.sh

# Configure sample count:
NUM_SAMPLES=5000 bash preprocess_redpajama_tiny.sh
```

**Development (small):**

```bash
bash preprocess_redpajama_small.sh
```

Processes only Wikipedia and StackExchange (~50/50 weight distribution).

**Production (full):**

```bash
bash preprocess_redpajama.sh

# Options:
bash preprocess_redpajama.sh --workers 64 --partitions 16
bash preprocess_redpajama.sh --dry-run                       # Preview only
bash preprocess_redpajama.sh --tokenizer-model Qwen/Qwen3-8B
bash preprocess_redpajama.sh --output-root /custom/output/path
```

#### Output Files

```
/workspace/processed_data/redpajama/
├── arxiv_text_document.bin          # Binary tokenized data
├── arxiv_text_document.idx          # Index file
├── c4_text_document.bin
├── c4_text_document.idx
├── wikipedia_text_document.bin
├── wikipedia_text_document.idx
├── ... (other sources)
├── data_blend.txt                   # Weight configuration file
└── logs/                            # Processing logs
```

**Data blend file format** (`data_blend.txt`):

```
0.025  /workspace/processed_data/redpajama/arxiv_text_document
0.15   /workspace/processed_data/redpajama/c4_text_document
0.02   /workspace/processed_data/redpajama/wikipedia_text_document
...
```

#### Using Processed Data in Training

**Option 1 — Multi-source blend (recommended for full training):**

```bash
python pretrain_gpt.py \
    --data-args-path /workspace/processed_data/redpajama/data_blend.txt \
    ... other args
```

**Option 2 — Single dataset:**

```bash
python pretrain_gpt.py \
    --data-path /workspace/processed_data/redpajama/wikipedia_text_document \
    ... other args
```

#### Default Configuration

| Parameter | Default Value |
|-----------|---------------|
| Tokenizer | `Qwen/Qwen3-8B` (HuggingFace) |
| Workers | 64 (full), 16 (small), 8 (tiny) |
| Partitions | 16 (adjusted automatically based on file count) |
| EOD Token | Appended at end of each document |
| JSON Key | `text` |

#### Data Source Weights (Full Dataset)

| Source | Weight |
|--------|--------|
| C4 | 15% |
| Common Crawl (5 snapshots) | 74% total |
| GitHub | 4.5% |
| arXiv | 2.5% |
| Wikipedia | 2% |
| StackExchange | 2% |

---

## 3. Training

**Location:** `launchers/training/`

Training scripts are organized into two parallel hierarchies — **lorentz/** (hyperbolic geometry) and **standard/** (Euclidean baseline) — plus a set of **top-level quick-start wrappers**.

### 3.1 Script Architecture

Training uses a **two-layer script structure**:

1. **Config scripts** (`single-node/`, `multi-node/`) — define model architecture, hyperparameters, and MoE/hyperbolic settings
2. **Base scripts** (`base/`) — handle container launch, distributed training setup, and `torchrun` invocation

```
Config script (exports MODEL_ARGS, HYPERBOLIC_ARGS, MOE_ARGS, etc.)
    |
    v
Base script (launches container, configures torchrun, runs pretrain_*.py)
```

Each config script exports environment variables and then sources the appropriate base script:

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

#### Container Backends

| Backend | Use Case | Base Scripts |
|---------|----------|-------------|
| Docker | Single-node testing | `_train_dense_base_docker.sh`, `_train_moe_base_docker.sh` |
| Apptainer | Single-node (cluster) | `_train_moe_base_apptainer.sh` |
| Slurm + Apptainer | Multi-node distributed | `_train_moe_base_slurm.sh` |
| Conda/venv | Native environment | Not implemented yet |

#### Entry Points

| Geometry | Training Script | Description |
|----------|----------------|-------------|
| Lorentz (hyperbolic) | `pretrain_lorentz_gpt.py` | Lorentz manifold transformer training |
| Standard (Euclidean) | `pretrain_gpt.py` | Standard Megatron GPT training |

#### Directory Layout

```
launchers/training/
├── README.md
│
├── lorentz/                              # ── Hyperbolic models ──
│   ├── base/
│   │   ├── _train_dense_base_docker.sh
│   │   ├── _train_moe_base_docker.sh
│   │   ├── _train_moe_base_apptainer.sh
│   │   └── _train_moe_base_slurm.sh
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

### 3.2 Model Inventory

Every Lorentz model has a matching Standard baseline for controlled comparison. All models use the Qwen3 family architecture.

#### Dense Models (HELM-D)

| Model | Layers | Hidden | Heads (Q/KV) | Parallelism | Nodes | Backend |
|-------|--------|--------|-------------|-------------|-------|---------|
| Qwen3-0.6B Lorentz | 28 | 1024 | 16 / 8 | DP | 1 | Docker / Apptainer |
| Qwen3-0.6B Standard | 28 | 1024 | 16 / 8 | DP | 1 | Docker / Apptainer |
| Qwen3-4B Lorentz | 36 | 2560 | 32 / 8 | TP=8 | 1 | Apptainer |
| Qwen3-4B Standard | 36 | 2560 | 32 / 8 | TP=8 | 1 | Apptainer |
| Qwen3-8B Lorentz | 36 | 4096 | 32 / 8 | TP=8 | 2 | Slurm |
| Qwen3-8B Standard | 36 | 4096 | 32 / 8 | TP=8 | 2 | Slurm |

#### MoE Models (HELM-MiCE)

| Model | Layers | Hidden | Experts (routed+shared) | Top-K | Parallelism | Nodes | Backend |
|-------|--------|--------|------------------------|-------|-------------|-------|---------|
| 80M MoE Lorentz (TE) | 4 | 256 | 4 + 1 shared | 2 | DP | 1 | Docker / Apptainer |
| 80M MoE Standard (TE) | 4 | 256 | 4 | 2 | DP | 1 | Docker |
| 80M MoE Standard (Seq) | 4 | 256 | 4 | 2 | DP | 1 | Docker |
| 80M MoE Lorentz 2-node | 4 | 256 | 4 + 1 shared | 2 | TP=2, EP=2 | 2 | Slurm |
| 80M MoE Standard 2-node | 4 | 256 | 4 | 2 | TP=2, EP=2 | 2 | Slurm |
| 30B-A3B MoE Lorentz | 48 | 2048 | 128 | 8 | TP=2, EP=8 | 2 | Slurm |
| 30B-A3B MoE Standard | 48 | 2048 | 128 | 8 | TP=2, EP=8 | 2 | Slurm |
| Mixtral 8x7B (reference) | 8 | 512 | 8 | 2 | EP=8 | 2 | Slurm |
| Qwen3 MoE Dummy | 8 | 1024 | 8 | 2 | EP=2 | 1 | Docker |

### 3.3 Lorentz (Hyperbolic) Training

All Lorentz scripts use `pretrain_lorentz_gpt.py` as the entry point.

#### Single-Node

```bash
# Dense 4B — Apptainer, TP=8 across 8 GPUs
./launchers/training/lorentz/single-node/train_4b_redpajama-small.sh

# MoE 80M (TEGroupedMLP) — Docker
./launchers/training/lorentz/single-node/train_moe_80M_te_redpajama-small.sh

# MoE 80M (TEGroupedMLP) — Apptainer
./launchers/training/lorentz/single-node/train_moe_80M_te_redpajama-small_apptainer.sh
```

#### Multi-Node (Slurm)

Submit with `sbatch`:

```bash
# Dense 8B — 2 nodes, 16 GPUs, TP=8
sbatch launchers/training/lorentz/multi-node/train_8b_redpajama-small_2node.sh

# MoE 80M — 2 nodes, TP=2, EP=2, sequence parallel
sbatch launchers/training/lorentz/multi-node/train_moe_80M_te_redpajama-small_2node.sh

# MoE 30B-A3B — 2 nodes, 128 experts, TP=2, EP=8 (production-scale)
sbatch launchers/training/lorentz/multi-node/train_moe_30b-a3b_redpajama-small_2node.sh
```

#### Lorentz-Specific Configuration

Hyperbolic models add these settings on top of standard Megatron arguments:

```bash
HYPERBOLIC_ARGS=(
    --use-lorentz-moe              # Enable hyperbolic geometry
    --hyperbolic-curvature 1.0     # Base manifold curvature
    --expert-curvature-min 0.1     # Per-expert curvature range (MoE only)
    --expert-curvature-max 2.0
)
```

MoE expert types for Lorentz:
- **`LorentzTEGroupedMLP`** — Full tangent-space operations via TransformerEngine (recommended)
- **`LorentzSequentialMLP`** — Per-expert `LorentzMLP` instances (slower, but correct fallback)
- **`LorentzGroupedMLP`** — Curvature scaling only (incomplete hyperbolic geometry)

### 3.4 Standard (Euclidean Baseline) Training

Standard scripts mirror the Lorentz scripts exactly (same architecture, same hyperparameters) but without hyperbolic geometry. They use `pretrain_gpt.py` as the entry point.

#### Single-Node

```bash
# Dense 4B — Apptainer, TP=8
./launchers/training/standard/single-node/train_4b_redpajama-small.sh

# MoE 80M (TEGroupedMLP) — Docker
./launchers/training/standard/single-node/train_moe_80M_te_redpajama-small.sh

# MoE 80M (SequentialMLP, no TE dependency) — Docker
./launchers/training/standard/single-node/train_moe_80M_sequential_redpajama-small.sh
```

#### Multi-Node (Slurm)

```bash
# Dense 8B — 2 nodes
sbatch launchers/training/standard/multi-node/train_8b_redpajama-small_2node.sh

# MoE 80M — 2 nodes
sbatch launchers/training/standard/multi-node/train_moe_80M_te_redpajama-small_2node.sh

# MoE 30B-A3B — 2 nodes, 128 experts
sbatch launchers/training/standard/multi-node/train_moe_30b-a3b_redpajama-small_2node.sh

# Mixtral 8x7B reference config — 2 nodes, EP=8
sbatch launchers/training/standard/multi-node/train_mixtral_8x7b_redpajama-small_2node.sh
```

### 3.5 Key Environment Variables

#### Common (All Scripts)

| Variable | Description | Typical Values |
|----------|-------------|----------------|
| `MODEL_NAME` | Checkpoint directory name | `lorentz-moe-80m-te`, `qwen3-4b-lorentz` |
| `BATCH_SIZE` | Per-node batch size | 1–4 |
| `MICRO_BATCH_SIZE` | Micro-batch (for gradient accumulation) | 1 |
| `LR` / `MIN_LR` | Learning rate / minimum LR | `3e-4` / `3e-5` |
| `MAX_STEPS` | Total training steps | 500–10000 |
| `WARMUP_STEPS` | LR warmup steps | 50–500 |
| `WEIGHT_DECAY` | AdamW weight decay | 0.1 |
| `GRAD_CLIP` | Gradient clipping norm | 1.0 |
| `SAVE_INTERVAL` | Checkpoint save frequency (steps) | 100–500 |

#### Parallelism

| Variable | Description | Range |
|----------|-------------|-------|
| `TP` | Tensor-model parallelism (splits model across GPUs) | 1, 2, 8 |
| `PP` | Pipeline-model parallelism (splits layers across stages) | 1 |
| `EP` | Expert parallelism (distributes MoE experts) | 1, 2, 8 |

#### MoE-Specific

| Variable | Description | Typical Values |
|----------|-------------|----------------|
| `NUM_EXPERTS` | Number of routed experts | 4, 8, 128 |
| `NUM_SHARED_EXPERTS` | Shared (always-on) experts | 0, 1 |
| `MOE_ROUTER_TOPK` | Experts activated per token | 2, 8 |
| `MOE_LAYER_FREQ` | MoE layer frequency (1 = every layer, 2 = every other) | 1, 2 |
| `MOE_AUX_LOSS_COEFF` | Load-balancing auxiliary loss coefficient | 0.001–0.01 |

#### Container

| Variable | Description |
|----------|-------------|
| `DOCKER_IMAGE` | Docker image (default: `nvcr.io/nvidia/pytorch:25.04-py3`) |
| `APPTAINER_IMAGE` | Apptainer `.sif` path (default: `~/images/lorentz-moe_25.04.sif`) |
| `HOST_DATA_DIR` | Host path to preprocessed data |
| `DATA_PATH` | Container-internal data path |

### 3.6 Outputs

| Artifact | Default Location |
|----------|-----------------|
| Checkpoints | `./checkpoints/<model-name>/` |
| TensorBoard logs | `./tensorboard/<model-name>/` |

Monitor training:

```bash
# Slurm job status
squeue -u $USER

# TensorBoard
tensorboard --logdir ./tensorboard/<model-name>/
```

---

## Further Reading

- **[CODE_MAP.md](./CODE_MAP.md)** — Detailed mapping of every hyperbolic/Lorentz code change in the repo, including file paths, key classes, line numbers, and common modification scenarios.

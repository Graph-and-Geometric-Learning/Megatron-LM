#!/bin/bash
# =============================================================================
# Lorentz Dense GPT: 193M with RedPajama-Small
# =============================================================================
# Single-node training with RedPajama small dataset (runs in Docker).
#
# Model: ~193M parameters (115M if embeddings tied)
# =============================================================================

set -e

# =============================================================================
# Data Configuration (RedPajama-Small)
# =============================================================================
HOST_DATA_DIR="/fsx/ubuntu/users/aosong/data"
HOST_DATA_PATH="${HOST_DATA_DIR}/processed_data/redpajama_small/stackexchange_text_document"

export DATA_PATH="/workspace/data/processed_data/redpajama_small/stackexchange_text_document"
export DOCKER_DATA_MOUNT="-v ${HOST_DATA_DIR}:/workspace/data"

# Validate data exists
if [ ! -f "${HOST_DATA_PATH}.bin" ]; then
    echo "ERROR: Data file not found: ${HOST_DATA_PATH}.bin"
    exit 1
fi

# =============================================================================
# Export Config for Base Script
# =============================================================================
export MODEL_NAME="lorentz-dense-193M-redpajama-small"

# Model Architecture (tuned for 8 GPU memory)
export MODEL_ARGS=(
    --hidden-size 512
    --num-layers 12
    --num-attention-heads 8
    --num-kv-heads 4
    --ffn-hidden-size 1536
    --vocab-size 151936
    --seq-length 1024
)

# Hyperbolic Configuration
export HYPERBOLIC_ARGS=(
    --curvature 1.0
)

# Training Hyperparameters
export BATCH_SIZE=8
export MICRO_BATCH_SIZE=1
export LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=5000
export WARMUP_STEPS=200
export WEIGHT_DECAY=0.1
export GRAD_CLIP=1.0
export LOG_INTERVAL=10
export SAVE_INTERVAL=500

# =============================================================================
# Run
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_train_dense_base_docker.sh"

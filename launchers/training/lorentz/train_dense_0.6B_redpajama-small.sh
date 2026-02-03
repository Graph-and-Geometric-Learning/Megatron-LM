#!/bin/bash
# =============================================================================
# Lorentz Dense GPT: 0.6B with RedPajama-Small
# =============================================================================
# Single-node training with RedPajama small dataset (runs in Docker).
#
# Model: ~621M parameters (Qwen3-0.6B architecture)
#   - Embeddings: 151936 × 1024 = 155.6M
#   - Attention per layer: Q(1M) + K(0.5M) + V(0.5M) + O(1M) = 3M
#   - FFN per layer: 3 × 1024 × 3072 = 9.4M
#   - 28 layers × 12.4M = 347M
#   - Output layer: 1024 × 151936 = 155.6M
#   - Total: ~621M (or ~465M if embeddings tied)
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
export MODEL_NAME="lorentz-dense-0.6B-redpajama-small"

# Model Architecture (Qwen3-0.6B)
export MODEL_ARGS=(
    --hidden-size 1024
    --num-layers 28
    --num-attention-heads 16
    --num-kv-heads 8
    --ffn-hidden-size 3072
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

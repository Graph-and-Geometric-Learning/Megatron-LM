#!/bin/bash
# =============================================================================
# Standard Megatron MoE GPT 80M: TEGroupedMLP Backend
# =============================================================================
# Standard (non-hyperbolic) MoE training with TEGroupedMLP.
# Baseline for comparison with Lorentz MoE.
#
# Expert Type: TEGroupedMLP
#   - TransformerEngine's efficient grouped linear
#   - Euclidean geometry (standard)
#
# Model: ~80M total parameters
#   - 4 layers, hidden_size 256, ffn 512
#   - 4 routed experts
#   - Top-2 routing, MoE every 2nd layer
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
    echo "Please prepare the data first."
    exit 1
fi

# =============================================================================
# Export Config for Base Script
# =============================================================================
export MODEL_NAME="standard-moe-80M-te"
export EXPERT_TYPE="TEGroupedMLP"

# Model Architecture (~80M params)
export MODEL_ARGS=(
    --num-layers 4
    --hidden-size 256
    --num-attention-heads 4
    --group-query-attention
    --num-query-groups 2
    --ffn-hidden-size 512
    --seq-length 256
    --max-position-embeddings 256
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --no-bias-dropout-fusion
    --no-persist-layer-norm
    --untie-embeddings-and-output-weights
    --tokenizer-type NullTokenizer
    --vocab-size 151936
)

# MoE Configuration (Megatron-style)
# --moe-grouped-gemm (without --moe-use-legacy-grouped-gemm) -> uses TEGroupedMLP
export MOE_ARGS=(
    --num-experts 4
    --moe-router-topk 2
    --moe-layer-freq 2
    --moe-aux-loss-coeff 0.01
    --moe-token-dispatcher-type allgather
    --moe-grouped-gemm
)

# Export settings for display
export NUM_EXPERTS=4
export MOE_ROUTER_TOPK=2
export MOE_LAYER_FREQ=2
export MOE_AUX_LOSS_COEFF=0.01

# Training Hyperparameters
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=1000
export WARMUP_STEPS=100
export WEIGHT_DECAY=0.1
export GRAD_CLIP=1.0
export LOG_INTERVAL=10
export SAVE_INTERVAL=500

# Parallelism
export TP=1
export PP=1
export EP=1

# =============================================================================
# Run
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_train_moe_base_docker.sh"

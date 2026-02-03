#!/bin/bash
# =============================================================================
# Standard Megatron MoE GPT: Legacy GroupedMLP
# =============================================================================
# Test standard Megatron MoE with legacy GroupedMLP expert type.
# Requires grouped_gemm package installation.
#
# Expert Type: GroupedMLP (Legacy)
#   - Requires: pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
#   - BF16 only (no FP8 support)
#   - Being deprecated in favor of TEGroupedMLP
#   - IMPORTANT: --disable-bias-linear is REQUIRED
#
# Model: ~100M total parameters (small test model)
#   - 12 layers, hidden_size 512
#   - 4 routed experts
#   - Top-2 routing per token
#   - MoE layer every 2nd layer
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
export MODEL_NAME="standard-moe-grouped-gemm"
export EXPERT_TYPE="GroupedMLP (Legacy)"

# Install grouped_gemm package inside Docker
export INSTALL_GROUPED_GEMM="pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"

# Model Architecture (Small test model) - Megatron args
# NOTE: --disable-bias-linear is REQUIRED for GroupedMLP
export MODEL_ARGS=(
    --num-layers 12
    --hidden-size 512
    --num-attention-heads 8
    --group-query-attention
    --num-query-groups 4
    --ffn-hidden-size 1536
    --seq-length 512
    --max-position-embeddings 512
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
# --moe-grouped-gemm + --moe-use-legacy-grouped-gemm -> uses GroupedMLP
export MOE_ARGS=(
    --num-experts 4
    --moe-router-topk 2
    --moe-layer-freq 2
    --moe-aux-loss-coeff 0.01
    --moe-token-dispatcher-type allgather
    --moe-grouped-gemm
    --moe-use-legacy-grouped-gemm
)

# Export settings for display
export NUM_EXPERTS=4
export MOE_ROUTER_TOPK=2
export MOE_LAYER_FREQ=2
export MOE_AUX_LOSS_COEFF=0.01

# Training Hyperparameters
export BATCH_SIZE=16
export MICRO_BATCH_SIZE=2
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

#!/bin/bash
# =============================================================================
# Lorentz Dense GPT 4B: Qwen3-4B Architecture (Single Node)
# =============================================================================
# Single-node training with 8 GPUs.
#
# Model: Qwen3-4B Dense (NO MoE)
#   - 36 layers, hidden_size 2560, ffn 9728
#   - GQA: 32 query heads, 8 KV heads
#   - Lorentz (hyperbolic) geometry
#
# Usage:
#   bash train_4b_redpajama-small.sh
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# =============================================================================
# Data Configuration (RedPajama-Small)
# =============================================================================
export HOST_DATA_DIR="/fsx/ubuntu/users/aosong/data"
HOST_DATA_PATH="${HOST_DATA_DIR}/processed_data/redpajama_small/stackexchange_text_document"

export DATA_PATH="/workspace/data/processed_data/redpajama_small/stackexchange_text_document"

# Validate data exists
if [ ! -f "${HOST_DATA_PATH}.bin" ]; then
    echo "ERROR: Data file not found: ${HOST_DATA_PATH}.bin"
    echo "Please prepare the data first."
    exit 1
fi

# =============================================================================
# Export Config for Base Script
# =============================================================================
export MODEL_NAME="lorentz-4b"

# Model Architecture (Qwen3-4B Dense)
# 36 layers, hidden 2560, 32 heads, 8 KV heads, ffn 9728
export MODEL_ARGS_STR="--num-layers 36 --hidden-size 2560 --num-attention-heads 32 --group-query-attention --num-query-groups 8 --ffn-hidden-size 9728 --seq-length 2048 --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 1000000 --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --disable-bias-linear --no-bias-dropout-fusion --no-persist-layer-norm --untie-embeddings-and-output-weights --tokenizer-type NullTokenizer --vocab-size 151936"

# NO MoE Configuration (Dense model)
export MOE_ARGS_STR=""

# Hyperbolic Configuration (Lorentz-specific, dense mode)
export HYPERBOLIC_ARGS_STR="--use-hyperbolic --hyperbolic-curvature 1.0"

# Export settings for display
export HYPERBOLIC_CURVATURE=1.0

# Training Hyperparameters
export BATCH_SIZE=1
export MICRO_BATCH_SIZE=1
export LR=1e-4
export MIN_LR=1e-5
export MAX_STEPS=10000
export WARMUP_STEPS=500
export WEIGHT_DECAY=0.1
export GRAD_CLIP=1.0
export LOG_INTERVAL=10
export SAVE_INTERVAL=500

# Checkpoint directory
export CHECKPOINT_DIR="${MEGATRON_DIR}/checkpoints/${MODEL_NAME}"

# Parallelism for single node (8 GPUs)
export TP=8
export PP=1
export EP=1
export GPUS_PER_NODE=8

# =============================================================================
# Run with Apptainer
# =============================================================================
# Use MoE base script with empty MOE_ARGS_STR for dense model
BASE_SCRIPT="${MEGATRON_DIR}/launchers/training/lorentz/base/_train_moe_base_apptainer.sh"
bash "${BASE_SCRIPT}"

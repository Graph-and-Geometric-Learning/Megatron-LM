#!/bin/bash
# =============================================================================
# Lorentz MoE GPT 250M: SequentialMLP Backend
# =============================================================================
# Lorentz MoE training with LorentzSequentialMLP (true Lorentz per expert).
#
# Expert Type: LorentzSequentialMLP
#   - True Lorentz geometry with LorentzMLP per expert
#   - No TE dependency (fallback, always works)
#   - Per-expert curvatures distributed across range
#
# Model: ~250M total parameters
#   - 12 layers, hidden_size 512, ffn 1536
#   - 4 routed experts + 1 shared expert
#   - Top-2 routing, MoE every 2nd layer
#   - Per-expert curvature (0.1 to 2.0)
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
export MODEL_NAME="lorentz-moe-250M-sequential"

# Model Architecture (~250M params)
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
export MOE_ARGS=(
    --num-experts 4
    --moe-shared-expert-intermediate-size 1536
    --moe-router-topk 2
    --moe-layer-freq 2
    --moe-aux-loss-coeff 0.01
    --moe-token-dispatcher-type allgather
)

# Hyperbolic Configuration (Lorentz-specific)
export HYPERBOLIC_ARGS=(
    --use-lorentz-moe
    --use-hyperbolic
    --hyperbolic-curvature 1.0
    --expert-curvature-min 0.1
    --expert-curvature-max 2.0
)

# Export settings for display
export NUM_EXPERTS=4
export NUM_SHARED_EXPERTS=1
export MOE_ROUTER_TOPK=2
export MOE_LAYER_FREQ=2
export MOE_AUX_LOSS_COEFF=0.01
export HYPERBOLIC_CURVATURE=1.0
export EXPERT_CURVATURE_MIN=0.1
export EXPERT_CURVATURE_MAX=2.0

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
source "${SCRIPT_DIR}/../base/_train_moe_base_docker.sh"

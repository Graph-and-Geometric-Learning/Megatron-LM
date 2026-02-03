#!/bin/bash
# =============================================================================
# Lorentz MoE GPT: Test Config (HELM-MiCE)
# =============================================================================
# Small MoE model for quick testing with mock data.
# Uses Megatron's full pretrain infrastructure.
#
# Model: ~10M parameters
#   - Small test architecture
#   - 4 routed experts + 1 shared expert
#   - Top-2 routing per token
#   - Per-expert curvature (0.1 to 2.0)
# =============================================================================

set -e

# =============================================================================
# Export Config for Base Script
# =============================================================================
export MODEL_NAME="lorentz-moe-test"

# Model Architecture (small test) - Megatron-style args
export MODEL_ARGS=(
    --num-layers 4
    --hidden-size 256
    --num-attention-heads 4
    --ffn-hidden-size 768
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
    --vocab-size 1024
)

# MoE Configuration (Megatron-style)
export MOE_ARGS=(
    --num-experts 4
    --moe-shared-expert-intermediate-size 768
    --moe-router-topk 2
    --moe-layer-freq 1
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
export MOE_LAYER_FREQ=1
export MOE_AUX_LOSS_COEFF=0.01
export HYPERBOLIC_CURVATURE=1.0
export EXPERT_CURVATURE_MIN=0.1
export EXPERT_CURVATURE_MAX=2.0

# Training Hyperparameters (quick test)
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=2
export LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=100
export WARMUP_STEPS=10
export WEIGHT_DECAY=0.1
export GRAD_CLIP=1.0
export LOG_INTERVAL=5
export SAVE_INTERVAL=50

# Parallelism
export TP=1
export PP=1
export EP=1

# No data path - will use mock data
unset DATA_PATH

# =============================================================================
# Run
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_train_moe_base_docker.sh"

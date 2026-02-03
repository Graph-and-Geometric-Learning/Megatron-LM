#!/bin/bash
# =============================================================================
# Lorentz GPT Training Script (Qwen3-0.6B-like architecture)
# =============================================================================
# Hyperbolic transformer with Lorentz geometry
# Uses PyTorch DDP for distributed training
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Model config: test, qwen3_0.6b, qwen3_0.6b_small
MODEL_CONFIG=${MODEL_CONFIG:-"qwen3_0.6b_small"}

# Training settings
MAX_STEPS=${MAX_STEPS:-1000}
BATCH_SIZE=${BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
LR=${LR:-3e-4}
WARMUP_STEPS=${WARMUP_STEPS:-100}

# Distributed settings
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${MEGATRON_DIR}/checkpoints/lorentz_${MODEL_CONFIG}"}

# Data (optional - will use dummy data if not provided)
DATA_PATH=${DATA_PATH:-""}

# Logging
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}

# =============================================================================
# Distributed Arguments
# =============================================================================

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# =============================================================================
# Training Arguments
# =============================================================================

TRAINING_ARGS=(
    --config $MODEL_CONFIG
    --batch-size $BATCH_SIZE
    --micro-batch-size $MICRO_BATCH_SIZE
    --lr $LR
    --max-steps $MAX_STEPS
    --warmup-steps $WARMUP_STEPS
    --log-interval $LOG_INTERVAL
    --save-interval $SAVE_INTERVAL
    --checkpoint-dir "$CHECKPOINT_DIR"
    --bf16
)

# Add data path if provided
if [ -n "$DATA_PATH" ]; then
    TRAINING_ARGS+=(--data-path "$DATA_PATH")
fi

# =============================================================================
# Run Training
# =============================================================================

echo "=============================================="
echo "Lorentz GPT Training"
echo "=============================================="
echo "Model config: $MODEL_CONFIG"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Num nodes: $NUM_NODES"
echo "Batch size: $BATCH_SIZE (micro: $MICRO_BATCH_SIZE)"
echo "Max steps: $MAX_STEPS"
echo "Learning rate: $LR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Data path: ${DATA_PATH:-'(dummy data)'}"
echo "=============================================="

cd "$MEGATRON_DIR"

torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_lorentz_gpt.py \
    ${TRAINING_ARGS[@]}

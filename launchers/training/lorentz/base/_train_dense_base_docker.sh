#!/bin/bash
# =============================================================================
# Base Training Script: Lorentz Dense GPT (HELM-D) - Docker
# =============================================================================
# This is the base script - do not run directly.
# Use the config-specific launcher scripts.
#
# Expected environment variables from launcher:
#   MODEL_NAME        - Name for checkpoints
#   MODEL_ARGS        - Model architecture arguments
#   HYPERBOLIC_ARGS   - Hyperbolic geometry arguments
#   DATA_PATH         - Path to data inside container
#   DOCKER_DATA_MOUNT - Docker mount for data directory
#   BATCH_SIZE, MICRO_BATCH_SIZE, LR, MAX_STEPS, etc.
# =============================================================================

set -e

# =============================================================================
# Validate Required Variables
# =============================================================================

if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: This is a base script. Use a config-specific launcher."
    exit 1
fi

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${MEGATRON_DIR}/checkpoints/${MODEL_NAME}"}
mkdir -p "$CHECKPOINT_DIR"

# =============================================================================
# GPU Configuration
# =============================================================================

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# =============================================================================
# Docker Configuration
# =============================================================================

DOCKER_IMAGE=${DOCKER_IMAGE:-"nvcr.io/nvidia/pytorch:25.04-py3"}

# =============================================================================
# Training Arguments
# =============================================================================

TRAINING_ARGS=(
    --batch-size "${BATCH_SIZE:-16}"
    --micro-batch-size "${MICRO_BATCH_SIZE:-2}"
    --lr "${LR:-3e-4}"
    --min-lr "${MIN_LR:-3e-5}"
    --max-steps "${MAX_STEPS:-1000}"
    --warmup-steps "${WARMUP_STEPS:-100}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --grad-clip "${GRAD_CLIP:-1.0}"
    --log-interval "${LOG_INTERVAL:-10}"
    --save-interval "${SAVE_INTERVAL:-500}"
    --checkpoint-dir /workspace/megatron/checkpoints/${MODEL_NAME}
    --bf16
)

# Add data path if provided
if [ -n "$DATA_PATH" ]; then
    TRAINING_ARGS+=(--data-path "$DATA_PATH")
else
    TRAINING_ARGS+=(--num-samples "${NUM_SAMPLES:-10000}")
fi

# =============================================================================
# Print Configuration
# =============================================================================

echo "=============================================="
echo "Lorentz Dense GPT Training (HELM-D)"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Data: ${DATA_PATH:-'dummy data'}"
echo "GPUs: $GPUS_PER_NODE x $NUM_NODES nodes = $WORLD_SIZE total"
echo "Batch: ${BATCH_SIZE:-16} (micro: ${MICRO_BATCH_SIZE:-2})"
echo "Steps: ${MAX_STEPS:-1000} (warmup: ${WARMUP_STEPS:-100})"
echo "LR: ${LR:-3e-4} -> ${MIN_LR:-3e-5}"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Docker: $DOCKER_IMAGE"
echo "=============================================="

# =============================================================================
# Run Training in Docker
# =============================================================================

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    -v "${MEGATRON_DIR}:/workspace/megatron" \
    -v "${CHECKPOINT_DIR}:/workspace/megatron/checkpoints/${MODEL_NAME}" \
    ${DOCKER_DATA_MOUNT:-} \
    -w /workspace/megatron \
    "$DOCKER_IMAGE" \
    torchrun \
        --nproc_per_node "$GPUS_PER_NODE" \
        --nnodes "$NUM_NODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        pretrain_lorentz_gpt.py \
        "${MODEL_ARGS[@]}" \
        "${HYPERBOLIC_ARGS[@]}" \
        "${TRAINING_ARGS[@]}"

echo "=============================================="
echo "Training completed!"
echo "=============================================="

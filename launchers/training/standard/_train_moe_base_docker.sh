#!/bin/bash
# =============================================================================
# Base Training Script: Standard Megatron MoE GPT - Docker
# =============================================================================
# This is the base script - do not run directly.
# Use the config-specific launcher scripts.
#
# Uses Megatron's full pretrain infrastructure with:
# - Megatron's DistributedDataParallel (NOT PyTorch DDP)
# - Megatron's distributed optimizer
# - Expert parallelism via Megatron's MoE infrastructure
#
# Expected environment variables from launcher:
#   MODEL_NAME        - Name for checkpoints
#   MODEL_ARGS        - Model architecture arguments
#   MOE_ARGS          - MoE configuration arguments
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
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${MEGATRON_DIR}/checkpoints/${MODEL_NAME}"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${MEGATRON_DIR}/tensorboard/${MODEL_NAME}"}
mkdir -p "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

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
# Parallelism Configuration (Megatron-style)
# =============================================================================

TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-1}

# =============================================================================
# Training Arguments (Megatron-style)
# =============================================================================

# Calculate global batch size
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((BATCH_SIZE * WORLD_SIZE))}

TRAINING_ARGS=(
    # Batch configuration
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE:-1}"

    # Optimizer
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr "${LR:-3e-4}"
    --min-lr "${MIN_LR:-3e-5}"
    --lr-decay-style cosine
    --lr-warmup-iters "${WARMUP_STEPS:-100}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --clip-grad "${GRAD_CLIP:-1.0}"

    # Training duration
    --train-iters "${MAX_STEPS:-1000}"
    --eval-interval "${EVAL_INTERVAL:-1000}"
    --eval-iters "${EVAL_ITERS:-10}"

    # Precision
    --bf16

    # Parallelism
    --tensor-model-parallel-size "${TP}"
    --pipeline-model-parallel-size "${PP}"
    --expert-model-parallel-size "${EP}"

    # Checkpointing
    --save "${CHECKPOINT_DIR}"
    --save-interval "${SAVE_INTERVAL:-500}"

    # Logging
    --log-interval "${LOG_INTERVAL:-10}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --log-throughput

    # Use Megatron's distributed optimizer
    --use-distributed-optimizer
)

# Add data path if provided, otherwise use mock data
if [ -n "$DATA_PATH" ]; then
    TRAINING_ARGS+=(
        --data-path "$DATA_PATH"
        --split "949,50,1"
    )
else
    TRAINING_ARGS+=(
        --mock-data
    )
fi

# =============================================================================
# Print Configuration
# =============================================================================

echo "=============================================="
echo "Standard Megatron MoE GPT Training"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Data: ${DATA_PATH:-'mock data'}"
echo "GPUs: $GPUS_PER_NODE x $NUM_NODES nodes = $WORLD_SIZE total"
echo "Parallelism: TP=$TP, PP=$PP, EP=$EP"
echo "Batch: ${GLOBAL_BATCH_SIZE} global (micro: ${MICRO_BATCH_SIZE:-1})"
echo "Steps: ${MAX_STEPS:-1000} (warmup: ${WARMUP_STEPS:-100})"
echo "LR: ${LR:-3e-4} -> ${MIN_LR:-3e-5}"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Docker: $DOCKER_IMAGE"
echo "=============================================="
echo "MoE Settings:"
echo "  Experts: ${NUM_EXPERTS:-8}"
echo "  Activated: ${MOE_ROUTER_TOPK:-2}"
echo "  Layer freq: ${MOE_LAYER_FREQ:-2}"
echo "  Aux loss coeff: ${MOE_AUX_LOSS_COEFF:-0.01}"
echo "  Expert type: ${EXPERT_TYPE:-SequentialMLP}"
echo "=============================================="

# =============================================================================
# Run Training in Docker
# =============================================================================

# Combine all arguments into a single string for passing to docker
ALL_ARGS="${MODEL_ARGS[*]} ${MOE_ARGS[*]} ${TRAINING_ARGS[*]}"

# Optional: Install grouped_gemm package if needed
INSTALL_CMD="${INSTALL_GROUPED_GEMM:-}"

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${MEGATRON_DIR}:/workspace/megatron" \
    -v "${CHECKPOINT_DIR}:/workspace/checkpoints" \
    -v "${TENSORBOARD_DIR}:/workspace/tensorboard" \
    ${DOCKER_DATA_MOUNT:-} \
    -w /workspace/megatron \
    -e GPUS_PER_NODE="$GPUS_PER_NODE" \
    -e NUM_NODES="$NUM_NODES" \
    -e NODE_RANK="$NODE_RANK" \
    -e MASTER_ADDR="$MASTER_ADDR" \
    -e MASTER_PORT="$MASTER_PORT" \
    -e ALL_ARGS="$ALL_ARGS" \
    -e INSTALL_CMD="$INSTALL_CMD" \
    "$DOCKER_IMAGE" \
    bash -c '
        echo "Starting Standard MoE training..."

        # Install optional packages if specified
        if [ -n "$INSTALL_CMD" ]; then
            echo "Installing additional packages..."
            eval "$INSTALL_CMD"
        fi

        torchrun \
            --nproc_per_node $GPUS_PER_NODE \
            --nnodes $NUM_NODES \
            --node_rank $NODE_RANK \
            --master_addr $MASTER_ADDR \
            --master_port $MASTER_PORT \
            pretrain_gpt.py \
            $ALL_ARGS
    '

echo "=============================================="
echo "Training completed!"
echo "=============================================="

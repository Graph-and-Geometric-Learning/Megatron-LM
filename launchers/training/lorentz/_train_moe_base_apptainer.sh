#!/bin/bash
# =============================================================================
# Base Training Script: Lorentz MoE GPT (HELM-MiCE) - Apptainer
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
#   HYPERBOLIC_ARGS   - Hyperbolic geometry arguments
#   DATA_PATH         - Path to data inside container
#   HOST_DATA_DIR     - Host path to data directory
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
# Apptainer Configuration
# =============================================================================

APPTAINER_IMAGE=${APPTAINER_IMAGE:-"${HOME}/images/lorentz-moe_25.04.sif"}

# Check if image exists
if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: Apptainer image not found: ${APPTAINER_IMAGE}"
    echo "Run: ./launchers/setup/build_lorentz_moe_image.sh"
    exit 1
fi

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

    # Use local transformer implementation for Lorentz components
    --transformer-impl local
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
echo "Lorentz MoE GPT Training (HELM-MiCE)"
echo "Apptainer Mode"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Data: ${DATA_PATH:-'mock data'}"
echo "GPUs: $GPUS_PER_NODE x $NUM_NODES nodes = $WORLD_SIZE total"
echo "Parallelism: TP=$TP, PP=$PP, EP=$EP"
echo "Batch: ${GLOBAL_BATCH_SIZE} global (micro: ${MICRO_BATCH_SIZE:-1})"
echo "Steps: ${MAX_STEPS:-1000} (warmup: ${WARMUP_STEPS:-100})"
echo "LR: ${LR:-3e-4} -> ${MIN_LR:-3e-5}"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Apptainer: $APPTAINER_IMAGE"
echo "=============================================="
echo "MoE Settings:"
echo "  Experts: ${NUM_EXPERTS:-8}"
echo "  Activated: ${MOE_ROUTER_TOPK:-2}"
echo "  Shared: ${NUM_SHARED_EXPERTS:-1}"
echo "  Layer freq: ${MOE_LAYER_FREQ:-2}"
echo "  Aux loss coeff: ${MOE_AUX_LOSS_COEFF:-0.01}"
echo "=============================================="
echo "Hyperbolic Settings:"
echo "  Curvature: ${HYPERBOLIC_CURVATURE:-1.0}"
echo "  Expert curvature range: [${EXPERT_CURVATURE_MIN:-0.1}, ${EXPERT_CURVATURE_MAX:-2.0}]"
echo "=============================================="

# =============================================================================
# Run Training in Apptainer
# =============================================================================

# Combine all arguments into a single string
ALL_ARGS="${MODEL_ARGS[*]} ${MOE_ARGS[*]} ${HYPERBOLIC_ARGS[*]} ${TRAINING_ARGS[*]}"

# Build bind mounts
BIND_MOUNTS="--bind ${MEGATRON_DIR}:/workspace/megatron"
BIND_MOUNTS="${BIND_MOUNTS} --bind ${CHECKPOINT_DIR}:/workspace/checkpoints"
BIND_MOUNTS="${BIND_MOUNTS} --bind ${TENSORBOARD_DIR}:/workspace/tensorboard"

if [ -n "$HOST_DATA_DIR" ]; then
    BIND_MOUNTS="${BIND_MOUNTS} --bind ${HOST_DATA_DIR}:/workspace/data"
fi

echo "Starting Lorentz MoE training in Apptainer..."

apptainer exec --nv \
    ${BIND_MOUNTS} \
    --pwd /workspace/megatron \
    "${APPTAINER_IMAGE}" \
    bash -c "
        export PYTHONPATH=/workspace/megatron:\${PYTHONPATH}

        torchrun \
            --nproc_per_node ${GPUS_PER_NODE} \
            --nnodes ${NUM_NODES} \
            --node_rank ${NODE_RANK} \
            --master_addr ${MASTER_ADDR} \
            --master_port ${MASTER_PORT} \
            pretrain_lorentz_moe_gpt.py \
            ${ALL_ARGS}
    "

echo "=============================================="
echo "Training completed!"
echo "=============================================="
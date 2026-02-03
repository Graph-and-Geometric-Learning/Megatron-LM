#!/bin/bash
# =============================================================================
# Docker wrapper for Lorentz GPT training
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# Configuration
# =============================================================================
DOCKER_IMAGE="nvcr.io/nvidia/pytorch:25.04-py3"
CHECKPOINT_HOST_DIR="${CHECKPOINT_HOST_DIR:-${HOME}/checkpoints/lorentz}"

# Training config (can be overridden)
MODEL_CONFIG=${MODEL_CONFIG:-"qwen3_0.6b_small"}
MAX_STEPS=${MAX_STEPS:-500}
BATCH_SIZE=${BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-250}

# Create checkpoint directory if needed
mkdir -p "${CHECKPOINT_HOST_DIR}"

# =============================================================================
# Cleanup function for Ctrl+C
# =============================================================================
cleanup() {
    echo "Caught interrupt, stopping container..."
    docker kill lorentz_gpt_training 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

# =============================================================================
# Run with Docker
# =============================================================================
echo "=============================================="
echo "Starting Lorentz GPT training in Docker"
echo "=============================================="
echo "Image: ${DOCKER_IMAGE}"
echo "Megatron: ${MEGATRON_DIR}"
echo "Checkpoints: ${CHECKPOINT_HOST_DIR}"
echo "Model: ${MODEL_CONFIG}"
echo "GPUs: ${GPUS_PER_NODE}"
echo "Max steps: ${MAX_STEPS}"
echo "=============================================="

docker run --rm --gpus all \
    --name lorentz_gpt_training \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${MEGATRON_DIR}:/workspace/megatron" \
    -v "${CHECKPOINT_HOST_DIR}:/workspace/checkpoints" \
    -w /workspace/megatron \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e MODEL_CONFIG="${MODEL_CONFIG}" \
    -e MAX_STEPS="${MAX_STEPS}" \
    -e BATCH_SIZE="${BATCH_SIZE}" \
    -e MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
    -e GPUS_PER_NODE="${GPUS_PER_NODE}" \
    -e LOG_INTERVAL="${LOG_INTERVAL}" \
    -e SAVE_INTERVAL="${SAVE_INTERVAL}" \
    -e CHECKPOINT_DIR="/workspace/checkpoints" \
    "${DOCKER_IMAGE}" \
    bash -c '
        set -e
        export PYTHONPATH="/workspace/megatron:${PYTHONPATH}"

        echo "Python version: $(python --version)"
        echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"
        echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
        echo "GPU count: $(python -c "import torch; print(torch.cuda.device_count())")"
        echo ""

        # Run training
        bash launchers/training/train_lorentz_qwen3_0.6b.sh
    '

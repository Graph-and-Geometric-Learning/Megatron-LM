#!/bin/bash
# =============================================================================
# Docker wrapper for Qwen3-0.6B training
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# Configuration
# =============================================================================
DOCKER_IMAGE="nvcr.io/nvidia/pytorch:25.04-py3"
DATA_HOST_DIR="${HOME}/users/aosong/data"
CHECKPOINT_HOST_DIR="${HOME}/users/aosong/checkpoints"

# Create checkpoint directory if needed
mkdir -p "${CHECKPOINT_HOST_DIR}"

# =============================================================================
# Run with Docker
# =============================================================================
echo "=============================================="
echo "Starting Qwen3-0.6B training in Docker"
echo "=============================================="
echo "Image: ${DOCKER_IMAGE}"
echo "Megatron: ${MEGATRON_DIR}"
echo "Data: ${DATA_HOST_DIR}"
echo "Checkpoints: ${CHECKPOINT_HOST_DIR}"
echo "=============================================="

# Cleanup function for Ctrl+C
cleanup() {
    echo "Caught interrupt, stopping container..."
    docker kill megatron_training 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

docker run --rm --gpus all \
    --name megatron_training \
    --user "$(id -u):$(id -g)" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${MEGATRON_DIR}:/workspace/megatron" \
    -v "${DATA_HOST_DIR}:/workspace/data" \
    -v "${CHECKPOINT_HOST_DIR}:/workspace/checkpoints" \
    -w /workspace/megatron \
    -e PIP_CONSTRAINT="" \
    "${DOCKER_IMAGE}" \
    bash -c '
        set -e

        # Install missing packages (writes to container writable layer)
        echo "Installing dependencies..."
        pip install transformers wandb -q

        # Set PYTHONPATH to use megatron source
        export PYTHONPATH="/workspace/megatron:${PYTHONPATH}"

        # Run training
        bash launchers/training/train_qwen3_0.6b_redpajama.sh
    '

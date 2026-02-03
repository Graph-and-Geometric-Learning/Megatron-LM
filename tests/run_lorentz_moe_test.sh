#!/bin/bash
# Run Lorentz MoE integration tests in Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DOCKER_IMAGE=${DOCKER_IMAGE:-"nvcr.io/nvidia/pytorch:25.04-py3"}

echo "=============================================="
echo "Running Lorentz MoE Integration Tests"
echo "=============================================="
echo "Megatron dir: $MEGATRON_DIR"
echo "Docker image: $DOCKER_IMAGE"
echo "=============================================="

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    -v "${MEGATRON_DIR}:/workspace/megatron" \
    -w /workspace/megatron \
    "$DOCKER_IMAGE" \
    python tests/test_lorentz_moe_integration.py

echo "=============================================="
echo "Tests completed!"
echo "=============================================="

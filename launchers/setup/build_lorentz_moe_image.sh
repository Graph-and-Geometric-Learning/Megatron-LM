#!/bin/bash
# =============================================================================
# Build Lorentz MoE Docker Image and Convert to Apptainer
# =============================================================================
# This script:
# 1. Builds a Docker image with all dependencies for Lorentz MoE training
# 2. Converts it to Apptainer .sif format for use on Slurm clusters
#
# Usage:
#   ./launchers/setup/build_lorentz_moe_image.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${MEGATRON_DIR}/docker/Dockerfile.lorentz-moe"

IMAGE_NAME="lorentz-moe"
IMAGE_TAG="25.04"
SIF_OUTPUT_DIR="${HOME}/images"
SIF_NAME="${IMAGE_NAME}_${IMAGE_TAG}.sif"

echo "=============================================="
echo "Building Lorentz MoE Image"
echo "=============================================="
echo "Dockerfile: ${DOCKERFILE}"
echo "Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Output SIF: ${SIF_OUTPUT_DIR}/${SIF_NAME}"
echo "=============================================="

# Check Dockerfile exists
if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "Error: Dockerfile not found: ${DOCKERFILE}"
    exit 1
fi

# Step 1: Build Docker image
echo ""
echo "[Step 1/2] Building Docker image..."
docker build \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${MEGATRON_DIR}"

echo ""
echo "Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"

# Step 2: Convert to Apptainer
echo ""
echo "[Step 2/2] Converting to Apptainer SIF..."
mkdir -p "${SIF_OUTPUT_DIR}"

# Check if SIF already exists
if [[ -f "${SIF_OUTPUT_DIR}/${SIF_NAME}" ]]; then
    echo "Warning: ${SIF_OUTPUT_DIR}/${SIF_NAME} already exists."
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -f "${SIF_OUTPUT_DIR}/${SIF_NAME}"
fi

apptainer build "${SIF_OUTPUT_DIR}/${SIF_NAME}" "docker-daemon://${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "=============================================="
echo "Build Complete!"
echo "=============================================="
echo "Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Apptainer SIF: ${SIF_OUTPUT_DIR}/${SIF_NAME}"
echo ""
echo "To use in training scripts:"
echo "  export APPTAINER_IMAGE=${SIF_OUTPUT_DIR}/${SIF_NAME}"
echo "=============================================="
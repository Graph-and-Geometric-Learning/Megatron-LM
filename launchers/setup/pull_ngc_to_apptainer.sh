#!/bin/bash
# Script to pull NGC PyTorch container and convert to Apptainer .sif format
# Usage: ./pull_ngc_to_apptainer.sh [NGC_TAG]
# Example: ./pull_ngc_to_apptainer.sh 25.04-py3

set -euo pipefail

# Configuration
NGC_TAG="${1:-25.04-py3}"
NGC_IMAGE="nvcr.io/nvidia/pytorch:${NGC_TAG}"
OUTPUT_DIR="${HOME}/users/images"
SIF_NAME="pytorch_${NGC_TAG}.sif"
SIF_PATH="${OUTPUT_DIR}/${SIF_NAME}"

echo "============================================"
echo "NGC PyTorch to Apptainer Conversion Script"
echo "============================================"
echo "NGC Image: ${NGC_IMAGE}"
echo "Output: ${SIF_PATH}"
echo "============================================"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Check if .sif already exists
if [[ -f "${SIF_PATH}" ]]; then
    echo "Warning: ${SIF_PATH} already exists."
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -f "${SIF_PATH}"
fi

# Convert Docker image to Apptainer .sif
echo "Pulling and converting Docker image to Apptainer format..."
echo "This may take a while depending on your network speed..."

apptainer pull "${SIF_PATH}" "docker://${NGC_IMAGE}"

# Verify the image was created
if [[ -f "${SIF_PATH}" ]]; then
    echo "============================================"
    echo "Success! Apptainer image created:"
    echo "  ${SIF_PATH}"
    echo ""
    echo "Image size: $(du -h "${SIF_PATH}" | cut -f1)"
    echo "============================================"
    echo ""
    echo "To test the image, run:"
    echo "  apptainer exec --nv ${SIF_PATH} python -c 'import torch; print(torch.cuda.is_available())'"
    echo ""
    echo "To start an interactive shell:"
    echo "  apptainer shell --nv ${SIF_PATH}"
else
    echo "Error: Failed to create Apptainer image."
    exit 1
fi

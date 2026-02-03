#!/bin/bash
# =============================================================================
# Apptainer wrapper for Qwen3-0.6B training
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# Configuration
# =============================================================================
APPTAINER_IMAGE="${HOME}/users/images/pytorch_25.04-py3.sif"
DATA_HOST_DIR="${HOME}/users/aosong/data"
CHECKPOINT_HOST_DIR="${HOME}/users/aosong/checkpoints"

# Create checkpoint directory if needed
mkdir -p "${CHECKPOINT_HOST_DIR}"

# =============================================================================
# Check prerequisites
# =============================================================================
if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "Error: Apptainer image not found at ${APPTAINER_IMAGE}"
    echo "Run: ./launchers/setup/pull_ngc_to_apptainer.sh"
    exit 1
fi

# =============================================================================
# Run with Apptainer
# =============================================================================
echo "=============================================="
echo "Starting Qwen3-0.6B training in Apptainer"
echo "=============================================="
echo "Image: ${APPTAINER_IMAGE}"
echo "Megatron: ${MEGATRON_DIR}"
echo "Data: ${DATA_HOST_DIR}"
echo "Checkpoints: ${CHECKPOINT_HOST_DIR}"
echo "=============================================="

apptainer exec --nv \
    --bind "${MEGATRON_DIR}:/workspace/megatron" \
    --bind "${DATA_HOST_DIR}:/workspace/data" \
    --bind "${CHECKPOINT_HOST_DIR}:/workspace/checkpoints" \
    "${APPTAINER_IMAGE}" \
    bash -c '
        set -e
        cd /workspace/megatron

        # Enable user site-packages (NGC container disables by default)
        export PYTHONNOUSERSITE=0

        # Use source directly via PYTHONPATH
        export PYTHONPATH="/workspace/megatron:${PYTHONPATH}"

        # Install missing packages to user directory (container is read-only)
        pip install --user transformers -q 2>/dev/null || true

        # Run training
        bash launchers/training/train_qwen3_0.6b_redpajama.sh
    '

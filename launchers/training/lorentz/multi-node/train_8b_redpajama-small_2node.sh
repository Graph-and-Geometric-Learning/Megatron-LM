#!/bin/bash
#SBATCH --job-name=lorentz-8b-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/lorentz-8b-2node-%j.out
#SBATCH --error=logs/lorentz-8b-2node-%j.err
# =============================================================================
# Lorentz Dense GPT 8B: Qwen3-8B Architecture (Multi-Node)
# =============================================================================
# Multi-node training with 2 nodes (16 GPUs total).
#
# Model: Qwen3-8B Dense (NO MoE)
#   - 36 layers, hidden_size 4096, ffn 12288
#   - GQA: 32 query heads, 8 KV heads
#   - Lorentz (hyperbolic) geometry
#
# Usage:
#   sbatch train_8b_redpajama-small_2node.sh
# =============================================================================

set -e

# Create logs directory
mkdir -p logs

# =============================================================================
# Data Configuration (RedPajama-Small)
# =============================================================================
export HOST_DATA_DIR="/fsx/ubuntu/users/aosong/data"
HOST_DATA_PATH="${HOST_DATA_DIR}/processed_data/redpajama_small/stackexchange_text_document"

export DATA_PATH="/workspace/data/processed_data/redpajama_small/stackexchange_text_document"

# Validate data exists
if [ ! -f "${HOST_DATA_PATH}.bin" ]; then
    echo "ERROR: Data file not found: ${HOST_DATA_PATH}.bin"
    echo "Please prepare the data first."
    exit 1
fi

# =============================================================================
# Export Config for Base Script
# =============================================================================
export MODEL_NAME="lorentz-8b-2node"

# Model Architecture (Qwen3-8B Dense)
# 36 layers, hidden 4096, 32 heads, 8 KV heads, ffn 12288
export MODEL_ARGS_STR="--num-layers 36 --hidden-size 4096 --num-attention-heads 32 --group-query-attention --num-query-groups 8 --ffn-hidden-size 12288 --seq-length 2048 --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 1000000 --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --disable-bias-linear --no-bias-dropout-fusion --no-persist-layer-norm --untie-embeddings-and-output-weights --tokenizer-type NullTokenizer --vocab-size 151936"

# NO MoE Configuration (Dense model)
export MOE_ARGS_STR=""

# Hyperbolic Configuration (Lorentz-specific, dense mode)
export HYPERBOLIC_ARGS_STR="--use-hyperbolic --hyperbolic-curvature 1.0"

# Export settings for display
export HYPERBOLIC_CURVATURE=1.0

# Training Hyperparameters
export BATCH_SIZE=1
export MICRO_BATCH_SIZE=1
export LR=1e-4
export MIN_LR=1e-5
export MAX_STEPS=10000
export WARMUP_STEPS=500
export WEIGHT_DECAY=0.1
export GRAD_CLIP=1.0
export LOG_INTERVAL=10
export SAVE_INTERVAL=500

# Checkpoint directory (on shared /fsx filesystem)
export CHECKPOINT_DIR="/fsx/ubuntu/users/aosong/ft/hyper/Megatron-LM/checkpoints/${MODEL_NAME}"

# Parallelism for 2 nodes (16 GPUs)
export TP=8
export PP=1
export EP=1
export GPUS_PER_NODE=8

# =============================================================================
# Run with Slurm + Apptainer
# =============================================================================
MEGATRON_DIR="/fsx/ubuntu/users/aosong/ft/hyper/Megatron-LM"
BASE_SCRIPT="${MEGATRON_DIR}/launchers/training/lorentz/base/_train_moe_base_slurm.sh"

srun --ntasks-per-node=1 bash "${BASE_SCRIPT}"

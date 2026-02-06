#!/bin/bash
#SBATCH --job-name=mixtral-8x7b-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/mixtral-8x7b-2node-%j.out
#SBATCH --error=logs/mixtral-8x7b-2node-%j.err
#SBATCH --nodelist=ip-10-4-245-4,ip-10-4-255-224
# =============================================================================
# Mixtral 8x7B Style MoE: Official Megatron Config (Slurm Multi-Node)
# =============================================================================
# Based on: examples/mixtral/train_mixtral_8x7b_distributed.sh
# Adapted for 2 nodes (16 GPUs total) with RedPajama-Small dataset.
#
# Key differences from our 80M test scripts:
#   - Uses alltoall token dispatcher (more efficient for MoE)
#   - Uses aux_loss load balancing type
#   - EP-focused parallelism (EP=8, TP=1) instead of TP-focused
#   - Includes overlap optimizations
#
# Model: Mixtral 8x7B architecture (scaled down for testing)
#   - 8 experts, top-2 routing
#   - Expert parallelism = 8
# =============================================================================

set -e

mkdir -p logs

# =============================================================================
# Data Configuration (RedPajama-Small)
# =============================================================================
export HOST_DATA_DIR="/fsx/ubuntu/users/aosong/data"
HOST_DATA_PATH="${HOST_DATA_DIR}/processed_data/redpajama_small/stackexchange_text_document"

export DATA_PATH="/workspace/data/processed_data/redpajama_small/stackexchange_text_document"

if [ ! -f "${HOST_DATA_PATH}.bin" ]; then
    echo "ERROR: Data file not found: ${HOST_DATA_PATH}.bin"
    exit 1
fi

# =============================================================================
# Export Config for Base Script
# =============================================================================
export MODEL_NAME="mixtral-8x7b-style-2node"
export EXPERT_TYPE="TEGroupedMLP"

# Model Architecture (Mixtral-style, scaled down for 16 GPUs)
# Official Mixtral: 32 layers, 4096 hidden, 32 heads
# Scaled down version for testing
export MODEL_ARGS_STR="--num-layers 8 --hidden-size 512 --num-attention-heads 8 --group-query-attention --num-query-groups 4 --ffn-hidden-size 1792 --seq-length 512 --max-position-embeddings 512 --position-embedding-type rope --normalization RMSNorm --swiglu --disable-bias-linear --no-bias-dropout-fusion --no-persist-layer-norm --untie-embeddings-and-output-weights --tokenizer-type NullTokenizer --vocab-size 151936"

# MoE Configuration (Official Mixtral style)
# Key: alltoall dispatcher, aux_loss balancing, EP-focused
export MOE_ARGS_STR="--num-experts 8 --moe-router-topk 2 --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-2 --moe-token-dispatcher-type alltoall --moe-grouped-gemm --sequence-parallel"

# Export settings for display
export NUM_EXPERTS=8
export MOE_ROUTER_TOPK=2
export MOE_LAYER_FREQ=1
export MOE_AUX_LOSS_COEFF=0.01

# Training Hyperparameters
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=500
export WARMUP_STEPS=50
export WEIGHT_DECAY=0.1
export GRAD_CLIP=1.0
export LOG_INTERVAL=10
export SAVE_INTERVAL=100

# Checkpoint directory
export CHECKPOINT_DIR="/fsx/ubuntu/users/aosong/ft/hyper/Megatron-LM/checkpoints/${MODEL_NAME}"

# Parallelism (Official Mixtral style: EP-focused)
# With 16 GPUs and 8 experts: EP=8 means each expert on 2 GPUs
# TP=1, PP=1, EP=8, DP=2
export TP=1
export PP=1
export EP=8
export GPUS_PER_NODE=8

# =============================================================================
# Run with Slurm + Apptainer
# =============================================================================
MEGATRON_DIR="/fsx/ubuntu/users/aosong/ft/hyper/Megatron-LM"
BASE_SCRIPT="${MEGATRON_DIR}/launchers/training/standard/base/_train_moe_base_slurm.sh"

srun --ntasks-per-node=1 bash "${BASE_SCRIPT}"

#!/bin/bash
#SBATCH --job-name=lorentz-moe-80M-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/lorentz-moe-80M-2node-%j.out
#SBATCH --error=logs/lorentz-moe-80M-2node-%j.err
#SBATCH --nodelist=ip-10-4-245-4,ip-10-4-255-224
# =============================================================================
# Lorentz MoE GPT 80M: TEGroupedMLP Backend (Slurm Multi-Node)
# =============================================================================
# Multi-node training with 2 nodes (16 GPUs total).
# Hardcoded nodes for debugging: ip-10-4-245-4, ip-10-4-255-224
#
# Expert Type: LorentzTEGroupedMLP
#   - True Lorentz geometry via tangent space approximation
#   - Uses TransformerEngine's efficient grouped linear operations
#   - Per-expert curvatures distributed across range
#
# Model: ~80M total parameters
#   - 4 layers, hidden_size 256, ffn 512
#   - 4 routed experts + 1 shared expert
#   - Top-2 routing, MoE every 2nd layer
#   - Per-expert curvature (0.1 to 2.0)
#
# Usage:
#   sbatch train_moe_80M_te_redpajama-small_2node.sh
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
export MODEL_NAME="lorentz-moe-80M-te-2node"
export EXPERT_TYPE="LorentzTEGroupedMLP"

# Model Architecture (~80M params)
# Note: Use strings (not arrays) for srun compatibility
export MODEL_ARGS_STR="--num-layers 4 --hidden-size 256 --num-attention-heads 4 --group-query-attention --num-query-groups 2 --ffn-hidden-size 512 --seq-length 256 --max-position-embeddings 256 --position-embedding-type rope --normalization RMSNorm --swiglu --disable-bias-linear --no-bias-dropout-fusion --no-persist-layer-norm --untie-embeddings-and-output-weights --tokenizer-type NullTokenizer --vocab-size 151936"

# MoE Configuration (Megatron-style)
# --moe-grouped-gemm enables TEGroupedMLP (without --moe-use-legacy-grouped-gemm)
# --sequence-parallel required when using MoE + TP together
# Note: lorentz_moe_layer_specs.py now uses TESpecProvider for non-MoE components
# (LayerNorm, Attention) which supports sequence parallelism
export MOE_ARGS_STR="--num-experts 4 --moe-shared-expert-intermediate-size 512 --moe-router-topk 2 --moe-layer-freq 2 --moe-aux-loss-coeff 0.01 --moe-token-dispatcher-type allgather --moe-grouped-gemm --sequence-parallel"

# Hyperbolic Configuration (Lorentz-specific)
export HYPERBOLIC_ARGS_STR="--use-lorentz-moe --use-hyperbolic --hyperbolic-curvature 1.0 --expert-curvature-min 0.1 --expert-curvature-max 2.0"

# Export settings for display
export NUM_EXPERTS=4
export NUM_SHARED_EXPERTS=1
export MOE_ROUTER_TOPK=2
export MOE_LAYER_FREQ=2
export MOE_AUX_LOSS_COEFF=0.01
export HYPERBOLIC_CURVATURE=1.0
export EXPERT_CURVATURE_MIN=0.1
export EXPERT_CURVATURE_MAX=2.0

# Training Hyperparameters
# With 16 GPUs, global batch = 4 * 16 = 64
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=1000
export WARMUP_STEPS=100
export WEIGHT_DECAY=0.1
export GRAD_CLIP=1.0
export LOG_INTERVAL=10
export SAVE_INTERVAL=100

# Checkpoint directory (on shared /fsx filesystem)
export CHECKPOINT_DIR="/fsx/ubuntu/users/aosong/ft/hyper/Megatron-LM/checkpoints/${MODEL_NAME}"

# Parallelism
# With 16 GPUs (2 nodes x 8), we can test:
# - TP=2: Tensor parallel (splits model layers across 2 GPUs)
# - EP=2: Expert parallel (distributes 4 experts across 2 groups)
# Constraint: TP * PP * DP = WORLD_SIZE, and EP divides num_experts
#
# Testing TP=2 with --sequence-parallel (fix: use TESpecProvider for LayerNorm)
# This matches the standard MoE config for fair comparison
export TP=2
export PP=1
export EP=2
export GPUS_PER_NODE=8

# =============================================================================
# Run with Slurm + Apptainer
# =============================================================================
# Use absolute path since Slurm copies scripts to spool directory
MEGATRON_DIR="/fsx/ubuntu/users/aosong/ft/hyper/Megatron-LM"
BASE_SCRIPT="${MEGATRON_DIR}/launchers/training/lorentz/base/_train_moe_base_slurm.sh"

# Use srun to launch on each node
srun --ntasks-per-node=1 bash "${BASE_SCRIPT}"

#!/bin/bash
#SBATCH --job-name=lorentz-moe-30b-a3b-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/lorentz-moe-30b-a3b-2node-%j.out
#SBATCH --error=logs/lorentz-moe-30b-a3b-2node-%j.err
# =============================================================================
# Lorentz MoE GPT 30B-A3B: Qwen3-30B-A3B Architecture (Multi-Node)
# =============================================================================
# Multi-node training with 2 nodes (16 GPUs total).
#
# Expert Type: LorentzTEGroupedMLP
#   - True Lorentz geometry via tangent space approximation
#   - Uses TransformerEngine's efficient grouped linear operations
#   - Per-expert curvatures distributed across range
#
# Model: Qwen3-30B-A3B (30B total, 3B active per token)
#   - 48 layers, hidden_size 2048, ffn 6144
#   - 128 routed experts, top-8 routing
#   - MoE every layer (moe_layer_freq=1)
#   - GQA: 32 query heads, 4 KV heads
#   - Per-expert FFN size: 768
#
# Usage:
#   sbatch train_moe_30b-a3b_te_redpajama-small_2node.sh
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
export MODEL_NAME="lorentz-moe-30b-a3b-te-2node"
export EXPERT_TYPE="LorentzTEGroupedMLP"

# Model Architecture (Qwen3-30B-A3B)
# 48 layers, hidden 2048, 32 heads, 4 KV heads, ffn 6144
export MODEL_ARGS_STR="--num-layers 48 --hidden-size 2048 --num-attention-heads 32 --group-query-attention --num-query-groups 4 --ffn-hidden-size 6144 --seq-length 2048 --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 1000000 --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --disable-bias-linear --no-bias-dropout-fusion --no-persist-layer-norm --untie-embeddings-and-output-weights --tokenizer-type NullTokenizer --vocab-size 151936"

# MoE Configuration (Qwen3-30B-A3B exact)
# 128 experts, top-8 routing, MoE every layer, per-expert FFN size 768
export MOE_ARGS_STR="--num-experts 128 --moe-ffn-hidden-size 768 --moe-router-topk 8 --moe-layer-freq 1 --moe-aux-loss-coeff 0.001 --moe-token-dispatcher-type alltoall --moe-grouped-gemm --sequence-parallel"

# Hyperbolic Configuration (Lorentz-specific)
export HYPERBOLIC_ARGS_STR="--use-lorentz-moe --use-hyperbolic --hyperbolic-curvature 1.0 --expert-curvature-min 0.1 --expert-curvature-max 2.0"

# Export settings for display
export NUM_EXPERTS=128
export NUM_SHARED_EXPERTS=0
export MOE_ROUTER_TOPK=8
export MOE_LAYER_FREQ=1
export MOE_AUX_LOSS_COEFF=0.001
export HYPERBOLIC_CURVATURE=1.0
export EXPERT_CURVATURE_MIN=0.1
export EXPERT_CURVATURE_MAX=2.0

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
# TP=2, PP=1, EP=8 for expert parallelism (128 experts / 8 = 16 experts per EP rank)
export TP=2
export PP=1
export EP=8
export GPUS_PER_NODE=8

# =============================================================================
# Run with Slurm + Apptainer
# =============================================================================
MEGATRON_DIR="/fsx/ubuntu/users/aosong/ft/hyper/Megatron-LM"
BASE_SCRIPT="${MEGATRON_DIR}/launchers/training/lorentz/base/_train_moe_base_slurm.sh"

srun --ntasks-per-node=1 bash "${BASE_SCRIPT}"

#!/bin/bash
# =============================================================================
# Qwen3 MoE Dummy Training Script
# =============================================================================
# A scaled-down version of Qwen3-30B-A3B for testing MoE features
# Full model: 48 layers, 128 experts, hidden=2048
# Dummy model: 8 layers, 8 experts, hidden=1024
# =============================================================================

set -e

# =============================================================================
# Paths Configuration
# =============================================================================
MEGATRON_DIR="/workspace/megatron"
DATA_DIR="/workspace/data"
CHECKPOINT_DIR="/workspace/checkpoints"

EXP_NAME="qwen3_moe_dummy"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${EXP_NAME}"
DATA_CACHE_PATH="${CHECKPOINT_DIR}/data_cache_${EXP_NAME}"

# Use stackexchange data (has both .bin and .idx)
DATA_PATH="${DATA_DIR}/processed_data/redpajama_small/stackexchange_text_document"

# Tokenizer
TOKENIZER_MODEL="Qwen/Qwen3-0.6B"

# Create directories
mkdir -p "${CHECKPOINT_PATH}" "${DATA_CACHE_PATH}"

# =============================================================================
# Distributed Training Setup
# =============================================================================
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# =============================================================================
# Dummy Qwen3 MoE Model Architecture
# =============================================================================
# Scaled down from Qwen3-30B-A3B:
#   - Layers: 48 -> 8
#   - Hidden: 2048 -> 1024
#   - Experts: 128 -> 8
#   - TopK: 8 -> 2
#   - MoE FFN: 768 -> 384

SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=8192

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 8
    --hidden-size 1024
    --ffn-hidden-size 3072
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.02
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --qk-layernorm
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --no-position-embedding
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
)

# =============================================================================
# MoE Configuration
# =============================================================================
MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-ffn-hidden-size 384
    --moe-aux-loss-coeff 0.001
    --moe-router-load-balancing-type aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-layer-freq 1
    --moe-grouped-gemm
    --moe-permute-fusion
)

# =============================================================================
# Training Hyperparameters
# =============================================================================
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 5000
    --lr-decay-samples 4500
    --lr-warmup-samples 250
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# =============================================================================
# Model Parallelism
# =============================================================================
# EP=2 distributes experts across 2 GPUs
# TP=1 keeps attention on single GPU
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 2
)

# =============================================================================
# Data Arguments
# =============================================================================
DATA_ARGS=(
    --data-path "$DATA_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --data-cache-path "$DATA_CACHE_PATH"
    --split "99,1,0"
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers 2
    --vocab-size 151936
)

# =============================================================================
# Logging and Checkpointing
# =============================================================================
LOGGING_ARGS=(
    --log-interval 2
    --eval-iters 2
    --eval-interval 10
    --save-interval 50
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
)

# =============================================================================
# Run Training
# =============================================================================
echo "=============================================="
echo "Qwen3 MoE Dummy Training"
echo "=============================================="
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "GPUs: $GPUS_PER_NODE (EP=2)"
echo "Model: 8 layers, 8 experts, top-2 routing"
echo "Batch: micro=$MICRO_BATCH_SIZE, global=$GLOBAL_BATCH_SIZE"
echo "Seq length: $SEQ_LENGTH"
echo "=============================================="

cd "${MEGATRON_DIR}"

torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${LOGGING_ARGS[@]}

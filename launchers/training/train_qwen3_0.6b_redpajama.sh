#!/bin/bash
# =============================================================================
# Qwen3-0.6B Training Script - Simple Version
# =============================================================================
# 1 node, 2 GPUs, bf16, RedPajama small dataset
# =============================================================================

set -e

# =============================================================================
# Paths Configuration
# =============================================================================
MEGATRON_DIR="/workspace/megatron"
DATA_DIR="/workspace/data"
CHECKPOINT_DIR="/workspace/checkpoints"

EXP_NAME="qwen3_0.6b_redpajama"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${EXP_NAME}"
DATA_CACHE_PATH="${CHECKPOINT_DIR}/data_cache_${EXP_NAME}"

# Use stackexchange data (has both .bin and .idx)
DATA_PATH="${DATA_DIR}/processed_data/redpajama_small/stackexchange_text_document"

# Tokenizer (HuggingFace will download if needed)
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
# Qwen3-0.6B Model Architecture
# =============================================================================
# From HuggingFace Qwen/Qwen3-0.6B config.json:
#   hidden_size: 1024
#   num_hidden_layers: 28
#   num_attention_heads: 16
#   num_key_value_heads: 8
#   intermediate_size: 3072
#   vocab_size: 151936
#   max_position_embeddings: 40960
#   rope_theta: 1000000
#   head_dim: 128

SEQ_LENGTH=2048  # Reduced for testing
MAX_POSITION_EMBEDDINGS=40960

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 28
    --hidden-size 1024
    --ffn-hidden-size 3072
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 8
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
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --no-position-embedding
)

# =============================================================================
# Training Hyperparameters
# =============================================================================
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 10000
    --lr-decay-samples 9000
    --lr-warmup-samples 500
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
# Model Parallelism (TP=1 for simplicity, use DP across 2 GPUs)
# =============================================================================
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
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
    --eval-interval 2
    --save-interval 2
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
)

# =============================================================================
# Run Training
# =============================================================================
echo "=============================================="
echo "Qwen3-0.6B Training"
echo "=============================================="
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "GPUs: $GPUS_PER_NODE"
echo "Batch: micro=$MICRO_BATCH_SIZE, global=$GLOBAL_BATCH_SIZE"
echo "Seq length: $SEQ_LENGTH"
echo "=============================================="

cd "${MEGATRON_DIR}"

torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${LOGGING_ARGS[@]}

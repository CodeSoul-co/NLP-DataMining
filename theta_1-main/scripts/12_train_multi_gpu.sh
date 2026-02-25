#!/bin/bash
# =============================================================================
# THETA Multi-GPU Training Script (DistributedDataParallel)
# =============================================================================
# Train THETA model using multiple GPUs with PyTorch DDP
#
# This script uses torchrun to launch distributed training across GPUs.
# Each GPU processes a portion of the batch, gradients are synchronized.
#
# Requirements:
#   - Multiple NVIDIA GPUs with CUDA support
#   - PyTorch with DDP support
#   - NCCL backend for GPU communication
#
# Scaling:
#   - Effective batch size = batch_size * num_gpus
#   - Learning rate may need adjustment for larger effective batch sizes
#
# Usage:
#   ./12_train_multi_gpu.sh --dataset <name> --num_gpus <n> [options]
#
# Examples:
#   ./12_train_multi_gpu.sh --dataset hatespeech --num_gpus 2 --model_size 0.6B
#   ./12_train_multi_gpu.sh --dataset mental_health --num_gpus 4 --model_size 4B --mode supervised
# =============================================================================

set -e

# Default values
DATASET=""
MODEL_SIZE="0.6B"
MODE="zero_shot"
NUM_TOPICS=20
EPOCHS=100
BATCH_SIZE=64
HIDDEN_DIM=512
LEARNING_RATE=0.002
NUM_GPUS=2
MASTER_PORT=29500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --num_topics) NUM_TOPICS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --hidden_dim) HIDDEN_DIM="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --master_port) MASTER_PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> --num_gpus <n> [options]"
            echo ""
            echo "Multi-GPU Training Options:"
            echo "  --dataset       Dataset name (required)"
            echo "  --num_gpus      Number of GPUs to use (default: 2)"
            echo "  --master_port   Master port for DDP (default: 29500)"
            echo ""
            echo "Model Options:"
            echo "  --model_size    Qwen model size: 0.6B, 4B, 8B (default: 0.6B)"
            echo "  --mode          Training mode: zero_shot, supervised, unsupervised (default: zero_shot)"
            echo "  --num_topics    Number of topics (default: 20)"
            echo "  --epochs        Training epochs (default: 100)"
            echo "  --batch_size    Batch size per GPU (default: 64)"
            echo "  --hidden_dim    Hidden dimension (default: 512)"
            echo "  --learning_rate Learning rate (default: 0.002)"
            echo ""
            echo "Examples:"
            echo "  # Train with 2 GPUs"
            echo "  $0 --dataset hatespeech --num_gpus 2 --num_topics 20"
            echo ""
            echo "  # Train with 4 GPUs and larger batch"
            echo "  $0 --dataset mental_health --num_gpus 4 --batch_size 128 --epochs 200"
            echo ""
            echo "  # Train with custom port (useful when running multiple jobs)"
            echo "  $0 --dataset socialTwitter --num_gpus 2 --master_port 29501"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo "=========================================="
echo "THETA Multi-GPU Training (DDP)"
echo "=========================================="
echo "Dataset:     $DATASET"
echo "Model Size:  $MODEL_SIZE"
echo "Mode:        $MODE"
echo "Num Topics:  $NUM_TOPICS"
echo "Epochs:      $EPOCHS"
echo "Batch Size:  $BATCH_SIZE (per GPU)"
echo "Num GPUs:    $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo ""

cd /root/autodl-tmp/ETM

# Use torchrun for distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main.py train \
    --dataset $DATASET \
    --mode $MODE \
    --num_topics $NUM_TOPICS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --learning_rate $LEARNING_RATE

echo ""
echo "=========================================="
echo "Multi-GPU Training completed!"
echo "Results saved to: /root/autodl-tmp/result/$MODEL_SIZE/$DATASET/$MODE/"
echo "=========================================="

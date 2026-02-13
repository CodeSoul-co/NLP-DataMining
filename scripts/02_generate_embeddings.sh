#!/bin/bash
# =============================================================================
# Embedding Generation Script
# =============================================================================
# Generate document embeddings using Qwen3-Embedding model.
#
# Modes:
#   - zero_shot:      Direct embedding without training (fastest)
#   - unsupervised:   LoRA fine-tuning with autoregressive LM loss (no labels needed)
#   - supervised:     LoRA fine-tuning with classification loss (requires labels)
#
# Usage:
#   bash scripts/02_generate_embeddings.sh --dataset <name> --mode <mode> [options]
#
# Examples:
#   bash scripts/02_generate_embeddings.sh --dataset edu_data --mode zero_shot
#   bash scripts/02_generate_embeddings.sh --dataset edu_data --mode unsupervised --epochs 10
#   bash scripts/02_generate_embeddings.sh --dataset hatespeech --mode supervised --epochs 10
# =============================================================================

set -e

# Default values
DATASET=""
MODE="zero_shot"
MODEL_SIZE="0.6B"
MODEL_PATH=""  # Auto-set based on MODEL_SIZE below
MAX_LENGTH=512
BATCH_SIZE=16
EPOCHS=10
LEARNING_RATE=2e-5
MAX_SAMPLES=""
USE_LORA=true
GPU=0
DEV=false
LABEL_COLUMN=""
EXP_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --max_length) MAX_LENGTH="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        --no_lora) USE_LORA=false; shift ;;
        --label_column) LABEL_COLUMN="$2"; shift 2 ;;
        --exp_dir) EXP_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --dev) DEV=true; shift ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> --mode <mode> [options]"
            echo ""
            echo "Options:"
            echo "  --dataset       Dataset name (required). Use 'all' for all datasets"
            echo "  --mode          Embedding mode: zero_shot, unsupervised, supervised (default: zero_shot)"
            echo "  --model_size    Qwen model size: 0.6B, 4B, 8B (default: 0.6B)"
            echo "  --model_path    Path to Qwen model (default: /root/autodl-tmp/qwen3_embedding_0.6B)"
            echo "  --max_length    Max sequence length (default: 512)"
            echo "  --batch_size    Batch size (default: 16)"
            echo "  --epochs        Training epochs for supervised/unsupervised (default: 10)"
            echo "  --learning_rate Learning rate for fine-tuning (default: 2e-5)"
            echo "  --max_samples   Max samples to process (default: all)"
            echo "  --no_lora       Disable LoRA, use full fine-tuning"
            echo "  --label_column  Label column for supervised mode (e.g. province, year)"
            echo "  --gpu           GPU device ID (default: 0)"
            echo "  --dev           Enable debug mode"
            echo ""
            echo "Modes:"
            echo "  zero_shot       Use pre-trained Qwen directly (no training)"
            echo "  unsupervised    LoRA fine-tuning with autoregressive LM loss"
            echo "  supervised      LoRA fine-tuning with classification loss (needs labels)"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset edu_data --mode zero_shot"
            echo "  $0 --dataset edu_data --mode unsupervised --epochs 10"
            echo "  $0 --dataset hatespeech --mode supervised --epochs 10"
            echo "  $0 --dataset edu_data --mode supervised --label_column province --epochs 10"
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

# Auto-set MODEL_PATH based on MODEL_SIZE if not explicitly provided
if [ -z "$MODEL_PATH" ]; then
    case $MODEL_SIZE in
        0.6B) MODEL_PATH="/root/autodl-tmp/embedding_models/qwen3_embedding_0.6B" ;;
        4B)   MODEL_PATH="/root/autodl-tmp/embedding_models/qwen3_embedding_4B" ;;
        8B)   MODEL_PATH="/root/autodl-tmp/embedding_models/qwen3_embedding_8B" ;;
        *)    echo "Error: Unknown model size '$MODEL_SIZE'"; exit 1 ;;
    esac
    # Fallback to legacy path if embedding_models path doesn't exist
    if [ ! -d "$MODEL_PATH" ]; then
        LEGACY_PATH="/root/autodl-tmp/qwen3_embedding_${MODEL_SIZE}"
        if [ -d "$LEGACY_PATH" ]; then
            MODEL_PATH="$LEGACY_PATH"
        fi
    fi
fi

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU

echo "=========================================="
echo "Embedding Generation"
echo "=========================================="
echo "Dataset:      $DATASET"
echo "Mode:         $MODE"
echo "Model Size:   $MODEL_SIZE"
echo "Model Path:   $MODEL_PATH"
echo "Max Length:    $MAX_LENGTH"
echo "Batch Size:   $BATCH_SIZE"
if [ "$MODE" != "zero_shot" ]; then
    echo "Epochs:       $EPOCHS"
    echo "Learning Rate: $LEARNING_RATE"
    echo "LoRA:         $USE_LORA"
fi
if [ "$MODE" = "supervised" ]; then
    if [ -n "$LABEL_COLUMN" ]; then
        echo "Label Column: $LABEL_COLUMN"
    else
        echo "Label Column: (interactive selection)"
        LABEL_COLUMN="select"
    fi
fi
echo ""

# Validate mode
case $MODE in
    zero_shot|supervised|unsupervised)
        ;;
    *)
        echo "Error: Unknown mode '$MODE'. Must be: zero_shot, supervised, unsupervised"
        exit 1
        ;;
esac

# Check model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download the Qwen3-Embedding model first."
    exit 1
fi

# Build command
CMD="python main.py --mode $MODE --dataset $DATASET"
CMD="$CMD --model_path $MODEL_PATH --model_size $MODEL_SIZE"
CMD="$CMD --max_length $MAX_LENGTH --batch_size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS --learning_rate $LEARNING_RATE"

if [ "$USE_LORA" = false ]; then
    CMD="$CMD --no_lora"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$LABEL_COLUMN" ]; then
    CMD="$CMD --label_column $LABEL_COLUMN"
fi

if [ -n "$EXP_DIR" ]; then
    CMD="$CMD --exp_dir $EXP_DIR"
fi

if [ "$DEV" = true ]; then
    CMD="$CMD --dev"
fi

echo "Running: $CMD"
echo ""

cd /root/autodl-tmp/embedding
eval $CMD

echo ""
echo "=========================================="
echo "Embedding generation completed!"
if [ -n "$EXP_DIR" ]; then
    echo "Results saved to: $EXP_DIR/embeddings/"
else
    echo "Results saved to: /root/autodl-tmp/result/$MODEL_SIZE/$DATASET/$MODE/embeddings/"
fi
echo "=========================================="

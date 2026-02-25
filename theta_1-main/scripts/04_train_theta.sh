#!/bin/bash
# =============================================================================
# THETA Model Training Script
# =============================================================================
# Train THETA (Textual Hybrid Embedding-based Topic Analysis) model
# using Qwen embeddings for document representation
#
# Model Sizes:
#   - 0.6B: 1024-dim embeddings (fastest, good for prototyping)
#   - 4B:   2560-dim embeddings (balanced performance)
#   - 8B:   4096-dim embeddings (best quality, requires more GPU memory)
#
# Embedding Modes:
#   - zero_shot:    No training, use pre-trained Qwen directly
#   - supervised:   Fine-tune with labeled data (requires labels)
#   - unsupervised: Fine-tune with SimCSE (no labels needed)
#
# Usage (non-interactive, all parameters via command line):
#   ./04_train_theta.sh --dataset <name> [options]
#
# Examples:
#   ./04_train_theta.sh --dataset hatespeech --model_size 0.6B --mode zero_shot --num_topics 20
#   ./04_train_theta.sh --dataset edu_data --mode zero_shot --num_topics 20 --language zh
#   ./04_train_theta.sh --dataset edu_data --mode unsupervised --num_topics 20 --epochs 50 --skip-viz
# =============================================================================

set -e

# Base directories - support environment variables for flexible deployment
BASE_DIR=${ETM_BASE_DIR:-/root/autodl-tmp}
RESULT_DIR=${ETM_RESULT_DIR:-$BASE_DIR/result}
ETM_CODE_DIR=${ETM_CODE_DIR:-$BASE_DIR/ETM}

# Default values
DATASET=""
MODEL_SIZE="0.6B"
MODE="zero_shot"
NUM_TOPICS=20
EPOCHS=100
BATCH_SIZE=64
HIDDEN_DIM=512
LEARNING_RATE=0.002
KL_START=0.0
KL_END=1.0
KL_WARMUP=50
PATIENCE=10
GPU=0
LANGUAGE="en"
SKIP_TRAIN=false
SKIP_VIZ=false
DATA_EXP=""
EXP_NAME=""

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
        --kl_start) KL_START="$2"; shift 2 ;;
        --kl_end) KL_END="$2"; shift 2 ;;
        --kl_warmup) KL_WARMUP="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-viz) SKIP_VIZ=true; shift ;;
        --data_exp) DATA_EXP="$2"; shift 2 ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> [options]"
            echo ""
            echo "Required:"
            echo "  --dataset       Dataset name (must have prepared data in result/ directory)"
            echo ""
            echo "Model Options:"
            echo "  --model_size    Qwen model size: 0.6B, 4B, 8B (default: 0.6B)"
            echo "  --mode          Embedding mode: zero_shot, supervised, unsupervised (default: zero_shot)"
            echo "  --data_exp      Data experiment ID from 03_prepare_data.sh (default: auto-select latest)"
            echo ""
            echo "Training Options:"
            echo "  --num_topics    Number of topics K (default: 20)"
            echo "  --epochs        Training epochs (default: 100)"
            echo "  --batch_size    Batch size (default: 64)"
            echo "  --hidden_dim    Hidden dimension (default: 512)"
            echo "  --learning_rate Learning rate (default: 0.002)"
            echo "  --patience      Early stopping patience (default: 10)"
            echo ""
            echo "KL Annealing Options:"
            echo "  --kl_start      KL annealing start weight (default: 0.0)"
            echo "  --kl_end        KL annealing end weight (default: 1.0)"
            echo "  --kl_warmup     KL warmup epochs (default: 50)"
            echo ""
            echo "Other Options:"
            echo "  --gpu           GPU device ID (default: 0)"
            echo "  --language      Visualization language: en, zh (default: en)"
            echo "                  Controls chart titles, axis labels, and legend text."
            echo "  --skip-train    Skip training, only evaluate existing model"
            echo "  --skip-viz      Skip visualization generation"
            echo "  --exp_name      Experiment name tag (default: auto-generated)"
            echo ""
            echo "Examples:"
            echo "  # Basic zero_shot training"
            echo "  $0 --dataset hatespeech --model_size 0.6B --mode zero_shot --num_topics 20"
            echo ""
            echo "  # Chinese dataset with Chinese visualization"
            echo "  $0 --dataset edu_data --mode zero_shot --num_topics 20 --language zh"
            echo ""
            echo "  # Unsupervised mode, skip visualization"
            echo "  $0 --dataset edu_data --mode unsupervised --num_topics 20 --epochs 50 --skip-viz"
            echo ""
            echo "  # Specify data experiment and custom KL annealing"
            echo "  $0 --dataset edu_data --mode zero_shot --num_topics 20 \\"
            echo "    --data_exp exp_20260208_151906_vocab3500_theta_0.6B_zero_shot \\"
            echo "    --kl_start 0.1 --kl_end 0.8 --kl_warmup 40 --language zh"
            echo ""
            echo "  # Skip training, only evaluate and visualize existing model"
            echo "  $0 --dataset edu_data --mode zero_shot --skip-train --language zh"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required parameters
if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# Auto-select data_exp if not provided
if [ -z "$DATA_EXP" ]; then
    THETA_DATA_DIR="$RESULT_DIR/$MODEL_SIZE/$DATASET/data"
    if [ -d "$THETA_DATA_DIR" ]; then
        # Find latest exp that has embeddings
        for EXP_CANDIDATE in $(ls -dt "$THETA_DATA_DIR"/exp_* 2>/dev/null); do
            # Check if this exp has embeddings
            if find "$EXP_CANDIDATE/embeddings" -name "*.npy" 2>/dev/null | head -1 | grep -q .; then
                LATEST_EXP="$EXP_CANDIDATE"
                break
            fi
        done
        if [ -n "$LATEST_EXP" ]; then
            DATA_EXP=$(basename "$LATEST_EXP")
            echo "[INFO] Auto-selected data experiment: $DATA_EXP"
            # Read mode from config if not explicitly set
            EXP_CONFIG="$THETA_DATA_DIR/$DATA_EXP/config.json"
            if [ -f "$EXP_CONFIG" ]; then
                CONFIG_MODE=$(python3 -c "import json; print(json.load(open('$EXP_CONFIG')).get('mode',''))" 2>/dev/null)
                if [ -n "$CONFIG_MODE" ] && [ "$MODE" = "zero_shot" ]; then
                    MODE="$CONFIG_MODE"
                    echo "[INFO] Mode from data experiment config: $MODE"
                fi
            fi
        fi
    fi
fi

echo "=========================================="
echo "THETA Model Training"
echo "=========================================="
echo "Dataset:     $DATASET"
echo "Model Size:  $MODEL_SIZE"
echo "Mode:        $MODE"
echo "Num Topics:  $NUM_TOPICS"
echo "Epochs:      $EPOCHS"
echo "Language:    $LANGUAGE"
echo ""

# Data path checking - supports both exp structure and legacy
THETA_DATA_BASE="$RESULT_DIR/$MODEL_SIZE/$DATASET"

if [ -n "$DATA_EXP" ]; then
    # New exp structure
    THETA_EXP_DIR="$THETA_DATA_BASE/data/$DATA_EXP"
    THETA_EMB_DIR="$THETA_EXP_DIR/embeddings"
    THETA_BOW_DIR="$THETA_EXP_DIR/bow"
    echo "Data experiment: $DATA_EXP"
else
    # Legacy structure
    THETA_EMB_DIR="$THETA_DATA_BASE/$MODE/embeddings"
    THETA_BOW_DIR="$THETA_DATA_BASE/bow"
fi

echo "Data path check:"
# Find embeddings file (check multiple possible locations)
EMB_FILE=$(find "$THETA_EMB_DIR" -name "*.npy" 2>/dev/null | head -1)
if [ -n "$EMB_FILE" ]; then
    echo "  ✓ Embeddings: $EMB_FILE"
else
    echo "  ✗ Embeddings not found in: $THETA_EMB_DIR"
    echo "    Please run first: bash scripts/03_prepare_data.sh --model theta"
    exit 1
fi

if [ -f "$THETA_BOW_DIR/bow_matrix.npy" ]; then
    echo "  ✓ BOW: $THETA_BOW_DIR/bow_matrix.npy"
else
    echo "  ✗ BOW not found: $THETA_BOW_DIR/bow_matrix.npy"
    echo "    Please run first: bash scripts/03_prepare_data.sh --model theta"
    exit 1
fi

# Check vocab_embeddings
if [ -f "$THETA_BOW_DIR/vocab_embeddings.npy" ]; then
    echo "  ✓ Vocab Embeddings: $THETA_BOW_DIR/vocab_embeddings.npy"
else
    echo "  ✗ Vocab Embeddings not found"
fi
echo ""

cd $ETM_CODE_DIR

# Build command
CMD="python run_pipeline.py --dataset $DATASET --models theta"
CMD="$CMD --model_size $MODEL_SIZE --mode $MODE"
CMD="$CMD --num_topics $NUM_TOPICS --epochs $EPOCHS --batch_size $BATCH_SIZE"
CMD="$CMD --hidden_dim $HIDDEN_DIM --learning_rate $LEARNING_RATE"
CMD="$CMD --kl_start $KL_START --kl_end $KL_END --kl_warmup $KL_WARMUP"
CMD="$CMD --patience $PATIENCE --gpu $GPU --language $LANGUAGE"

# Pass data_exp if using new exp structure
if [ -n "$DATA_EXP" ]; then
    CMD="$CMD --data_exp $DATA_EXP"
fi

if [ "$SKIP_TRAIN" = true ]; then
    CMD="$CMD --skip-train"
fi

if [ "$SKIP_VIZ" = true ]; then
    CMD="$CMD --skip-viz"
fi

# Auto-generate exp_name from parameters if not provided
if [ -z "$EXP_NAME" ]; then
    EXP_NAME="k${NUM_TOPICS}_e${EPOCHS}_${MODE}"
fi
CMD="$CMD --exp_name $EXP_NAME"

echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: $RESULT_DIR/$MODEL_SIZE/$DATASET/models/"
echo "=========================================="

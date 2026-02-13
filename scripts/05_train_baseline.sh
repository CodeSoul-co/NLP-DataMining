#!/bin/bash
# =============================================================================
# Baseline Model Training Script
# =============================================================================
# Train baseline topic models for comparison with THETA
#
# Supported Models (12 total):
#   Traditional Models:
#     - lda:      Latent Dirichlet Allocation (sklearn, fast, no GPU needed)
#     - hdp:      Hierarchical Dirichlet Process (auto topic number)
#     - stm:      Structural Topic Model (requires covariates/metadata)
#     - btm:      Biterm Topic Model (for short texts like tweets)
#
#   Neural Models:
#     - etm:      Embedded Topic Model (Word2Vec + VAE)
#     - ctm:      Contextualized Topic Model (SBERT + VAE)
#     - dtm:      Dynamic Topic Model (time-aware, requires timestamp)
#     - nvdm:     Neural Variational Document Model (basic VAE)
#     - gsm:      Gaussian Softmax Model (NVDM with softmax constraint)
#     - prodlda:  Product of Experts LDA (better topic separation)
#     - bertopic: BERT-based topic modeling (auto topic number)
#
# Usage (non-interactive, all parameters via command line):
#   ./05_train_baseline.sh --dataset <name> --models <model_list> [options]
#
# Examples:
#   ./05_train_baseline.sh --dataset hatespeech --models lda --num_topics 20
#   ./05_train_baseline.sh --dataset hatespeech --models lda,hdp,btm --num_topics 20
#   ./05_train_baseline.sh --dataset hatespeech --models nvdm,gsm,prodlda --num_topics 20 --epochs 100
#   ./05_train_baseline.sh --dataset edu_data --models dtm --num_topics 20 --language zh
# =============================================================================

set -e

# Default values
DATASET=""
MODELS="lda"
NUM_TOPICS=20
VOCAB_SIZE=5000
EPOCHS=100
BATCH_SIZE=64
HIDDEN_DIM=512
LEARNING_RATE=0.002
GPU=0
LANGUAGE="en"
SKIP_TRAIN=false
SKIP_VIZ=false
DATA_EXP=""
EXP_NAME=""

# Model-specific defaults
MAX_ITER=100
MAX_TOPICS=150
N_ITER=100
ALPHA=1.0
BETA=0.01
INFERENCE_TYPE="zeroshot"
DROPOUT=0.2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --num_topics) NUM_TOPICS="$2"; shift 2 ;;
        --vocab_size) VOCAB_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --hidden_dim) HIDDEN_DIM="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-viz) SKIP_VIZ=true; shift ;;
        --with-viz) SKIP_VIZ=false; shift ;;
        --max_iter) MAX_ITER="$2"; shift 2 ;;
        --max_topics) MAX_TOPICS="$2"; shift 2 ;;
        --n_iter) N_ITER="$2"; shift 2 ;;
        --alpha) ALPHA="$2"; shift 2 ;;
        --beta) BETA="$2"; shift 2 ;;
        --inference_type) INFERENCE_TYPE="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --data_exp) DATA_EXP="$2"; shift 2 ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> --models <model_list> [options]"
            echo ""
            echo "Supported Models:"
            echo "  Traditional: lda, hdp, stm (requires covariates), btm"
            echo "  Neural:      etm, ctm, dtm, nvdm, gsm, prodlda, bertopic"
            echo ""
            echo "Required:"
            echo "  --dataset       Dataset name (must have prepared data in result/baseline/)"
            echo "  --models        Model list, comma-separated (e.g. lda,hdp,nvdm)"
            echo ""
            echo "Training Options:"
            echo "  --num_topics    Number of topics (default: 20, ignored for hdp/bertopic)"
            echo "  --vocab_size    Vocabulary size (default: 5000)"
            echo "  --epochs        Training epochs for neural models (default: 100)"
            echo "  --batch_size    Batch size (default: 64)"
            echo "  --hidden_dim    Hidden dimension (default: 512)"
            echo "  --learning_rate Learning rate (default: 0.002)"
            echo "  --dropout       Dropout rate for neural models (default: 0.2)"
            echo ""
            echo "Model-specific Options:"
            echo "  --max_iter      Max iterations for LDA (default: 100)"
            echo "  --max_topics    Max topics for HDP (default: 150)"
            echo "  --n_iter        Gibbs sampling iterations for BTM (default: 100)"
            echo "  --alpha         Alpha prior for HDP/BTM (default: 1.0)"
            echo "  --beta          Beta prior for BTM (default: 0.01)"
            echo "  --inference_type CTM inference type: zeroshot, combined (default: zeroshot)"
            echo ""
            echo "Other Options:"
            echo "  --gpu           GPU device ID (default: 0)"
            echo "  --language      Visualization language: en, zh (default: en)"
            echo "  --skip-train    Skip training, only evaluate existing model"
            echo "  --skip-viz      Skip visualization (default: skipped)"
            echo "  --with-viz      Enable visualization after training"
            echo "  --data_exp      Data experiment ID (default: auto-select latest)"
            echo "  --exp_name      Experiment name tag (default: auto-generated)"
            echo ""
            echo "Examples:"
            echo "  # Train single model"
            echo "  $0 --dataset hatespeech --models lda --num_topics 20"
            echo ""
            echo "  # Train multiple traditional models"
            echo "  $0 --dataset hatespeech --models lda,hdp,btm --num_topics 20"
            echo ""
            echo "  # Train multiple neural models"
            echo "  $0 --dataset hatespeech --models nvdm,gsm,prodlda --num_topics 20 --epochs 150"
            echo ""
            echo "  # Train DTM with Chinese visualization"
            echo "  $0 --dataset edu_data --models dtm --num_topics 20 --language zh"
            echo ""
            echo "  # Specify data experiment"
            echo "  $0 --dataset edu_data --models lda,hdp --num_topics 20 --data_exp exp_20260208_153424_vocab3500_lda"
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

if [ -z "$MODELS" ]; then
    echo "Error: --models is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# Experiment manager script
EXP_MANAGER="/root/autodl-tmp/ETM/experiment_manager.py"

# Auto-select data_exp if not provided
if [ -z "$DATA_EXP" ]; then
    BASELINE_DATA_DIR="/root/autodl-tmp/result/baseline/$DATASET/data"
    if [ -d "$BASELINE_DATA_DIR" ]; then
        LATEST_EXP=$(ls -dt "$BASELINE_DATA_DIR"/exp_* 2>/dev/null | head -1)
        if [ -n "$LATEST_EXP" ]; then
            DATA_EXP=$(basename "$LATEST_EXP")
            echo "[INFO] Auto-selected data experiment: $DATA_EXP"
        fi
    fi
fi

# If data_exp is provided, read vocab_size from its config.json (override default)
if [ -n "$DATA_EXP" ]; then
    DATA_EXP_DIR="/root/autodl-tmp/result/baseline/$DATASET/data/$DATA_EXP"
    CONFIG_FILE="$DATA_EXP_DIR/config.json"
    if [ -f "$CONFIG_FILE" ]; then
        REAL_VOCAB_SIZE=$(python -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('vocab_size', $VOCAB_SIZE))" 2>/dev/null)
        if [ -n "$REAL_VOCAB_SIZE" ] && [ "$REAL_VOCAB_SIZE" != "$VOCAB_SIZE" ]; then
            echo "[INFO] Read vocab_size from data_exp config.json: $REAL_VOCAB_SIZE (overriding default $VOCAB_SIZE)"
            VOCAB_SIZE="$REAL_VOCAB_SIZE"
        fi
    fi
fi

# Config script for querying model parameters
CONFIG_SCRIPT="/root/autodl-tmp/ETM/models_config/model_config.py"

# Get first model to determine type (for display purposes)
FIRST_MODEL=$(echo "$MODELS" | cut -d',' -f1)
if [ -f "$CONFIG_SCRIPT" ]; then
    MODEL_TYPE=$(python "$CONFIG_SCRIPT" --model "$FIRST_MODEL" --query type 2>/dev/null || echo "unknown")
    MODEL_FULL_NAME=$(python "$CONFIG_SCRIPT" --model "$FIRST_MODEL" --query name 2>/dev/null || echo "$FIRST_MODEL")
    MODEL_PARAMS=$(python "$CONFIG_SCRIPT" --model "$FIRST_MODEL" --query params 2>/dev/null || echo "")
else
    MODEL_TYPE="unknown"
    MODEL_FULL_NAME="$FIRST_MODEL"
    MODEL_PARAMS=""
fi

echo "=========================================="
echo "Baseline Model Training"
echo "=========================================="
echo "Dataset:    $DATASET"
echo "Models:     $MODELS"
echo "Vocab Size: $VOCAB_SIZE"
if [ "$MODEL_TYPE" = "traditional" ]; then
    echo "Type:       Traditional (no GPU needed)"
else
    echo "Type:       Neural (GPU: $GPU)"
fi
echo ""
echo "Parameters for $FIRST_MODEL ($MODEL_FULL_NAME):"
echo "  Available: $MODEL_PARAMS"
echo ""

cd /root/autodl-tmp/ETM

# Build command - base parameters
CMD="python run_pipeline.py --dataset $DATASET --models $MODELS"
CMD="$CMD --vocab_size $VOCAB_SIZE --language $LANGUAGE"

# Add num_topics only if model supports it (not for hdp/bertopic which auto-detect)
AUTO_TOPICS=$(python "$CONFIG_SCRIPT" --model "$FIRST_MODEL" --query auto_topics 2>/dev/null || echo "false")
if [ "$AUTO_TOPICS" = "false" ]; then
    CMD="$CMD --num_topics $NUM_TOPICS"
fi

# Add parameters based on model type
if [ "$MODEL_TYPE" = "neural" ]; then
    # Neural models need epochs, batch_size, learning_rate, gpu
    CMD="$CMD --epochs $EPOCHS --batch_size $BATCH_SIZE"
    CMD="$CMD --hidden_dim $HIDDEN_DIM --learning_rate $LEARNING_RATE"
    CMD="$CMD --gpu $GPU --dropout $DROPOUT"
fi

# Model-specific parameters (only add what each model needs)
# LDA: max_iter
if echo "$MODELS" | grep -qE "^lda$|,lda,|,lda$|^lda,"; then
    CMD="$CMD --max_iter $MAX_ITER"
fi

# HDP: max_topics, alpha
if echo "$MODELS" | grep -qE "^hdp$|,hdp,|,hdp$|^hdp,"; then
    CMD="$CMD --max_topics $MAX_TOPICS --alpha $ALPHA"
fi

# STM: max_iter
if echo "$MODELS" | grep -qE "^stm$|,stm,|,stm$|^stm,"; then
    CMD="$CMD --max_iter $MAX_ITER"
fi

# BTM: n_iter, alpha, beta
if echo "$MODELS" | grep -qE "^btm$|,btm,|,btm$|^btm,"; then
    CMD="$CMD --n_iter $N_ITER --alpha $ALPHA --beta $BETA"
fi

# CTM: inference_type
if echo "$MODELS" | grep -qE "^ctm$|,ctm,|,ctm$|^ctm,"; then
    CMD="$CMD --inference_type $INFERENCE_TYPE"
fi

if [ "$SKIP_TRAIN" = true ]; then
    CMD="$CMD --skip-train"
fi

if [ "$SKIP_VIZ" = true ]; then
    CMD="$CMD --skip-viz"
fi

# Add data_exp if provided
if [ -n "$DATA_EXP" ]; then
    CMD="$CMD --data_exp $DATA_EXP"
fi

# Auto-generate exp_name from parameters if not provided
if [ -z "$EXP_NAME" ]; then
    EXP_NAME="k${NUM_TOPICS}_e${EPOCHS}"
fi
CMD="$CMD --exp_name $EXP_NAME"

echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: /root/autodl-tmp/result/baseline/$DATASET/"
echo "=========================================="

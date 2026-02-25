#!/bin/bash
# =============================================================================
# THETA Data Preparation Script
# =============================================================================
# Generate embeddings and BOW matrices for topic modeling
#
# Supported Models and their data requirements:
#   - lda, hdp, btm, nvdm, gsm, prodlda: BOW only
#   - stm: BOW + covariates (metadata)
#   - ctm: BOW + SBERT embeddings
#   - etm: BOW + Word2Vec embeddings
#   - dtm: BOW + SBERT + time slices
#   - bertopic: SBERT + raw text (no BOW needed)
#   - theta: Qwen embeddings + BOW
#
# Output Structure:
#   /root/autodl-tmp/result/baseline/{dataset}/    (for baseline models)
#   /root/autodl-tmp/result/{model_size}/{dataset}/ (for THETA)
#
# Usage (non-interactive, all parameters via command line):
#   ./03_prepare_data.sh --dataset <name> --model <model_name> [options]
#
# Examples:
#   ./03_prepare_data.sh --dataset edu_data --model lda --vocab_size 5000 --language chinese
#   ./03_prepare_data.sh --dataset edu_data --model ctm --vocab_size 5000 --language chinese
#   ./03_prepare_data.sh --dataset edu_data --model theta --model_size 0.6B --language chinese
# =============================================================================

set -e

# Base directories - support environment variables for flexible deployment
BASE_DIR=${ETM_BASE_DIR:-/root/autodl-tmp}
RESULT_DIR=${ETM_RESULT_DIR:-$BASE_DIR/result}
DATA_DIR=${ETM_DATA_DIR:-$BASE_DIR/data}
ETM_CODE_DIR=${ETM_CODE_DIR:-$BASE_DIR/ETM}

# Default values
DATASET=""
MODEL=""
MODEL_SIZE="0.6B"
MODE="zero_shot"
VOCAB_SIZE=5000
BATCH_SIZE=32
MAX_LENGTH=512
GPU=0
LANGUAGE="english"
CLEAN=false
RAW_INPUT=""
BOW_ONLY=false
CHECK_ONLY=false
EXP_NAME=""
TIME_COLUMN="year"
# Embedding training params (for supervised/unsupervised)
EMB_EPOCHS=10
EMB_LR="2e-5"
EMB_MAX_LENGTH=512
EMB_BATCH_SIZE=8  # Smaller than BOW batch_size; CausalLM needs more VRAM

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --vocab_size) VOCAB_SIZE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --max_length) MAX_LENGTH="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --clean) CLEAN=true; shift ;;
        --raw-input) RAW_INPUT="$2"; shift 2 ;;
        --bow-only) BOW_ONLY=true; shift ;;
        --check-only) CHECK_ONLY=true; shift ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --time_column) TIME_COLUMN="$2"; shift 2 ;;
        --label_column) LABEL_COLUMN="$2"; shift 2 ;;
        --emb_epochs) EMB_EPOCHS="$2"; shift 2 ;;
        --emb_lr) EMB_LR="$2"; shift 2 ;;
        --emb_max_length) EMB_MAX_LENGTH="$2"; shift 2 ;;
        --emb_batch_size) EMB_BATCH_SIZE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> --model <model_name> [options]"
            echo ""
            echo "Supported Models:"
            echo "  BOW only:        lda, hdp, btm, nvdm, gsm, prodlda"
            echo "  BOW + covariates: stm (requires document-level metadata)"
            echo "  BOW + SBERT:     ctm"
            echo "  BOW + Word2Vec:  etm"
            echo "  BOW + SBERT + Time: dtm"
            echo "  SBERT + Text:    bertopic"
            echo "  Qwen + BOW:      theta"
            echo ""
            echo "Required:"
            echo "  --dataset      Dataset name (must exist in /root/autodl-tmp/data/)"
            echo "  --model        Target model: lda, hdp, stm, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta"
            echo ""
            echo "Common Options:"
            echo "  --vocab_size   Vocabulary size (default: 5000)"
            echo "  --batch_size   Batch size for embedding generation (default: 32)"
            echo "  --max_length   Max sequence length (default: 512)"
            echo "  --gpu          GPU device ID (default: 0)"
            echo "  --language     Data language: english, chinese (default: english)"
            echo "                 Determines tokenization and stopword filtering for BOW."
            echo "  --bow-only     Only generate BOW, skip embeddings"
            echo "  --check-only   Only check if files exist, do not generate"
            echo "  --exp_name     Experiment name tag (default: auto-generated)"
            echo "  --clean        Clean raw data first (use with --raw-input)"
            echo "  --raw-input    Path to raw input file (use with --clean)"
            echo ""
            echo "THETA-specific Options (--model theta):"
            echo "  --model_size   Qwen model size: 0.6B, 4B, 8B (default: 0.6B)"
            echo "  --mode         Embedding mode: zero_shot, unsupervised, supervised (default: zero_shot)"
            echo "  --label_column Label column name for supervised mode (required if --mode supervised)"
            echo "  --emb_epochs   Embedding fine-tuning epochs (default: 10, for supervised/unsupervised)"
            echo "  --emb_lr       Embedding fine-tuning learning rate (default: 2e-5)"
            echo "  --emb_max_length  Embedding max sequence length (default: 512)"
            echo "  --emb_batch_size  Embedding batch size (default: 8, smaller for CausalLM VRAM)"
            echo ""
            echo "DTM-specific Options (--model dtm):"
            echo "  --time_column  Time column name in CSV (default: year)"
            echo ""
            echo "Examples:"
            echo "  # Baseline BOW-only models"
            echo "  $0 --dataset edu_data --model lda --vocab_size 5000 --language chinese"
            echo ""
            echo "  # CTM (BOW + SBERT)"
            echo "  $0 --dataset edu_data --model ctm --vocab_size 5000 --language chinese"
            echo ""
            echo "  # DTM (BOW + SBERT + time slices)"
            echo "  $0 --dataset edu_data --model dtm --vocab_size 5000 --language chinese --time_column year"
            echo ""
            echo "  # THETA zero_shot"
            echo "  $0 --dataset edu_data --model theta --model_size 0.6B --mode zero_shot --vocab_size 3500 --language chinese"
            echo ""
            echo "  # THETA unsupervised (LoRA fine-tuning)"
            echo "  $0 --dataset edu_data --model theta --model_size 0.6B --mode unsupervised --vocab_size 3500 --language chinese --emb_epochs 10 --emb_batch_size 8"
            echo ""
            echo "  # THETA supervised (requires label column)"
            echo "  $0 --dataset edu_data --model theta --model_size 0.6B --mode supervised --vocab_size 3500 --language chinese --label_column province --emb_epochs 10"
            echo ""
            echo "  # BOW only (skip embedding generation)"
            echo "  $0 --dataset edu_data --model theta --model_size 0.6B --bow-only --vocab_size 3500 --language chinese"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Additional variable for supervised label
LABEL_COLUMN=${LABEL_COLUMN:-""}

# Validate required parameters
if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

if [ -z "$MODEL" ]; then
    echo "Error: --model is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# Validate model name
case $MODEL in
    lda|hdp|stm|btm|nvdm|gsm|prodlda|ctm|etm|dtm|bertopic|theta|baseline) ;;
    *) echo "Error: Unknown model '$MODEL'. Must be: lda, hdp, stm, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta"; exit 1 ;;
esac

# Validate supervised mode requires label_column
if [ "$MODEL" = "theta" ] && [ "$MODE" = "supervised" ] && [ -z "$LABEL_COLUMN" ]; then
    echo "Error: --label_column is required for supervised mode"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# Query model configuration from config file
CONFIG_SCRIPT="$ETM_CODE_DIR/models_config/model_config.py"

if [ -f "$CONFIG_SCRIPT" ]; then
    # Use configuration file to determine data requirements
    NEED_BOW=$(python "$CONFIG_SCRIPT" --model "$MODEL" --query needs_bow)
    NEED_SBERT=$(python "$CONFIG_SCRIPT" --model "$MODEL" --query needs_sbert)
    NEED_WORD2VEC=$(python "$CONFIG_SCRIPT" --model "$MODEL" --query needs_word2vec)
    NEED_TIME=$(python "$CONFIG_SCRIPT" --model "$MODEL" --query needs_time)
    NEED_QWEN=$(python "$CONFIG_SCRIPT" --model "$MODEL" --query needs_qwen)
    MODEL_FULL_NAME=$(python "$CONFIG_SCRIPT" --model "$MODEL" --query name)
    
    # Determine data type
    if [ "$NEED_QWEN" = "true" ]; then
        DATA_TYPE="theta"
    else
        DATA_TYPE="baseline"
    fi
else
    # Fallback: hardcoded logic if config file not found
    echo "Warning: Config file not found, using fallback logic"
    NEED_BOW=true
    NEED_SBERT=false
    NEED_WORD2VEC=false
    NEED_TIME=false
    NEED_QWEN=false
    MODEL_FULL_NAME="$MODEL"
    DATA_TYPE="baseline"
    
    case "$MODEL" in
        ctm) NEED_SBERT=true ;;
        etm) NEED_WORD2VEC=true ;;
        dtm) NEED_SBERT=true; NEED_TIME=true ;;
        bertopic) NEED_BOW=false; NEED_SBERT=true ;;
        theta) NEED_QWEN=true; DATA_TYPE="theta" ;;
    esac
fi

echo "=========================================="
echo "THETA Data Preparation"
echo "=========================================="
echo "Dataset:    $DATASET"
echo "Model:      $MODEL ($MODEL_FULL_NAME)"
if [ "$MODEL" = "theta" ]; then
    echo "Model Size: $MODEL_SIZE"
    echo "Mode:       $MODE"
fi
echo "Vocab Size: $VOCAB_SIZE"
echo ""
echo "Data Requirements:"
echo "  - BOW Matrix:    $([ "$NEED_BOW" = "true" ] && echo "Yes" || echo "No")"
echo "  - SBERT Embed:   $([ "$NEED_SBERT" = "true" ] && echo "Yes" || echo "No")"
echo "  - Word2Vec:      $([ "$NEED_WORD2VEC" = "true" ] && echo "Yes" || echo "No")"
echo "  - Time Slices:   $([ "$NEED_TIME" = "true" ] && echo "Yes" || echo "No")"
echo "  - Qwen Embed:    $([ "$NEED_QWEN" = "true" ] && echo "Yes" || echo "No")"
echo ""

# DTM time_column validation
if [ "$NEED_TIME" = "true" ]; then
    echo "DTM time column: $TIME_COLUMN"
fi

cd $ETM_CODE_DIR

# Build command based on data type
if [ "$DATA_TYPE" = "theta" ]; then
    # All THETA modes: BOW only via prepare_data, then embeddings via 02_generate_embeddings.sh
    CMD="python prepare_data.py --dataset $DATASET --model theta --model_size $MODEL_SIZE --mode $MODE --bow-only"
elif [ "$NEED_TIME" = "true" ]; then
    # DTM uses --model dtm with --time_column
    CMD="python prepare_data.py --dataset $DATASET --model dtm --time_column $TIME_COLUMN"
else
    CMD="python prepare_data.py --dataset $DATASET --model baseline"
    
    # Add flags based on requirements
    # Note: When MODEL is "baseline", we generate SBERT by default (CTM needs it)
    # Only skip SBERT for specific models that don't need it
    if [ "$NEED_SBERT" = "false" ] && [ "$MODEL" != "bertopic" ] && [ "$MODEL" != "baseline" ]; then
        CMD="$CMD --skip-sbert"
    fi
fi

# Only add vocab_size (always needed for BOW generation)
CMD="$CMD --vocab_size $VOCAB_SIZE"

# Only add batch_size if model needs it (neural models for embedding generation)
MODEL_TYPE=$(python "$CONFIG_SCRIPT" --model "$MODEL" --query type 2>/dev/null || echo "traditional")
if [ "$MODEL_TYPE" = "neural" ] || [ "$NEED_SBERT" = "true" ] || [ "$NEED_QWEN" = "true" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

# GPU is needed for neural models
if [ "$MODEL_TYPE" = "neural" ] || [ "$NEED_SBERT" = "true" ] || [ "$NEED_QWEN" = "true" ]; then
    CMD="$CMD --gpu $GPU"
fi

# Always pass language (used for tokenization)
CMD="$CMD --language $LANGUAGE"

if [ "$CLEAN" = true ]; then
    CMD="$CMD --clean --raw-input $RAW_INPUT"
fi

if [ "$BOW_ONLY" = true ] && [ "$DATA_TYPE" != "theta" ]; then
    # For THETA, --bow-only is already in CMD (always BOW-first flow)
    # User's --bow-only flag controls whether step 2+3 (embeddings) run
    CMD="$CMD --bow-only"
fi

if [ "$CHECK_ONLY" = true ]; then
    CMD="$CMD --check-only"
fi

# Auto-generate exp_name from parameters if not provided
if [ -z "$EXP_NAME" ]; then
    EXP_NAME="vocab${VOCAB_SIZE}_${MODEL}"
    if [ "$MODEL" = "theta" ]; then
        EXP_NAME="vocab${VOCAB_SIZE}_theta_${MODEL_SIZE}_${MODE}"
    elif [ "$MODEL" = "dtm" ]; then
        EXP_NAME="vocab${VOCAB_SIZE}_dtm_${TIME_COLUMN}"
    fi
fi
CMD="$CMD --exp_name $EXP_NAME"

echo "Running: $CMD"
echo ""

# For THETA: capture output to extract exp_id
if [ "$DATA_TYPE" = "theta" ]; then
    PREP_OUTPUT=$(eval $CMD 2>&1 | tee /dev/stderr)
    # Extract exp_id from output line like "  Experiment: exp_20260207_140000_vocab3500_theta_0.6B_zero_shot"
    THETA_EXP_ID=$(echo "$PREP_OUTPUT" | grep -oP 'Experiment: \K(exp_\S+)' | head -1)
    if [ -n "$THETA_EXP_ID" ]; then
        THETA_EXP_DIR="$RESULT_DIR/$MODEL_SIZE/$DATASET/data/$THETA_EXP_ID"
        echo ""
        echo "  THETA data experiment: $THETA_EXP_ID"
        echo "  Directory: $THETA_EXP_DIR"
    fi
else
    eval $CMD
fi

# ============================================================
# For ALL THETA modes: run embedding generation after BOW
# ============================================================
if [ "$DATA_TYPE" = "theta" ] && [ "$CHECK_ONLY" != true ] && [ "$BOW_ONLY" != true ]; then
    if [ -z "$THETA_EXP_DIR" ]; then
        echo "Error: Could not determine THETA experiment directory"
        exit 1
    fi
    echo ""
    echo "=========================================="
    echo "Step 2: Embedding Generation ($MODE)"
    echo "=========================================="
    echo "Experiment:   $THETA_EXP_ID"
    if [ "$MODE" = "zero_shot" ]; then
        echo "Type:         Pre-trained (no fine-tuning)"
    elif [ "$MODE" = "supervised" ]; then
        echo "Type:         LoRA fine-tuning (supervised)"
        echo "Label Column: $LABEL_COLUMN"
    else
        echo "Type:         LoRA fine-tuning (unsupervised)"
    fi
    echo "Epochs:       $EMB_EPOCHS"
    echo "LR:           $EMB_LR"
    echo "Max Length:    $EMB_MAX_LENGTH"
    echo "Batch Size:   $EMB_BATCH_SIZE"
    echo ""
    SCRIPTS_DIR=$(dirname "$0")
    EMB_CMD="bash $SCRIPTS_DIR/02_generate_embeddings.sh"
    EMB_CMD="$EMB_CMD --dataset $DATASET --mode $MODE"
    EMB_CMD="$EMB_CMD --model_size $MODEL_SIZE"
    EMB_CMD="$EMB_CMD --epochs $EMB_EPOCHS --learning_rate $EMB_LR"
    EMB_CMD="$EMB_CMD --max_length $EMB_MAX_LENGTH --batch_size $EMB_BATCH_SIZE"
    EMB_CMD="$EMB_CMD --gpu $GPU"
    EMB_CMD="$EMB_CMD --exp_dir $THETA_EXP_DIR"
    if [ "$MODE" = "supervised" ] && [ -n "$LABEL_COLUMN" ]; then
        EMB_CMD="$EMB_CMD --label_column $LABEL_COLUMN"
    fi
    echo "Running: $EMB_CMD"
    echo ""
    eval $EMB_CMD

    # Check if embeddings were actually generated (check for any .npy file in embeddings dir)
    EMB_FILE=$(find "$THETA_EXP_DIR/embeddings" -name "*.npy" 2>/dev/null | head -1)
    if [ -z "$EMB_FILE" ]; then
        echo ""
        echo "==========================================" 
        echo "ERROR: Embedding generation failed!"
        echo "==========================================" 
        echo "  No embedding .npy file was created."
        echo "  Common causes:"
        echo "    - CUDA out of memory: try smaller batch size (current: $EMB_BATCH_SIZE)"
        echo "    - GPU occupied by another process: check nvidia-smi"
        echo ""
        echo "  To retry with smaller batch size:"
        echo "    bash scripts/02_generate_embeddings.sh --dataset $DATASET --mode $MODE \\"
        echo "      --model_size $MODEL_SIZE --batch_size 4 --gpu $GPU \\"
        echo "      --exp_dir $THETA_EXP_DIR"
        echo ""
        exit 1
    fi
    echo "✓ Embeddings generated successfully: $EMB_FILE"

    # Step 3: Generate vocabulary embeddings
    echo ""
    echo "=========================================="
    echo "Step 3: Vocabulary Embeddings"
    echo "=========================================="
    VOCAB_FILE="$THETA_EXP_DIR/bow/vocab.json"
    if [ -f "$VOCAB_FILE" ]; then
        echo "Vocab file:   $VOCAB_FILE"
        echo "Output:       $THETA_EXP_DIR/bow/vocab_embeddings.npy"
        echo ""
        cd $ETM_CODE_DIR
        python -c "
import sys, json, numpy as np
sys.path.insert(0, '.')
from model.vocab_embedder import VocabEmbedder
from config import get_qwen_model_path

model_path = get_qwen_model_path('$MODEL_SIZE')
with open('$VOCAB_FILE', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
print(f'Generating vocab embeddings for {len(vocab)} words...')
embedder = VocabEmbedder(model_path=model_path, batch_size=$BATCH_SIZE, normalize=True)
embeddings = embedder.embed_vocab(vocab)
np.save('$THETA_EXP_DIR/bow/vocab_embeddings.npy', embeddings)
print(f'✓ Saved vocab_embeddings.npy: {embeddings.shape}')
"
    else
        echo "Warning: Vocab file not found: $VOCAB_FILE"
        echo "Skipping vocab embeddings generation"
    fi
fi

echo ""
echo "Data preparation completed!"

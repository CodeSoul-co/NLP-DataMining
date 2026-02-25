#!/bin/bash
# =============================================================================
# THETA Visualization Script
# =============================================================================
# Generate visualizations for trained topic models
#
# Supported Visualizations:
#   - Topic word clouds
#   - Topic-word distribution heatmaps
#   - Document-topic distribution
#   - Topic coherence metrics
#   - Training loss curves
#   - t-SNE/UMAP document embeddings
#   - Topic evolution (for DTM)
#
# Output Formats:
#   - PNG images (configurable DPI)
#   - Interactive HTML (pyLDAvis)
#
# Languages:
#   - en: English labels
#   - zh: Chinese labels
#
# Usage (non-interactive, all parameters via command line):
#   ./06_visualize.sh --dataset <name> [options]
#
# Examples:
#   ./06_visualize.sh --dataset hatespeech --model_size 0.6B --mode zero_shot --language en
#   ./06_visualize.sh --baseline --dataset hatespeech --model lda --num_topics 20 --language en
# =============================================================================

set -e

# Default values
RESULT_DIR=""
DATASET=""
MODEL=""
MODEL_SIZE="0.6B"
MODE="zero_shot"
NUM_TOPICS=20
VOCAB_SIZE=5000
LANGUAGE="en"
DPI=300
BASELINE=false
MODEL_EXP=""

# Experiment manager script
EXP_MANAGER="/root/autodl-tmp/ETM/experiment_manager.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --result_dir) RESULT_DIR="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --num_topics) NUM_TOPICS="$2"; shift 2 ;;
        --vocab_size) VOCAB_SIZE="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --dpi) DPI="$2"; shift 2 ;;
        --baseline) BASELINE=true; shift ;;
        --model_exp) MODEL_EXP="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> [options]"
            echo ""
            echo "Required:"
            echo "  --dataset       Dataset name"
            echo ""
            echo "THETA Options:"
            echo "  --model_size    Qwen model size: 0.6B, 4B, 8B (default: 0.6B)"
            echo "  --mode          Embedding mode: zero_shot, supervised, unsupervised (default: zero_shot)"
            echo ""
            echo "Baseline Options:"
            echo "  --baseline      Use baseline model mode"
            echo "  --model         Baseline model name: lda, hdp, stm (requires covariates), btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic"
            echo "  --model_exp     Model experiment ID (default: auto-select latest)"
            echo ""
            echo "Common Options:"
            echo "  --num_topics    Number of topics (default: 20)"
            echo "  --vocab_size    Vocabulary size (default: 5000)"
            echo "  --language      Visualization language: en, zh (default: en)"
            echo "  --dpi           Image DPI (default: 300)"
            echo "  --result_dir    Override result directory path"
            echo ""
            echo "Examples:"
            echo "  # THETA model"
            echo "  $0 --dataset hatespeech --model_size 0.6B --mode zero_shot --language en"
            echo ""
            echo "  # Baseline model"
            echo "  $0 --baseline --dataset hatespeech --model lda --num_topics 20 --language en"
            echo ""
            echo "  # Baseline with specific experiment"
            echo "  $0 --baseline --dataset edu_data --model lda --model_exp exp_20260208_xxx --language zh"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-select model experiment for baseline if not provided
if [ "$BASELINE" = true ] && [ -z "$MODEL_EXP" ] && [ -n "$MODEL" ]; then
    MODEL_EXP=$(python "$EXP_MANAGER" --find-model latest --dataset "$DATASET" --model "$MODEL" 2>/dev/null | xargs basename 2>/dev/null)
    if [ -n "$MODEL_EXP" ]; then
        echo "[INFO] Auto-selected model experiment: $MODEL_EXP"
    fi
fi

if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

echo "=========================================="
echo "THETA Visualization"
echo "=========================================="
echo "Dataset:  $DATASET"
echo "Language: $LANGUAGE"
echo ""

cd /root/autodl-tmp/ETM

if [ "$BASELINE" = true ]; then
    if [ -z "$MODEL" ]; then
        echo "Error: --model is required for baseline mode"
        exit 1
    fi
    
    # Use new experiment directory structure
    if [ -n "$MODEL_EXP" ]; then
        RESULT_DIR="/root/autodl-tmp/result/baseline/$DATASET/models/$MODEL/$MODEL_EXP"
        echo "Model:      $MODEL (baseline)"
        echo "Experiment: $MODEL_EXP"
    else
        # Auto-select latest experiment
        MODEL_EXP=$(python "$EXP_MANAGER" --find-model latest --dataset "$DATASET" --model "$MODEL" 2>/dev/null | xargs basename 2>/dev/null)
        if [ -z "$MODEL_EXP" ]; then
            echo "Error: No model experiments found for $MODEL on $DATASET"
            echo "Please run training first or specify --model_exp"
            exit 1
        fi
        RESULT_DIR="/root/autodl-tmp/result/baseline/$DATASET/models/$MODEL/$MODEL_EXP"
        echo "Model:      $MODEL (baseline)"
        echo "Experiment: $MODEL_EXP (auto-selected latest)"
    fi
    
    CMD="python -m visualization.run_visualization --baseline"
    CMD="$CMD --result_dir $RESULT_DIR --dataset $DATASET --model $MODEL"
    CMD="$CMD --language $LANGUAGE --dpi $DPI"
else
    RESULT_DIR="/root/autodl-tmp/result/$MODEL_SIZE"
    echo "Model:    THETA $MODEL_SIZE"
    echo "Mode:     $MODE"
    
    CMD="python -m visualization.run_visualization"
    CMD="$CMD --result_dir $RESULT_DIR --dataset $DATASET"
    CMD="$CMD --mode $MODE --model_size $MODEL_SIZE --language $LANGUAGE --dpi $DPI"
fi

echo ""
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Visualization completed!"
echo "=========================================="

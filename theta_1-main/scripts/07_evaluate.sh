#!/bin/bash
# =============================================================================
# THETA Evaluation Script
# =============================================================================
# Evaluate trained topic models using unified metrics
#
# Metrics (7 total):
#   - TD:         Topic Diversity (higher is better)
#   - iRBO:       Inverse Rank-Biased Overlap (higher is better)
#   - NPMI:       Normalized PMI Coherence (higher is better)
#   - C_V:        C_V Coherence (higher is better)
#   - UMass:      UMass Coherence (closer to 0 is better)
#   - Exclusivity: Topic Exclusivity (higher is better)
#   - PPL:        Perplexity (lower is better)
#
# Usage:
#   ./07_evaluate.sh --dataset <name> --model <model> [options]
#
# Examples:
#   ./07_evaluate.sh --dataset edu_data --model lda --num_topics 20
#   ./07_evaluate.sh --dataset edu_data --model prodlda --num_topics 15
#   ./07_evaluate.sh --baseline --dataset edu_data --model ctm --num_topics 20
# =============================================================================

set -e

# Default values
DATASET=""
MODEL=""
NUM_TOPICS=20
VOCAB_SIZE=5000
BASELINE=false
MODEL_SIZE="0.6B"
MODE="zero_shot"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --num_topics) NUM_TOPICS="$2"; shift 2 ;;
        --vocab_size) VOCAB_SIZE="$2"; shift 2 ;;
        --baseline) BASELINE=true; shift ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> --model <model> [options]"
            echo ""
            echo "Options:"
            echo "  --dataset      Dataset name (required)"
            echo "  --model        Model name (required)"
            echo "  --num_topics   Number of topics (default: 20)"
            echo "  --vocab_size   Vocabulary size (default: 5000)"
            echo "  --baseline     Use baseline model directory"
            echo "  --model_size   THETA model size: 0.6B, 4B, 8B (default: 0.6B)"
            echo "  --mode         THETA mode: zero_shot, supervised (default: zero_shot)"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset edu_data --model lda --num_topics 20"
            echo "  $0 --dataset edu_data --model prodlda --num_topics 15"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
    echo "Error: --dataset and --model are required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

echo "=========================================="
echo "THETA Evaluation"
echo "=========================================="
echo "Dataset:    $DATASET"
echo "Model:      $MODEL"
echo "Num Topics: $NUM_TOPICS"
echo "Vocab Size: $VOCAB_SIZE"
echo ""

cd /root/autodl-tmp/ETM

# Determine result directory - now includes vocab_size
if [ "$BASELINE" = true ] || [ "$MODEL" != "theta" ]; then
    RESULT_DIR="/root/autodl-tmp/result/baseline/$DATASET/vocab_$VOCAB_SIZE"
    MODEL_DIR="$RESULT_DIR/$MODEL"
else
    RESULT_DIR="/root/autodl-tmp/result/$MODEL_SIZE/$DATASET/$MODE"
    MODEL_DIR="$RESULT_DIR"
fi

# Check if model files exist
THETA_PATH="$MODEL_DIR/theta_k${NUM_TOPICS}.npy"
BETA_PATH="$MODEL_DIR/beta_k${NUM_TOPICS}.npy"

# Check model/ subdirectory for neural models
if [ ! -f "$THETA_PATH" ]; then
    THETA_PATH="$MODEL_DIR/model/theta_k${NUM_TOPICS}.npy"
    BETA_PATH="$MODEL_DIR/model/beta_k${NUM_TOPICS}.npy"
fi

if [ ! -f "$THETA_PATH" ]; then
    echo "Error: Model files not found at $MODEL_DIR"
    echo "Please train the model first using:"
    echo "  bash scripts/05_train_baseline.sh --dataset $DATASET --models $MODEL --num_topics $NUM_TOPICS"
    exit 1
fi

echo "Model files found at: $MODEL_DIR"
echo ""

# Run evaluation
python -c "
import sys
sys.path.insert(0, '/root/autodl-tmp/ETM')
import numpy as np
import json
from evaluation.unified_evaluator import UnifiedEvaluator

# Load data
result_dir = '$RESULT_DIR'
model_dir = '$MODEL_DIR'
num_topics = $NUM_TOPICS

# Load BOW and vocab
bow_matrix = np.load(f'{result_dir}/bow_matrix.npy')
with open(f'{result_dir}/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# Load theta and beta
theta_path = f'{model_dir}/theta_k{num_topics}.npy'
beta_path = f'{model_dir}/beta_k{num_topics}.npy'
import os
if not os.path.exists(theta_path):
    theta_path = f'{model_dir}/model/theta_k{num_topics}.npy'
    beta_path = f'{model_dir}/model/beta_k{num_topics}.npy'

theta = np.load(theta_path)
beta = np.load(beta_path)

print(f'Loaded theta: {theta.shape}')
print(f'Loaded beta: {beta.shape}')
print(f'Loaded vocab: {len(vocab)} words')
print()

# Run evaluation
evaluator = UnifiedEvaluator(
    beta=beta, theta=theta, bow_matrix=bow_matrix, vocab=vocab,
    model_name='$MODEL', dataset='$DATASET', 
    output_dir=model_dir, num_topics=num_topics
)

metrics = evaluator.compute_all_metrics()
evaluator.save_metrics()
evaluator.generate_metrics_plots()

print()
print('Evaluation completed!')
print(f'Metrics saved to: {model_dir}/metrics_k{num_topics}.json')
"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="

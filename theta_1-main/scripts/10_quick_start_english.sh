#!/bin/bash
# THETA Quick Start - English Dataset
# Complete workflow for a new English dataset

set -e

DATASET=${1:-"my_dataset"}

echo "=========================================="
echo "THETA Quick Start - English Dataset"
echo "=========================================="
echo "Dataset: $DATASET"
echo ""

# Step 1: Create dataset directory
echo "[1/4] Creating dataset directory..."
mkdir -p /root/autodl-tmp/data/$DATASET

echo ""
echo "Expecting cleaned CSV file at:"
echo "  /root/autodl-tmp/data/$DATASET/${DATASET}_cleaned.csv"
echo "CSV should have a 'text' column (or 'content', 'cleaned_content')"
echo ""

# Validate data file exists
CSV_FILE=$(ls /root/autodl-tmp/data/$DATASET/*_cleaned.csv 2>/dev/null | head -1)
if [ -z "$CSV_FILE" ]; then
    CSV_FILE=$(ls /root/autodl-tmp/data/$DATASET/*.csv 2>/dev/null | head -1)
fi
if [ -z "$CSV_FILE" ]; then
    echo "Error: No CSV file found in /root/autodl-tmp/data/$DATASET/"
    echo "Please place your data file there first."
    exit 1
fi
echo "Found: $CSV_FILE"

# Step 2: Prepare data
echo ""
echo "[2/4] Preparing data (generating embeddings and BOW)..."
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset $DATASET --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0

# Step 3: Train model
echo ""
echo "[3/4] Training THETA model..."
python run_pipeline.py --dataset $DATASET --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en

# Step 4: Done
echo ""
echo "[4/4] Complete!"
echo ""
echo "=========================================="
echo "Results saved to:"
echo "  /root/autodl-tmp/result/0.6B/$DATASET/zero_shot/"
echo ""
echo "Evaluation metrics:"
echo "  /root/autodl-tmp/result/0.6B/$DATASET/zero_shot/metrics/"
echo ""
echo "Visualizations:"
echo "  /root/autodl-tmp/result/0.6B/$DATASET/zero_shot/visualizations/"
echo "=========================================="

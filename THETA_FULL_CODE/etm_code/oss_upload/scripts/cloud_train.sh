#!/bin/bash
# THETA 云端训练脚本 - PAI-DLC 使用
#
# Usage:
#   bash scripts/cloud_train.sh <dataset> <mode> <num_topics> <epochs>
#
# Example:
#   bash scripts/cloud_train.sh hatespeech zero_shot 20 50
#
# 环境变量:
#   THETA_BASE: OSS基础路径 (如 oss://theta-bucket)
#   THETA_MODEL_SIZE: 模型大小 (0.6B, 4B, 8B)

set -e

# 参数
DATASET=${1:-"hatespeech"}
MODE=${2:-"zero_shot"}
NUM_TOPICS=${3:-20}
EPOCHS=${4:-50}
BATCH_SIZE=${5:-64}

# 环境变量默认值
export THETA_BASE=${THETA_BASE:-"oss://theta-bucket"}
export THETA_MODEL_SIZE=${THETA_MODEL_SIZE:-"0.6B"}

echo "=============================================="
echo "THETA Cloud Training"
echo "=============================================="
echo "  THETA_BASE: $THETA_BASE"
echo "  THETA_MODEL_SIZE: $THETA_MODEL_SIZE"
echo "  Dataset: $DATASET"
echo "  Mode: $MODE"
echo "  Num Topics: $NUM_TOPICS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "=============================================="

# 切换到ETM目录
cd /workspace/ETM || cd ETM || cd .

# 安装依赖 (如果需要)
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
fi

# 运行训练
echo ""
echo "[Step 1/1] Running THETA Training Pipeline..."
python run_pipeline.py \
    --dataset "$DATASET" \
    --models theta \
    --model_size "$THETA_MODEL_SIZE" \
    --mode "$MODE" \
    --num_topics "$NUM_TOPICS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "=============================================="
echo "Training completed!"
echo "Results saved to: $THETA_BASE/result/$THETA_MODEL_SIZE/$DATASET/$MODE/"
echo "=============================================="

#!/bin/bash
# THETA 云端推理服务启动脚本 - PAI-EAS 使用
#
# Usage:
#   bash scripts/cloud_inference.sh <dataset> <mode> <num_topics>
#
# Example:
#   bash scripts/cloud_inference.sh hatespeech zero_shot 20
#
# 环境变量:
#   THETA_BASE: OSS基础路径
#   THETA_MODEL_SIZE: 模型大小

set -e

# 参数
DATASET=${1:-"hatespeech"}
MODE=${2:-"zero_shot"}
NUM_TOPICS=${3:-20}
PORT=${4:-8080}

# 环境变量默认值
export THETA_BASE=${THETA_BASE:-"oss://theta-bucket"}
export THETA_MODEL_SIZE=${THETA_MODEL_SIZE:-"0.6B"}

# 模型目录
MODEL_DIR="$THETA_BASE/result/$THETA_MODEL_SIZE/$DATASET/$MODE/model"

echo "=============================================="
echo "THETA Inference Server"
echo "=============================================="
echo "  THETA_BASE: $THETA_BASE"
echo "  THETA_MODEL_SIZE: $THETA_MODEL_SIZE"
echo "  Model Dir: $MODEL_DIR"
echo "  Port: $PORT"
echo "=============================================="

# 切换到ETM目录
cd /workspace/ETM || cd ETM || cd .

# 安装依赖 (如果需要)
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
fi

# 启动推理服务
python inference_server.py \
    --model_dir "$MODEL_DIR" \
    --model_size "$THETA_MODEL_SIZE" \
    --num_topics "$NUM_TOPICS" \
    --port "$PORT"

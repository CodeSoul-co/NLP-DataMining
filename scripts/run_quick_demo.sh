#!/bin/bash
# ============================================================================
# 快速演示脚本 - 使用已有数据集快速运行
# ============================================================================
# 用法: ./run_quick_demo.sh [dataset] [mode]
# 示例: ./run_quick_demo.sh hatespeech zero_shot
# ============================================================================

set -e

DATASET="${1:-hatespeech}"
MODE="${2:-zero_shot}"

PROJECT_ROOT="/root/autodl-tmp"
ETM_DIR="${PROJECT_ROOT}/ETM"

echo "=============================================="
echo "快速演示: ${DATASET} / ${MODE}"
echo "=============================================="

cd "${ETM_DIR}"

# 使用pipeline_api运行
python pipeline_api.py train \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --num_topics 20 \
    --vocab_size 5000

echo ""
echo "完成! 查看结果: ${PROJECT_ROOT}/result/${DATASET}/${MODE}/"

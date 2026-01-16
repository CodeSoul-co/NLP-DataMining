#!/bin/bash
# ============================================================================
# 步骤5: ETM主题模型训练
# ============================================================================
# 功能: 训练ETM主题模型
# 输入: 
#   - 文档embedding (result/{dataset}/{mode}/embeddings/)
#   - BOW矩阵 (result/{dataset}/bow/)
#   - 词汇embedding (result/{dataset}/{mode}/embeddings/vocab_embeddings.npy)
# 输出:
#   - 训练好的模型 (result/{dataset}/{mode}/model/)
#   - Theta矩阵 (文档-主题分布)
#   - Beta矩阵 (主题-词分布)
#   - 主题词列表
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
MODE="${2:-zero_shot}"
NUM_TOPICS="${3:-20}"
VOCAB_SIZE="${4:-5000}"
EPOCHS="${5:-50}"
BATCH_SIZE="${6:-64}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
ETM_DIR="${PROJECT_ROOT}/ETM"

echo "=============================================="
echo "步骤5: ETM主题模型训练"
echo "=============================================="
echo "数据集: ${DATASET_NAME}"
echo "模式: ${MODE}"
echo "主题数: ${NUM_TOPICS}"
echo "词表大小: ${VOCAB_SIZE}"
echo "训练轮数: ${EPOCHS}"
echo "批次大小: ${BATCH_SIZE}"
echo "=============================================="

cd "${ETM_DIR}"

# 运行ETM训练
python main.py \
    --dataset "${DATASET_NAME}" \
    --mode "${MODE}" \
    --num_topics "${NUM_TOPICS}" \
    --vocab_size "${VOCAB_SIZE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}"

echo ""
echo "=============================================="
echo "ETM训练完成!"
echo "输出目录: ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/model/"
echo "=============================================="
echo ""
echo "下一步: 运行 06_evaluate.sh ${DATASET_NAME} ${MODE}"

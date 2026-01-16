#!/bin/bash
# ============================================================================
# 步骤4: 词汇Embedding生成
# ============================================================================
# 功能: 为词汇表中的每个词生成embedding向量
# 输入: 词汇表 (result/{dataset}/bow/vocab.json)
# 输出: 词汇embedding矩阵 (result/{dataset}/{mode}/embeddings/vocab_embeddings.npy)
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
MODE="${2:-zero_shot}"
BATCH_SIZE="${3:-64}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
EMBEDDING_DIR="${PROJECT_ROOT}/embedding"

echo "=============================================="
echo "步骤4: 词汇Embedding生成"
echo "=============================================="
echo "数据集: ${DATASET_NAME}"
echo "模式: ${MODE}"
echo "批次大小: ${BATCH_SIZE}"
echo "=============================================="

cd "${EMBEDDING_DIR}"

# 生成词汇embedding
python main.py \
    --mode generate_vocab_embeddings \
    --dataset "${DATASET_NAME}" \
    --vocab_file "${PROJECT_ROOT}/result/${DATASET_NAME}/bow/vocab.json" \
    --batch_size "${BATCH_SIZE}"

echo ""
echo "=============================================="
echo "词汇Embedding生成完成!"
echo "输出: ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/embeddings/vocab_embeddings.npy"
echo "=============================================="
echo ""
echo "下一步: 运行 05_etm_train.sh ${DATASET_NAME} ${MODE}"

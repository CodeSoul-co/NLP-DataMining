#!/bin/bash
# ============================================================================
# 完整Pipeline一键运行脚本
# ============================================================================
# 功能: 按顺序执行所有步骤
# 用法: ./run_full_pipeline.sh <dataset_name> <mode> [num_topics] [vocab_size]
# 示例: ./run_full_pipeline.sh hatespeech supervised 20 5000
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
MODE="${2:-zero_shot}"
NUM_TOPICS="${3:-20}"
VOCAB_SIZE="${4:-5000}"
EPOCHS="${5:-50}"

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "ETM主题模型完整Pipeline"
echo "=============================================="
echo "数据集: ${DATASET_NAME}"
echo "模式: ${MODE}"
echo "主题数: ${NUM_TOPICS}"
echo "词表大小: ${VOCAB_SIZE}"
echo "=============================================="
echo ""

# 步骤2: Embedding生成
echo "[1/5] Embedding生成..."
bash "${SCRIPT_DIR}/02_embedding_generate.sh" "${DATASET_NAME}" "${MODE}" 3

# 步骤3: BOW生成
echo "[2/5] BOW矩阵生成..."
bash "${SCRIPT_DIR}/03_bow_generate.sh" "${DATASET_NAME}" "${VOCAB_SIZE}"

# 步骤4: 词汇Embedding
echo "[3/5] 词汇Embedding生成..."
bash "${SCRIPT_DIR}/04_vocab_embedding.sh" "${DATASET_NAME}" "${MODE}"

# 步骤5: ETM训练
echo "[4/5] ETM模型训练..."
bash "${SCRIPT_DIR}/05_etm_train.sh" "${DATASET_NAME}" "${MODE}" "${NUM_TOPICS}" "${VOCAB_SIZE}" "${EPOCHS}"

# 步骤6: 评估
echo "[5/5] 模型评估..."
bash "${SCRIPT_DIR}/06_evaluate.sh" "${DATASET_NAME}" "${MODE}"

# 步骤7: 可视化
echo "[Bonus] 生成可视化..."
bash "${SCRIPT_DIR}/07_visualize.sh" "${DATASET_NAME}" "${MODE}"

echo ""
echo "=============================================="
echo "Pipeline完成!"
echo "结果目录: /root/autodl-tmp/result/${DATASET_NAME}/${MODE}/"
echo "=============================================="

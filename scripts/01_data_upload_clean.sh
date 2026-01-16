#!/bin/bash
# ============================================================================
# 步骤1: 数据上传与清洗
# ============================================================================
# 功能: 将原始文本/CSV文件清洗处理成标准格式
# 输入: 原始数据文件 (txt/csv)
# 输出: 清洗后的CSV文件 (data/{dataset}/{dataset}_cleaned.csv)
# ============================================================================

set -e  # 遇到错误立即退出

# 默认参数
DATASET_NAME="${1:-my_dataset}"
INPUT_PATH="${2:-}"
LANGUAGE="${3:-english}"

# 项目根目录
PROJECT_ROOT="/root/autodl-tmp"
DATA_DIR="${PROJECT_ROOT}/data"
ETM_DIR="${PROJECT_ROOT}/ETM"

echo "=============================================="
echo "步骤1: 数据上传与清洗"
echo "=============================================="
echo "数据集名称: ${DATASET_NAME}"
echo "输入路径: ${INPUT_PATH}"
echo "语言: ${LANGUAGE}"
echo "=============================================="

# 创建数据集目录
mkdir -p "${DATA_DIR}/${DATASET_NAME}"

# 如果提供了输入路径，复制文件
if [ -n "${INPUT_PATH}" ]; then
    echo "复制输入文件到数据目录..."
    cp -r "${INPUT_PATH}" "${DATA_DIR}/${DATASET_NAME}/"
fi

# 运行数据清洗
echo "运行数据清洗..."
cd "${ETM_DIR}"

# 方式1: 批量处理目录中的所有文件
python -m dataclean.main batch \
    --input_dir "${DATA_DIR}/${DATASET_NAME}" \
    --output "${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_cleaned.csv" \
    --language "${LANGUAGE}" \
    --operations lowercase stopwords lemmatize

# 方式2: 如果是单个CSV文件
# python -m dataclean.main clean \
#     --input "${DATA_DIR}/${DATASET_NAME}/raw.csv" \
#     --output "${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_cleaned.csv" \
#     --text_column "text" \
#     --language "${LANGUAGE}"

echo ""
echo "=============================================="
echo "数据清洗完成!"
echo "输出文件: ${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_cleaned.csv"
echo "=============================================="
echo ""
echo "下一步: 运行 02_embedding_generate.sh ${DATASET_NAME}"

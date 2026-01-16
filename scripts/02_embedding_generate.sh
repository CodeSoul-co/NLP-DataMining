#!/bin/bash
# ============================================================================
# 步骤2: Embedding生成
# ============================================================================
# 功能: 使用Qwen3-Embedding生成文档向量
# 模式: 
#   - zero_shot: 直接生成，无需训练（最快）
#   - supervised: 有标签数据，使用LoRA微调
#   - unsupervised: 无标签数据，使用SimCSE自监督训练
# 输入: 清洗后的CSV文件
# 输出: 文档embedding矩阵 (result/{dataset}/{mode}/embeddings/)
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
MODE="${2:-zero_shot}"           # zero_shot / supervised / unsupervised
EPOCHS="${3:-3}"
BATCH_SIZE="${4:-16}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
EMBEDDING_DIR="${PROJECT_ROOT}/embedding"

echo "=============================================="
echo "步骤2: Embedding生成"
echo "=============================================="
echo "数据集: ${DATASET_NAME}"
echo "模式: ${MODE}"
echo "训练轮数: ${EPOCHS}"
echo "批次大小: ${BATCH_SIZE}"
echo "=============================================="

cd "${EMBEDDING_DIR}"

case "${MODE}" in
    "zero_shot")
        echo "运行Zero-shot Embedding生成..."
        python main.py \
            --mode zero_shot \
            --dataset "${DATASET_NAME}" \
            --batch_size "${BATCH_SIZE}"
        ;;
    
    "supervised")
        echo "运行Supervised LoRA训练..."
        echo "注意: 需要数据集包含标签列"
        python main.py \
            --mode supervised \
            --dataset "${DATASET_NAME}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --use_lora
        ;;
    
    "unsupervised")
        echo "运行Unsupervised SimCSE训练..."
        python main.py \
            --mode unsupervised \
            --dataset "${DATASET_NAME}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --use_lora
        ;;
    
    *)
        echo "错误: 未知模式 ${MODE}"
        echo "可用模式: zero_shot, supervised, unsupervised"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Embedding生成完成!"
echo "输出目录: ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/embeddings/"
echo "=============================================="
echo ""
echo "下一步: 运行 03_bow_generate.sh ${DATASET_NAME}"

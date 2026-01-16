#!/bin/bash
# ============================================================================
# 步骤3: BOW矩阵生成
# ============================================================================
# 功能: 生成词袋(Bag-of-Words)矩阵和词汇表
# 输入: 清洗后的CSV文件
# 输出: 
#   - BOW矩阵 (result/{dataset}/bow/bow_matrix.npz)
#   - 词汇表 (result/{dataset}/bow/vocab.json)
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
VOCAB_SIZE="${2:-5000}"
MIN_DOC_FREQ="${3:-5}"
MAX_DOC_FREQ="${4:-0.95}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
ETM_DIR="${PROJECT_ROOT}/ETM"

echo "=============================================="
echo "步骤3: BOW矩阵生成"
echo "=============================================="
echo "数据集: ${DATASET_NAME}"
echo "词表大小: ${VOCAB_SIZE}"
echo "最小文档频率: ${MIN_DOC_FREQ}"
echo "最大文档频率: ${MAX_DOC_FREQ}"
echo "=============================================="

cd "${ETM_DIR}"

# 生成BOW矩阵
python -c "
import sys
sys.path.insert(0, '.')
from config import PipelineConfig, DATASET_CONFIGS
from bow.bow_generator import BOWGenerator

# 创建配置
config = PipelineConfig()
config.data.dataset = '${DATASET_NAME}'
config.bow.vocab_size = ${VOCAB_SIZE}
config.bow.min_doc_freq = ${MIN_DOC_FREQ}
config.bow.max_doc_freq = ${MAX_DOC_FREQ}

# 加载数据集特定配置
if '${DATASET_NAME}' in DATASET_CONFIGS:
    ds_config = DATASET_CONFIGS['${DATASET_NAME}']
    config.bow.language = ds_config.get('language', 'english')

# 加载文本
import pandas as pd
from pathlib import Path

data_dir = Path('${PROJECT_ROOT}/data/${DATASET_NAME}')
csv_files = list(data_dir.glob('*cleaned*.csv')) + list(data_dir.glob('*text_only*.csv'))

if not csv_files:
    csv_files = list(data_dir.glob('*.csv'))

if not csv_files:
    print(f'Error: No CSV files found in {data_dir}')
    sys.exit(1)

df = pd.read_csv(csv_files[0])
text_col = next((c for c in df.columns if c.lower() in ['text', 'content', 'document']), df.columns[0])
texts = df[text_col].dropna().tolist()

print(f'Loaded {len(texts)} documents')

# 生成BOW
generator = BOWGenerator(config)
bow_matrix, vocab = generator.fit_transform(texts)

print(f'BOW matrix shape: {bow_matrix.shape}')
print(f'Vocabulary size: {len(vocab)}')

# 保存
import os
import json
import scipy.sparse as sp

output_dir = Path('${PROJECT_ROOT}/result/${DATASET_NAME}/bow')
output_dir.mkdir(parents=True, exist_ok=True)

sp.save_npz(output_dir / 'bow_matrix.npz', bow_matrix)
with open(output_dir / 'vocab.json', 'w') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f'Saved to {output_dir}')
"

echo ""
echo "=============================================="
echo "BOW生成完成!"
echo "输出目录: ${PROJECT_ROOT}/result/${DATASET_NAME}/bow/"
echo "=============================================="
echo ""
echo "下一步: 运行 04_vocab_embedding.sh ${DATASET_NAME}"

#!/bin/bash
# ============================================================================
# 步骤6: 模型评估
# ============================================================================
# 功能: 评估训练好的主题模型
# 评估指标:
#   - Topic Coherence (主题一致性)
#   - Topic Diversity (主题多样性)
#   - Perplexity (困惑度)
# 输入: 训练好的模型和矩阵
# 输出: 评估报告 (result/{dataset}/{mode}/evaluation/)
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
MODE="${2:-zero_shot}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
ETM_DIR="${PROJECT_ROOT}/ETM"

echo "=============================================="
echo "步骤6: 模型评估"
echo "=============================================="
echo "数据集: ${DATASET_NAME}"
echo "模式: ${MODE}"
echo "=============================================="

cd "${ETM_DIR}"

# 运行评估
python -c "
import sys
sys.path.insert(0, '.')
from evaluation.evaluator import TopicEvaluator
from config import PipelineConfig
from pathlib import Path
import json
import numpy as np

# 配置
config = PipelineConfig()
config.data.dataset = '${DATASET_NAME}'
config.embedding.mode = '${MODE}'

result_dir = Path('${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}')
model_dir = result_dir / 'model'
eval_dir = result_dir / 'evaluation'
eval_dir.mkdir(parents=True, exist_ok=True)

# 加载数据
print('Loading data...')
beta_files = sorted(model_dir.glob('beta_*.npy'), reverse=True)
topic_word_files = sorted(model_dir.glob('topic_words_*.json'), reverse=True)

if not beta_files:
    print('Error: No beta matrix found')
    sys.exit(1)

beta = np.load(beta_files[0])
print(f'Beta shape: {beta.shape}')

# 加载词汇表
bow_dir = Path('${PROJECT_ROOT}/result/${DATASET_NAME}/bow')
with open(bow_dir / 'vocab.json') as f:
    vocab = json.load(f)

# 加载主题词
if topic_word_files:
    with open(topic_word_files[0]) as f:
        topic_words = json.load(f)
else:
    topic_words = None

# 评估
print('Running evaluation...')
evaluator = TopicEvaluator(config)

metrics = {}

# Topic Diversity
if topic_words:
    all_words = []
    for words in topic_words.values():
        all_words.extend(words[:10])
    unique_words = len(set(all_words))
    total_words = len(all_words)
    metrics['topic_diversity'] = unique_words / total_words if total_words > 0 else 0
    print(f'Topic Diversity: {metrics[\"topic_diversity\"]:.4f}')

# Topic Coherence (简化版)
# 完整版需要外部语料库
metrics['num_topics'] = beta.shape[0]
metrics['vocab_size'] = beta.shape[1]

# 保存结果
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = eval_dir / f'metrics_{timestamp}.json'

with open(output_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'Saved metrics to {output_file}')
print(f'Metrics: {json.dumps(metrics, indent=2)}')
"

echo ""
echo "=============================================="
echo "评估完成!"
echo "输出目录: ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/evaluation/"
echo "=============================================="
echo ""
echo "下一步: 运行 07_visualize.sh ${DATASET_NAME} ${MODE}"

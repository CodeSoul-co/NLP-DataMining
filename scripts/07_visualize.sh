#!/bin/bash
# ============================================================================
# 步骤7: 可视化
# ============================================================================
# 功能: 生成主题模型可视化图表
# 输出:
#   - 主题词云 (topic_wordclouds.png)
#   - 主题相似度热力图 (topic_similarity.png)
#   - 文档-主题分布 (doc_topic_dist.png)
#   - 主题演化图 (如果有时间信息)
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
MODE="${2:-zero_shot}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
ETM_DIR="${PROJECT_ROOT}/ETM"

echo "=============================================="
echo "步骤7: 可视化"
echo "=============================================="
echo "数据集: ${DATASET_NAME}"
echo "模式: ${MODE}"
echo "=============================================="

cd "${ETM_DIR}"

# 运行可视化
python -c "
import sys
sys.path.insert(0, '.')
from visualization.topic_visualizer import TopicVisualizer
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
viz_dir = result_dir / 'visualization'
viz_dir.mkdir(parents=True, exist_ok=True)

# 加载数据
print('Loading data...')

# Beta矩阵
beta_files = sorted(model_dir.glob('beta_*.npy'), reverse=True)
if beta_files:
    beta = np.load(beta_files[0])
    print(f'Beta shape: {beta.shape}')
else:
    print('Warning: No beta matrix found')
    beta = None

# Theta矩阵
theta_files = sorted(model_dir.glob('theta_*.npy'), reverse=True)
if theta_files:
    theta = np.load(theta_files[0])
    print(f'Theta shape: {theta.shape}')
else:
    print('Warning: No theta matrix found')
    theta = None

# 主题词
topic_word_files = sorted(model_dir.glob('topic_words_*.json'), reverse=True)
if topic_word_files:
    with open(topic_word_files[0]) as f:
        topic_words = json.load(f)
    print(f'Loaded {len(topic_words)} topics')
else:
    print('Warning: No topic words found')
    topic_words = None

# 词汇表
bow_dir = Path('${PROJECT_ROOT}/result/${DATASET_NAME}/bow')
vocab_file = bow_dir / 'vocab.json'
if vocab_file.exists():
    with open(vocab_file) as f:
        vocab = json.load(f)
else:
    vocab = None

# 生成可视化
print('Generating visualizations...')
visualizer = TopicVisualizer(output_dir=str(viz_dir))

if topic_words:
    print('  - Generating word clouds...')
    visualizer.plot_topic_wordclouds(topic_words)

if beta is not None:
    print('  - Generating topic similarity heatmap...')
    visualizer.plot_topic_similarity(beta)

if theta is not None:
    print('  - Generating document-topic distribution...')
    visualizer.plot_doc_topic_distribution(theta)

print(f'Visualizations saved to {viz_dir}')
"

echo ""
echo "=============================================="
echo "可视化完成!"
echo "输出目录: ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/visualization/"
echo "=============================================="
echo ""
echo "全部流程完成! 查看结果: ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/"

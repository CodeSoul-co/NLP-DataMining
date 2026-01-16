# ETM Project Structure

## Pipeline Overview

```
Raw Data → Data Cleaning → Embedding → ETM Training → Evaluation & Visualization
   │            │              │            │                │
   ▼            ▼              ▼            ▼                ▼
 data/     dataclean/     embedding/      ETM/          results/
           (格式转换)     (Qwen嵌入)    (主题模型)      (评估可视化)
           (NLP清洗)      N×D矩阵       N×V矩阵
                                        训练theta,beta
```

## Data Flow

### Stage 1: Raw Data (data/)
- **Input**: 各种格式的原始文本数据 (CSV, JSON, TXT, PDF等)
- **Output**: 标准化的CSV文件

### Stage 2: Data Cleaning (dataclean/)
- **Input**: 原始数据文件
- **Output**: 清洗后的CSV文件，格式为:
  - `text`: 清洗后的文本
  - `label`: 标签(可选)
  - `timestamp`: 时间戳(可选，用于时序分析)

### Stage 3: Embedding (embedding/)
- **Input**: 清洗后的CSV文件
- **Process**: 使用Qwen模型生成文档嵌入
- **Output**: 
  - `{dataset}_{mode}_embeddings.npy`: N×D矩阵 (N=文档数, D=1024嵌入维度)
  - `{dataset}_{mode}_labels.npy`: 标签数组
  - `{dataset}_{mode}_metadata.json`: 元数据

### Stage 4: ETM Training (ETM/)
- **Input**: 
  - 清洗后的文本 (用于生成BOW)
  - N×D文档嵌入矩阵
- **Process**:
  1. 生成词汇表和BOW矩阵 (N×V)
  2. 生成词汇嵌入 (V×D)
  3. ETM模型训练
- **Output**:
  - `theta_{timestamp}.npy`: 文档-主题分布 (N×K)
  - `beta_{timestamp}.npy`: 主题-词分布 (K×V)
  - `topic_words_{timestamp}.json`: 主题词
  - `etm_model_{timestamp}.pt`: 模型权重

### Stage 5: Evaluation & Visualization (evaluation/, visualization/)
- **Input**: ETM训练结果
- **Output**: 
  - 评估指标 (Coherence, Diversity, Stability等)
  - 可视化图表 (词云, 热力图, 时序分析等)

## Directory Structure

```
/root/autodl-tmp/
│
├── data/                          # 原始数据（只读）
│   ├── socialTwitter/
│   ├── hatespeech/
│   ├── mental_health/
│   ├── FCPB/
│   └── germanCoal/
│
├── embedding/                     # Qwen嵌入生成代码
│   ├── main.py                    # 嵌入生成主入口
│   ├── embedder.py                # 嵌入器
│   ├── trainer.py                 # 训练器(supervised/unsupervised)
│   ├── data_loader.py             # 数据加载
│   ├── scripts/                   # 运行脚本
│   ├── checkpoints/               # 训练checkpoint
│   └── logs/                      # 嵌入训练日志
│
├── ETM/                           # ETM主题模型代码
│   ├── main.py                    # 统一入口 ★
│   ├── config.py                  # 配置管理 ★
│   │
│   ├── dataclean/                 # 数据清洗模块
│   │   ├── main.py
│   │   └── src/
│   │
│   ├── bow/                       # BOW生成模块
│   │   ├── vocab_builder.py
│   │   └── bow_generator.py
│   │
│   ├── model/                     # ETM核心模型
│   │   ├── etm.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── vocab_embedder.py
│   │
│   ├── evaluation/                # 评估模块
│   │   ├── metrics.py
│   │   ├── topic_metrics.py
│   │   └── advanced_metrics.py
│   │
│   ├── visualization/             # 可视化模块
│   │   ├── topic_visualizer.py
│   │   └── temporal_analysis.py
│   │
│   └── logs/                      # ETM训练日志
│
├── qwen3_embedding_0.6B/          # Qwen预训练模型（只读）
│
└── result/                        # ★ 所有输出结果（统一存放）
    └── {dataset}/                 # 按数据集分
        └── {mode}/                # 按模式分 (zero_shot/supervised/unsupervised)
            ├── embeddings/        # 文档嵌入 N×D
            │   ├── {dataset}_{mode}_embeddings.npy
            │   ├── {dataset}_{mode}_labels.npy
            │   └── {dataset}_{mode}_metadata.json
            │
            ├── bow/               # BOW矩阵 N×V + 词汇表
            │   ├── bow_matrix.npz
            │   ├── vocab.txt
            │   └── vocab_embeddings.npy
            │
            ├── model/             # ETM模型 + theta, beta
            │   ├── theta_{timestamp}.npy
            │   ├── beta_{timestamp}.npy
            │   ├── topic_embeddings_{timestamp}.npy
            │   ├── topic_words_{timestamp}.json
            │   ├── etm_model_{timestamp}.pt
            │   ├── training_history_{timestamp}.json
            │   └── config_{timestamp}.json
            │
            ├── evaluation/        # 评估指标
            │   └── metrics_{timestamp}.json
            │
            └── visualization/     # 可视化结果
                ├── topic_words_*.png
                ├── topic_similarity_*.png
                ├── doc_topics_*.png
                ├── topic_proportions_*.png
                └── pyldavis.html
```

## Key Matrices

| Matrix | Shape | Description |
|--------|-------|-------------|
| Doc Embeddings | N × D | Qwen文档嵌入 (D=1024) |
| BOW Matrix | N × V | 词袋矩阵 (V=vocab_size) |
| Vocab Embeddings | V × D | 词汇嵌入 |
| Theta | N × K | 文档-主题分布 |
| Beta | K × V | 主题-词分布 |
| Topic Embeddings | K × D | 主题嵌入 |

## Configuration Parameters

### Training Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| num_topics | 20 | 5-100 | 主题数量 |
| vocab_size | 5000 | 1000-20000 | 词汇表大小 |
| epochs | 50 | 20-200 | 训练轮数 |
| batch_size | 64 | 32-256 | 批次大小 |
| learning_rate | 0.002 | 0.0001-0.01 | 学习率 |
| hidden_dim | 512 | 256-1024 | 隐藏层维度 |
| dropout | 0.2 | 0.1-0.5 | Dropout率 |

### KL Annealing Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| kl_start | 0.0 | 0.0-0.5 | KL权重起始值 |
| kl_end | 1.0 | 0.5-1.0 | KL权重终止值 |
| kl_warmup | 50 | 10-100 | KL预热轮数 |

## Evaluation Metrics

### Topic Quality
- **Topic Coherence (NPMI)**: 主题内词语共现一致性
- **Topic Diversity (TD)**: 主题间词语多样性
- **Topic Stability**: 多次运行的主题稳定性

### Model Quality
- **Perplexity**: 困惑度
- **Reconstruction Loss**: 重建损失
- **KL Divergence**: KL散度

### Visualization
- **Word Clouds**: 主题词云
- **Topic Similarity Heatmap**: 主题相似度热力图
- **Document-Topic Distribution**: 文档-主题分布(t-SNE/PCA)
- **Topic Proportions**: 主题比例
- **Temporal Analysis**: 时序主题演变(如有时间戳)
- **pyLDAvis**: 交互式主题可视化

## Usage

### Quick Start
```bash
# 1. 数据清洗
python ETM/dataclean/main.py convert data/raw/ data/cleaned/ --language english

# 2. 生成嵌入
python embedding/main.py --dataset socialTwitter --mode zero_shot

# 3. ETM训练
python ETM/main.py train --dataset socialTwitter --mode zero_shot --num_topics 20

# 4. 评估和可视化
python ETM/main.py evaluate --dataset socialTwitter --mode zero_shot
python ETM/main.py visualize --dataset socialTwitter --mode zero_shot
```

### Full Pipeline
```bash
python ETM/main.py pipeline --dataset socialTwitter --mode zero_shot --num_topics 20
```

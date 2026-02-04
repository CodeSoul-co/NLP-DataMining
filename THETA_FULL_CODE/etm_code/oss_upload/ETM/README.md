# ETM Topic Model Pipeline

Unified framework for topic model training, evaluation, and visualization.

## Directory Structure

### Result Directory Organization

All results are now organized in a standardized structure:

```
/root/autodl-tmp/result/baseline/{dataset}/{model}/
├── bow/                    # BOW data and vocabulary
│   ├── bow_matrix.npz
│   ├── vocab.json
│   ├── vocab.txt
│   └── vocab_embeddings.npy (optional)
├── model/                  # Model parameter files
│   ├── theta_k{K}.npy
│   ├── beta_k{K}.npy
│   ├── beta_over_time_k{K}.npy (DTM only)
│   ├── model_k{K}.pt
│   └── training_history_k{K}.json
├── evaluation/             # Evaluation results
│   ├── metrics_k{K}.json
│   └── metrics_k{K}.csv
├── topicwords/             # Topic words related
│   ├── topic_words_k{K}.json
│   └── topic_evolution_k{K}.json (DTM only)
└── visualization_k{K}_{lang}_{timestamp}/  # Visualization results
```

### Using ResultManager

The `ResultManager` class provides a unified interface for saving and loading results:

```python
from utils.result_manager import ResultManager

# Initialize manager
manager = ResultManager(
    result_dir='/root/autodl-tmp/result/baseline',
    dataset='edu_data',
    model='dtm',
    num_topics=20
)

# Save all results
manager.save_all(
    theta=theta,
    beta=beta,
    vocab=vocab,
    topic_words=topic_words,
    metrics=metrics,
    beta_over_time=beta_over_time,  # DTM only
    topic_evolution=topic_evolution  # DTM only
)

# Load all results
data = manager.load_all(num_topics=20)
```

### Migrating Old Results

If you have results in the old flat directory structure, use the migration utility:

```bash
# Migrate old structure to new structure
python -m utils.result_manager --action migrate --dataset edu_data --model dtm

# Or in Python
from utils.result_manager import migrate_baseline_results
migrate_baseline_results(dataset='edu_data', model='dtm')
```

## Supported Models

| Model | Description | Embedding |
|-------|-------------|-----------|
| **THETA** | Qwen Embedding-based Topic Model (Our Method) | Qwen 0.6B/4B/8B |
| **LDA** | Latent Dirichlet Allocation (sklearn) | None |
| **ETM** | Embedded Topic Model (Original) | Word2Vec |
| **CTM** | Contextualized Topic Model | SBERT |
| **DTM** | Dynamic Topic Model (Temporal) | SBERT |

## 快速开始

### 统一入口：`run_pipeline.py`

```bash
# THETA模型（需指定模型大小和模式）
python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --mode zero_shot
python run_pipeline.py --dataset socialTwitter --models theta --model_size 4B --mode supervised

# Baseline模型（不需要model_size）
python run_pipeline.py --dataset socialTwitter --models lda
python run_pipeline.py --dataset socialTwitter --models lda,etm,ctm

# DTM模型（需要时间戳数据）
python run_pipeline.py --dataset edu_data --models dtm --num_topics 20 --epochs 100

# 检查数据文件是否存在（运行前先检查）
python run_pipeline.py --dataset socialTwitter --models theta --model_size 4B --check-only

# 跳过训练，只做评估和可视化
python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --skip-train

# 只训练，不评估和可视化
python run_pipeline.py --dataset socialTwitter --models lda --skip-eval --skip-viz
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集名称 | 必填 |
| `--models` | 模型列表（逗号分隔） | 必填 |
| `--model_size` | Qwen模型大小 (THETA专用) | 0.6B |
| `--mode` | THETA模式 | zero_shot |
| `--num_topics` | 主题数量 | 20 |
| `--epochs` | 训练轮数 | 100 |
| `--batch_size` | 批次大小 | 64 |
| `--skip-train` | 跳过训练 | False |
| `--skip-eval` | 跳过评估 | False |
| `--skip-viz` | 跳过可视化 | False |
| `--check-only` | 只检查数据文件 | False |
| `--gpu` | GPU ID | 0 |
| `--language` | 可视化语言 (en/zh) | en |

### Qwen模型大小

| 模型 | 参数量 | Embedding维度 | 说明 |
|------|--------|---------------|------|
| 0.6B | 6亿 | 1024 | 默认，速度快 |
| 4B | 40亿 | 2560 | 中等 |
| 8B | 80亿 | 4096 | 最大，效果最好 |

### 支持的数据集

- `socialTwitter`
- `hatespeech`
- `mental_health`
- `FCPB`
- `germanCoal`
- `edu_data` (中文教育政策文档，带时间戳，适合DTM)

## 新数据集使用流程

### 完整数据流

```
原始数据 → [dataclean] → 清洗后CSV → [prepare_data] → embedding/BOW → [run_pipeline] → 训练/评估/可视化
```

### 方式一：已有清洗好的CSV

如果你已经有清洗好的CSV文件：

```bash
# 1. 放置CSV文件
mkdir -p data/my_dataset
cp your_cleaned_data.csv data/my_dataset/my_dataset_cleaned.csv

# 2. 生成预处理数据
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B

# 3. 运行训练
python run_pipeline.py --dataset my_dataset --models theta --model_size 0.6B
```

### 方式二：从原始数据开始（包含清洗）

如果你有原始的CSV/TXT文件需要清洗：

```bash
# 一步完成：清洗 + 预处理
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B \
    --clean --raw-input /path/to/raw_data.csv --language english

# 中文数据
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B \
    --clean --raw-input /path/to/raw_data.csv --language chinese
```

### 方式三：分步执行

```bash
# 步骤1: 数据清洗（可选，如果原始数据需要清洗）
python prepare_data.py --dataset my_dataset --model baseline \
    --clean --raw-input /path/to/raw_data.csv

# 步骤2: 生成预处理数据
# THETA模型
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B --mode zero_shot

# Baseline模型
python prepare_data.py --dataset my_dataset --model baseline

# 步骤3: 检查数据是否就绪
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B --check-only

# 步骤4: 运行训练
python run_pipeline.py --dataset my_dataset --models theta --model_size 0.6B
```

### CSV文件要求

- **文本列**（必须）：`cleaned_content`, `clean_text`, `text`, `content` 之一
- **标签列**（可选）：`label`, `category`

### 各模型所需的预处理数据

| 模型 | 所需数据 | 生成命令 |
|------|----------|----------|
| **THETA** | Qwen embedding + BOW + vocab_embedding | `--model theta` |
| **LDA** | BOW | `--model baseline` |
| **ETM** | BOW + Word2Vec (自动生成) | `--model baseline` |
| **CTM** | BOW + SBERT embedding | `--model baseline` |
| **DTM** | BOW + SBERT embedding + 时间片信息 | `--model dtm` |

## DTM模型使用指南

DTM (Dynamic Topic Model) 用于分析主题随时间的演化，需要带时间戳的数据。

### DTM数据要求

CSV文件需要包含：
- **文本列**：`cleaned_content`, `text` 等
- **时间列**：`year`, `timestamp`, `date` 等（用于划分时间片）

### DTM使用示例

```bash
# 1. 从docx目录处理数据（自动使用dataclean模块）
python prepare_data.py --dataset edu_data --model dtm \
    --clean --raw-input /path/to/docx_directory --language chinese

# 2. 从已有CSV准备DTM数据
python prepare_data.py --dataset my_data --model dtm --time_column year

# 3. 训练DTM模型
python run_pipeline.py --dataset edu_data --models dtm --num_topics 20 --epochs 100
```

### DTM输出文件

```
result/baseline/{dataset}/dtm/
├── theta_k20.npy              # 文档-主题分布
├── beta_k20.npy               # 主题-词分布（最后时间片）
├── beta_over_time_k20.npy     # 所有时间片的主题-词分布
├── topic_words_k20.json       # 主题词
├── topic_evolution_k20.json   # 主题词随时间演化
├── training_history_k20.json  # 训练历史（含loss和perplexity）
├── metrics_k20.json           # 评估指标
└── visualization/             # 可视化图表
```

## 项目结构

```
ETM/
├── run_pipeline.py      # 统一入口脚本 ⭐
├── prepare_data.py      # 数据预处理脚本 ⭐
├── main.py              # THETA模型主入口
├── config.py            # 配置管理
│
├── model/               # 模型定义
│   ├── baseline_trainer.py  # Baseline训练器（LDA/ETM/CTM/DTM）
│   ├── etm.py              # THETA模型
│   ├── lda.py              # LDA模型
│   ├── ctm.py              # CTM模型
│   ├── dtm.py              # DTM模型（动态主题模型）
│   └── etm_original.py     # 原始ETM模型
│
├── evaluation/          # 评估模块
│   ├── unified_evaluator.py  # 统一评估器 ⭐
│   └── topic_metrics.py      # 评估指标
│
├── visualization/       # 可视化模块
│   ├── run_visualization.py
│   └── visualization_generator.py
│
├── bow/                 # BOW生成
├── data/                # 数据加载
├── dataclean/           # 数据清洗
└── preprocessing/       # 预处理
```

## 数据流

```
数据处理 → 模型训练 → 评估 → 可视化 → 结果保存
```

### 结果保存位置

- **THETA**: `result/0.6B/{dataset}/{mode}/`
- **Baseline**: `result/baseline/{dataset}/{model}/`

### 评估指标（7个）

1. **TD** - Topic Diversity
2. **iRBO** - Inverse Rank-Biased Overlap
3. **NPMI** - Normalized PMI Coherence
4. **C_V** - C_V Coherence
5. **UMass** - UMass Coherence
6. **Exclusivity** - Topic Exclusivity
7. **PPL** - Perplexity

## 示例

### 训练LDA并查看结果

```bash
# 训练
python run_pipeline.py --dataset socialTwitter --models lda

# 结果位置
ls result/baseline/socialTwitter/lda/
# theta_k20.npy, beta_k20.npy, topic_words_k20.json, metrics_k20.json
# visualization/global/*.png
```

### 对比多个模型

```bash
# 同时训练LDA、ETM、CTM
python run_pipeline.py --dataset socialTwitter --models lda,etm,ctm

# 查看评估结果
cat result/baseline/socialTwitter/lda/metrics_k20.json
cat result/baseline/socialTwitter/etm/metrics_k20.json
cat result/baseline/socialTwitter/ctm_zeroshot/metrics_k20.json
```

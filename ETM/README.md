# THETA: Topic Modeling with Hybrid Embedding Training Architecture

基于Qwen-Embedding的主题模型系统，支持ETM、DTM、LDA等多种主题模型。

## 项目概述

本项目实现了一个完整的主题建模Pipeline，核心特点：

- **双层模型架构**: Qwen-Embedding + 主题模型（ETM/DTM/LDA）
- **多种Embedding模式**: Zero-shot / Supervised / Unsupervised
- **可扩展设计**: 支持添加新的Embedding模型和主题模型
- **统一接口**: 前后端人员可直接调用，无需了解内部实现

---

## 目录结构

```
/root/autodl-tmp/
├── README.md                    # 本文件 - 项目说明
├── data/                        # 数据目录
│   ├── hatespeech/             # 数据集: 仇恨言论检测
│   ├── socialTwitter/          # 数据集: 社交媒体
│   ├── germanCoal/             # 数据集: 德语煤炭
│   ├── mental_health/          # 数据集: 心理健康
│   └── FCPB/                   # 数据集: FCPB
│
├── embedding/                   # Embedding模块
│   ├── main.py                 # Embedding训练入口
│   ├── embedder.py             # Qwen-Embedding封装
│   ├── trainer.py              # Supervised/Unsupervised训练
│   ├── data_loader.py          # 数据加载器
│   ├── registry.py             # Embedding模型注册表 ⭐
│   ├── checkpoints/            # 训练检查点
│   └── outputs/                # 临时输出
│
├── ETM/                         # 主题模型模块
│   ├── main.py                 # ETM训练入口
│   ├── pipeline_api.py         # 统一API接口 ⭐⭐⭐
│   ├── config.py               # 配置管理
│   ├── model/                  # 模型定义
│   │   ├── etm.py             # ETM模型
│   │   ├── dtm.py             # DTM动态主题模型
│   │   ├── lda.py             # LDA经典主题模型
│   │   ├── encoder.py         # 编码器
│   │   ├── decoder.py         # 解码器
│   │   └── registry.py        # 主题模型注册表 ⭐
│   ├── bow/                    # BOW生成
│   │   └── bow_generator.py   # 词袋矩阵生成器
│   ├── dataclean/              # 数据清洗
│   │   └── main.py            # 清洗CLI工具
│   ├── evaluation/             # 评估模块
│   │   └── evaluator.py       # 主题评估器
│   ├── visualization/          # 可视化
│   │   └── topic_visualizer.py # 可视化工具
│   └── logs/                   # 训练日志
│
├── result/                      # 结果输出目录
│   └── {dataset}/              # 按数据集组织
│       ├── bow/                # BOW矩阵和词汇表
│       ├── zero_shot/          # Zero-shot模式结果
│       │   ├── embeddings/    # 文档/词汇embedding
│       │   ├── model/         # 训练好的模型
│       │   ├── evaluation/    # 评估结果
│       │   └── visualization/ # 可视化图表
│       ├── supervised/         # Supervised模式结果
│       └── unsupervised/       # Unsupervised模式结果
│
├── scripts/                     # 运行脚本 ⭐
│   ├── 01_data_upload_clean.sh # 数据上传与清洗
│   ├── 02_embedding_generate.sh # Embedding生成
│   ├── 03_bow_generate.sh      # BOW矩阵生成
│   ├── 04_vocab_embedding.sh   # 词汇Embedding生成
│   ├── 05_etm_train.sh         # ETM模型训练
│   ├── 06_evaluate.sh          # 模型评估
│   ├── 07_visualize.sh         # 可视化生成
│   ├── run_full_pipeline.sh    # 完整Pipeline一键运行
│   └── run_quick_demo.sh       # 快速演示
│
└── qwen3_embedding_0.6B/        # Qwen3-Embedding模型文件
```

---

## 数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           完整数据流程                                    │
└─────────────────────────────────────────────────────────────────────────┘

1. 数据上传与清洗
   ┌──────────┐     ┌──────────┐     ┌──────────────────┐
   │ 原始数据  │ ──▶ │ 数据清洗  │ ──▶ │ {dataset}_cleaned.csv │
   │ (txt/csv) │     │ dataclean │     │                  │
   └──────────┘     └──────────┘     └──────────────────┘

2. Embedding生成
   ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
   │ {dataset}_cleaned.csv │ ──▶ │ Qwen-Embedding  │ ──▶ │ doc_embeddings.npy │
   │                  │     │ (zero/sup/unsup) │     │ (D × 1024)       │
   └──────────────────┘     └─────────────────┘     └──────────────────┘

3. BOW生成
   ┌──────────────────┐     ┌──────────────┐     ┌──────────────────┐
   │ {dataset}_cleaned.csv │ ──▶ │ BOW Generator │ ──▶ │ bow_matrix.npz   │
   │                  │     │              │     │ vocab.json       │
   └──────────────────┘     └──────────────┘     └──────────────────┘

4. 词汇Embedding
   ┌──────────────┐     ┌─────────────────┐     ┌────────────────────┐
   │ vocab.json   │ ──▶ │ Qwen-Embedding  │ ──▶ │ vocab_embeddings.npy │
   │ (V words)    │     │                 │     │ (V × 1024)         │
   └──────────────┘     └─────────────────┘     └────────────────────┘

5. 主题模型训练
   ┌────────────────────┐
   │ doc_embeddings.npy │ ──┐
   │ (D × 1024)         │   │     ┌─────────────┐     ┌──────────────────┐
   └────────────────────┘   ├──▶  │ ETM/DTM/LDA │ ──▶ │ theta (D × K)    │
   ┌────────────────────┐   │     │             │     │ beta (K × V)     │
   │ vocab_embeddings   │ ──┤     └─────────────┘     │ topic_words.json │
   │ (V × 1024)         │   │                         └──────────────────┘
   └────────────────────┘   │
   ┌────────────────────┐   │
   │ bow_matrix.npz     │ ──┘
   │ (D × V)            │
   └────────────────────┘

6. 评估与可视化
   ┌──────────────────┐     ┌──────────────┐     ┌──────────────────┐
   │ theta, beta      │ ──▶ │ Evaluator    │ ──▶ │ metrics.json     │
   │ topic_words      │     │ Visualizer   │     │ *.png            │
   └──────────────────┘     └──────────────┘     └──────────────────┘
```

**矩阵说明**:
- `D`: 文档数量
- `V`: 词汇表大小 (vocab_size)
- `K`: 主题数量 (num_topics)
- `E`: Embedding维度 (1024 for Qwen3-0.6B)

---

## 快速开始

### 方式1: 使用统一API (推荐)

```python
from ETM.pipeline_api import get_available_options, run_pipeline, PipelineRequest

# 1. 查看所有可配置选项
options = get_available_options()
print(options["num_topics"])        # [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
print(options["embedding_modes"])   # ['zero_shot', 'supervised', 'unsupervised']
print(options["topic_models"])      # {'etm': {...}, 'dtm': {...}, 'lda': {...}}

# 2. 创建训练请求
request = PipelineRequest(
    dataset="hatespeech",
    embedding_mode="zero_shot",
    num_topics=20,
    vocab_size=5000,
)

# 3. 运行Pipeline
response = run_pipeline(request)

if response.success:
    print(f"结果目录: {response.result_dir}")
    print(f"主题词: {response.topic_words}")
else:
    print(f"错误: {response.error_message}")
```

### 方式2: 使用脚本

```bash
# 完整Pipeline一键运行
cd /root/autodl-tmp/scripts
./run_full_pipeline.sh hatespeech zero_shot 20 5000

# 或分步执行
./02_embedding_generate.sh hatespeech zero_shot
./03_bow_generate.sh hatespeech 5000
./04_vocab_embedding.sh hatespeech zero_shot
./05_etm_train.sh hatespeech zero_shot 20 5000
./06_evaluate.sh hatespeech zero_shot
./07_visualize.sh hatespeech zero_shot
```

### 方式3: 命令行

```bash
# ETM训练
cd /root/autodl-tmp/ETM
python main.py --dataset hatespeech --mode zero_shot --num_topics 20

# Embedding生成
cd /root/autodl-tmp/embedding
python main.py --mode zero_shot --dataset hatespeech
python main.py --mode supervised --dataset hatespeech --epochs 3
python main.py --mode unsupervised --dataset hatespeech --epochs 3
```

---

## 可配置参数

### 核心参数

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `num_topics` | 主题数量 | 5-100 | 20 |
| `vocab_size` | 词表大小 | 1000-20000 | 5000 |
| `embedding_mode` | Embedding模式 | zero_shot/supervised/unsupervised | zero_shot |
| `topic_model` | 主题模型 | etm/dtm/lda | etm |
| `epochs` | 训练轮数 | 10-500 | 50 |
| `batch_size` | 批次大小 | 16-256 | 64 |
| `learning_rate` | 学习率 | 0.0001-0.01 | 0.002 |

### Embedding模式说明

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `zero_shot` | 直接使用预训练模型生成embedding | 快速实验、无标签数据 |
| `supervised` | 使用标签进行LoRA微调 | 有标签数据、需要更好的语义区分 |
| `unsupervised` | SimCSE自监督训练 | 无标签数据、需要领域适应 |

### 主题模型说明

| 模型 | 说明 | 特点 |
|------|------|------|
| `etm` | Embedded Topic Model | 使用预训练词向量，效果最好 |
| `dtm` | Dynamic Topic Model | 支持时间序列，追踪主题演化 |
| `lda` | Latent Dirichlet Allocation | 经典模型，不需要embedding |

---

## 扩展指南

### 添加新的Embedding模型

编辑 `embedding/registry.py`:

```python
EMBEDDING_MODEL_REGISTRY["new_model"] = EmbeddingModelInfo(
    name="New Embedding Model",
    path="/path/to/model",
    embedding_dim=1024,
    max_length=512,
    description="模型描述",
    languages=["chinese", "english"],
    model_size="1B",
    requires_gpu=True,
    default_batch_size=16
)
```

### 添加新的主题模型

1. 创建模型文件 `ETM/model/new_model.py`
2. 实现统一接口:

```python
class NewModel(nn.Module):
    def __init__(self, vocab_size, num_topics, word_embeddings=None, **kwargs):
        ...
    
    def forward(self, doc_embeddings, bow) -> Dict:
        # 返回 {'loss', 'theta', 'recon_loss', 'kl_loss'}
        ...
    
    def get_beta(self) -> torch.Tensor:
        # 返回 (K, V) 主题-词分布
        ...
    
    def get_topic_words(self, vocab, top_k=10) -> Dict[str, List[str]]:
        # 返回主题词
        ...
```

3. 在 `ETM/model/registry.py` 中注册

### 添加新的数据集

1. 将数据放入 `data/{dataset_name}/`
2. 确保有CSV文件，包含 `text` 列
3. (可选) 在 `ETM/config.py` 的 `DATASET_CONFIGS` 中添加推荐配置

---

## 输出文件说明

训练完成后，结果保存在 `result/{dataset}/{mode}/`:

```
result/hatespeech/zero_shot/
├── embeddings/
│   ├── doc_embeddings.npy      # 文档embedding (D × 1024)
│   └── vocab_embeddings.npy    # 词汇embedding (V × 1024)
├── model/
│   ├── etm_model_*.pt          # 训练好的模型
│   ├── theta_*.npy             # 文档-主题分布 (D × K)
│   ├── beta_*.npy              # 主题-词分布 (K × V)
│   ├── topic_words_*.json      # 主题词列表
│   └── config_*.json           # 训练配置
├── evaluation/
│   └── metrics_*.json          # 评估指标
└── visualization/
    ├── topic_wordclouds.png    # 主题词云
    ├── topic_similarity.png    # 主题相似度热力图
    └── doc_topic_dist.png      # 文档-主题分布
```

---

## 前后端开发人员指南

### API接口

所有功能通过 `ETM/pipeline_api.py` 暴露:

```python
# 获取所有可选参数 (用于生成UI下拉框)
from ETM.pipeline_api import get_available_options
options = get_available_options()

# 获取数据集列表
from ETM.pipeline_api import list_datasets
datasets = list_datasets()

# 获取训练结果
from ETM.pipeline_api import list_results, get_topic_words
results = list_results(dataset="hatespeech")
topic_words = get_topic_words("hatespeech", "zero_shot")

# 运行训练
from ETM.pipeline_api import run_pipeline, PipelineRequest
request = PipelineRequest(dataset="hatespeech", num_topics=20)
response = run_pipeline(request)
```

### 参数验证

```python
from ETM.config import validate_params

is_valid, msg = validate_params({"num_topics": 200})
# False, "num_topics=200 exceeds maximum 100"
```

---

## 依赖环境

```
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.30
peft >= 0.5 (用于LoRA训练)
scipy
numpy
pandas
scikit-learn
matplotlib
wordcloud
tqdm
```

---

## 许可证

MIT License

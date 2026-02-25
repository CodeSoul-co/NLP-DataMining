# THETA 主题模型系统 - 模型技术总结

本文档详细说明 THETA 主题模型系统中所有模型的技术原理、运行逻辑和使用方法。

---

## 目录

1. [系统架构概述](#1-系统架构概述)
2. [模型分类与技术原理](#2-模型分类与技术原理)
3. [各模型详细说明](#3-各模型详细说明)
4. [运行命令参考](#4-运行命令参考)
5. [数据流程](#5-数据流程)
6. [评估指标](#6-评估指标)
7. [可视化系统](#7-可视化系统)
8. [当前进展总结](#8-当前进展总结)

---

## 1. 系统架构概述

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        THETA 主题模型系统                            │
├─────────────────────────────────────────────────────────────────────┤
│  数据层                                                              │
│  ├── dataclean/     数据清洗模块 (中英文支持)                         │
│  ├── preprocessing/ 预处理模块 (分词、去停用词)                        │
│  └── bow/           BOW生成模块 (词袋矩阵、词表)                       │
├─────────────────────────────────────────────────────────────────────┤
│  模型层                                                              │
│  ├── THETA         我们的方法 (Qwen Embedding + VAE)                 │
│  ├── 传统模型       LDA, HDP, BTM, STM                               │
│  └── 神经网络模型   CTM, DTM, ETM, NVDM, GSM, ProdLDA                │
├─────────────────────────────────────────────────────────────────────┤
│  评估层                                                              │
│  └── evaluation/    统一评估器 (7项指标)                              │
├─────────────────────────────────────────────────────────────────────┤
│  可视化层                                                            │
│  └── visualization/ 可视化生成器 (70+图表类型)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心文件结构

```
/root/autodl-tmp/ETM/
├── run_pipeline.py          # 统一入口脚本 ⭐
├── prepare_data.py          # 数据预处理脚本 ⭐
├── main.py                  # THETA 模型主入口
├── config.py                # 配置管理
│
├── model/                   # 模型定义
│   ├── base.py              # 基类定义
│   ├── etm.py               # THETA 模型实现
│   ├── baseline_trainer.py  # Baseline 统一训练器
│   └── baselines/           # Baseline 模型实现
│       ├── lda.py           # LDA 模型
│       ├── hdp.py           # HDP 模型
│       ├── btm.py           # BTM 模型
│       ├── stm.py           # STM 模型
│       ├── ctm.py           # CTM 模型
│       ├── dtm.py           # DTM 模型
│       ├── etm_original.py  # 原始 ETM 模型
│       └── ntm.py           # NVDM, GSM, ProdLDA
│
├── evaluation/              # 评估模块
│   ├── unified_evaluator.py # 统一评估器
│   └── topic_metrics.py     # 评估指标计算
│
└── visualization/           # 可视化模块
    ├── run_visualization.py
    ├── visualization_generator.py
    └── topic_visualizer.py
```

---

## 2. 模型分类与技术原理

### 2.1 模型总览

| 类别 | 模型 | Embedding | 核心技术 | 适用场景 |
|------|------|-----------|----------|----------|
| **我们的方法** | THETA | Qwen 0.6B/4B/8B | VAE + KL退火 | 通用，高质量 |
| **传统模型** | LDA | 无 | Gibbs采样/变分推断 | 基准对比 |
| | HDP | 无 | 层次Dirichlet过程 | 自动确定主题数 |
| | BTM | 无 | Biterm + Gibbs采样 | 短文本 |
| | STM | 无 | 结构化主题模型 | 带协变量 |
| **神经网络模型** | CTM | SBERT | VAE + 文档嵌入 | 上下文感知 |
| | DTM | SBERT | 时序VAE | 主题演化分析 |
| | ETM | Word2Vec | VAE + 词嵌入 | 嵌入式主题 |
| | NVDM | 无 | VAE | 神经变分 |
| | GSM | 无 | VAE + Softmax约束 | 稀疏主题 |
| | ProdLDA | 无 | VAE + PoE | 高质量主题 |

### 2.2 技术原理分类

#### A. 概率图模型 (传统方法)

```
文档 d → θ_d (主题分布) → z_n (主题) → w_n (词)
                              ↑
                         β_k (主题-词分布)
```

- **LDA**: p(θ|α) = Dir(α), p(z|θ) = Cat(θ), p(w|z,β) = Cat(β_z)
- **HDP**: 非参数贝叶斯，自动推断主题数
- **BTM**: 建模词对(biterm)而非文档，适合短文本

#### B. 变分自编码器 (神经网络方法)

```
文档 x → Encoder → μ, σ → z ~ N(μ, σ²) → Decoder → x̂
                    ↓
              KL(q(z|x) || p(z))
```

- **损失函数**: L = Reconstruction Loss + β * KL Divergence
- **KL退火**: β 从 0 逐渐增加到 1，防止后验坍塌

---

## 3. 各模型详细说明

### 3.1 THETA 模型 (我们的方法)

**技术架构**:
```
输入文档 → Qwen Embedding (1024/2560/4096维)
                ↓
         Encoder (MLP + Softplus)
                ↓
         μ, log_σ² → z ~ N(μ, σ²)  [主题分布]
                ↓
         Decoder (词嵌入矩阵)
                ↓
         重构的词分布
```

**核心特点**:
- 使用 Qwen3-Embedding 作为文档表示
- 支持三种训练模式: zero_shot, supervised, unsupervised
- KL退火策略防止后验坍塌

**模型规格**:

| 模型 | 参数量 | Embedding维度 | 显存需求 | 推荐场景 |
|------|--------|---------------|----------|----------|
| 0.6B | 600M | 1024 | ~4GB | 快速实验 |
| 4B | 4B | 2560 | ~12GB | 平衡效果 |
| 8B | 8B | 4096 | ~24GB | 最佳效果 |

**运行命令**:
```bash
# 数据预处理
python prepare_data.py --dataset edu_data --model theta --model_size 0.6B --mode zero_shot

# 训练
python run_pipeline.py --dataset edu_data --models theta --model_size 0.6B --mode zero_shot \
    --num_topics 20 --epochs 100 --batch_size 64 --learning_rate 0.002 \
    --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --language zh
```

**输出位置**: `/root/autodl-tmp/result/{model_size}/{dataset}/{mode}/`

---

### 3.2 LDA (Latent Dirichlet Allocation)

**技术原理**:
- 生成模型: 每个文档是主题的混合，每个主题是词的分布
- 推断方法: sklearn 使用变分推断 (Variational Bayes)

**数学公式**:
```
p(θ, z, w | α, β) = p(θ|α) ∏_n p(z_n|θ) p(w_n|z_n, β)
```

**核心参数**:
- `n_components`: 主题数量
- `max_iter`: 最大迭代次数 (默认100)

**运行命令**:
```bash
bash scripts/05_train_baseline.sh --dataset edu_data --models lda \
    --num_topics 20 --language zh
```

**输出文件**:
- `theta_k{K}.npy`: 文档-主题分布 (n_docs × K)
- `beta_k{K}.npy`: 主题-词分布 (K × vocab_size)
- `topic_words_k{K}.json`: 每个主题的Top词

---

### 3.3 HDP (Hierarchical Dirichlet Process)

**技术原理**:
- 非参数贝叶斯方法，自动推断主题数量
- 使用 Gensim 的 HdpModel 实现

**核心特点**:
- 不需要预先指定主题数
- 通过数据自动确定最优主题数
- 适合探索性分析

**核心参数**:
- `max_topics`: 最大主题数上限 (默认150)
- `alpha`: 文档-主题先验
- `gamma`: 主题-词先验

**运行命令**:
```bash
bash scripts/05_train_baseline.sh --dataset edu_data --models hdp --language zh
```

---

### 3.4 BTM (Biterm Topic Model)

**技术原理**:
- 专为短文本设计
- 建模词对(biterm)而非整个文档
- 使用 Gibbs 采样进行推断

**数学模型**:
```
对于每个 biterm (w_i, w_j):
  1. 采样主题 z ~ p(z|θ)
  2. 采样词 w_i ~ p(w|z, φ)
  3. 采样词 w_j ~ p(w|z, φ)
```

**核心参数**:
- `alpha`: 主题先验 (默认0.1)
- `beta`: 词先验 (默认0.01)
- `n_iter`: Gibbs采样迭代次数 (默认100)

**运行命令**:
```bash
bash scripts/05_train_baseline.sh --dataset edu_data --models btm \
    --num_topics 20 --language zh
```

**注意**: BTM 对长文档效率较低（会生成大量biterms）

---

### 3.5 CTM (Contextualized Topic Model)

**技术原理**:
- 结合 SBERT 文档嵌入和 VAE
- 支持 ZeroShot 和 Combined 两种推断模式

**架构**:
```
文档 → SBERT → 文档嵌入 (384维)
                  ↓
            Encoder (MLP)
                  ↓
            μ, σ → z (主题分布)
                  ↓
            Decoder → 词分布
```

**推断模式**:
- **ZeroShot**: 只使用 SBERT 嵌入，不需要 BOW
- **Combined**: 同时使用 SBERT 嵌入和 BOW

**核心参数**:
- `inference_type`: 'zeroshot' 或 'combined'
- `hidden_sizes`: 编码器隐藏层 (默认 (100, 100))
- `epochs`: 训练轮数
- `batch_size`: 批大小

**运行命令**:
```bash
bash scripts/05_train_baseline.sh --dataset edu_data --models ctm \
    --num_topics 20 --epochs 100 --batch_size 64 --language zh
```

---

### 3.6 DTM (Dynamic Topic Model)

**技术原理**:
- 时序主题模型，分析主题随时间的演化
- 每个时间片有独立的主题-词分布

**架构**:
```
时间片 t:
  文档 → SBERT嵌入 + 时间编码 → Encoder → θ_t (主题分布)
                                           ↓
  主题演化: β_t = f(β_{t-1}, Δ_t)  →  Decoder → 词分布
```

**数据要求**:
- CSV 必须包含时间列 (`year`, `timestamp`, `date`)
- 系统自动按时间分片

**核心参数**:
- `num_time_slices`: 时间片数量 (自动从数据推断)
- `epochs`: 训练轮数
- `batch_size`: 批大小

**运行命令**:
```bash
# 数据预处理 (生成时间片信息)
python prepare_data.py --dataset edu_data --model dtm --time_column year

# 训练
bash scripts/05_train_baseline.sh --dataset edu_data --models dtm \
    --num_topics 20 --epochs 100 --language zh
```

**输出文件**:
- `beta_over_time_k{K}.npy`: 所有时间片的主题-词分布 (T × K × V)
- `topic_evolution_k{K}.json`: 主题词随时间的演化

---

### 3.7 NVDM (Neural Variational Document Model)

**技术原理**:
- 最基础的神经主题模型
- 标准 VAE 架构

**架构**:
```
BOW → Encoder (MLP) → μ, σ → z ~ N(μ, σ²) → Decoder (MLP) → 词分布
```

**核心参数**:
- `hidden_dim`: 隐藏层维度 (默认512)
- `dropout`: Dropout率 (默认0.2)
- `epochs`: 训练轮数
- `learning_rate`: 学习率

**运行命令**:
```bash
bash scripts/05_train_baseline.sh --dataset edu_data --models nvdm \
    --num_topics 20 --epochs 100 --hidden_dim 512 --learning_rate 0.002 --language zh
```

---

### 3.8 GSM (Gaussian Softmax Model)

**技术原理**:
- 在 NVDM 基础上增加 Softmax 约束
- 主题分布通过 Softmax 归一化

**特点**:
- 主题分布更稀疏
- 更好的可解释性

**运行命令**:
```bash
bash scripts/05_train_baseline.sh --dataset edu_data --models gsm \
    --num_topics 20 --epochs 100 --hidden_dim 512 --learning_rate 0.002 --language zh
```

---

### 3.9 ProdLDA (Product of Experts LDA)

**技术原理**:
- 使用 Dirichlet 先验的近似
- Product of Experts 架构

**数学公式**:
```
p(θ) ≈ Logistic-Normal(μ_0, Σ_0)
q(θ|x) = Logistic-Normal(μ(x), Σ(x))
```

**特点**:
- 主题质量通常优于 NVDM 和 GSM
- 更接近 LDA 的概率解释

**运行命令**:
```bash
bash scripts/05_train_baseline.sh --dataset edu_data --models prodlda \
    --num_topics 20 --epochs 100 --hidden_dim 512 --learning_rate 0.002 --language zh
```

---

### 3.10 ETM (Embedded Topic Model - 原始版本)

**技术原理**:
- 使用 Word2Vec 词嵌入
- 主题和词共享嵌入空间

**架构**:
```
词嵌入矩阵 ρ (V × E)
主题嵌入矩阵 α (K × E)
β_k = softmax(α_k · ρ^T)  # 主题-词分布
```

**特点**:
- 词嵌入提供语义信息
- 主题更具语义连贯性

**运行命令**:
```bash
bash scripts/05_train_baseline.sh --dataset edu_data --models etm \
    --num_topics 20 --epochs 100 --hidden_dim 512 --learning_rate 0.002 --language zh
```

---

## 4. 运行命令参考

### 4.1 Bash 脚本统一入口

```bash
# 基本格式
bash scripts/05_train_baseline.sh \
    --dataset <数据集名> \
    --models <模型列表> \
    --num_topics <主题数> \
    --epochs <训练轮数> \
    --batch_size <批大小> \
    --hidden_dim <隐藏层维度> \
    --learning_rate <学习率> \
    --gpu <GPU编号> \
    --language <zh/en> \
    [--skip-train] \
    [--skip-viz]
```

### 4.2 完整参数列表

| 参数 | 说明 | 默认值 | 适用模型 |
|------|------|--------|----------|
| `--dataset` | 数据集名称 | 必需 | 所有 |
| `--models` | 模型列表(逗号分隔) | 必需 | 所有 |
| `--num_topics` | 主题数量 | 20 | 所有 |
| `--epochs` | 训练轮数 | 100 | 神经网络模型 |
| `--batch_size` | 批大小 | 64 | 神经网络模型 |
| `--hidden_dim` | 隐藏层维度 | 512 | 神经网络模型 |
| `--learning_rate` | 学习率 | 0.002 | 神经网络模型 |
| `--gpu` | GPU编号 | 0 | 神经网络模型 |
| `--language` | 可视化语言 | en | 所有 |
| `--skip-train` | 跳过训练 | false | 所有 |
| `--skip-viz` | 跳过可视化 | false | 所有 |

### 4.3 常用命令示例

```bash
# 传统模型 (不需要GPU)
bash scripts/05_train_baseline.sh --dataset edu_data --models lda --num_topics 15 --language zh
bash scripts/05_train_baseline.sh --dataset edu_data --models hdp --language zh

# 神经网络模型 (需要GPU)
CUDA_VISIBLE_DEVICES=0 bash scripts/05_train_baseline.sh --dataset edu_data --models nvdm,gsm,prodlda \
    --num_topics 15 --epochs 100 --batch_size 64 --learning_rate 0.002 --language zh

# CTM 模型
CUDA_VISIBLE_DEVICES=0 bash scripts/05_train_baseline.sh --dataset edu_data --models ctm \
    --num_topics 15 --epochs 100 --batch_size 64 --language zh

# DTM 动态主题模型
CUDA_VISIBLE_DEVICES=0 bash scripts/05_train_baseline.sh --dataset edu_data --models dtm \
    --num_topics 15 --epochs 100 --language zh

# 多模型同时训练
bash scripts/05_train_baseline.sh --dataset edu_data --models lda,nvdm,prodlda \
    --num_topics 15 --epochs 100 --language zh
```

---

## 5. 数据流程

### 5.1 完整数据流

```
原始数据 (CSV/TXT/DOCX)
        ↓
[dataclean] 数据清洗
        ↓
清洗后CSV ({dataset}_cleaned.csv)
        ↓
[prepare_data.py] 数据预处理
        ↓
┌───────────────────────────────────────┐
│  BOW矩阵 (bow_matrix.npy)             │
│  词表 (vocab.json)                    │
│  SBERT嵌入 (sbert_embeddings.npy)     │  ← CTM/DTM需要
│  Qwen嵌入 (embeddings.npy)            │  ← THETA需要
│  时间片信息 (time_slices.json)        │  ← DTM需要
└───────────────────────────────────────┘
        ↓
[run_pipeline.py] 模型训练
        ↓
┌───────────────────────────────────────┐
│  theta (文档-主题分布)                 │
│  beta (主题-词分布)                    │
│  topic_words (主题词)                  │
│  training_history (训练历史)           │
└───────────────────────────────────────┘
        ↓
[UnifiedEvaluator] 评估
        ↓
metrics_k{K}.json (7项评估指标)
        ↓
[VisualizationGenerator] 可视化
        ↓
visualization_k{K}_{lang}_{timestamp}/ (70+图表)
```

### 5.2 数据格式要求

**输入CSV文件**:

| 列名 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `text` / `content` / `cleaned_content` | string | ✅ | 文本内容 |
| `label` / `category` | string/int | ❌ | 标签 (supervised模式) |
| `year` / `timestamp` / `date` | int/string | ❌ | 时间戳 (DTM模型) |

---

## 6. 评估指标

### 6.1 七项核心指标

| 指标 | 全称 | 范围 | 说明 |
|------|------|------|------|
| **TD** | Topic Diversity | [0,1] | 主题词的多样性，越高越好 |
| **iRBO** | Inverse Rank-Biased Overlap | [0,1] | 主题间的差异性，越高越好 |
| **NPMI** | Normalized PMI | [-1,1] | 主题一致性，越高越好 |
| **C_V** | C_V Coherence | [0,1] | 主题一致性(另一种度量) |
| **UMass** | UMass Coherence | (-∞,0] | 主题一致性，越接近0越好 |
| **Exclusivity** | Topic Exclusivity | [0,1] | 主题排他性，越高越好 |
| **PPL** | Perplexity | (0,∞) | 困惑度，越低越好 |

### 6.2 指标计算示例

```python
from evaluation.unified_evaluator import UnifiedEvaluator

evaluator = UnifiedEvaluator(
    beta=beta,           # 主题-词分布 (K × V)
    theta=theta,         # 文档-主题分布 (D × K)
    bow_matrix=bow,      # BOW矩阵 (D × V)
    vocab=vocab,         # 词表 (list)
    model_name='lda',
    dataset='edu_data',
    output_dir='./output',
    num_topics=20
)

metrics = evaluator.compute_all_metrics()
evaluator.save_metrics()
```

---

## 7. 可视化系统

### 7.1 可视化类型

**全局图表** (global/):
- 主题表.png - 主题词表格
- 主题网络图.png - 主题关系网络
- 主题占比饼图.png - 主题分布
- 文档聚类图.png - UMAP文档聚类
- topic_wordclouds.png - 主题词云
- topic_similarity.png - 主题相似度热力图
- pyldavis_interactive.html - 交互式主题探索

**时序图表** (DTM专用):
- 代表性主题演化图.png - 主题随时间变化
- 词汇演化图.png - 词汇随时间变化
- 主题桑基图.png - 主题流动
- KL散度图.png - 主题稳定性

**每主题图表** (topics/):
- topic_{i}_wordcloud.png - 单主题词云
- topic_{i}_documents.png - 代表性文档
- topic_{i}_evolution.png - 单主题演化

### 7.2 可视化命令

```bash
# 单独运行可视化
python -m visualization.run_visualization \
    --baseline \
    --result_dir /root/autodl-tmp/result/baseline \
    --dataset edu_data \
    --model lda \
    --num_topics 20 \
    --language zh \
    --dpi 300
```

---

## 8. 当前进展总结

### 8.1 已完成测试的模型

| 模型 | 状态 | 测试数据集 | 训练时间 | 备注 |
|------|------|------------|----------|------|
| ✅ LDA | 完成 | edu_data | ~2s | 训练+评估+可视化 |
| ✅ HDP | 完成 | edu_data | ~84s | 自动推断150个主题 |
| ✅ NVDM | 完成 | edu_data | ~5s | 神经网络VAE |
| ✅ GSM | 完成 | edu_data | ~7s | Softmax约束 |
| ✅ ProdLDA | 完成 | edu_data | ~6s | Product of Experts |
| ✅ CTM | 完成 | edu_data | ~44s | SBERT + VAE, TD=0.81 |
| ✅ DTM | 完成 | edu_data | ~75s | 21个时间片 |
| ⚠️ BTM | 长时间运行 | edu_data | >30min | 27M+ biterms |

### 8.2 已修复的问题

1. **抽象类实例化错误**: 修复 `TraditionalTopicModel` 和 `NeuralTopicModel` 的 `num_topics`/`vocab_size` 属性实现
2. **稀疏矩阵处理**: 修复 `toarray()` 调用，支持稀疏和稠密矩阵
3. **CUDA设备管理**: 修复 DataLoader 的 CUDA 初始化错误，数据保持在CPU直到训练循环
4. **路径问题**: 修复评估时的文件路径查找（神经网络模型保存在 model/ 子目录）
5. **CTM初始化**: 修复父类初始化参数传递
6. **DTM导入**: 修复模块导入路径 `.dtm` → `.baselines.dtm`

### 8.3 代码修改记录

| 文件 | 修改内容 |
|------|----------|
| `model/base.py` | 添加 `TraditionalTopicModel` 和 `NeuralTopicModel` 的 `__init__` 和属性实现 |
| `model/baseline_trainer.py` | 修复 HDP/STM/BTM/DTM 的稀疏矩阵处理；修复神经网络模型的CUDA设备管理 |
| `model/baselines/ntm.py` | 添加 NVDM/GSM/ProdLDA 的 `get_config` 和 `get_topic_words` 方法 |
| `model/baselines/ctm.py` | 修复 CTM 父类初始化参数传递 |
| `run_pipeline.py` | 修复评估时的文件路径查找逻辑 |

### 8.4 待优化项

1. 统一结果输出目录结构
2. 评估和可视化的路径一致性
3. BTM 对长文档的效率优化
4. 添加更多评估指标的可视化

---

## 9. 附录：模型类继承关系

```
BaseTopicModel (ABC)
├── TraditionalTopicModel
│   ├── LDA (SklearnLDA)
│   ├── HDP
│   ├── BTM
│   └── STM
│
└── NeuralTopicModel (+ nn.Module)
    ├── THETA (ETM)
    ├── CTM
    ├── DTM
    ├── OriginalETM
    ├── NVDM
    ├── GSM
    └── ProdLDA
```

---

*文档更新日期: 2026-02-05*
*作者: THETA 开发团队*

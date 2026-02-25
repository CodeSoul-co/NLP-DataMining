# 主题模型对比说明

## THETA ETM vs 原始 ETM vs 其他Baseline

### 架构对比

| 特性 | THETA ETM (你的方法) | 原始 ETM (Baseline) | LDA | CTM |
|------|---------------------|---------------------|-----|-----|
| **编码器输入** | Qwen doc embedding (1024维) | BOW | BOW | BOW + SBERT |
| **词向量** | Qwen word embedding (1024维) | Word2Vec (300维) | 无 | 无 |
| **推断方式** | VAE | VAE | 变分/Gibbs | VAE |
| **语义信息** | ✅ 预训练LLM语义 | ⚠️ Word2Vec语义 | ❌ 无 | ⚠️ SBERT语义 |

### 详细区别

#### 1. THETA ETM (你的方法)

```
输入: Qwen文档embedding (1024维)
      ↓
编码器: MLP → μ, σ → 重参数化 → z
      ↓
主题分布: θ = softmax(z)  [D × K]
      ↓
解码器: θ × (topic_emb @ word_emb.T) → β
      ↓
输出: 词分布重建BOW
```

**特点:**
- 使用Qwen预训练模型的文档embedding作为输入
- 使用Qwen词向量作为解码器的语义基础
- 主题embedding在Qwen语义空间中学习
- 能够捕获深层语义信息

**优势:**
- 利用大语言模型的语义理解能力
- 主题更具语义连贯性
- 适合需要深度语义理解的任务

#### 2. 原始 ETM (Baseline)

```
输入: BOW (词频向量)
      ↓
编码器: MLP → μ, σ → 重参数化 → z
      ↓
主题分布: θ = softmax(z)  [D × K]
      ↓
解码器: θ × (α @ ρ.T) → β
        α: 主题embedding [K × E]
        ρ: Word2Vec词向量 [V × E]
      ↓
输出: 词分布重建BOW
```

**特点:**
- 使用BOW作为编码器输入
- 使用Word2Vec/GloVe词向量 (通常300维)
- 主题embedding在词向量空间中学习

**优势:**
- 经典方法，有理论保证
- 计算效率高
- 不依赖大模型

#### 3. LDA (Baseline)

```
输入: BOW
      ↓
生成过程:
  - 文档 → Dirichlet(α) → θ (主题分布)
  - 主题 → Dirichlet(β) → φ (词分布)
  - 词 ~ Multinomial(φ[z])
      ↓
推断: 变分推断 / Gibbs采样
```

**特点:**
- 纯概率生成模型
- 不使用任何embedding
- 基于词共现统计

**优势:**
- 可解释性强
- 稳定可靠
- 适合作为baseline

#### 4. CTM (Baseline)

```
输入: BOW + SBERT embedding
      ↓
推断网络:
  - ZeroShot: 仅用SBERT
  - Combined: BOW + SBERT融合
      ↓
主题分布: θ = softmax(z)
      ↓
解码器: θ × β → 词分布
```

**特点:**
- 结合BOW和预训练embedding
- 使用SBERT (较小的预训练模型)
- 支持零样本跨语言

**优势:**
- 平衡语义和统计信息
- 支持跨语言主题建模

### 数据流对比

```
THETA ETM:
  原始文本 → Qwen → doc_embedding → THETA ETM → 主题
                  ↘ word_embedding ↗

原始 ETM (Baseline):
  原始文本 → sklearn BOW → 原始ETM → 主题
           → Word2Vec ↗

LDA (Baseline):
  原始文本 → sklearn BOW → LDA → 主题

CTM (Baseline):
  原始文本 → sklearn BOW → CTM → 主题
           → SBERT ↗
```

### 评估指标

所有模型使用相同的评估指标，确保公平对比：

| 指标 | 说明 | 方向 |
|------|------|------|
| **Perplexity** | 困惑度，衡量模型拟合程度 | ↓ 越低越好 |
| **Topic Diversity (TD)** | 主题词多样性 | ↑ 越高越好 |
| **Topic Diversity (iRBO)** | 基于排名的多样性 | ↑ 越高越好 |
| **Coherence (NPMI)** | 归一化点互信息 | ↑ 越高越好 |
| **Coherence (C_V)** | C_V一致性 | ↑ 越高越好 |
| **Coherence (UMass)** | UMass一致性 | ↑ 越高越好 |
| **Exclusivity** | 主题独占性 | ↑ 越高越好 |

### 使用方式

#### 训练Baseline模型

```bash
cd /root/autodl-tmp/ETM
python -m model.baseline_trainer \
    --dataset hatespeech \
    --models lda,etm,ctm \
    --num_topics 20 \
    --vocab_size 5000
```

#### 评估和对比

```python
from model import compare_all_models, print_comparison_table

# 对比所有模型（包括THETA）
results = compare_all_models(
    dataset='hatespeech',
    baseline_dir='/root/autodl-tmp/result/baseline',
    theta_dir='/root/autodl-tmp/result/0.6B',
    mode='zero_shot',
    num_topics=20
)

# 打印对比表格
print_comparison_table(results)
```

#### 可视化

```python
from model import BaselineEvaluator

evaluator = BaselineEvaluator(
    result_dir='/root/autodl-tmp/result/baseline/hatespeech',
    dataset='hatespeech'
)

# 可视化各模型结果
evaluator.visualize_from_files('lda', num_topics=20)
evaluator.visualize_from_files('etm', num_topics=20)
evaluator.visualize_from_files('ctm', num_topics=20)
```

### 文件结构

```
/root/autodl-tmp/result/
├── 0.6B/                          # THETA结果
│   └── {dataset}/
│       └── {mode}/
│           ├── embeddings/        # Qwen embedding
│           └── model/             # THETA模型结果
│               ├── theta_*.npy
│               ├── beta_*.npy
│               └── topic_words_*.json
│
└── baseline/                      # Baseline结果
    └── {dataset}/
        ├── bow_matrix.npz         # sklearn BOW
        ├── vocab.json
        ├── sbert_embeddings.npy   # CTM用
        ├── word2vec_embeddings_300.npy  # ETM用
        ├── lda/
        │   ├── theta_k20.npy
        │   ├── beta_k20.npy
        │   └── topic_words_k20.json
        ├── etm/
        │   └── ...
        └── ctm_zeroshot/
            └── ...
```

### 预期结果

基于文献和经验，预期结果排序：

**语义质量 (Coherence):**
THETA ETM > CTM > 原始ETM > LDA

**多样性 (Diversity):**
取决于数据集和主题数

**可解释性:**
LDA ≈ THETA ETM > 原始ETM > CTM

**计算效率:**
LDA > 原始ETM > CTM > THETA ETM

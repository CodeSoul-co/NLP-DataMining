# THETA主题模型

**基于通义千问嵌入的先进主题建模**

---

THETA是一个先进的主题建模框架，利用**Qwen3-Embedding**模型在主题发现和分析中实现卓越性能。THETA设计为对LDA和ETM等传统主题模型的改进，结合了大语言模型嵌入的强大能力与先进的神经主题建模架构。

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **快速入门**

    ---

    几分钟内安装THETA并训练您的第一个主题模型

    [:octicons-arrow-right-24: 快速开始](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **用户指南**

    ---

    从数据准备到结果分析的完整工作流程

    [:octicons-arrow-right-24: 用户指南](user-guide/data-preparation.md)

-   :material-brain:{ .lg .middle } **模型**

    ---

    THETA和基线模型的架构细节

    [:octicons-arrow-right-24: 模型](models/theta.md)

-   :material-api:{ .lg .middle } **API参考**

    ---

    所有CLI工具的完整参数文档

    [:octicons-arrow-right-24: API参考](api/prepare-data.md)

</div>

---

## 主要特点

| 特点 | 描述 |
|---------|-------------|
| :material-chip: **强大嵌入** | 基于Qwen3-Embedding（0.6B / 4B / 8B）实现卓越语义理解 |
| :material-tune: **灵活训练** | 零样本、有监督和无监督模式 |
| :material-chart-box: **丰富可视化** | 主题分布、热图、UMAP投影、pyLDAvis |
| :material-translate: **多语言** | 完全支持英文和中文数据 |
| :material-cog: **可扩展** | 通过新数据集和配置轻松定制 |
| :material-speedometer: **全面评估** | TD、TC、NPMI等更多指标 |

---

## 模型比较

| 模型 | 嵌入 | 类型 | 特点 |
|-------|-----------|------|----------------|
| **THETA** | Qwen3-Embedding | 神经模型 | 我们的方法 — 最佳性能 |
| LDA | — | 概率模型 | 经典生成式模型 |
| ETM | Word2Vec | 神经模型 | 嵌入主题模型 |
| CTM | SBERT | 神经模型 | 上下文模型 |
| DTM | SBERT | 神经模型 | 动态时序模型 |

---

## 快速示例

```bash
# 1. 预处理数据
python prepare_data.py \
    --dataset 20ng \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --gpu 0

# 2. 训练模型
python run_pipeline.py \
    --dataset 20ng \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --gpu 0
```

---

## 引用

如果您在研究中使用THETA，请引用：

```bibtex
@article{theta2025,
  title={THETA: Advanced Topic Modeling with Qwen Embeddings},
  author={CodeSoul},
  year={2025}
}
```

---

## 链接


- [:fontawesome-brands-github: GitHub Repository](https://github.com/CodeSoul-co/THETA)
- [:material-web: Website](https://theta.code-soul.com)

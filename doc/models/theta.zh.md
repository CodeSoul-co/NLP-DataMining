# THETA模型

**[English](theta.md)** | **[中文](theta.zh.md)**

---

THETA是一个结合变分自编码器和Qwen3-Embedding表示的神经主题模型。

---

## 架构

该模型由三个主要组件组成：

**编码器网络**
- 输入：Qwen嵌入（根据模型规模为1024/2560/4096维）
- 架构：具有可配置隐藏层维度的多层感知机
- 输出：变分后验 q(θ|x) 的参数
  - 均值 μ ∈ R^K（K = 主题数量）
  - 对数方差 log σ^2 ∈ R^K

**重参数化**
- 使用重参数化技巧采样主题分布
- θ = μ + σ ⊙ ε，其中 ε ~ N(0, I)
- 通过随机采样实现基于梯度的训练

**解码器网络**
- 主题-词矩阵 β ∈ R^(K×V)（V = 词汇表大小）
- 重建：p(w|θ) = softmax(θ^T β)
- 损失：负ELBO = -E_q[log p(w|θ)] + KL[q(θ|x) || p(θ)]

---

## 训练目标

模型最大化证据下界（ELBO）：

```
ELBO = E_q(θ|x)[log p(w|θ)] - KL[q(θ|x) || p(θ)]
```

组成部分：
- 重建项：观测词的期望对数似然
- KL散度：向先验 p(θ) = Dir(α) 的正则化

应用KL退火以防止后验坍塌：
```
损失 = -重建项 + β_t * KL
```
其中 β_t 在预热期间从0增加到1。

---

## 模型规格

**0.6B模型**

| 属性 | 值 |
|----------|-------|
| 参数量 | 6亿 |
| 嵌入维度 | 1024 |
| 显存需求 | ~4GB |
| 处理速度 | ~1000篇文档/分钟 |
| 预处理批大小 | 32 |
| 训练批大小 | 64 |

特点：
- 最快的处理速度
- 适合开发和迭代
- 在大多数数据集上质量良好
- 推荐的起点

**4B模型**

| 属性 | 值 |
|----------|-------|
| 参数量 | 40亿 |
| 嵌入维度 | 2560 |
| 显存需求 | ~12GB |
| 处理速度 | ~400篇文档/分钟 |
| 预处理批大小 | 16 |
| 训练批大小 | 32 |

特点：
- 性能和成本平衡
- 语义理解优于0.6B
- 适合生产环境部署
- 推荐用于最终结果

**8B模型**

| 属性 | 值 |
|----------|-------|
| 参数量 | 80亿 |
| 嵌入维度 | 4096 |
| 显存需求 | ~24GB |
| 处理速度 | ~200篇文档/分钟 |
| 预处理批大小 | 8 |
| 训练批大小 | 16 |

特点：
- 最高质量的嵌入
- 所有指标上最佳性能
- 需要高端GPU（A100、H100）
- 推荐用于研究和关键应用

---

## 训练模式

**zero_shot模式**

标准的无监督主题建模：
- 不使用标签信息
- 主题纯粹从文本模式中涌现
- 无标签可用时的默认选择

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20
```

**supervised模式**

标签引导的主题发现：
- 训练过程中融入标签信息
- 主题与提供的类别对齐
- CSV中需要标签列

模型增加分类目标：
```
损失 = -ELBO + λ * 交叉熵(y_pred, y_true)
```

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 20
```

**unsupervised模式**

明确的无监督学习：
- 类似zero_shot，但明确忽略标签（如有）
- 用于带标签数据的比较实验
- 有助于消融研究

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode unsupervised \
    --num_topics 20
```

---

## 超参数指南

**主题数量**

选择取决于语料库特征：
- 小型语料库（< 1千篇文档）：10-20个主题
- 中型语料库（1千-1万篇文档）：20-50个主题
- 大型语料库（> 1万篇文档）：50-100个主题

**隐藏层维度**

控制编码器容量：
- 256：最小容量，训练更快
- 512：默认选择，适用于大多数情况
- 768-1024：更高容量，适用于复杂语料库

**学习率**

影响收敛速度和稳定性：
- 0.001：保守，稳定收敛
- 0.002：默认，平衡性能
- 0.005：激进，更快但不稳定

**KL退火**

标准调度：
- 起始：0.0（无KL惩罚）
- 结束：1.0（完全KL惩罚）
- 预热：50轮（逐渐增加）

---

## 实现细节

### 检查点管理

训练过程中保存模型检查点：
- `best_model.pt`：验证损失最佳模型
- `last_model.pt`：最终轮次模型
- `training_history.json`：损失曲线和指标

加载检查点进行推理：
```python
from src.model import etm
model = etm.THETA(num_topics=20, vocab_size=5000)
model.load_state_dict(torch.load('best_model.pt'))
```

### 内存管理

显存使用量随以下因素线性扩展：
- 批大小（线性）
- 嵌入维度（线性）
- 词汇表大小（线性）
- 隐藏层维度（线性）

通过以下方式减少内存使用：
- 减小批大小
- 使用较小模型（0.6B而非4B）
- 减小词汇表大小
- 减小隐藏层维度

### 可重复性

设置随机种子以实现可重复结果：
```python
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
```

确定性操作：
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

注意：即使在设置种子的情况下，GPU上的某些操作也可能是非确定性的。
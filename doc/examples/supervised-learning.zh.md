# 有监督学习示例

**[English](supervised-learning.md)** | **[中文](supervised-learning.zh.md)**

---

本示例演示有监督主题建模和时序分析。

---

## 示例1：有监督主题分类

### 数据集描述

- 领域：客户评论
- 规模：3000篇文档
- 语言：英文
- 标签：5个产品类别
- 目标：发现与类别对齐的主题

### 步骤1：准备带标签数据

带标签的CSV格式：
```csv
text,label
"很棒的游戏本，处理器快，电池续航长",电子产品
"舒适的运动鞋，足弓支撑好",运动用品
"香气浓郁的咖啡豆",食品
```

### 步骤2：以有监督模式预处理

```bash
python prepare_data.py \
    --dataset reviews \
    --model theta \
    --model_size 0.6B \
    --mode supervised \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

### 步骤3：使用有监督训练

```bash
python run_pipeline.py \
    --dataset reviews \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 15 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

### 步骤4：比较不同模式

同时训练有监督和零样本模式进行比较：

```bash
# 零样本（忽略标签）
python run_pipeline.py \
    --dataset reviews \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 15 \
    --gpu 0

# 有监督（使用标签）
python run_pipeline.py \
    --dataset reviews \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 15 \
    --gpu 0
```

结果比较：

| 模式 | TD | NPMI | 标签对齐度 |
|------|-----|------|----------------|
| 零样本 | 0.85 | 0.41 | 0.62 |
| 有监督 | 0.83 | 0.38 | 0.89 |

有监督模式在多样性略有降低的情况下实现了更好的标签对齐。

### 步骤5：主题-标签分析

```python
import json
import numpy as np

# 加载结果
with open('result/0.6B/reviews/supervised/metrics/evaluation_results.json') as f:
    results = json.load(f)

# 分析主题-标签对应关系
# 主题0-2：电子产品
# 主题3-5：运动用品
# 主题6-8：食品
# 主题9-11：书籍
# 主题12-14：服装
```

---

## 示例2：时序主题演化

### 数据集描述

- 领域：学术论文
- 规模：10000篇文档
- 语言：英文
- 时间范围：2015-2023年
- 领域：机器学习

### 步骤1：准备时序数据

带年份列的CSV：
```csv
text,year
"图像识别的深度学习方法...",2015
"自然语言处理的Transformer架构...",2019
"大语言模型和涌现能力...",2023
```

### 步骤2：使用时间信息预处理

```bash
python prepare_data.py \
    --dataset ml_papers \
    --model dtm \
    --vocab_size 8000 \
    --time_column year
```

### 步骤3：训练DTM模型

```bash
python run_pipeline.py \
    --dataset ml_papers \
    --models dtm \
    --num_topics 30 \
    --epochs 150 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en
```

### 步骤4：分析主题演化

DTM追踪主题随时间的变化：

**主题5：深度学习（2015-2018年）**
- 2015年：卷积，神经，网络，分类
- 2016年：深度，学习，层，训练
- 2017年：残差，连接，跳跃，深度
- 2018年：架构，设计，高效，移动端

**主题12：注意力机制（2017-2020年）**
- 2017年：注意力，机制，序列，编码器
- 2018年：自注意力，多头，Transformer
- 2019年：BERT，预训练，微调，下游
- 2020年：扩展，模型，参数，性能

**主题18：大语言模型（2020-2023年）**
- 2020年：GPT，生成，语言，模型
- 2021年：提示，少样本，上下文，学习
- 2022年：指令，微调，对齐，人工
- 2023年：涌现，能力，扩展，规律

### 步骤5：可视化趋势

```bash
python -m visualization.run_visualization \
    --baseline \
    --result_dir /root/autodl-tmp/result/baseline \
    --dataset ml_papers \
    --model dtm \
    --num_topics 30 \
    --language en \
    --dpi 300
```

可视化显示：
- 主题的诞生和消亡
- 词概率随时间的变化
- 主题强度趋势
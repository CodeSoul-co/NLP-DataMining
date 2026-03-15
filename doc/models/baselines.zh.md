# 基线模型

**[English](baselines.md)** | **[中文](baselines.zh.md)**

---

THETA包含几个用于比较的基线模型。

---

## LDA（潜在狄利克雷分配）

使用狄利克雷-多项分布的经典概率主题模型。

**架构：**
- 无神经组件
- 主题和词分布上的狄利克雷先验
- 通过吉布斯采样或变分推断进行推理

**优势：**
- 成熟的理论基础
- 可解释的概率框架
- 在CPU上高效运行
- 给定随机种子时结果确定

**局限性：**
- 无语义词关系
- 词袋假设
- 在大词汇量上性能停滞

**训练：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda \
    --num_topics 20 \
    --epochs 100
```

---

## ETM（嵌入主题模型）

使用Word2Vec嵌入的神经主题模型。

**架构：**
- 类似THETA的VAE框架
- Word2Vec嵌入（300维）
- 主题嵌入与词嵌入在同一空间

**优势：**
- 捕捉语义关系
- 比LDA更连贯的主题
- 使用GPU高效训练
- 原始ETM实现

**局限性：**
- Word2Vec限于静态嵌入
- 300维嵌入不如Qwen表达能力强
- 需要预训练的Word2Vec模型

**训练：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models etm \
    --num_topics 20 \
    --epochs 100 \
    --hidden_dim 512 \
    --learning_rate 0.002
```

---

## CTM（上下文主题模型）

使用SBERT上下文嵌入的神经主题模型。

**架构：**
- 带有SBERT编码器的VAE框架
- SBERT嵌入（768维）
- 上下文化的文档表示

**优势：**
- 上下文语义表示
- 大多数指标优于ETM
- 比大型THETA模型更快
- 在近期研究中广泛使用

**局限性：**
- SBERT嵌入固定为768维
- 不如Qwen 4B/8B模型强大
- 需要下载SBERT模型

**训练：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models ctm \
    --num_topics 20 \
    --epochs 100 \
    --hidden_dim 512 \
    --learning_rate 0.002
```

---

## DTM（动态主题模型）

CTM的时间扩展版本，用于追踪主题随时间的演化。

**架构：**
- 基于CTM架构
- 额外的时态动态层
- 建模时间切片间的主题转换

**优势：**
- 捕捉时间动态
- 揭示新兴和衰退的主题
- 对趋势分析有用
- 处理可变时间切片大小

**局限性：**
- 数据中需要时间列
- 需要估计更多参数
- 训练时间更长
- 每个时间切片需要足够文档

**训练：**
```bash
python run_pipeline.py \
    --dataset temporal_dataset \
    --models dtm \
    --num_topics 20 \
    --epochs 100 \
    --hidden_dim 512 \
    --learning_rate 0.002
```

数据要求：
- CSV必须包含时间列（year、timestamp或date）
- 预处理必须通过 `--time_column` 指定时间列
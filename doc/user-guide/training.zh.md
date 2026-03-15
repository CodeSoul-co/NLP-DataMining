# 训练模型

**[English](training.md)** | **[中文](training.zh.md)**

---

本指南涵盖使用各种配置训练THETA和基线模型。

---

## THETA模型训练

### 基本训练

使用默认超参数训练THETA模型：

```bash
cd /root/autodl-tmp/ETM

python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 50 \
    --patience 10 \
    --gpu 0 \
    --language en
```

训练通常在20-40分钟内完成，具体取决于数据集规模和硬件。

### 主题数量选择

主题数量是影响粒度的关键超参数：

| 主题数 | 适用场景 |
|--------|----------------|
| 10-15 | 小型语料库，宽泛类别，高层级概览 |
| 20-30 | 中型语料库，平衡粒度，默认选择 |
| 40-100 | 大型多样化语料库，细粒度分析 |

### 学习率调优

| 学习率 | 使用场景 |
|--------------|----------|
| 0.001 | 训练不稳定，损失值震荡 |
| 0.002 | 大多数数据集的默认选择 |
| 0.005 | 训练过慢，需要更快收敛 |

### KL退火配置

KL退火在训练过程中逐渐增加KL散度权重，以防止后验坍塌。

**标准KL退火：**
权重在50轮内从0.0线性增加到1.0。

**慢速KL退火：**
`--kl_warmup 80` — 更长的预热期有助于防止早期后验坍塌。

**部分KL退火：**
`--kl_start 0.1 --kl_end 0.9 --kl_warmup 30` — 从非零权重开始，在达到完全权重前停止。

### 隐藏层维度配置

| 隐藏层维度 | 使用场景 |
|-----------|----------|
| 256 | 小型数据集，训练更快，显存有限 |
| 512 | 大多数数据集的默认选择 |
| 768-1024 | 大型复杂数据集，显存充足 |

### 早停

早停通过监控验证性能来防止过拟合：

- **默认**：`--patience 10` — 10轮无改善后停止
- **禁用**：`--no_early_stopping` — 训练所有指定轮数

### 中文数据训练

```bash
python run_pipeline.py \
    --dataset chinese_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language zh
```

语言参数影响可视化渲染（字体、布局），但不改变训练算法。

### 有监督训练

对于带标签的数据集：

```bash
python run_pipeline.py \
    --dataset labeled_dataset \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

模型融入标签信息以引导主题发现。

---

## 基线模型训练

### LDA训练

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

LDA使用吉布斯采样，不利用GPU加速。

### ETM训练

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models etm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en
```

ETM使用Word2Vec嵌入（300维）。

### CTM训练

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models ctm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en
```

CTM使用SBERT嵌入（768维）。

### DTM训练

```bash
python run_pipeline.py \
    --dataset temporal_dataset \
    --models dtm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en
```

DTM对由预处理中时间列定义的时间切片上的主题演化进行建模。

### 训练多个模型

同时比较多个模型：

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

模型顺序训练。结果保存在单独的目录中以供比较。
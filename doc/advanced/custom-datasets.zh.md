# 使用自定义数据集

**[English](custom-datasets.md)** | **[中文](custom-datasets.zh.md)**

---

本指南涵盖了处理新数据集的完整工作流程。

---

## 英文数据完整流程

以名为 `news_articles` 的新数据集为例：

**步骤 1：创建数据集目录**

```bash
mkdir -p /root/autodl-tmp/data/news_articles
```

**步骤 2：放置清洗后的 CSV 文件**

```bash
cp /path/to/cleaned_news.csv /root/autodl-tmp/data/news_articles/news_articles_cleaned.csv
```

文件命名规则：`{数据集名称}_cleaned.csv`

**步骤 3：预处理数据**

```bash
cd /root/autodl-tmp/ETM

python prepare_data.py \
    --dataset news_articles \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --gpu 0
```

**步骤 4：验证预处理文件**

```bash
python prepare_data.py \
    --dataset news_articles \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --check-only
```

**步骤 5：训练模型**

```bash
python run_pipeline.py \
    --dataset news_articles \
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

**步骤 6：查看结果**

结果位置：
```
/root/autodl-tmp/result/0.6B/news_articles/zero_shot/
├── metrics/evaluation_results.json
└── visualizations/
```

---

## 中文数据完整流程

以名为 `weibo_posts` 的数据集为例：

**步骤 1-2：设置**

```bash
mkdir -p /root/autodl-tmp/data/weibo_posts
cp /path/to/cleaned_weibo.csv /root/autodl-tmp/data/weibo_posts/weibo_posts_cleaned.csv
```

**步骤 3：预处理中文数据**

```bash
cd /root/autodl-tmp/ETM

python prepare_data.py \
    --dataset weibo_posts \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --gpu 0
```

通义千问模型原生支持中文，无需特殊配置。

**步骤 4：使用中文语言设置训练**

```bash
python run_pipeline.py \
    --dataset weibo_posts \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language zh
```

`--language zh` 参数确保可视化中的字体正确渲染。

---

## 从原始数据开始

在单个流程中处理未清洗数据：

**英文原始数据：**

```bash
cd /root/autodl-tmp/ETM

python prepare_data.py \
    --dataset news_articles \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --clean \
    --raw-input /root/autodl-tmp/data/news_articles/raw_data.csv \
    --language english \
    --gpu 0
```

**中文原始数据：**

```bash
python prepare_data.py \
    --dataset weibo_posts \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --clean \
    --raw-input /root/autodl-tmp/data/weibo_posts/raw_data.csv \
    --language chinese \
    --gpu 0
```

---

## 有监督学习场景

对于在 `label` 或 `category` 列中包含标签的数据集：

**步骤 1：验证数据格式**

CSV 必须包含：
```csv
text,label
"关于气候政策的文章",环境
"关于人工智能进展的报告",技术
```

**步骤 2：以有监督模式预处理**

```bash
python prepare_data.py \
    --dataset labeled_news \
    --model theta \
    --model_size 0.6B \
    --mode supervised \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

**步骤 3：使用有监督模式训练**

```bash
python run_pipeline.py \
    --dataset labeled_news \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

---

## 时序数据处理

对于 DTM 分析，数据必须包含时间信息：

**步骤 1：验证时间列**

CSV 格式：
```csv
text,year
"2020年的文章",2020
"2021年的文章",2021
```

接受的列名：`year`、`timestamp`、`date`

**步骤 2：使用时间列预处理**

```bash
python prepare_data.py \
    --dataset temporal_news \
    --model dtm \
    --vocab_size 5000 \
    --time_column year
```

**步骤 3：训练 DTM 模型**

```bash
python run_pipeline.py \
    --dataset temporal_news \
    --models dtm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

---

## 流程控制

### 跳过阶段

**跳过训练（评估现有模型）：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --skip-train \
    --gpu 0
```

**跳过可视化：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --skip-viz \
    --gpu 0
```

**仅训练（无评估或可视化）：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --skip-eval \
    --skip-viz \
    --gpu 0
```

### 仅生成词袋

仅生成词袋，不生成嵌入：

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --bow-only
```
# 中文数据集示例

**[English](chinese-dataset.md)** | **[中文](chinese-dataset.zh.md)**

---

本示例演示如何使用THETA处理中文文本。

---

## 数据集描述

- 领域：微博帖子
- 规模：8000篇文档
- 语言：中文
- 来源：微博公开API
- 主题：各类社会讨论

---

## 步骤1：数据清洗

清洗原始中文文本：

```bash
cd /root/autodl-tmp/ETM

python -m dataclean.main \
    --input /root/autodl-tmp/data/weibo/raw_data.csv \
    --output /root/autodl-tmp/data/weibo/weibo_cleaned.csv \
    --language chinese
```

清洗移除：
- URL和提及
- 特殊符号
- 过多标点
- 非中文字符

---

## 步骤2：预处理

```bash
python prepare_data.py \
    --dataset weibo \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --gpu 0
```

通义千问模型原生支持中文分词。

---

## 步骤3：训练模型

```bash
python run_pipeline.py \
    --dataset weibo \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language zh
```

注意：`--language zh` 确保正确渲染中文字体。

---

## 步骤4：结果

发现的主题包括：
- 生活，分享，日常，今天，开心（日常生活）
- 工作，公司，同事，加班，项目（工作）
- 美食，餐厅，好吃，推荐，味道（美食）
- 旅游，景点，风景，拍照，美丽（旅游）

可视化使用适当字体正确显示中文字符。

---

## 步骤5：时序分析

如果微博数据包含时间戳，使用DTM：

```bash
python prepare_data.py \
    --dataset weibo \
    --model dtm \
    --vocab_size 5000 \
    --time_column year

python run_pipeline.py \
    --dataset weibo \
    --models dtm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language zh
```

DTM揭示主题随时间演化：
- 远程工作讨论的兴起（2020-2021）
- 环保意识增强（2021-2023）
- 技术采纳趋势（2020-2023）
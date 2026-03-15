# Chinese Dataset Example

This example demonstrates Chinese text processing with THETA.

---

## Dataset Description

- Domain: Weibo posts
- Size: 8000 documents
- Language: Chinese
- Source: Weibo public API
- Topics: Various social discussions

---

## Step 1: Data Cleaning

Clean raw Chinese text:

```bash
cd /root/autodl-tmp/ETM

python -m dataclean.main \
    --input /root/autodl-tmp/data/weibo/raw_data.csv \
    --output /root/autodl-tmp/data/weibo/weibo_cleaned.csv \
    --language chinese
```

Cleaning removes:
- URLs and mentions
- Special symbols
- Excessive punctuation
- Non-Chinese characters

---

## Step 2: Preprocess

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

Qwen models handle Chinese tokenization natively.

---

## Step 3: Train Model

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

Note: `--language zh` ensures proper Chinese font rendering.

---

## Step 4: Results

Discovered topics include:
- 生活, 分享, 日常, 今天, 开心 (daily life)
- 工作, 公司, 同事, 加班, 项目 (work)
- 美食, 餐厅, 好吃, 推荐, 味道 (food)
- 旅游, 景点, 风景, 拍照, 美丽 (travel)

Visualizations render Chinese characters correctly with appropriate fonts.

---

## Step 5: Temporal Analysis

If Weibo data includes timestamps, use DTM:

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

DTM reveals topic evolution over time:
- Rise of remote work discussions (2020-2021)
- Increasing environmental awareness (2021-2023)
- Technology adoption trends (2020-2023)

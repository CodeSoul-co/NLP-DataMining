# Working with Custom Datasets

This guide covers complete workflows for processing new datasets.

---

## Complete Workflow for English Data

Example using a new dataset named `news_articles`:

**Step 1: Create dataset directory**

```bash
mkdir -p /root/autodl-tmp/data/news_articles
```

**Step 2: Place cleaned CSV file**

```bash
cp /path/to/cleaned_news.csv /root/autodl-tmp/data/news_articles/news_articles_cleaned.csv
```

File naming convention: `{dataset_name}_cleaned.csv`

**Step 3: Preprocess data**

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

**Step 4: Verify preprocessed files**

```bash
python prepare_data.py \
    --dataset news_articles \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --check-only
```

**Step 5: Train model**

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

**Step 6: Review results**

Results location:
```
/root/autodl-tmp/result/0.6B/news_articles/zero_shot/
├── metrics/evaluation_results.json
└── visualizations/
```

---

## Complete Workflow for Chinese Data

Example using a dataset named `weibo_posts`:

**Step 1-2: Setup**

```bash
mkdir -p /root/autodl-tmp/data/weibo_posts
cp /path/to/cleaned_weibo.csv /root/autodl-tmp/data/weibo_posts/weibo_posts_cleaned.csv
```

**Step 3: Preprocess Chinese data**

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

Qwen models handle Chinese natively without special configuration.

**Step 4: Train with Chinese language setting**

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

The `--language zh` parameter ensures proper font rendering in visualizations.

---

## Starting from Raw Data

Process uncleaned data in a single pipeline:

**English raw data:**

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

**Chinese raw data:**

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

## Supervised Learning Scenario

For datasets with labels in a `label` or `category` column:

**Step 1: Verify data format**

CSV must contain:
```csv
text,label
"Article about climate policy",Environment
"Report on AI advances",Technology
```

**Step 2: Preprocess in supervised mode**

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

**Step 3: Train with supervision**

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

## Temporal Data Processing

For DTM analysis, data must include temporal information:

**Step 1: Verify temporal column**

CSV format:
```csv
text,year
"Article from 2020",2020
"Article from 2021",2021
```

Accepted column names: `year`, `timestamp`, `date`

**Step 2: Preprocess with time column**

```bash
python prepare_data.py \
    --dataset temporal_news \
    --model dtm \
    --vocab_size 5000 \
    --time_column year
```

**Step 3: Train DTM model**

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

## Pipeline Control

### Skipping Stages

**Skip training (evaluate existing model):**
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

**Skip visualization:**
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

**Training only (no evaluation or visualization):**
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

### BOW-Only Generation

Generate only bag-of-words without embeddings:

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --bow-only
```

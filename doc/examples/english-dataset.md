# English Dataset Examples

Complete tutorials demonstrating THETA usage with English data.

---

## Example 1: English News Dataset

This example demonstrates the complete workflow for analyzing news articles.

### Dataset Description

- Domain: News articles
- Size: 5000 documents
- Language: English
- Source: Online news aggregator
- Time period: 2020-2023

### Step 1: Prepare Data

Create dataset directory and place CSV file:

```bash
mkdir -p /root/autodl-tmp/data/news_corpus
```

CSV format:
```csv
text
"Federal Reserve raises interest rates to combat inflation..."
"Climate summit reaches historic agreement on emissions..."
"Technology companies announce layoffs amid economic uncertainty..."
```

Save as `/root/autodl-tmp/data/news_corpus/news_corpus_cleaned.csv`

### Step 2: Preprocess

```bash
cd /root/autodl-tmp/ETM

python prepare_data.py \
    --dataset news_corpus \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --gpu 0
```

Processing time: ~5 minutes on V100 GPU

### Step 3: Train Model

Train with 25 topics to capture diverse news categories:

```bash
python run_pipeline.py \
    --dataset news_corpus \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 25 \
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

Training time: ~25 minutes

### Step 4: Analyze Results

```bash
cat /root/autodl-tmp/result/0.6B/news_corpus/zero_shot/metrics/evaluation_results.json
```

Example output:
```json
{
  "TD": 0.87,
  "iRBO": 0.73,
  "NPMI": 0.39,
  "C_V": 0.62,
  "UMass": -2.56,
  "Exclusivity": 0.81,
  "PPL": 152.34
}
```

### Step 5: Compare with Baselines

```bash
python prepare_data.py \
    --dataset news_corpus \
    --model baseline \
    --vocab_size 5000

python run_pipeline.py \
    --dataset news_corpus \
    --models lda,etm,ctm \
    --num_topics 25 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

Comparison results:

| Model | TD | NPMI | C_V | PPL |
|-------|-----|------|-----|-----|
| LDA | 0.72 | 0.24 | 0.48 | 185.2 |
| ETM | 0.79 | 0.31 | 0.55 | 168.5 |
| CTM | 0.83 | 0.36 | 0.59 | 158.7 |
| THETA | 0.87 | 0.39 | 0.62 | 152.3 |

---

## Example 2: Large-Scale Dataset with 4B Model

This example demonstrates scaling to larger models and datasets.

### Dataset Description

- Domain: Wikipedia articles
- Size: 50000 documents
- Language: English
- Complexity: Diverse topics and vocabulary

### Step 1: Preprocess with 4B Model

```bash
python prepare_data.py \
    --dataset wikipedia \
    --model theta \
    --model_size 4B \
    --mode zero_shot \
    --vocab_size 10000 \
    --batch_size 16 \
    --max_length 512 \
    --gpu 0
```

Processing time: ~4 hours on A100 GPU

### Step 2: Train with Increased Capacity

```bash
python run_pipeline.py \
    --dataset wikipedia \
    --models theta \
    --model_size 4B \
    --mode zero_shot \
    --num_topics 50 \
    --epochs 150 \
    --batch_size 32 \
    --hidden_dim 768 \
    --learning_rate 0.001 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 80 \
    --patience 15 \
    --gpu 0 \
    --language en
```

### Step 3: Compare Model Sizes

| Model | TD | NPMI | C_V | PPL | Time |
|-------|-----|------|-----|-----|------|
| 0.6B | 0.89 | 0.43 | 0.66 | 138.5 | 90 min |
| 4B | 0.92 | 0.49 | 0.71 | 128.2 | 180 min |

4B model provides significant quality improvements at 2x cost.

---

## Example 3: Multi-Model Comparison

### Dataset Description

- Domain: Movie reviews
- Size: 4000 documents
- Language: English

### Step 1: Preprocess for All Models

```bash
python prepare_data.py \
    --dataset movie_reviews \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --gpu 0

python prepare_data.py \
    --dataset movie_reviews \
    --model baseline \
    --vocab_size 5000
```

### Step 2: Train All Models

```bash
python run_pipeline.py \
    --dataset movie_reviews \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en

python run_pipeline.py \
    --dataset movie_reviews \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

### Step 3: Comparison Table

| Model | TD | iRBO | NPMI | C_V | Exclusivity | PPL | Time |
|-------|-----|------|------|-----|------------|-----|------|
| LDA | 0.74 | 0.68 | 0.26 | 0.49 | 0.76 | 178.3 | 12 min |
| ETM | 0.81 | 0.71 | 0.33 | 0.56 | 0.79 | 163.5 | 18 min |
| CTM | 0.84 | 0.74 | 0.37 | 0.60 | 0.82 | 154.2 | 22 min |
| THETA | 0.88 | 0.77 | 0.41 | 0.64 | 0.85 | 147.8 | 26 min |

---

## Example 4: Hyperparameter Grid Search

### Setup

Dataset: 2000 news articles
Goal: Find optimal hyperparameters for topic quality

### Grid Search Script

```bash
#!/bin/bash

topics=(15 20 25 30)
learning_rates=(0.001 0.002 0.005)
hidden_dims=(256 512 768)

for K in "${topics[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for hd in "${hidden_dims[@]}"; do
            echo "Training K=$K, lr=$lr, hd=$hd"
            
            python run_pipeline.py \
                --dataset news \
                --models theta \
                --model_size 0.6B \
                --mode zero_shot \
                --num_topics $K \
                --learning_rate $lr \
                --hidden_dim $hd \
                --epochs 100 \
                --batch_size 64 \
                --gpu 0 \
                --language en
            
            mkdir -p results_grid/K${K}_lr${lr}_hd${hd}
            cp -r result/0.6B/news/zero_shot/* results_grid/K${K}_lr${lr}_hd${hd}/
        done
    done
done
```

### Best Configuration

Analysis reveals optimal settings:
- Number of topics: 20
- Learning rate: 0.002
- Hidden dimension: 512

---

## Best Practices Summary

### Data Preparation
1. Clean data thoroughly before preprocessing
2. Ensure CSV follows naming convention
3. Verify data quality with exploratory analysis

### Model Selection
1. Start with THETA 0.6B for prototyping
2. Compare with CTM baseline
3. Scale to 4B for production if needed

### Hyperparameter Tuning
1. Begin with default parameters
2. Adjust number of topics based on corpus
3. Tune learning rate if training is unstable

### Evaluation
1. Review multiple metrics, not just one
2. Examine visualizations for qualitative assessment
3. Compare with baseline models

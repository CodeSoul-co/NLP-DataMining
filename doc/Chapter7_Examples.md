# Examples

Complete tutorials demonstrating THETA usage in various scenarios.

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

Generate embeddings and BOW:

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

View evaluation metrics:

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

### Step 5: Interpret Topics

Examine discovered topics in visualizations:

```
/root/autodl-tmp/result/0.6B/news_corpus/zero_shot/visualizations/
├── topic_words_bars.png      # Top words per topic
├── topic_similarity.png       # Topic relationships
├── doc_topic_umap.png         # Document clustering
└── pyldavis.html             # Interactive exploration
```

Example discovered topics:
- Topic 0: economy, inflation, federal, reserve, rates
- Topic 5: climate, emissions, energy, renewable, carbon
- Topic 12: technology, layoffs, companies, workers, jobs
- Topic 18: election, voters, campaign, polls, candidate

### Step 6: Compare with Baselines

Train baseline models for comparison:

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

THETA shows consistent improvements across all metrics.

---

## Example 2: Chinese Social Media

This example demonstrates Chinese text processing.

### Dataset Description

- Domain: Weibo posts
- Size: 8000 documents
- Language: Chinese
- Source: Weibo public API
- Topics: Various social discussions

### Step 1: Data Cleaning

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

### Step 2: Preprocess

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

### Step 3: Train Model

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

### Step 4: Results

Discovered topics include:
- 生活, 分享, 日常, 今天, 开心 (daily life)
- 工作, 公司, 同事, 加班, 项目 (work)
- 美食, 餐厅, 好吃, 推荐, 味道 (food)
- 旅游, 景点, 风景, 拍照, 美丽 (travel)

Visualizations render Chinese characters correctly with appropriate fonts.

### Step 5: Temporal Analysis

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

---

## Example 3: Supervised Topic Classification

This example demonstrates supervised learning with labeled data.

### Dataset Description

- Domain: Customer reviews
- Size: 3000 documents
- Language: English
- Labels: 5 product categories
- Goal: Discover category-aligned topics

### Step 1: Prepare Labeled Data

CSV format with labels:
```csv
text,label
"Great laptop with fast processor and long battery life",Electronics
"Comfortable running shoes with good arch support",Sports
"Delicious coffee beans with rich aroma",Food
```

### Step 2: Preprocess in Supervised Mode

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

### Step 3: Train with Supervision

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

Supervised mode incorporates label information during training.

### Step 4: Compare Modes

Train both supervised and zero-shot for comparison:

```bash
# Zero-shot (ignores labels)
python run_pipeline.py \
    --dataset reviews \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 15 \
    --gpu 0

# Supervised (uses labels)
python run_pipeline.py \
    --dataset reviews \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 15 \
    --gpu 0
```

Results comparison:

| Mode | TD | NPMI | Label Alignment |
|------|-----|------|----------------|
| Zero-shot | 0.85 | 0.41 | 0.62 |
| Supervised | 0.83 | 0.38 | 0.89 |

Supervised mode achieves better label alignment with slight reduction in diversity.

### Step 5: Topic-Label Analysis

Examine how topics align with labels:

```python
import json
import numpy as np

# Load results
with open('result/0.6B/reviews/supervised/metrics/evaluation_results.json') as f:
    results = json.load(f)

# Analyze topic-label correspondence
# Topics 0-2: Electronics
# Topics 3-5: Sports
# Topics 6-8: Food
# Topics 9-11: Books
# Topics 12-14: Clothing
```

Supervised mode discovers topics that correspond to product categories.

---

## Example 4: Temporal Topic Evolution

This example analyzes topic dynamics over time using DTM.

### Dataset Description

- Domain: Academic papers
- Size: 10000 documents
- Language: English
- Time range: 2015-2023
- Field: Machine learning

### Step 1: Prepare Temporal Data

CSV with year column:
```csv
text,year
"Deep learning approaches for image recognition...",2015
"Transformer architectures for natural language...",2019
"Large language models and emergent capabilities...",2023
```

### Step 2: Preprocess with Time Information

```bash
python prepare_data.py \
    --dataset ml_papers \
    --model dtm \
    --vocab_size 8000 \
    --time_column year
```

Documents are grouped by year automatically.

### Step 3: Train DTM Model

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

Training time: ~45 minutes

### Step 4: Analyze Topic Evolution

DTM tracks topic changes over time:

**Topic 5: Deep Learning (2015-2018)**
- 2015: convolutional, neural, network, classification
- 2016: deep, learning, layers, training
- 2017: residual, connections, skip, depth
- 2018: architectures, design, efficient, mobile

**Topic 12: Attention Mechanisms (2017-2020)**
- 2017: attention, mechanism, sequence, encoder
- 2018: self-attention, multi-head, transformer
- 2019: bert, pre-training, fine-tuning, downstream
- 2020: scaling, models, parameters, performance

**Topic 18: Large Language Models (2020-2023)**
- 2020: gpt, generation, language, model
- 2021: prompting, few-shot, in-context, learning
- 2022: instruction, tuning, alignment, human
- 2023: emergent, capabilities, scaling, laws

### Step 5: Visualize Trends

Generate temporal visualizations:

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

Visualizations show:
- Topic birth and death
- Word probability changes over time
- Topic intensity trends

---

## Example 5: Large-Scale Dataset with 4B Model

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
Memory usage: ~14GB VRAM

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

Training time: ~3 hours

### Step 3: Compare with 0.6B Model

Train 0.6B model for comparison:

```bash
python prepare_data.py \
    --dataset wikipedia \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 10000 \
    --batch_size 32 \
    --gpu 0

python run_pipeline.py \
    --dataset wikipedia \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 50 \
    --epochs 150 \
    --batch_size 64 \
    --gpu 0
```

Performance comparison:

| Model | TD | NPMI | C_V | PPL | Time |
|-------|-----|------|-----|-----|------|
| 0.6B | 0.89 | 0.43 | 0.66 | 138.5 | 90 min |
| 4B | 0.92 | 0.49 | 0.71 | 128.2 | 180 min |

4B model provides significant quality improvements at 2x cost.

### Step 4: Topic Quality Analysis

4B model discovers more coherent topics:

**0.6B Topics (Example):**
- history, war, world, century, historical
- science, research, study, scientists, theory

**4B Topics (Example):**
- wwii, allies, axis, normandy, victory
- quantum, mechanics, particles, wave, uncertainty

4B model captures finer semantic distinctions.

### Step 5: Scaling to 8B

For maximum quality on critical applications:

```bash
python prepare_data.py \
    --dataset wikipedia \
    --model theta \
    --model_size 8B \
    --mode zero_shot \
    --vocab_size 10000 \
    --batch_size 8 \
    --gpu 0

python run_pipeline.py \
    --dataset wikipedia \
    --models theta \
    --model_size 8B \
    --mode zero_shot \
    --num_topics 50 \
    --epochs 150 \
    --batch_size 16 \
    --gpu 0
```

Requires 24-32GB VRAM (A100 or H100).

---

## Example 6: Multi-Model Comparison

This example demonstrates comprehensive model comparison.

### Dataset Description

- Domain: Movie reviews
- Size: 4000 documents
- Language: English
- Task: Compare all available models

### Step 1: Preprocess for All Models

THETA preprocessing:
```bash
python prepare_data.py \
    --dataset movie_reviews \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --gpu 0
```

Baseline preprocessing:
```bash
python prepare_data.py \
    --dataset movie_reviews \
    --model baseline \
    --vocab_size 5000
```

### Step 2: Train All Models

Single command for baseline models:
```bash
python run_pipeline.py \
    --dataset movie_reviews \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

THETA model:
```bash
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

### Step 3: Collect Results

Extract metrics from all models:

```bash
# LDA
cat result/baseline/movie_reviews/lda/K20/metrics/evaluation_results.json

# ETM
cat result/baseline/movie_reviews/etm/K20/metrics/evaluation_results.json

# CTM
cat result/baseline/movie_reviews/ctm/K20/metrics/evaluation_results.json

# THETA
cat result/0.6B/movie_reviews/zero_shot/metrics/evaluation_results.json
```

### Step 4: Comparison Table

Aggregate results:

| Model | TD | iRBO | NPMI | C_V | Exclusivity | PPL | Time |
|-------|-----|------|------|-----|------------|-----|------|
| LDA | 0.74 | 0.68 | 0.26 | 0.49 | 0.76 | 178.3 | 12 min |
| ETM | 0.81 | 0.71 | 0.33 | 0.56 | 0.79 | 163.5 | 18 min |
| CTM | 0.84 | 0.74 | 0.37 | 0.60 | 0.82 | 154.2 | 22 min |
| THETA | 0.88 | 0.77 | 0.41 | 0.64 | 0.85 | 147.8 | 26 min |

### Step 5: Qualitative Comparison

Examine topic quality for each model:

**Topic: Action Movies**

LDA: action, film, movie, scene, character
ETM: action, stunts, sequences, explosions, chase
CTM: action, blockbuster, CGI, special-effects, thriller
THETA: action, adrenaline, choreography, set-pieces, visceral

THETA captures more nuanced semantic distinctions.

### Step 6: Visualization Comparison

Generate visualizations for all models:

```bash
# LDA
python -m visualization.run_visualization \
    --baseline --result_dir result/baseline \
    --dataset movie_reviews --model lda --num_topics 20 \
    --language en --dpi 300

# ETM
python -m visualization.run_visualization \
    --baseline --result_dir result/baseline \
    --dataset movie_reviews --model etm --num_topics 20 \
    --language en --dpi 300

# CTM
python -m visualization.run_visualization \
    --baseline --result_dir result/baseline \
    --dataset movie_reviews --model ctm --num_topics 20 \
    --language en --dpi 300

# THETA
python -m visualization.run_visualization \
    --result_dir result/0.6B \
    --dataset movie_reviews --mode zero_shot --model_size 0.6B \
    --language en --dpi 300
```

Compare visualizations side-by-side to assess topic coherence.

---

## Example 7: Hyperparameter Grid Search

This example demonstrates systematic hyperparameter optimization.

### Setup

Dataset: 2000 news articles
Goal: Find optimal hyperparameters for topic quality

### Grid Search Parameters

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
            
            # Move results to labeled directory
            mkdir -p results_grid/K${K}_lr${lr}_hd${hd}
            cp -r result/0.6B/news/zero_shot/* results_grid/K${K}_lr${lr}_hd${hd}/
        done
    done
done
```

### Results Analysis

Collect metrics from all configurations:

```bash
#!/bin/bash

echo "Config,TD,NPMI,C_V,PPL" > grid_results.csv

for dir in results_grid/*/; do
    config=$(basename $dir)
    metrics=$(cat $dir/metrics/evaluation_results.json)
    
    td=$(echo $metrics | jq -r '.TD')
    npmi=$(echo $metrics | jq -r '.NPMI')
    cv=$(echo $metrics | jq -r '.C_V')
    ppl=$(echo $metrics | jq -r '.PPL')
    
    echo "$config,$td,$npmi,$cv,$ppl" >> grid_results.csv
done
```

### Best Configuration

Analysis reveals optimal settings:
- Number of topics: 20
- Learning rate: 0.002
- Hidden dimension: 512

These become default recommendations for similar datasets.

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
4. Use 8B only for critical applications

### Hyperparameter Tuning

1. Begin with default parameters
2. Adjust number of topics based on corpus
3. Tune learning rate if training is unstable
4. Increase hidden dimension for complex data
5. Adjust KL annealing for large datasets

### Evaluation

1. Review multiple metrics, not just one
2. Examine visualizations for qualitative assessment
3. Compare with baseline models
4. Validate topics with domain experts

### Scaling

1. Process small sample first
2. Optimize on sample before full corpus
3. Monitor memory usage
4. Use batch processing for multiple datasets

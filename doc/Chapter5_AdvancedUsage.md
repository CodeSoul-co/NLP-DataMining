# Advanced Usage

This section covers advanced features and specialized use cases.

---

## Working with New Datasets

### Complete Workflow for English Data

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

Expected output confirms presence of:
- Embeddings file
- BOW representations
- Vocabulary
- Document indices

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

### Complete Workflow for Chinese Data

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

### Starting from Raw Data

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

The `--clean` flag triggers automatic cleaning. Cleaned output is saved as `news_articles_cleaned.csv`.

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

Chinese cleaning handles character encoding, punctuation normalization, and traditional/simplified conversion.

### Supervised Learning Scenario

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

The model learns topics that align with provided labels while maintaining topic diversity.

### Temporal Data Processing

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

Documents are grouped by time slice automatically.

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

DTM tracks topic evolution across time periods.

---

## Hyperparameter Tuning

### Learning Rate Scheduling

Manual learning rate tuning:

**Conservative approach (unstable training):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --learning_rate 0.0005 \
    --epochs 150 \
    --gpu 0
```

**Standard approach:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --learning_rate 0.002 \
    --epochs 100 \
    --gpu 0
```

**Aggressive approach (slow convergence):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --learning_rate 0.01 \
    --epochs 80 \
    --gpu 0
```

Monitor training loss curves to determine if adjustment is needed.

### Batch Size Optimization

Batch size affects memory usage and training dynamics:

**Small batches (noisy gradients):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --batch_size 32 \
    --gpu 0
```

Advantages: Lower memory usage, better exploration
Disadvantages: Noisy updates, slower convergence

**Large batches (stable gradients):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --batch_size 128 \
    --gpu 0
```

Advantages: Stable updates, faster epochs
Disadvantages: Higher memory, may overfit

Default of 64 balances these tradeoffs.

### KL Annealing Strategies

**No annealing (immediate full KL):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --kl_start 1.0 \
    --kl_end 1.0 \
    --kl_warmup 0 \
    --gpu 0
```

Risk: Posterior collapse, poor topic quality

**Standard annealing:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 50 \
    --gpu 0
```

Recommended default for most datasets.

**Slow annealing (complex data):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 80 \
    --gpu 0
```

Use for large or complex corpora.

**Partial annealing (fine-tuning):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --kl_start 0.2 \
    --kl_end 0.8 \
    --kl_warmup 40 \
    --gpu 0
```

Use when standard annealing causes issues.

### Hidden Dimension Tuning

**Minimal capacity:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --hidden_dim 256 \
    --gpu 0
```

Use for small datasets or when memory is constrained.

**Standard capacity:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --hidden_dim 512 \
    --gpu 0
```

Default choice for most applications.

**High capacity:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --hidden_dim 1024 \
    --gpu 0
```

Use for large complex datasets when VRAM permits.

### Early Stopping Configuration

**Tight patience (fast stopping):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --patience 5 \
    --gpu 0
```

Stops quickly if validation loss plateaus.

**Standard patience:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --patience 10 \
    --gpu 0
```

Default setting.

**Loose patience (more training):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --patience 20 \
    --gpu 0
```

Allows longer training before stopping.

**Disabled (train full epochs):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 200 \
    --no_early_stopping \
    --gpu 0
```

Trains for all specified epochs regardless of validation performance.

### Vocabulary Size Selection

**Small vocabulary:**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 3000 \
    --gpu 0
```

Appropriate for: Domain-specific text, smaller corpora, faster training

**Medium vocabulary:**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --gpu 0
```

Default choice for most datasets.

**Large vocabulary:**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 10000 \
    --gpu 0
```

Appropriate for: Large diverse corpora, capturing rare terms

Trade-off: Larger vocabularies increase computation but capture more terms.

---

## Custom Configurations

### Combining Multiple Settings

Complex configuration example:

```bash
python run_pipeline.py \
    --dataset complex_corpus \
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

This configuration:
- Uses larger 4B model for better quality
- Discovers 50 topics for fine-grained analysis
- Trains longer (150 epochs) with patient early stopping
- Uses smaller batch size due to 4B model memory requirements
- Increases hidden dimension for complex data
- Reduces learning rate for stability
- Applies slow KL annealing for gradual learning

### Skipping Pipeline Stages

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

Loads checkpoint and runs evaluation only.

**Skip evaluation:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --skip-eval \
    --gpu 0
```

Trains model but skips metric computation.

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

Trains and evaluates but does not generate visualizations.

**Combined skipping:**
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

Training only, no evaluation or visualization.

### Preprocessing-Only Mode

Generate embeddings without training:

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

Then train later:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --gpu 0
```

Useful for preprocessing large datasets once and training multiple times with different hyperparameters.

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

Faster preprocessing when embeddings are not immediately needed.

---

## Chinese Data Processing

### Specialized Preprocessing

Chinese text requires different handling than English:

**Data cleaning:**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/chinese_corpus/raw_data.csv \
    --output /root/autodl-tmp/data/chinese_corpus/chinese_corpus_cleaned.csv \
    --language chinese
```

Cleaning operations for Chinese:
- Remove HTML entities
- Normalize full-width and half-width characters
- Handle Chinese punctuation
- Preserve Chinese word boundaries
- Convert traditional to simplified (optional)

**Preprocessing:**
```bash
python prepare_data.py \
    --dataset chinese_corpus \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

Qwen models handle Chinese tokenization internally.

**Training:**
```bash
python run_pipeline.py \
    --dataset chinese_corpus \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language zh
```

The `--language zh` setting ensures Chinese fonts in visualizations.

### Chinese Visualization

Chinese visualizations require proper font configuration:

```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset chinese_corpus \
    --mode zero_shot \
    --model_size 0.6B \
    --language zh \
    --dpi 300
```

The visualization module automatically:
- Selects Chinese-compatible fonts
- Handles character encoding
- Adjusts layout for Chinese text
- Renders word clouds with Chinese characters

### Chinese-English Mixed Data

For datasets containing both languages:

1. Clean as Chinese (preserves both languages)
2. Preprocess normally (Qwen handles multilingual)
3. Train with appropriate language setting
4. Visualizations may show mixed text

Primary language should be specified in `--language` parameter based on majority content.

---

## Using Different Model Sizes

### Scaling Strategy

**Development workflow:**
1. Start with 0.6B model
2. Optimize hyperparameters
3. Scale to 4B for production
4. Use 8B for final results if needed

**Quick comparison:**
```bash
# Train all three sizes
for size in 0.6B 4B 8B; do
    python run_pipeline.py \
        --dataset my_dataset \
        --models theta \
        --model_size $size \
        --mode zero_shot \
        --num_topics 20 \
        --gpu 0
done
```

Compare metrics to determine cost-benefit tradeoff.

### Memory-Efficient Training

For limited VRAM:

**0.6B with reduced batch size:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --batch_size 32 \
    --gpu 0
```

**4B with minimal batch size:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 4B \
    --mode zero_shot \
    --num_topics 20 \
    --batch_size 16 \
    --gpu 0
```

**8B requiring high-end GPU:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 8B \
    --mode zero_shot \
    --num_topics 20 \
    --batch_size 8 \
    --gpu 0
```

Reduce batch size if out-of-memory errors occur.

### Quality vs Cost Analysis

Typical improvement from model scaling:

**0.6B → 4B:**
- Topic diversity: +3-5%
- Coherence (NPMI): +10-15%
- Perplexity: -5-8%
- Training time: +60-80%
- Cost: +200-250%

**4B → 8B:**
- Topic diversity: +1-2%
- Coherence (NPMI): +5-8%
- Perplexity: -3-5%
- Training time: +80-100%
- Cost: +200-250%

Diminishing returns suggest 4B is often the best choice for production.

### Mixed-Size Workflow

Efficient workflow using multiple sizes:

1. Rapid prototyping with 0.6B
2. Hyperparameter search with 0.6B
3. Validate best configuration with 4B
4. Final results with 8B if necessary

This minimizes compute cost while maintaining quality.

---

## Batch Processing

### Processing Multiple Datasets

Process several datasets sequentially:

```bash
#!/bin/bash
datasets=("news" "reviews" "papers" "social")

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    
    python prepare_data.py \
        --dataset $dataset \
        --model theta \
        --model_size 0.6B \
        --mode zero_shot \
        --vocab_size 5000 \
        --gpu 0
    
    python run_pipeline.py \
        --dataset $dataset \
        --models theta \
        --model_size 0.6B \
        --mode zero_shot \
        --num_topics 20 \
        --gpu 0
done
```

### Grid Search Over Hyperparameters

Systematic hyperparameter exploration:

```bash
#!/bin/bash
topics=(10 20 30 40)
learning_rates=(0.001 0.002 0.005)

for K in "${topics[@]}"; do
    for lr in "${learning_rates[@]}"; do
        echo "Training with K=$K, lr=$lr"
        
        python run_pipeline.py \
            --dataset my_dataset \
            --models theta \
            --model_size 0.6B \
            --mode zero_shot \
            --num_topics $K \
            --learning_rate $lr \
            --gpu 0
        
        # Save results with descriptive names
        mv result/0.6B/my_dataset/zero_shot \
           result/0.6B/my_dataset/zero_shot_K${K}_lr${lr}
    done
done
```

### Parallel Processing on Multiple GPUs

Train different configurations in parallel:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset dataset1 --models theta --gpu 0 &

# Terminal 2  
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --dataset dataset2 --models theta --gpu 0 &

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python run_pipeline.py \
    --dataset dataset3 --models theta --gpu 0 &
```

Each process uses a different GPU.

# API Reference

Complete parameter documentation for all THETA command-line tools.

---

## prepare_data.py

Data preprocessing script for generating embeddings and bag-of-words representations.

### Basic Usage

```bash
python prepare_data.py --dataset DATASET --model MODEL [OPTIONS]
```

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--dataset` | string | Dataset name (must match directory name in `/root/autodl-tmp/data/`) |
| `--model` | string | Model type: `theta`, `baseline`, or `dtm` |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_size` | string | `0.6B` | Qwen model size: `0.6B`, `4B`, or `8B` (THETA only) |
| `--mode` | string | `zero_shot` | Training mode: `zero_shot`, `supervised`, or `unsupervised` (THETA only) |

### Data Processing

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--vocab_size` | int | `5000` | 1000-20000 | Vocabulary size for BOW representation |
| `--batch_size` | int | `32` | 8-128 | Batch size for embedding generation |
| `--max_length` | int | `512` | 128-2048 | Maximum sequence length for embeddings |

### GPU Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gpu` | int | `0` | GPU device ID (0, 1, 2, ...) |

### Data Cleaning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--clean` | flag | False | Clean data before preprocessing |
| `--raw-input` | string | None | Path to raw CSV file (requires `--clean`) |
| `--language` | string | `english` | Cleaning language: `english` or `chinese` |

### Utility Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--bow-only` | flag | False | Generate BOW only, skip embeddings |
| `--check-only` | flag | False | Check if preprocessed files exist |
| `--time_column` | string | `year` | Time column name for DTM (DTM only) |

### Examples

**Basic THETA preprocessing:**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000
```

**Baseline model preprocessing:**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model baseline \
    --vocab_size 5000
```

**Combined cleaning and preprocessing:**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --clean \
    --raw-input /path/to/raw.csv \
    --language english
```

**Check preprocessed files:**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --check-only
```

### Output Files

Preprocessed data is saved to:
```
/root/autodl-tmp/result/{model_size}/{dataset}/bow/
```

Generated files:
- `qwen_embeddings_{mode}.npy`: Document embeddings
- `vocab.pkl`: Vocabulary dictionary
- `doc_indices.npy`: Document-term indices
- `bow_matrix.npz`: Sparse BOW matrix

---

## run_pipeline.py

Unified training, evaluation, and visualization pipeline.

### Basic Usage

```bash
python run_pipeline.py --dataset DATASET --models MODELS [OPTIONS]
```

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--dataset` | string | Dataset name |
| `--models` | string | Comma-separated model list: `theta`, `lda`, `etm`, `ctm`, `dtm` |

### Model Configuration (THETA)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_size` | string | `0.6B` | Qwen model size: `0.6B`, `4B`, or `8B` |
| `--mode` | string | `zero_shot` | Training mode: `zero_shot`, `supervised`, or `unsupervised` |

### Topic Model Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--num_topics` | int | `20` | 5-100 | Number of topics to discover |
| `--epochs` | int | `100` | 10-500 | Maximum training epochs |
| `--batch_size` | int | `64` | 8-512 | Training batch size |

### Neural Network Architecture

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--hidden_dim` | int | `512` | 128-1024 | Encoder hidden dimension |

### Optimization

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--learning_rate` | float | `0.002` | 0.00001-0.1 | Learning rate for optimizer |

### KL Annealing

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--kl_start` | float | `0.0` | 0.0-1.0 | Initial KL divergence weight |
| `--kl_end` | float | `1.0` | 0.0-1.0 | Final KL divergence weight |
| `--kl_warmup` | int | `50` | 0-200 | Number of warmup epochs for KL annealing |

### Early Stopping

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--patience` | int | `10` | 1-50 | Epochs to wait before early stopping |
| `--no_early_stopping` | flag | False | N/A | Disable early stopping |

### Hardware Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gpu` | int | `0` | GPU device ID |

### Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--language` | string | `en` | Visualization language: `en` or `zh` |

### Pipeline Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--skip-train` | flag | False | Skip training, evaluate only |
| `--skip-eval` | flag | False | Skip evaluation |
| `--skip-viz` | flag | False | Skip visualization |
| `--check-only` | flag | False | Check data files only |
| `--prepare` | flag | False | Run preprocessing before training |

### Examples

**Basic THETA training:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --gpu 0
```

**Multiple baseline models:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --gpu 0
```

**Custom hyperparameters:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 30 \
    --epochs 150 \
    --batch_size 32 \
    --hidden_dim 768 \
    --learning_rate 0.001 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 80 \
    --patience 15 \
    --gpu 0
```

**Evaluate existing model:**
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

**Training without visualization:**
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

### Output Files

Training results are saved to:

**THETA models:**
```
/root/autodl-tmp/result/{model_size}/{dataset}/{mode}/
├── checkpoints/
│   ├── best_model.pt
│   └── training_history.json
├── metrics/
│   └── evaluation_results.json
└── visualizations/
    ├── topic_words_bars.png
    ├── topic_similarity.png
    ├── doc_topic_umap.png
    ├── topic_wordclouds.png
    ├── metrics.png
    └── pyldavis.html
```

**Baseline models:**
```
/root/autodl-tmp/result/baseline/{dataset}/{model}/K{num_topics}/
├── checkpoints/
├── metrics/
└── visualizations/
```

---

## visualization.run_visualization

Separate visualization generation tool.

### Basic Usage

```bash
python -m visualization.run_visualization --result_dir DIR --dataset DATASET [OPTIONS]
```

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--result_dir` | string | Results directory path |
| `--dataset` | string | Dataset name |

### THETA Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mode` | string | `zero_shot` | Training mode (for THETA models) |
| `--model_size` | string | `0.6B` | Qwen model size (for THETA models) |

### Baseline Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--baseline` | flag | False | Indicates baseline model |
| `--model` | string | None | Baseline model name: `lda`, `etm`, `ctm`, or `dtm` |
| `--num_topics` | int | `20` | Number of topics (for baseline models) |

### Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--language` | string | `en` | Visualization language: `en` or `zh` |
| `--dpi` | int | `300` | Image resolution (dots per inch) |

### Examples

**THETA model visualization:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 300
```

**LDA model visualization:**
```bash
python -m visualization.run_visualization \
    --baseline \
    --result_dir /root/autodl-tmp/result/baseline \
    --dataset my_dataset \
    --model lda \
    --num_topics 20 \
    --language en \
    --dpi 300
```

**High-resolution visualization:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 600
```

**Chinese visualization:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset chinese_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language zh \
    --dpi 300
```

### Output Files

Visualizations are saved to the same directory as the model results:
- `topic_words_bars.png`: Bar charts of topic words
- `topic_similarity.png`: Topic similarity heatmap
- `doc_topic_umap.png`: Document-topic UMAP projection
- `topic_wordclouds.png`: Word clouds for each topic
- `metrics.png`: Evaluation metrics comparison
- `pyldavis.html`: Interactive visualization

---

## dataclean.main

Data cleaning module for preprocessing raw text.

### Basic Usage

```bash
python -m dataclean.main --input INPUT --output OUTPUT --language LANGUAGE
```

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--input` | string | Input CSV file path or directory |
| `--output` | string | Output CSV file path or directory |
| `--language` | string | Language: `english` or `chinese` |

### Examples

**Clean single file (English):**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language english
```

**Clean single file (Chinese):**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language chinese
```

**Clean directory:**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw/ \
    --output /root/autodl-tmp/data/cleaned/ \
    --language english
```

### Cleaning Operations

**English cleaning:**
- Remove HTML tags and entities
- Remove URLs and email addresses
- Remove special characters (except basic punctuation)
- Normalize whitespace
- Remove non-ASCII characters (optional)
- Lowercase text (optional)

**Chinese cleaning:**
- Remove HTML tags and entities
- Remove URLs and email addresses
- Normalize full-width and half-width characters
- Handle Chinese punctuation
- Remove non-Chinese characters (optional)
- Preserve word boundaries

### Output Format

Cleaned CSV maintains the same structure as input. The text column is cleaned while other columns are preserved.

---

## Parameter Interactions

### Model Size and Batch Size

Recommended batch sizes for different model sizes:

| Model Size | Preprocessing Batch | Training Batch | VRAM Usage |
|-----------|-------------------|---------------|------------|
| 0.6B | 32 | 64 | ~8GB |
| 4B | 16 | 32 | ~16GB |
| 8B | 8 | 16 | ~28GB |

Reduce batch sizes if encountering out-of-memory errors.

### KL Annealing and Epochs

Recommended warmup periods:

| Total Epochs | Warmup Period | KL Schedule |
|-------------|--------------|-------------|
| 50-80 | 30 | Fast annealing |
| 80-120 | 50 | Standard annealing |
| 120-200 | 80 | Slow annealing |

Warmup should be 40-60% of total epochs.

### Learning Rate and Batch Size

Adjust learning rate based on batch size:

| Batch Size | Learning Rate | Rationale |
|-----------|--------------|-----------|
| 16-32 | 0.001-0.002 | Small batches, noisy gradients |
| 64 | 0.002 | Default configuration |
| 128-256 | 0.003-0.005 | Large batches, stable gradients |

Larger batches support higher learning rates.

### Hidden Dimension and Topics

Recommended hidden dimensions:

| Number of Topics | Hidden Dimension | Model Capacity |
|-----------------|-----------------|---------------|
| 5-15 | 256-384 | Low |
| 15-30 | 512 | Medium |
| 30-50 | 512-768 | High |
| 50-100 | 768-1024 | Very High |

More topics require higher encoder capacity.

### Vocabulary Size and Corpus Size

Recommended vocabulary sizes:

| Corpus Size | Vocabulary Size | Coverage |
|------------|----------------|----------|
| < 1K docs | 2000-3000 | ~85% |
| 1K-10K docs | 5000 | ~90% |
| 10K-100K docs | 8000-10000 | ~92% |
| > 100K docs | 10000-15000 | ~95% |

Larger corpora support larger vocabularies.

---

## Common Parameter Combinations

### Rapid Prototyping

Fast training for initial exploration:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 10 \
    --epochs 50 \
    --batch_size 64 \
    --gpu 0
```

### Standard Configuration

Balanced quality and speed:

```bash
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
    --gpu 0
```

### High-Quality Results

Maximum quality for publication:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 8B \
    --mode zero_shot \
    --num_topics 50 \
    --epochs 200 \
    --batch_size 16 \
    --hidden_dim 1024 \
    --learning_rate 0.001 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 100 \
    --patience 20 \
    --gpu 0
```

### Memory-Constrained Training

Minimal VRAM usage:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 16 \
    --hidden_dim 256 \
    --gpu 0
```

### Baseline Comparison

Train all baseline models:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0
```

---

## Environment Variables

### CUDA Configuration

Control GPU visibility:

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --dataset my_dataset --models theta

# Use multiple GPUs (not for single training)
CUDA_VISIBLE_DEVICES=0,1 python run_pipeline.py --dataset my_dataset --models theta
```

### Memory Management

PyTorch memory settings:

```bash
# Limit GPU memory fraction
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python run_pipeline.py ...

# Enable memory debugging
PYTORCH_NO_CUDA_MEMORY_CACHING=1 python run_pipeline.py ...
```

### Logging

Control output verbosity:

```bash
# Disable progress bars
TQDM_DISABLE=1 python run_pipeline.py ...

# Reduce logging
export PYTHONWARNINGS="ignore"
python run_pipeline.py ...
```

---

## Return Codes

Scripts return standard exit codes:

| Exit Code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | General error |
| 2 | File not found |
| 3 | Invalid parameters |
| 4 | CUDA out of memory |
| 5 | Convergence failure |

Check exit codes in scripts:

```bash
python run_pipeline.py ...
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with code $?"
fi
```

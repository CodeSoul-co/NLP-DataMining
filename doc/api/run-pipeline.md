# run_pipeline.py

Unified training, evaluation, and visualization pipeline.

---

## Basic Usage

```bash
python run_pipeline.py --dataset DATASET --models MODELS [OPTIONS]
```

---

## Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--dataset` | string | Dataset name |
| `--models` | string | Comma-separated model list: `theta`, `lda`, `etm`, `ctm`, `dtm` |

## Model Configuration (THETA)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_size` | string | `0.6B` | Qwen model size: `0.6B`, `4B`, or `8B` |
| `--mode` | string | `zero_shot` | Training mode: `zero_shot`, `supervised`, or `unsupervised` |

## Topic Model Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--num_topics` | int | `20` | 5-100 | Number of topics to discover |
| `--epochs` | int | `100` | 10-500 | Maximum training epochs |
| `--batch_size` | int | `64` | 8-512 | Training batch size |

## Neural Network Architecture

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--hidden_dim` | int | `512` | 128-1024 | Encoder hidden dimension |

## Optimization

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--learning_rate` | float | `0.002` | 0.00001-0.1 | Learning rate for optimizer |

## KL Annealing

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--kl_start` | float | `0.0` | 0.0-1.0 | Initial KL divergence weight |
| `--kl_end` | float | `1.0` | 0.0-1.0 | Final KL divergence weight |
| `--kl_warmup` | int | `50` | 0-200 | Number of warmup epochs for KL annealing |

## Early Stopping

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--patience` | int | `10` | 1-50 | Epochs to wait before early stopping |
| `--no_early_stopping` | flag | False | N/A | Disable early stopping |

## Hardware Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gpu` | int | `0` | GPU device ID |

## Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--language` | string | `en` | Visualization language: `en` or `zh` |

## Pipeline Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--skip-train` | flag | False | Skip training, evaluate only |
| `--skip-eval` | flag | False | Skip evaluation |
| `--skip-viz` | flag | False | Skip visualization |
| `--check-only` | flag | False | Check data files only |
| `--prepare` | flag | False | Run preprocessing before training |

---

## Examples

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

---

## Output Files

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

## Return Codes

| Exit Code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | General error |
| 2 | File not found |
| 3 | Invalid parameters |
| 4 | CUDA out of memory |
| 5 | Convergence failure |

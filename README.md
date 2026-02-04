<div align="center">

<img src="assets/THETA.png" width="40%" alt="THETA Logo"/>

<h1>THETA (θ)</h1>

Textual Hybrid Embedding–based Topic Analysis  

</div>

## Overview

THETA (θ) is an open-source, research-oriented platform for LLM-enhanced topic analysis in social science. It combines:

- Domain-adaptive document embeddings from Qwen-3 models (0.6B/4B/8B)
  - Zero-shot embedding (no training), or
  - Supervised/Unsupervised fine-tuning modes
- Generative topic models with 12 baseline models for comparison:
  - **THETA**: Main model using Qwen embeddings (0.6B/4B/8B)
  - **Traditional**: LDA, HDP (auto topics), STM (covariates), BTM (short texts)
  - **Neural**: ETM, CTM, DTM (time-aware), NVDM, GSM, ProdLDA, BERTopic
- Scientific validation via 7 intrinsic metrics (PPL, TD, iRBO, NPMI, C_V, UMass, Exclusivity)
- Comprehensive visualization with bilingual support (English/Chinese)

THETA aims to move topic modeling from "clustering with pretty plots" to a reproducible, validated scientific workflow.

---

## Key Features

- **Hybrid embedding topic analysis**: Zero-shot / Supervised / Unsupervised modes
- **Multiple Qwen model sizes**: 0.6B (1024-dim), 4B (2560-dim), 8B (4096-dim)
- **12 Baseline models**: LDA, HDP, STM, BTM, ETM, CTM, DTM, NVDM, GSM, ProdLDA, BERTopic for comparison
- **Data governance**: Domain-aware cleaning for multiple languages (English, Chinese, German, Spanish)
- **Unified evaluation**: 7 metrics with JSON/CSV export
- **Rich visualization**: 20+ chart types with bilingual labels

---

## Supported Models

### Model Overview

| Model | Type | Description | Auto Topics | Best For |
|-------|------|-------------|-------------|----------|
| `theta` | Neural | THETA with Qwen embeddings (0.6B/4B/8B) | No | General purpose, high quality |
| `lda` | Traditional | Latent Dirichlet Allocation (sklearn) | No | Fast baseline, interpretable |
| `hdp` | Traditional | Hierarchical Dirichlet Process | **Yes** | Unknown topic count |
| `stm` | Traditional | Structural Topic Model | No | With metadata/covariates |
| `btm` | Traditional | Biterm Topic Model | No | Short texts (tweets, titles) |
| `etm` | Neural | Embedded Topic Model (Word2Vec + VAE) | No | Word embedding integration |
| `ctm` | Neural | Contextualized Topic Model (SBERT + VAE) | No | Semantic understanding |
| `dtm` | Neural | Dynamic Topic Model | No | Time-series analysis |
| `nvdm` | Neural | Neural Variational Document Model | No | VAE-based baseline |
| `gsm` | Neural | Gaussian Softmax Model | No | Better topic separation |
| `prodlda` | Neural | Product of Experts LDA | No | State-of-the-art neural LDA |
| `bertopic` | Neural | BERT-based topic modeling | **Yes** | Clustering-based topics |

### Model Selection Guide

```
Choose your model based on:

┌─────────────────────────────────────────────────────────────────┐
│ Do you know the number of topics?                               │
│   ├─ NO  → Use HDP or BERTopic (auto-detect topics)            │
│   └─ YES → Continue below                                       │
├─────────────────────────────────────────────────────────────────┤
│ What is your text length?                                       │
│   ├─ SHORT (tweets, titles) → Use BTM                          │
│   └─ NORMAL/LONG → Continue below                               │
├─────────────────────────────────────────────────────────────────┤
│ Do you have time-series data?                                   │
│   ├─ YES → Use DTM                                              │
│   └─ NO  → Continue below                                       │
├─────────────────────────────────────────────────────────────────┤
│ What's your priority?                                           │
│   ├─ SPEED      → Use LDA (fastest)                            │
│   ├─ QUALITY    → Use THETA (best with Qwen embeddings)        │
│   └─ COMPARISON → Use multiple: lda,nvdm,prodlda,theta         │
└─────────────────────────────────────────────────────────────────┘
```

### Python API Usage

#### THETA Model Training

```python
import sys
sys.path.insert(0, '/root/autodl-tmp/ETM')

from main import run_training
from config import config_from_args
import argparse

# Create configuration
args = argparse.Namespace(
    dataset='hatespeech',
    mode='zero_shot',           # zero_shot / supervised / unsupervised
    model_size='0.6B',          # 0.6B / 4B / 8B
    num_topics=20,
    epochs=100,
    batch_size=64,
    hidden_dim=512,
    learning_rate=0.002,
    kl_start=0.0,
    kl_end=1.0,
    kl_warmup=50,
    patience=10,
    gpu_id=0,
    dev_mode=False
)

config = config_from_args(args)
run_training(config)
```

#### Baseline Model Training

```python
import sys
sys.path.insert(0, '/root/autodl-tmp/ETM')

from model.baseline_trainer import BaselineTrainer

# Initialize trainer
trainer = BaselineTrainer(
    dataset='hatespeech',
    num_topics=20,
    vocab_size=5000,
    data_dir='/root/autodl-tmp/data',
    result_dir='/root/autodl-tmp/result/baseline'
)

# Prepare data (generates BOW and SBERT embeddings)
trainer.prepare_data(generate_sbert=True)

# Train individual models
lda_result = trainer.train_lda(max_iter=100)
hdp_result = trainer.train_hdp(max_topics=150)      # Auto topic number
btm_result = trainer.train_btm(n_iter=100)
nvdm_result = trainer.train_nvdm(epochs=100, batch_size=64)
gsm_result = trainer.train_gsm(epochs=100, batch_size=64)
prodlda_result = trainer.train_prodlda(epochs=100, batch_size=64)

# Or train multiple models at once
results = trainer.train_all(
    models=['lda', 'hdp', 'btm', 'nvdm', 'prodlda'],
    batch_size=64
)

# Access results
print(f"LDA theta shape: {lda_result['theta'].shape}")      # (N, K)
print(f"LDA beta shape: {lda_result['beta'].shape}")        # (K, V)
print(f"HDP actual topics: {hdp_result['actual_num_topics']}")
```

### Training Parameters Reference

#### THETA Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--num_topics` | int | 20 | 5-100 | Number of topics |
| `--epochs` | int | 100 | 10-500 | Training epochs |
| `--batch_size` | int | 64 | 8-512 | Batch size |
| `--hidden_dim` | int | 512 | 128-1024 | Encoder hidden dimension |
| `--learning_rate` | float | 0.002 | 1e-5 - 0.1 | Learning rate |
| `--kl_start` | float | 0.0 | 0-1 | KL annealing start weight |
| `--kl_end` | float | 1.0 | 0-1 | KL annealing end weight |
| `--kl_warmup` | int | 50 | 0-epochs | KL warmup epochs |
| `--patience` | int | 10 | 1-50 | Early stopping patience |

#### Baseline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_topics` | int | 20 | Number of topics (ignored for HDP/BERTopic) |
| `--epochs` | int | 100 | Training epochs (neural models only) |
| `--batch_size` | int | 64 | Batch size (neural models only) |
| `--hidden_dim` | int | 512 | Hidden dimension (neural models only) |
| `--learning_rate` | float | 0.002 | Learning rate (neural models only) |

---

## Project Structure

```
/root/
├── ETM/                          # Main codebase
│   ├── run_pipeline.py           # Unified entry point
│   ├── prepare_data.py           # Data preprocessing
│   ├── config.py                 # Configuration management
│   ├── dataclean/                # Data cleaning module
│   ├── model/                    # Model implementations
│   │   ├── theta/                # THETA main model
│   │   ├── baselines/            # 12 baseline models
│   │   └── _reference/           # Reference implementations
│   ├── evaluation/               # Evaluation metrics
│   ├── visualization/            # Visualization tools
│   └── utils/                    # Utilities 
├── agent/                        # Agent system
│   ├── api.py                    # FastAPI endpoints
│   ├── core/                     # Agent implementations
│   ├── config/                   # Configuration management
│   ├── prompts/                  # Prompt templates
│   ├── utils/                    # LLM and vision utilities
│   └── docs/                     # API documentation
├── scripts/                      # Shell scripts for automation
├── embedding/                     # Qwen Embedding
│   ├── main.py                    # Embedding generation main codebase
│   ├── embedder.py                # Embedding
│   ├── trainer.py                 # Training (supervised/unsupervised)
│   ├── data_loader.py             # Dataloader
```

---

## Requirements

- Python 3.10+
- CUDA recommended for GPU acceleration
- Key dependencies:

```
numpy>=1.20.0
scipy>=1.7.0
torch>=1.10.0
transformers>=4.30.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
gensim>=4.1.0
wordcloud>=1.8.0
pyLDAvis>=3.3.0
jieba>=0.42.0
```

---

## Installation

```bash
git clone https://github.com/<YOUR_ORG>/THETA.git
cd THETA

# Install dependencies
pip install -r ETM/requirements.txt

# Or use the setup script
bash scripts/01_setup.sh
```

### Pre-trained Data from HuggingFace

If pre-trained embeddings and BOW data are not available locally, download from HuggingFace:

**Repository**: [https://huggingface.co/CodeSoulco/THETA](https://huggingface.co/CodeSoulco/THETA)

```bash
# Download pre-trained data and LoRA weights
bash scripts/09_download_from_hf.sh

# Or manually using Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='CodeSoulco/THETA',
    local_dir='/root/autodl-tmp/hf_cache/THETA'
)
"
```

The HuggingFace repository contains:
- Pre-computed embeddings for benchmark datasets
- BOW matrices and vocabularies
- LoRA fine-tuned weights (optional)

---

## Shell Scripts

All commands are available as shell scripts in the `scripts/` folder for easy frontend integration:

| Script | Description |
|--------|-------------|
| `01_setup.sh` | Install dependencies and download data from HuggingFace |
| `02_clean_data.sh` | Clean raw data |
| `03_prepare_data.sh` | Generate embeddings and BOW |
| `04_train_theta.sh` | Train THETA model |
| `05_train_baseline.sh` | Train baseline models (LDA, HDP, STM, BTM, ETM, CTM, DTM, NVDM, GSM, ProdLDA) |
| `06_visualize.sh` | Generate visualizations |
| `07_start_agent_api.sh` | Start the Agent API server |
| `08_test_agent.sh` | Test agent functionality |
| `09_download_from_hf.sh` | Download pre-trained data from HuggingFace |
| `10_quick_start_english.sh` | Quick start for English datasets |
| `11_quick_start_chinese.sh` | Quick start for Chinese datasets |
| `12_train_multi_gpu.sh` | Multi-GPU training with DistributedDataParallel |

---

## Quickstart

### Using Shell Scripts (Recommended)

```bash
# Quick start for English dataset
bash scripts/10_quick_start_english.sh my_dataset

# Quick start for Chinese dataset
bash scripts/11_quick_start_chinese.sh my_chinese_dataset

# Train THETA model
bash scripts/04_train_theta.sh --dataset hatespeech --model_size 0.6B --mode zero_shot --num_topics 20

# Train baseline models (traditional + neural)
bash scripts/05_train_baseline.sh --dataset hatespeech --models lda,hdp,btm,nvdm,prodlda --num_topics 20

# Generate visualizations
bash scripts/06_visualize.sh --dataset hatespeech --model_size 0.6B --mode zero_shot --language en
```

### A) THETA Model (Zero-shot Mode)

```bash
# 1) Prepare data (generate Qwen embeddings + BOW)
bash scripts/03_prepare_data.sh --dataset socialTwitter --model theta --model_size 0.6B --mode zero_shot

# 2) Run complete pipeline (train + evaluate + visualize)
bash scripts/04_train_theta.sh --dataset socialTwitter --model_size 0.6B --mode zero_shot --num_topics 20

# 3) Skip training, only evaluate and visualize existing results
bash scripts/04_train_theta.sh --dataset socialTwitter --model_size 0.6B --mode zero_shot --skip-train

# 4) With custom training parameters
bash scripts/04_train_theta.sh --dataset socialTwitter \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 30 \
    --epochs 150 \
    --batch_size 128 \
    --hidden_dim 768 \
    --learning_rate 0.001 \
    --kl_warmup 30 \
    --patience 15
```

### B) THETA Model (Supervised Mode with 4B/8B)

```bash
# Use larger model with supervised fine-tuning
bash scripts/04_train_theta.sh --dataset hatespeech --model_size 4B --mode supervised --num_topics 20

# Use 8B model for best quality
bash scripts/04_train_theta.sh --dataset mental_health --model_size 8B --mode supervised --num_topics 25

# With KL annealing customization
bash scripts/04_train_theta.sh --dataset hatespeech \
    --model_size 4B \
    --mode supervised \
    --num_topics 20 \
    --kl_start 0.1 \
    --kl_end 0.8 \
    --kl_warmup 40
```

### C) Baseline Models (12 Models Available)

**Traditional Models:**
- `lda` - Latent Dirichlet Allocation (sklearn)
- `hdp` - Hierarchical Dirichlet Process (auto topic number)
- `stm` - Structural Topic Model (with covariates)
- `btm` - Biterm Topic Model (for short texts)

**Neural Models:**
- `etm` - Embedded Topic Model (Word2Vec + VAE)
- `ctm` - Contextualized Topic Model (SBERT + VAE)
- `dtm` - Dynamic Topic Model (time-aware)
- `nvdm` - Neural Variational Document Model
- `gsm` - Gaussian Softmax Model
- `prodlda` - Product of Experts LDA
- `bertopic` - BERT-based topic modeling (auto topic number)

```bash
# Train single baseline model
bash scripts/05_train_baseline.sh --dataset socialTwitter --models lda --num_topics 20

# Train multiple traditional models
bash scripts/05_train_baseline.sh --dataset socialTwitter --models lda,hdp,stm,btm --num_topics 20

# Train multiple neural models
bash scripts/05_train_baseline.sh --dataset socialTwitter --models nvdm,gsm,prodlda --num_topics 20

# Train all baselines for comparison
bash scripts/05_train_baseline.sh --dataset hatespeech --models lda,hdp,btm,etm,ctm,nvdm,gsm,prodlda --num_topics 20

# With custom parameters
bash scripts/05_train_baseline.sh --dataset hatespeech \
    --models nvdm,prodlda \
    --num_topics 30 \
    --epochs 200 \
    --batch_size 128
```

### D) Dynamic Topic Model (DTM)

```bash
# Prepare data with time slices (requires timestamp column)
bash scripts/03_prepare_data.sh --dataset edu_data --model dtm --time_column year

# Train DTM model
bash scripts/05_train_baseline.sh --dataset edu_data --models dtm --num_topics 20 --epochs 100 --language zh

# Skip training, only visualize
bash scripts/05_train_baseline.sh --dataset edu_data --models dtm --skip-train --language zh
```

### E) Data Preparation

```bash
# Prepare THETA data with custom vocabulary size
bash scripts/03_prepare_data.sh --dataset mydata \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 8000 \
    --batch_size 64 \
    --max_length 1024

# Prepare baseline data
bash scripts/03_prepare_data.sh --dataset mydata --model baseline --vocab_size 5000

# Clean raw data first, then prepare
bash scripts/03_prepare_data.sh --dataset mydata \
    --model theta \
    --clean \
    --raw-input /path/to/raw.csv \
    --language english

# Check if data files exist
bash scripts/03_prepare_data.sh --dataset mydata --model theta --check-only
```

### F) Visualization

```bash
# THETA model visualization
bash scripts/06_visualize.sh --dataset hatespeech --model_size 0.6B --mode zero_shot --language en --dpi 300

# Baseline model visualization
bash scripts/06_visualize.sh --baseline --dataset hatespeech --model lda --num_topics 20 --language en

# Chinese visualization
bash scripts/06_visualize.sh --dataset edu_data --model_size 0.6B --mode zero_shot --language zh --dpi 300

# High-resolution output
bash scripts/06_visualize.sh --baseline --dataset mental_health --model etm --num_topics 20 --dpi 600
```

### G) Agent API Server

```bash
# Start the Agent API server (default port 8000)
bash scripts/07_start_agent_api.sh

# Start on custom port
bash scripts/07_start_agent_api.sh --port 8080

# Test agent functionality
bash scripts/08_test_agent.sh
```

### H) Data Download from HuggingFace

```bash
# Download pre-trained embeddings, BOW, and LoRA weights
bash scripts/09_download_from_hf.sh

# Setup everything (install dependencies + download data)
bash scripts/01_setup.sh
```

### I) Data Cleaning

```bash
# Clean English text data
bash scripts/02_clean_data.sh \
    --input /root/autodl-tmp/data/mydata/raw.csv \
    --output /root/autodl-tmp/data/mydata/mydata_cleaned.csv \
    --language english

# Clean Chinese text data
bash scripts/02_clean_data.sh \
    --input /root/autodl-tmp/data/chinese_data/raw.csv \
    --output /root/autodl-tmp/data/chinese_data/chinese_data_cleaned.csv \
    --language chinese
```

### J) Advanced Training Examples

```bash
# THETA with all custom parameters
bash scripts/04_train_theta.sh --dataset hatespeech \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 25 \
    --epochs 200 \
    --batch_size 128 \
    --hidden_dim 768 \
    --learning_rate 0.001 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 50 \
    --patience 20 \
    --gpu 0 \
    --language en

# Baseline with all custom parameters
bash scripts/05_train_baseline.sh --dataset mental_health \
    --models lda,etm,ctm \
    --num_topics 30 \
    --epochs 150 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en

# Skip training, only run evaluation
bash scripts/04_train_theta.sh --dataset hatespeech --skip-train --language en

# Skip visualization, only train and evaluate
bash scripts/04_train_theta.sh --dataset hatespeech --skip-viz

# Prepare data with all options
bash scripts/03_prepare_data.sh --dataset newdata \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 10000 \
    --batch_size 64 \
    --max_length 1024 \
    --gpu 0

# DTM with time-aware data
bash scripts/03_prepare_data.sh --dataset temporal_data \
    --model dtm \
    --time_column year \
    --vocab_size 5000
```

### K) Batch Processing Examples

```bash
# Train THETA on multiple datasets
for dataset in hatespeech mental_health socialTwitter; do
    bash scripts/04_train_theta.sh --dataset $dataset \
        --model_size 0.6B --mode zero_shot --num_topics 20
done

# Compare different topic numbers
for k in 10 15 20 25 30; do
    bash scripts/04_train_theta.sh --dataset hatespeech \
        --model_size 0.6B --mode zero_shot --num_topics $k
done

# Train all baseline models on a dataset
bash scripts/05_train_baseline.sh --dataset hatespeech \
    --models lda,etm,ctm --num_topics 20

# Generate visualizations for all trained models
for model in lda etm ctm; do
    bash scripts/06_visualize.sh --baseline --dataset hatespeech \
        --model $model --num_topics 20 --language en
done
```

### L) Multi-GPU Training (DistributedDataParallel)

THETA supports multi-GPU training using PyTorch's DistributedDataParallel (DDP) for faster training on large datasets.

```bash
# Train with 2 GPUs
bash scripts/12_train_multi_gpu.sh --dataset hatespeech --num_gpus 2 --num_topics 20

# Train with 4 GPUs and larger batch size
bash scripts/12_train_multi_gpu.sh --dataset mental_health \
    --num_gpus 4 \
    --batch_size 128 \
    --epochs 200

# Train with custom master port (useful when running multiple jobs)
bash scripts/12_train_multi_gpu.sh --dataset socialTwitter \
    --num_gpus 2 \
    --master_port 29501

# Full example with all parameters
bash scripts/12_train_multi_gpu.sh --dataset hatespeech \
    --num_gpus 4 \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 25 \
    --epochs 150 \
    --batch_size 64 \
    --hidden_dim 768 \
    --learning_rate 0.001

# Or use torchrun directly
torchrun --nproc_per_node=2 --master_port=29500 \
    ETM/main.py train \
    --dataset hatespeech \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100
```

**Multi-GPU Benefits**:
- **Faster training**: Linear speedup with more GPUs
- **Larger effective batch size**: Each GPU processes `batch_size` samples
- **Memory efficiency**: Model is replicated, gradients are synchronized

**Requirements**:
- Multiple NVIDIA GPUs with NCCL support
- PyTorch with CUDA support
- Sufficient GPU memory for model replication

---

## Parameter Reference

### run_pipeline.py Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | string | Required | Dataset name |
| `--models` | string | Required | Model list (comma-separated): theta / lda / etm / ctm / dtm |
| `--model_size` | string | 0.6B | Qwen model size: 0.6B / 4B / 8B |
| `--mode` | string | zero_shot | THETA mode: zero_shot / supervised / unsupervised |
| `--num_topics` | int | 20 | Number of topics (5-100) |
| `--epochs` | int | 100 | Training epochs (10-500) |
| `--batch_size` | int | 64 | Batch size (8-512) |
| `--hidden_dim` | int | 512 | Encoder hidden dimension (128-1024) |
| `--learning_rate` | float | 0.002 | Learning rate (0.00001-0.1) |
| `--kl_start` | float | 0.0 | KL annealing start weight (0-1) |
| `--kl_end` | float | 1.0 | KL annealing end weight (0-1) |
| `--kl_warmup` | int | 50 | KL warmup epochs |
| `--patience` | int | 10 | Early stopping patience (1-50) |
| `--no_early_stopping` | flag | False | Disable early stopping |
| `--gpu` | int | 0 | GPU device ID |
| `--language` | string | en | Visualization language: en / zh |
| `--skip-train` | flag | False | Skip training |
| `--skip-eval` | flag | False | Skip evaluation |
| `--skip-viz` | flag | False | Skip visualization |
| `--check-only` | flag | False | Check files only |
| `--prepare` | flag | False | Preprocess data first |

### visualization.run_visualization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--result_dir` | string | Required | Results directory path |
| `--dataset` | string | Required | Dataset name |
| `--mode` | string | zero_shot | THETA mode (for THETA models) |
| `--model_size` | string | 0.6B | Qwen model size (for THETA models) |
| `--baseline` | flag | False | Is baseline model |
| `--model` | string | None | Baseline model name: lda / etm / ctm / dtm |
| `--num_topics` | int | 20 | Number of topics (for baseline models) |
| `--language` | string | en | Visualization language: en / zh |
| `--dpi` | int | 300 | Image DPI |
| `--output_dir` | string | auto | Output directory |
| `--all` | flag | False | Run for all datasets and models (baseline mode only) |

### prepare_data.py Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | string | Required | Dataset name |
| `--model` | string | Required | Model type: theta / baseline / dtm |
| `--model_size` | string | 0.6B | Qwen model size: 0.6B / 4B / 8B |
| `--mode` | string | zero_shot | Training mode: zero_shot / supervised / unsupervised |
| `--vocab_size` | int | 5000 | Vocabulary size (1000-20000) |
| `--batch_size` | int | 32 | Batch size for embedding (8-128) |
| `--max_length` | int | 512 | Embedding max input length (128-2048) |
| `--gpu` | int | 0 | GPU device ID |
| `--language` | string | english | Cleaning language: english / chinese |
| `--clean` | flag | False | Clean data first |
| `--raw-input` | string | None | Raw data path (use with --clean) |
| `--bow-only` | flag | False | Only generate BOW |
| `--check-only` | flag | False | Only check files |
| `--time_column` | string | year | Time column name (DTM only) |

---

## Data Governance & Preprocessing

The `dataclean` module provides domain-aware text cleaning:

```bash
cd ETM/dataclean

# Convert text files to CSV with NLP cleaning
python main.py convert /path/to/documents output.csv --language chinese --recursive

# Available cleaning operations
python main.py convert input.txt output.csv \
  -p remove_urls \
  -p remove_html_tags \
  -p remove_stopwords \
  -p normalize_whitespace
```

**Supported file formats**: TXT, DOCX, PDF

**Cleaning operations**:
- `remove_urls` - Remove URLs
- `remove_html_tags` - Strip HTML tags
- `remove_punctuation` - Remove punctuation
- `remove_stopwords` - Remove stopwords (language-aware)
- `normalize_whitespace` - Normalize whitespace
- `remove_numbers` - Remove numbers
- `remove_special_chars` - Remove special characters

---

## Semantic Enhancement (Embeddings)

THETA uses Qwen-3 embedding models with three size options:

| Model Size | Embedding Dim | Use Case |
|------------|---------------|----------|
| 0.6B | 1024 | Fast, default |
| 4B | 2560 | Balanced |
| 8B | 4096 | Best quality |

**Embedding modes**:
- `zero_shot` - Direct embedding without fine-tuning
- `supervised` - Fine-tuned with labeled data
- `unsupervised` - Fine-tuned without labels

```bash
# Generate embeddings for a dataset
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B --mode zero_shot

# Check if embeddings exist
python prepare_data.py --dataset my_dataset --model theta --model_size 4B --check-only
```

**Output artifacts**:
- `{dataset}_{mode}_embeddings.npy` - Embedding matrix (N x D)
- `bow_matrix.npz` - Bag-of-words matrix
- `vocab.json` - Vocabulary list

---

## Topic Modeling

THETA supports multiple topic modeling approaches:

| Model | Description | Time-aware |
|-------|-------------|------------|
| THETA | Qwen embedding + ETM | No |
| LDA | Latent Dirichlet Allocation | No |
| ETM | Embedded Topic Model | No |
| CTM | Contextualized Topic Model | No |
| DTM | Dynamic Topic Model | Yes |

**Training outputs** (organized by ResultManager):
- `model/theta_k{K}.npy` - Document-topic distribution
- `model/beta_k{K}.npy` - Topic-word distribution
- `model/training_history_k{K}.json` - Training history
- `topicwords/topic_words_k{K}.json` - Top words per topic
- `topicwords/topic_evolution_k{K}.json` - Topic evolution (DTM only)

---

## Validation & Evaluation

THETA provides unified evaluation with 7 metrics:

| Metric | Description |
|--------|-------------|
| PPL | Perplexity - model fit |
| TD | Topic Diversity |
| iRBO | Inverse Rank-Biased Overlap |
| NPMI | Normalized PMI coherence |
| C_V | C_V coherence |
| UMass | UMass coherence |
| Exclusivity | Topic exclusivity |

```python
from evaluation.unified_evaluator import UnifiedEvaluator

evaluator = UnifiedEvaluator(
    beta=beta,
    theta=theta,
    bow_matrix=bow_matrix,
    vocab=vocab,
    model_name="dtm",
    dataset="edu_data",
    num_topics=20
)

metrics = evaluator.evaluate_all()
evaluator.save_results()  # Saves to evaluation/metrics_k20.json and .csv
```

**Evaluation outputs**:
- `evaluation/metrics_k{K}.json` - All metrics in JSON format
- `evaluation/metrics_k{K}.csv` - All metrics in CSV format

---

## Visualization

THETA provides comprehensive visualization with bilingual support (English/Chinese):

```bash
# Generate visualizations after training
python run_pipeline.py --dataset edu_data --models dtm --skip-train --language en

# Or use visualization module directly
python -c "
from visualization.run_visualization import run_baseline_visualization
run_baseline_visualization(
    result_dir='/root/autodl-tmp/result/baseline',
    dataset='edu_data',
    model='dtm',
    num_topics=20,
    language='zh'
)
"
```

**Generated charts** (20+ types):
- Topic word bars, word clouds, topic similarity heatmap
- Document clustering (UMAP), topic network graph
- Topic evolution (DTM), sankey diagrams
- Training convergence, coherence metrics
- pyLDAvis interactive HTML

**Output structure**:
```
visualization_k{K}_{lang}_{timestamp}/
├── global/                    # Global charts
│   ├── topic_table.png
│   ├── topic_network.png
│   ├── clustering_heatmap.png
│   ├── topic_wordclouds.png
│   └── ...
├── topics/                    # Per-topic charts
│   ├── topic_0/
│   ├── topic_1/
│   └── ...
└── README.md                  # Summary report
```

---

## Result Directory Structure

All results are organized using `ResultManager`:

```
/root/autodl-tmp/result/baseline/{dataset}/{model}/
├── bow/                    # BOW data and vocabulary
│   ├── bow_matrix.npz
│   ├── vocab.json
│   └── vocab.txt
├── model/                  # Model parameters
│   ├── theta_k{K}.npy
│   ├── beta_k{K}.npy
│   └── training_history_k{K}.json
├── evaluation/             # Evaluation results
│   ├── metrics_k{K}.json
│   └── metrics_k{K}.csv
├── topicwords/             # Topic words
│   ├── topic_words_k{K}.json
│   └── topic_evolution_k{K}.json
└── visualization_k{K}_{lang}_{timestamp}/
```

**Using ResultManager**:

```python
from utils.result_manager import ResultManager

# Initialize
manager = ResultManager(
    result_dir='/root/autodl-tmp/result/baseline',
    dataset='edu_data',
    model='dtm',
    num_topics=20
)

# Save all results
manager.save_all(theta, beta, vocab, topic_words, metrics=metrics)

# Load all results
data = manager.load_all(num_topics=20)

# Migrate old flat structure to new structure
from utils.result_manager import migrate_baseline_results
migrate_baseline_results(dataset='edu_data', model='dtm')
```

---

## Configuration

Dataset configurations are defined in `config.py`:

```python
DATASET_CONFIGS = {
    "socialTwitter": {
        "vocab_size": 5000,
        "num_topics": 20,
        "min_doc_freq": 5,
        "language": "multi",
    },
    "hatespeech": {
        "vocab_size": 8000,
        "num_topics": 20,
        "min_doc_freq": 10,
        "language": "english",
    },
    "edu_data": {
        "vocab_size": 5000,
        "num_topics": 20,
        "min_doc_freq": 3,
        "language": "chinese",
        "has_timestamp": True,
    },
}
```

**Command-line parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name | Required |
| `--models` | Model list (comma-separated) | Required |
| `--model_size` | Qwen model size (THETA) | 0.6B |
| `--mode` | THETA mode | zero_shot |
| `--num_topics` | Number of topics | 20 |
| `--epochs` | Training epochs | 100 |
| `--batch_size` | Batch size | 64 |
| `--language` | Visualization language | en |
| `--skip-train` | Skip training | False |
| `--skip-eval` | Skip evaluation | False |
| `--skip-viz` | Skip visualization | False |

---

## Supported Datasets

| Dataset | Documents | Language | Time-aware |
|---------|-----------|----------|------------|
| socialTwitter | ~40K | Spanish/English | No |
| hatespeech | ~437K | English | No |
| mental_health | ~1M | English | No |
| FCPB | ~854K | English | No |
| germanCoal | ~9K | German | No |
| edu_data | ~857 | Chinese | Yes |

---

## Roadmap

- v0.1: Unified dataset interface + zero-shot embeddings + ETM baseline
- v0.2: Multiple Qwen model sizes + coherence/perplexity reports
- v0.3: DTM topic evolution + bilingual visualizations
- v0.4: ResultManager + standardized output structure
- v1.0: Reproducible benchmark suite (datasets, baselines, downstream tasks)

---

## Citation

```bibtex
@software{theta_topic_analysis,
  title  = {THETA: Textual Hybrid Embedding-based Topic Analysis},
  author = {Duan, Zhenke and Pan, Jiqun and Li, Xin},
  year   = {2026}
}
```

---

## License

Apache-2.0

---

## Contributing

Contributions are welcome:

- New dataset adapters
- Topic visualization modules
- Evaluation and reproducibility scripts
- Documentation improvements

**Suggested workflow**:
1. Fork the repo and create a feature branch
2. Add a minimal reproducible example or tests
3. Open a pull request

---

## Ethics & Safety

This project analyzes social text and may involve sensitive content.

- Do not include personally identifiable information (PII)
- Ensure dataset usage complies with platform terms and research ethics
- Interpret outputs cautiously; topic discovery does not replace scientific conclusions
- Be responsible with sensitive domains such as self-harm, hate speech, and political polarization

---

## FAQ

**Q: Is this only for Qwen-3?**

A: No. Qwen-3 is the reference backbone, but THETA is designed to be model-agnostic. You can adapt it for other embedding models.

**Q: What is the difference between ETM and DTM?**

A: ETM learns static topics across the corpus; DTM (Dynamic Topic Model) models topic evolution over time and requires timestamps.

**Q: Can I add my own dataset?**

A: Yes. Prepare a cleaned CSV with `text` column (and optionally `year` for DTM), then add configuration to `config.py`:

```python
DATASET_CONFIGS["my_dataset"] = {
    "vocab_size": 5000,
    "num_topics": 20,
    "min_doc_freq": 5,
    "language": "english",
}
```

---

## Agent System

THETA includes an intelligent agent system for interactive analysis and Q&A:

### Features

- **LLM-powered Q&A**: Ask questions about your topic modeling results
- **Metric Interpretation**: Get human-readable explanations of evaluation metrics
- **Topic Interpretation**: Semantic analysis of discovered topics
- **Vision Analysis**: Analyze charts and visualizations using Qwen3-VL
- **Multi-turn Conversation**: Context-aware dialogue with session management

### Starting the Agent API

```bash
# Start the agent API server
bash scripts/07_start_agent_api.sh

# Or manually
cd /root/autodl-tmp
python -m agent.api
```

API will be available at `http://localhost:8000` with Swagger docs at `/docs`.

### Configuration

Create a `.env` file in the `agent/` directory:

```bash
# LLM Provider (deepseek, qwen, openai)
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Vision API (Qwen3-VL)
QWEN_VISION_API_KEY=your-dashscope-api-key
QWEN_VISION_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Simple Q&A chat |
| `/api/chat/v2` | POST | Multi-turn conversation |
| `/api/interpret/metrics` | POST | Interpret evaluation metrics |
| `/api/interpret/topics` | POST | Interpret topic semantics |
| `/api/interpret/summary` | POST | Generate analysis summary |
| `/api/vision/analyze` | POST | Analyze image with Qwen3-VL |
| `/api/vision/analyze-chart` | POST | Analyze chart from job results |

See `agent/docs/API_REFERENCE.md` for complete API documentation.

---

## Contact

Please contact us if you have any questions:
- duanzhenke@code-soul.com
- panjiqun@code-soul.com
- lixin@code-soul.com

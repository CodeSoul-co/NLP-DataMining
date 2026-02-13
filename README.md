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
├── embedding/                     # Qwen embedding generation
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

All scripts are **non-interactive** (pure command-line parameters), suitable for DLC/batch environments. No stdin input required:

| Script | Description |
|--------|-------------|
| `01_setup.sh` | Install dependencies and download data from HuggingFace |
| `02_clean_data.sh` | Clean raw text data (tokenization, stopword removal, lemmatization) |
| `02_generate_embeddings.sh` | Generate Qwen embeddings (sub-script of 03, for failure recovery) |
| `03_prepare_data.sh` | One-stop data preparation: BOW + embeddings for all 12 models |
| `04_train_theta.sh` | Train THETA model (train + evaluate + visualize) |
| `05_train_baseline.sh` | Train 11 baseline models for comparison with THETA |
| `06_visualize.sh` | Generate visualizations for trained models |
| `07_evaluate.sh` | Standalone evaluation with 7 unified metrics |
| `08_compare_models.sh` | Cross-model metric comparison table |
| `09_download_from_hf.sh` | Download pre-trained data from HuggingFace |
| `10_quick_start_english.sh` | Quick start for English datasets |
| `11_quick_start_chinese.sh` | Quick start for Chinese datasets |
| `12_train_multi_gpu.sh` | Multi-GPU training with DistributedDataParallel |
| `13_test_agent.sh` | Test LLM Agent connection and functionality |
| `14_start_agent_api.sh` | Start the Agent API server (FastAPI) |

---

## Quickstart

### Quick Start (One Command)

```bash
# English dataset — one-stop data prep + THETA training
bash scripts/10_quick_start_english.sh my_dataset

# Chinese dataset — one-stop data prep + THETA training (Chinese visualization)
bash scripts/11_quick_start_chinese.sh my_chinese_dataset
```

### End-to-End Pipeline (Step by Step)

```bash
# Step 1: Install dependencies
bash scripts/01_setup.sh

# Step 2: Clean raw data (preview columns first, then clean with explicit text column)
bash scripts/02_clean_data.sh --input data/edu_data/edu_data_raw.csv --preview
bash scripts/02_clean_data.sh --input data/edu_data/edu_data_raw.csv --language chinese --text_column cleaned_content

# Step 3: Prepare data (BOW + embeddings)
bash scripts/03_prepare_data.sh --dataset edu_data --model theta --model_size 0.6B --mode zero_shot --vocab_size 3500

# Step 4: Train THETA
bash scripts/04_train_theta.sh --dataset edu_data --model_size 0.6B --mode zero_shot --num_topics 20 --language zh

# Step 5: Train baselines for comparison
bash scripts/05_train_baseline.sh --dataset edu_data --models lda,prodlda,etm --num_topics 20 --epochs 100

# Step 6: Compare all models
bash scripts/08_compare_models.sh --dataset edu_data --models lda,prodlda,etm --num_topics 20
```

---

## Detailed Script Usage

### A) Data Cleaning — `02_clean_data.sh`

Row-by-row text cleaning with user-specified column selection. Two modes:
- **CSV mode**: User specifies `--text_column` (cleaned) and `--label_columns` (preserved as-is)
- **Directory mode**: Convert docx/txt files into a single cleaned CSV

**Supported languages**: `english`, `chinese`, `german`, `spanish`

```bash
# 1. Preview columns (recommended first step for CSV)
bash scripts/02_clean_data.sh \
    --input data/FCPB/complaints_text_only.csv --preview

# 2. Clean text column only
bash scripts/02_clean_data.sh \
    --input data/FCPB/complaints_text_only.csv \
    --language english \
    --text_column 'Consumer complaint narrative'

# 3. Clean text + keep label column
bash scripts/02_clean_data.sh \
    --input data/hatespeech/hatespeech_text_only.csv \
    --language english \
    --text_column cleaned_content --label_columns Label

# 4. Keep ALL columns, only clean the text column
bash scripts/02_clean_data.sh \
    --input raw.csv --language english \
    --text_column text --keep_all

# 5. Directory mode (docx/txt → CSV)
bash scripts/02_clean_data.sh \
    --input data/edu_data/ --language chinese
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--input` | ✓ | Input CSV file or directory (docx/txt) | - |
| `--language` | ✓ (not for preview) | Data language: english, chinese, german, spanish | - |
| `--text_column` | ✓ (CSV mode) | Name of the text column to clean | - |
| `--label_columns` | | Comma-separated label/metadata columns to keep as-is | - |
| `--keep_all` | | Keep ALL original columns (only text column is cleaned) | false |
| `--preview` | | Show CSV columns and sample rows, then exit | false |
| `--output` | | Output CSV path | auto-generated |
| `--min_words` | | Min words per document after cleaning | 3 |

**Output**: `data/{dataset}/{dataset}_cleaned.csv`

### B) Data Preparation — `03_prepare_data.sh`

One-stop data preparation for all 12 models. Generates BOW matrix and model-specific embeddings.

**Data requirements by model**:

| Model | Type | Data Needed |
|-------|------|-------------|
| lda, hdp, stm, btm | Traditional | BOW only |
| nvdm, gsm, prodlda | Neural | BOW only |
| etm | Neural | BOW + Word2Vec |
| ctm | Neural | BOW + SBERT |
| dtm | Neural | BOW + SBERT + time slices |
| bertopic | Neural | SBERT + raw text |
| theta | THETA | BOW + Qwen embeddings |

> **Note**: Models 1-7 (BOW-only) share the same data experiment. Prepare once, train all.

```bash
# ---- Baseline models ----

# BOW-only models (lda, hdp, stm, btm, nvdm, gsm, prodlda share this)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model lda --vocab_size 3500 --language chinese

# CTM (BOW + SBERT embeddings)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model ctm --vocab_size 3500 --language chinese

# ETM (BOW + Word2Vec embeddings)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model etm --vocab_size 3500 --language chinese

# DTM (BOW + SBERT + time slices, requires time column)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model dtm --vocab_size 3500 --language chinese --time_column year

# BERTopic (SBERT + raw text)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model bertopic --vocab_size 3500 --language chinese

# ---- THETA model ----

# Zero-shot (fastest, no training needed)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode zero_shot \
    --vocab_size 3500 --language chinese

# Unsupervised (LoRA fine-tuned Qwen embeddings)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode unsupervised \
    --vocab_size 3500 --language chinese

# Supervised (requires label column)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode supervised \
    --vocab_size 3500 --language chinese

# ---- Advanced options ----

# BOW only (skip embedding generation)
bash scripts/03_prepare_data.sh --dataset mydata --model theta --bow-only --vocab_size 5000

# Check if data files already exist
bash scripts/03_prepare_data.sh --dataset mydata --model theta --check-only

# Custom vocabulary size and max sequence length
bash scripts/03_prepare_data.sh --dataset mydata \
    --model theta --model_size 0.6B --mode zero_shot \
    --vocab_size 10000 --batch_size 64 --gpu 0
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--dataset` | ✓ | Dataset name | - |
| `--model` | ✓ | Target model: lda, hdp, stm, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta | - |
| `--model_size` | | Qwen model size (theta only): 0.6B, 4B, 8B | 0.6B |
| `--mode` | | Embedding mode (theta only): zero_shot, unsupervised, supervised | zero_shot |
| `--vocab_size` | | Vocabulary size | 5000 |
| `--batch_size` | | Embedding generation batch size | 32 |
| `--gpu` | | GPU device ID | 0 |
| `--language` | | Data language: english, chinese (controls tokenization) | english |
| `--bow-only` | | Only generate BOW, skip embeddings | false |
| `--check-only` | | Only check if files exist | false |
| `--time_column` | | Time column name (DTM only) | year |
| `--label_column` | | Label column (theta supervised only) | - |
| `--emb_epochs` | | Embedding fine-tuning epochs (theta only) | 10 |
| `--emb_batch_size` | | Embedding fine-tuning batch size (theta only) | 8 |
| `--exp_name` | | Experiment name tag | auto-generated |

**Embedding recovery** — If embedding generation fails (e.g., OOM), re-run only the embedding step:

```bash
bash scripts/02_generate_embeddings.sh \
    --dataset edu_data --mode zero_shot --model_size 0.6B \
    --batch_size 4 --exp_dir result/0.6B/edu_data/data/exp_xxx
```

### C) THETA Model Training — `04_train_theta.sh`

Train THETA model with integrated training + evaluation + visualization.

```bash
# ---- Basic usage ----

# Zero-shot mode (simplest command)
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot --num_topics 20

# Unsupervised mode
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode unsupervised --num_topics 20

# Supervised mode (requires label column)
bash scripts/04_train_theta.sh \
    --dataset hatespeech --model_size 0.6B --mode supervised --num_topics 20

# Larger model for better quality
bash scripts/04_train_theta.sh \
    --dataset hatespeech --model_size 4B --mode zero_shot --num_topics 20

# ---- Full parameters ----

bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --num_topics 20 --epochs 100 --batch_size 64 \
    --hidden_dim 512 --learning_rate 0.002 \
    --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 \
    --patience 10 --gpu 0 --language zh

# Custom KL annealing
bash scripts/04_train_theta.sh \
    --dataset hatespeech --model_size 0.6B --mode zero_shot \
    --num_topics 20 --epochs 200 \
    --kl_start 0.1 --kl_end 0.8 --kl_warmup 40

# ---- Specify data experiment ----

# Use a specific preprocessed data experiment
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --data_exp exp_20260208_151906_vocab3500_theta_0.6B_zero_shot \
    --num_topics 20 --epochs 50 --language zh

# ---- Skip options ----

# Skip visualization (train + evaluate only, faster)
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --num_topics 20 --skip-viz

# Skip training (evaluate + visualize existing model)
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --skip-train --language zh
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--dataset` | ✓ | Dataset name | - |
| `--model_size` | | Qwen model size: 0.6B, 4B, 8B | 0.6B |
| `--mode` | | Embedding mode: zero_shot, unsupervised, supervised | zero_shot |
| `--num_topics` | | Number of topics K | 20 |
| `--epochs` | | Training epochs | 100 |
| `--batch_size` | | Training batch size | 64 |
| `--hidden_dim` | | Encoder hidden dimension | 512 |
| `--learning_rate` | | Learning rate | 0.002 |
| `--kl_start` | | KL annealing start weight | 0.0 |
| `--kl_end` | | KL annealing end weight | 1.0 |
| `--kl_warmup` | | KL warmup epochs | 50 |
| `--patience` | | Early stopping patience | 10 |
| `--gpu` | | GPU device ID | 0 |
| `--language` | | Visualization language: en, zh | en |
| `--skip-train` | | Skip training, only evaluate | false |
| `--skip-viz` | | Skip visualization | false |
| `--data_exp` | | Data experiment ID | auto latest |
| `--exp_name` | | Experiment name tag | auto-generated |

### D) Baseline Model Training — `05_train_baseline.sh`

Train 11 baseline topic models for comparison with THETA.

```bash
# ============================================================
# 1. LDA — Latent Dirichlet Allocation
#    Type: Traditional | Data: BOW only
#    Specific params: --max_iter
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models lda --num_topics 20
# Full:
bash scripts/05_train_baseline.sh --dataset edu_data --models lda \
    --num_topics 20 --max_iter 200 --gpu 0 --language zh --with-viz

# ============================================================
# 2. HDP — Hierarchical Dirichlet Process
#    Type: Traditional | Data: BOW only
#    Note: Auto topic count, --num_topics is IGNORED
#    Specific params: --max_topics, --alpha
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models hdp
# Full:
bash scripts/05_train_baseline.sh --dataset edu_data --models hdp \
    --max_topics 150 --alpha 1.0 --gpu 0 --language zh --with-viz

# ============================================================
# 3. STM — Structural Topic Model
#    Type: Traditional | Data: BOW only
#    Specific params: --max_iter
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models stm --num_topics 20
# Full:
bash scripts/05_train_baseline.sh --dataset edu_data --models stm \
    --num_topics 20 --max_iter 200 --gpu 0 --language zh --with-viz

# ============================================================
# 4. BTM — Biterm Topic Model (best for short texts)
#    Type: Traditional | Data: BOW only
#    Specific params: --n_iter, --alpha, --beta
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models btm --num_topics 20
# Full:
bash scripts/05_train_baseline.sh --dataset edu_data --models btm \
    --num_topics 20 --n_iter 100 --alpha 1.0 --beta 0.01 --gpu 0 --language zh --with-viz

# ============================================================
# 5. NVDM — Neural Variational Document Model
#    Type: Neural | Data: BOW only
#    Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate, --dropout
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models nvdm --num_topics 20
# Full:
bash scripts/05_train_baseline.sh --dataset edu_data --models nvdm \
    --num_topics 20 --epochs 200 --batch_size 128 --hidden_dim 512 \
    --learning_rate 0.002 --dropout 0.2 --gpu 0 --language zh --with-viz

# ============================================================
# 6. GSM — Gaussian Softmax Model
#    Type: Neural | Data: BOW only
#    Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate, --dropout
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models gsm --num_topics 20
# Full:
bash scripts/05_train_baseline.sh --dataset edu_data --models gsm \
    --num_topics 20 --epochs 200 --batch_size 128 --hidden_dim 512 \
    --learning_rate 0.002 --dropout 0.2 --gpu 0 --language zh --with-viz

# ============================================================
# 7. ProdLDA — Product of Experts LDA
#    Type: Neural | Data: BOW only
#    Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate, --dropout
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models prodlda --num_topics 20
# Full:
bash scripts/05_train_baseline.sh --dataset edu_data --models prodlda \
    --num_topics 20 --epochs 200 --batch_size 128 --hidden_dim 512 \
    --learning_rate 0.002 --dropout 0.2 --gpu 0 --language zh --with-viz

# ============================================================
# 8. CTM — Contextualized Topic Model (requires SBERT data_exp)
#    Type: Neural | Data: BOW + SBERT embeddings
#    Specific params: --epochs, --inference_type (zeroshot | combined)
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models ctm --num_topics 20
# Zeroshot inference:
bash scripts/05_train_baseline.sh --dataset edu_data --models ctm \
    --num_topics 20 --epochs 100 --inference_type zeroshot \
    --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language zh --with-viz
# Combined inference:
bash scripts/05_train_baseline.sh --dataset edu_data --models ctm \
    --num_topics 20 --epochs 100 --inference_type combined --gpu 0 --with-viz

# ============================================================
# 9. ETM — Embedded Topic Model (uses Word2Vec from BOW data_exp)
#    Type: Neural | Data: BOW + Word2Vec embeddings
#    Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models etm --num_topics 20
# Full:
bash scripts/05_train_baseline.sh --dataset edu_data --models etm \
    --num_topics 20 --epochs 200 --batch_size 64 --hidden_dim 512 \
    --learning_rate 0.002 --gpu 0 --language zh --with-viz

# ============================================================
# 10. DTM — Dynamic Topic Model (requires time_slices data_exp)
#     Type: Neural | Data: BOW + SBERT + time slices
#     Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models dtm --num_topics 20
# Full:
bash scripts/05_train_baseline.sh --dataset edu_data --models dtm \
    --num_topics 20 --epochs 200 --batch_size 64 --hidden_dim 512 \
    --learning_rate 0.002 --gpu 0 --language zh --with-viz

# ============================================================
# 11. BERTopic — BERT-based Topic Model (requires SBERT data_exp)
#     Type: Neural | Data: SBERT + raw text
#     Note: Auto topic count, --num_topics is IGNORED
#     Can reuse CTM's SBERT data_exp
# ============================================================
bash scripts/05_train_baseline.sh --dataset edu_data --models bertopic
# With explicit data_exp:
bash scripts/05_train_baseline.sh --dataset edu_data --models bertopic \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_154645_vocab3500_ctm

# ============================================================
# Batch training (multiple models at once)
# ============================================================

# All BOW-only models (share the same data_exp)
bash scripts/05_train_baseline.sh --dataset edu_data \
    --models lda,hdp,stm,btm,nvdm,gsm,prodlda \
    --num_topics 20 --epochs 100

# Models needing special data (train separately)
bash scripts/05_train_baseline.sh --dataset edu_data --models etm --num_topics 20 --epochs 100
bash scripts/05_train_baseline.sh --dataset edu_data --models ctm,bertopic --num_topics 20 --epochs 100
bash scripts/05_train_baseline.sh --dataset edu_data --models dtm --num_topics 20 --epochs 100

# ============================================================
# Skip / visualization options
# ============================================================

# Skip training, only evaluate existing model
bash scripts/05_train_baseline.sh --dataset edu_data --models lda --num_topics 20 --skip-train

# Enable visualization (disabled by default for speed)
bash scripts/05_train_baseline.sh --dataset edu_data --models lda --num_topics 20 --with-viz --language zh
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--dataset` | ✓ | Dataset name | - |
| `--models` | ✓ | Model list (comma-separated) | - |
| `--num_topics` | Number of topics (ignored for hdp/bertopic) | 20 |
| `--vocab_size` | Vocabulary size | 5000 |
| `--epochs` | Training epochs (neural models) | 100 |
| `--batch_size` | Batch size | 64 |
| `--hidden_dim` | Hidden dimension | 512 |
| `--learning_rate` | Learning rate | 0.002 |
| `--gpu` | GPU device ID | 0 |
| `--language` | Visualization language: en, zh | en |
| `--skip-train` | Skip training | false |
| `--skip-viz` | Skip visualization (default) | true |
| `--with-viz` | Enable visualization | false |
| `--data_exp` | Data experiment ID | auto latest |
| `--exp_name` | Experiment name tag | auto-generated |

**Model-specific parameters**:

| Parameter | Models | Description | Default |
|-----------|--------|-------------|---------|
| `--max_iter` | lda, stm | Max iterations | 100 |
| `--max_topics` | hdp | Max topic count | 150 |
| `--n_iter` | btm | Gibbs sampling iterations | 100 |
| `--alpha` | hdp, btm | Alpha prior | 1.0 |
| `--beta` | btm | Beta prior | 0.01 |
| `--inference_type` | ctm | Inference type: zeroshot, combined | zeroshot |
| `--dropout` | neural models | Dropout rate | 0.2 |

### E) Visualization — `06_visualize.sh`

Generate visualizations for trained models without re-training.

```bash
# ---- THETA model ----

bash scripts/06_visualize.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot --language zh

bash scripts/06_visualize.sh \
    --dataset edu_data --model_size 0.6B --mode unsupervised --language en --dpi 600

# ---- Baseline models (all 11) ----

bash scripts/06_visualize.sh --baseline --dataset edu_data --model lda --num_topics 20 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model hdp --num_topics 150 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model stm --num_topics 20 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model btm --num_topics 20 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model nvdm --num_topics 20 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model gsm --num_topics 20 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model prodlda --num_topics 20 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model ctm --num_topics 20 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model etm --num_topics 20 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model dtm --num_topics 20 --language zh
bash scripts/06_visualize.sh --baseline --dataset edu_data --model bertopic --num_topics 20 --language zh

# Specify model experiment explicitly
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model ctm --model_exp exp_20260208_xxx --language zh

# High DPI for publications
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model lda --num_topics 20 --language en --dpi 600
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | — |
| `--baseline` | Baseline model mode | false |
| `--model` | Baseline model name | — |
| `--model_exp` | Model experiment ID | auto latest |
| `--model_size` | THETA model size | 0.6B |
| `--mode` | THETA mode | zero_shot |
| `--language` | Visualization language: en, zh | en |
| `--dpi` | Image DPI | 300 |

**Generated charts** (20+ types): topic word bars, word clouds, topic similarity heatmap, document clustering (UMAP), topic network graph, topic evolution (DTM), training convergence, coherence metrics, pyLDAvis interactive HTML, per-topic word importance, and more.

### F) Evaluation — `07_evaluate.sh`

Standalone evaluation with 7 unified metrics.

```bash
# Evaluate all 11 baseline models
bash scripts/07_evaluate.sh --dataset edu_data --model lda --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model hdp --num_topics 150
bash scripts/07_evaluate.sh --dataset edu_data --model stm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model btm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model nvdm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model gsm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model prodlda --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model ctm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model etm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model dtm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model bertopic --num_topics 20

# Evaluate THETA model
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 0.6B --mode zero_shot
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 0.6B --mode unsupervised
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 4B --mode supervised
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | — |
| `--model` | Model name (required) | — |
| `--num_topics` | Number of topics | 20 |
| `--vocab_size` | Vocabulary size | 5000 |
| `--baseline` | Baseline model mode | false |
| `--model_size` | THETA model size: 0.6B, 4B, 8B | 0.6B |
| `--mode` | THETA mode: zero_shot, unsupervised, supervised | zero_shot |

### G) Model Comparison — `08_compare_models.sh`

Cross-model metric comparison table.

```bash
# Compare all baseline models
bash scripts/08_compare_models.sh \
    --dataset edu_data \
    --models lda,hdp,stm,btm,nvdm,gsm,prodlda,ctm,etm,dtm,bertopic \
    --num_topics 20

# Compare specific models
bash scripts/08_compare_models.sh \
    --dataset edu_data --models lda,prodlda,ctm --num_topics 20

# Export to CSV
bash scripts/08_compare_models.sh \
    --dataset edu_data --models lda,hdp,nvdm,gsm,prodlda,ctm,etm,stm,dtm \
    --num_topics 20 --output comparison.csv
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | — |
| `--models` | Comma-separated model list (required) | — |
| `--num_topics` | Number of topics | 20 |
| `--output` | Output CSV file path | terminal only |

### H) Multi-GPU Training — `12_train_multi_gpu.sh`

THETA supports multi-GPU training using PyTorch DistributedDataParallel (DDP).

```bash
# Train with 2 GPUs
bash scripts/12_train_multi_gpu.sh --dataset hatespeech --num_gpus 2 --num_topics 20

# Full parameters
bash scripts/12_train_multi_gpu.sh --dataset hatespeech \
    --num_gpus 4 --model_size 0.6B --mode zero_shot \
    --num_topics 25 --epochs 150 --batch_size 64 \
    --hidden_dim 768 --learning_rate 0.001

# Custom master port (for multiple concurrent jobs)
bash scripts/12_train_multi_gpu.sh --dataset socialTwitter \
    --num_gpus 2 --master_port 29501

# Or use torchrun directly
torchrun --nproc_per_node=2 --master_port=29500 \
    ETM/main.py train \
    --dataset hatespeech --mode zero_shot --num_topics 20 --epochs 100
```

### I) Agent API — `14_start_agent_api.sh`

Start the AI Agent API server for interactive analysis and Q&A.

```bash
# Start agent API (default port 8000)
bash scripts/14_start_agent_api.sh --port 8000

# Test agent connection
bash scripts/13_test_agent.sh
```

API endpoints: `POST /chat`, `POST /api/chat/v2`, `POST /api/interpret/metrics`, `POST /api/interpret/topics`, `POST /api/vision/analyze`. See `agent/docs/API_REFERENCE.md` for details.

### J) Batch Processing Examples

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

# Generate visualizations for all trained baseline models
for model in lda etm ctm prodlda; do
    bash scripts/06_visualize.sh --baseline --dataset hatespeech \
        --model $model --num_topics 20 --language en
done
```

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

THETA includes an intelligent agent system built on **LangChain + LangGraph**, providing:

### Features

- **LangChain ReAct Agent**: Autonomous tool-calling agent that can execute the full pipeline (clean → prepare → train → evaluate → visualize) via natural language
- **11 Built-in Tools**: `list_datasets`, `list_experiments`, `clean_data`, `prepare_data`, `train_theta`, `train_baseline`, `visualize`, `evaluate_model`, `compare_models`, `get_training_results`, `list_visualizations`
- **Multi-provider LLM**: Supports DeepSeek, Qwen, OpenAI via unified `ChatOpenAI` interface
- **Metric Interpretation**: Human-readable explanations of evaluation metrics
- **Topic Interpretation**: Semantic analysis of discovered topics
- **Vision Analysis**: Analyze charts using Qwen3-VL
- **Multi-turn Conversation**: Session-based dialogue with context management
- **Streaming**: SSE streaming responses for real-time feedback

### Starting the Agent API

```bash
# Start the agent API server
bash scripts/14_start_agent_api.sh

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

# Or use Qwen
# LLM_PROVIDER=qwen
# DASHSCOPE_API_KEY=your-dashscope-api-key

# Vision API (Qwen3-VL)
QWEN_VISION_API_KEY=your-dashscope-api-key
QWEN_VISION_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# LLM Settings
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
LLM_TIMEOUT=120
```

### Python Usage

```python
from agent import THETAAgent

# Create agent (reads config from .env)
agent = THETAAgent(provider="deepseek", temperature=0.3)

# Chat with the agent
response = agent.chat("列出所有可用的数据集")
print(response)

# Multi-turn conversation
response = agent.chat("用 edu_data 训练一个 LDA 模型，20 个主题", session_id="s1")
response = agent.chat("训练结果怎么样？", session_id="s1")
```

### API Endpoints

**LangChain Agent (v3 — recommended)**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agent/chat` | POST | Agent chat (auto tool-calling) |
| `/api/agent/chat/stream` | POST | Agent chat with SSE streaming |
| `/api/agent/sessions` | GET | List active sessions |
| `/api/agent/sessions/{id}` | DELETE | Clear session history |
| `/api/agent/tools` | GET | List available tools |

**Legacy endpoints** (still available):

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

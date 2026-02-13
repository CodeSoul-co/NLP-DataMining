<div align="center">

<img src="assets/THETA.png" width="40%" alt="THETA Logo"/>

<h1>THETA (θ)</h1>

Textual Hybrid Embedding–based Topic Analysis

</div>

## Overview

THETA (θ) is an open-source, research-oriented platform for LLM-enhanced topic analysis in social science. It combines:

- Domain-adaptive document embeddings from Qwen-3 models (0.6B / 4B / 8B)
  - Zero-shot embedding (no training)
  - Supervised fine-tuning (with labels)
  - Unsupervised fine-tuning (SimCSE, no labels)
- 12 topic models for benchmarking:
  - **THETA**: Main model using Qwen embeddings
  - **Traditional**: LDA, HDP (auto topics), STM (covariates), BTM (short texts)
  - **Neural**: ETM, CTM, DTM (time-aware), NVDM, GSM, ProdLDA, BERTopic
- Scientific validation via 7 intrinsic metrics (PPL, TD, iRBO, NPMI, C_V, UMass, Exclusivity)
- Comprehensive visualization with bilingual support (English / Chinese)
- Fully non-interactive CLI — all scripts accept command-line parameters only, suitable for batch / DLC environments

---

## Key Features

- **Hybrid embedding topic analysis**: Zero-shot / Supervised / Unsupervised modes
- **Multiple Qwen model sizes**: 0.6B (1024-dim), 4B (2560-dim), 8B (4096-dim)
- **12 baseline models**: LDA, HDP, STM, BTM, ETM, CTM, DTM, NVDM, GSM, ProdLDA, BERTopic
- **Data governance**: Domain-aware cleaning for multiple languages (English, Chinese, German, Spanish)
- **Unified evaluation**: 7 metrics with JSON/CSV export
- **Rich visualization**: 20+ chart types with bilingual labels
- **Non-interactive CLI**: All scripts are pure command-line, no stdin prompts

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

---

## Project Core Structure

```
/THETA/
├── ETM/                            # Main codebase
│   ├── run_pipeline.py             # Unified entry point (train + eval + viz)
│   ├── prepare_data.py             # Data preprocessing (BOW + embeddings)
│   ├── config.py                   # Configuration management
│   ├── experiment_manager.py       # Experiment directory management
│   ├── dataclean/                  # Data cleaning module
│   ├── model/                      # Model implementations
│   │   ├── theta/                  # THETA main model (Qwen + ETM)
│   │   ├── baselines/              # 11 baseline models
│   │   └── vocab_embedder.py       # Vocabulary embedding generator
│   ├── evaluation/                 # Unified evaluation (7 metrics)
│   │   └── unified_evaluator.py    # TD, iRBO, NPMI, C_V, UMass, Excl, PPL
│   ├── visualization/              # Visualization tools (20+ chart types)
│   │   └── run_visualization.py    # Bilingual visualization entry point
│   ├── models_config/              # Model configuration registry
│   │   └── model_config.py         # Data requirements per model
│   └── utils/                      # Utilities
│       └── result_manager.py       # Result directory management
├── embedding/                      # Qwen embedding generation
│   ├── main.py                     # Embedding generation entry point
│   ├── embedder.py                 # Zero-shot / fine-tuned embedding
│   ├── trainer.py                  # LoRA fine-tuning (supervised/unsupervised)
│   └── data_loader.py              # Dataset loader
├── scripts/                        # Shell scripts (all non-interactive)
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

---

## Quickstart

### Quick Start (One Command)

```bash
# English dataset
bash scripts/10_quick_start_english.sh my_dataset

# Chinese dataset (Chinese visualization labels)
bash scripts/11_quick_start_chinese.sh my_chinese_dataset
```

### Step-by-Step Pipeline

```bash
cd /your_file_path
S=scripts

# Step 1: Install dependencies
bash $S/01_setup.sh

# Step 2: Clean raw data (preview columns first, then clean)
bash $S/02_clean_data.sh --input data/hatespeech/hatespeech_text_only.csv --preview
bash $S/02_clean_data.sh --input data/hatespeech/hatespeech_text_only.csv \
    --language english --text_column cleaned_content --label_columns Label --keep_all

# Step 3a: Prepare THETA data (BOW + Qwen embeddings)
bash $S/03_prepare_data.sh --dataset hatespeech --model theta \
    --model_size 4B --mode zero_shot --vocab_size 5000 --gpu 0 --language english

# Step 3b: Prepare baseline data (BOW for traditional/neural models)
bash $S/03_prepare_data.sh --dataset hatespeech --model lda --vocab_size 5000 --language english

# Step 4: Train THETA (includes training + evaluation + visualization)
bash $S/04_train_theta.sh --dataset hatespeech --model_size 4B --mode zero_shot \
    --num_topics 20 --epochs 50 --gpu 0 --language en

# Step 5: Train baselines
bash $S/05_train_baseline.sh --dataset hatespeech --models lda \
    --num_topics 20 --vocab_size 5000 --language en --with-viz
bash $S/05_train_baseline.sh --dataset hatespeech --models prodlda \
    --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 0 --language en --with-viz

# Step 6: Compare models
bash $S/08_compare_models.sh --dataset hatespeech \
    --models lda,prodlda --num_topics 20
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

### I) Batch Processing Examples

```bash
cd /root/autodl-tmp
S=scripts

# Train THETA 4B on all 5 datasets (each with its own mode)
for ds_mode in "FCPB zero_shot" "FCPB unsupervised" \
               "germanCoal zero_shot" "germanCoal unsupervised" \
               "hatespeech zero_shot" "hatespeech supervised" \
               "mental_health zero_shot" "mental_health supervised" \
               "socialTwitter zero_shot" "socialTwitter supervised"; do
    set -- $ds_mode
    bash $S/04_train_theta.sh --dataset $1 --model_size 4B --mode $2 \
        --num_topics 20 --epochs 50 --gpu 0 --language en
done

# Train all traditional baselines (CPU, no GPU needed)
for model in lda stm btm; do
    for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
        bash $S/05_train_baseline.sh --dataset $ds --models $model \
            --num_topics 20 --vocab_size 5000 --language en --with-viz
    done
done

# HDP (auto topic number, no --num_topics)
for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
    bash $S/05_train_baseline.sh --dataset $ds --models hdp \
        --vocab_size 5000 --language en --with-viz
done

# Train all neural baselines (GPU required, --epochs needed)
for model in nvdm gsm prodlda etm ctm; do
    for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
        bash $S/05_train_baseline.sh --dataset $ds --models $model \
            --num_topics 20 --epochs 50 --vocab_size 5000 --gpu 0 --language en --with-viz
    done
done

# BERTopic (auto topic number, no --num_topics or --epochs)
for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
    bash $S/05_train_baseline.sh --dataset $ds --models bertopic \
        --vocab_size 5000 --gpu 0 --language en --with-viz
done

# Compare different topic numbers for THETA
for k in 10 15 20 25 30; do
    bash $S/04_train_theta.sh --dataset hatespeech \
        --model_size 4B --mode zero_shot --num_topics $k --epochs 50 --gpu 0
done

# Visualize all trained baselines for a dataset
for model in lda hdp stm btm nvdm gsm prodlda etm ctm bertopic; do
    bash $S/06_visualize.sh --baseline --dataset hatespeech \
        --model $model --num_topics 20 --language en
done

# Evaluate all baselines for a dataset
for model in lda hdp stm btm nvdm gsm prodlda etm ctm bertopic; do
    bash $S/07_evaluate.sh --dataset hatespeech --model $model \
        --num_topics 20 --baseline
done

# Compare all models on all datasets
for ds in FCPB germanCoal hatespeech mental_health socialTwitter; do
    bash $S/08_compare_models.sh --dataset $ds \
        --models lda,hdp,stm,btm,nvdm,gsm,prodlda,etm,ctm,bertopic --num_topics 20
done
```

---

## Evaluation Metrics

THETA provides unified evaluation with 7 intrinsic metrics:

| Metric | Direction | Description |
|--------|-----------|-------------|
| TD | Higher is better | Topic Diversity — ratio of unique words across topics |
| iRBO | Higher is better | Inverse Rank-Biased Overlap — penalizes redundant topics |
| NPMI | Higher is better | Normalized PMI coherence — word co-occurrence quality |
| C_V | Higher is better | C_V coherence — sliding window coherence |
| UMass | Closer to 0 | UMass coherence — document co-occurrence |
| Exclusivity | Higher is better | Topic exclusivity — words unique to each topic |
| PPL | Lower is better | Perplexity — model fit to held-out data |

**Evaluation outputs**:
- `evaluation/metrics_k{K}.json` — all metrics in JSON
- `evaluation/metrics_k{K}.csv` — all metrics in CSV

---

## Visualization

20+ chart types with bilingual support (English / Chinese):

- Topic word bars, word clouds, topic similarity heatmap
- Document clustering (UMAP), topic network graph
- Topic evolution (DTM only), sankey diagrams
- Training convergence curves, coherence metric plots
- pyLDAvis interactive HTML

**Output structure**:
```
visualization_k{K}_{lang}_{timestamp}/
├── global/                    # Cross-topic charts
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

**THETA model results**:
```
result/{model_size}/{dataset}/
├── data/exp_*/                # Preprocessed data (BOW + embeddings)
│   ├── bow/
│   │   ├── bow_matrix.npy
│   │   ├── vocab.json
│   │   └── vocab_embeddings.npy
│   └── embeddings/
│       └── embeddings.npy
├── models/exp_*/              # Trained model outputs
│   ├── model/
│   │   ├── theta_k{K}.npy    # Document-topic distribution (N × K)
│   │   ├── beta_k{K}.npy     # Topic-word distribution (K × V)
│   │   └── training_history_k{K}.json
│   ├── evaluation/
│   │   └── metrics_k{K}.json
│   ├── topicwords/
│   │   └── topic_words_k{K}.json
│   └── visualization_k{K}_{lang}_{timestamp}/
```

**Baseline model results**:
```
result/baseline/{dataset}/
├── data/exp_*/                # Preprocessed data
│   ├── bow_matrix.npy
│   ├── vocab.json
│   └── sbert_embeddings.npy   # CTM/BERTopic only
├── models/{model}/exp_*/      # Per-model outputs
│   ├── theta_k{K}.npy
│   ├── beta_k{K}.npy
│   ├── metrics_k{K}.json
│   └── visualization_k{K}_{lang}_{timestamp}/
```

---

## Supported Datasets

| Dataset | Documents | Language | THETA 4B Modes |
|---------|-----------|----------|----------------|
| FCPB | ~854K | English | zero_shot, unsupervised |
| germanCoal | ~9K | German | zero_shot, unsupervised |
| hatespeech | ~437K | English | zero_shot, supervised |
| mental_health | ~1M | English | zero_shot, supervised |
| socialTwitter | ~40K | English/Spanish | zero_shot, supervised |

**Adding a custom dataset**: Place a cleaned CSV with a `text` column (and optionally `label`, `year`) in `data/{dataset_name}/`, then use the scripts directly with `--dataset {dataset_name}`.

---

## Qwen Embedding Sizes

| Model Size | Embedding Dim | VRAM (approx.) | Use Case |
|------------|---------------|----------------|----------|
| 0.6B | 1024 | ~2 GB | Fast prototyping |
| 4B | 2560 | ~10 GB | Balanced quality/speed |
| 8B | 4096 | ~20 GB | Best quality |

**Embedding modes**:
- `zero_shot` — use pre-trained Qwen directly (no fine-tuning)
- `supervised` — LoRA fine-tuning with classification loss (requires `--label_column`)
- `unsupervised` — LoRA fine-tuning with SimCSE / autoregressive LM loss

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

A: Yes. Place a cleaned CSV with a `text` column in `data/my_dataset/`, then run:

```bash
bash scripts/03_prepare_data.sh --dataset my_dataset --model theta \
    --model_size 4B --mode zero_shot --vocab_size 5000 --language english --gpu 0
bash scripts/04_train_theta.sh --dataset my_dataset --model_size 4B \
    --mode zero_shot --num_topics 20 --epochs 50 --gpu 0 --language en
```

---

## Contact

Please contact us if you have any questions:
- duanzhenke@code-soul.com
- panjiqun@code-soul.com
- lixin@code-soul.com

# THETA Scripts Usage Guide

> This document provides detailed instructions for all scripts in the `scripts/` directory, including parameters, data flow, and output paths. Examples use `edu_data` (Chinese education dataset, 823 documents) for end-to-end demonstration.
>
> **⚠️ All scripts run in pure CLI mode (non-interactive), suitable for DLC/batch environments. All parameters are specified via command-line arguments — no stdin input required.**

---

## Table of Contents

- [Overview: Full Pipeline](#overview-full-pipeline)
- [Directory Structure](#directory-structure)
- [Step 0: Setup — 01_setup.sh](#step-0-setup)
- [Step 1: Data Cleaning — 02_clean_data.sh](#step-1-data-cleaning)
- [Step 2: Data Preparation — 03_prepare_data.sh](#step-2-data-preparation) (main entry, all-in-one)
  - [Appendix: Re-run Embeddings — 02_generate_embeddings.sh](#appendix-re-run-embeddings) (sub-script of 03, for recovery)
- [Step 3: THETA Training — 04_train_theta.sh](#step-3-theta-training)
- [Step 4: Baseline Training — 05_train_baseline.sh](#step-4-baseline-training)
- [Step 5: Standalone Visualization — 06_visualize.sh](#step-5-standalone-visualization)
- [Step 6: Standalone Evaluation — 07_evaluate.sh](#step-6-standalone-evaluation)
- [Step 7: Model Comparison — 08_compare_models.sh](#step-7-model-comparison)
- [Utility Scripts](#utility-scripts)
- [End-to-End Example: edu_data](#end-to-end-example-edu_data)
- [FAQ](#faq)

---

## Overview: Full Pipeline

```
Raw Data (CSV/DOCX)
    │
    ▼
┌─────────────────────┐
│  01_setup.sh        │  Install dependencies, download pretrained data
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  02_clean_data.sh   │  Text cleaning (tokenization, stopwords, lemmatization)
└─────────────────────┘
    │
    ▼  data/{dataset}/{dataset}_cleaned.csv
┌─────────────────────┐
│  03_prepare_data.sh │  Generate BOW + Embeddings (all-in-one)
└─────────────────────┘
    │
    ├── Baseline models ──▶ result/baseline/{dataset}/data/exp_xxx/
    │                         ├── bow_matrix.npy, vocab.json
    │                         ├── sbert_embeddings.npy  (ctm/dtm/bertopic)
    │                         ├── word2vec_embeddings.npy (etm)
    │                         └── time_slices.json (dtm)
    │
    └── THETA model ──────▶ result/{model_size}/{dataset}/data/exp_xxx/
                              ├── bow/  (bow_matrix.npy, vocab.json, vocab_embeddings.npy)
                              └── embeddings/  (embeddings.npy, metadata.json)
    │
    ▼
┌─────────────────────┐     ┌──────────────────────┐
│  04_train_theta.sh  │     │  05_train_baseline.sh │
│  (THETA: train +    │     │  (11 baseline models) │
│   eval + visualize) │     │                      │
└─────────────────────┘     └──────────────────────┘
    │                            │
    ▼                            ▼
result/{size}/{dataset}/     result/baseline/{dataset}/
  models/exp_xxx/              models/{model}/exp_xxx/
    │
    ▼
┌─────────────────────┐     ┌──────────────────────┐
│  06_visualize.sh    │     │  07_evaluate.sh       │
│  (standalone viz)   │     │  (standalone eval)    │
└─────────────────────┘     └──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  08_compare_models.sh│  Cross-model metric comparison
└──────────────────────┘
```

---

## Directory Structure

### Raw Data Directory

```
data/
├── edu_data/
│   ├── edu_data_cleaned.csv          # Cleaned CSV (must have a 'text' column)
│   └── edu_data_raw.csv              # Raw data (optional)
├── hatespeech/
├── mental_health/
├── germanCoal/
├── socialTwitter/
└── FCPB/
```

### Result Directory (Experiment Structure)

**THETA model**:
```
result/{model_size}/{dataset}/
├── data/                                          # Data experiments (preprocessing outputs)
│   └── exp_{timestamp}_{exp_name}/
│       ├── config.json                            # Experiment configuration
│       ├── bow/
│       │   ├── bow_matrix.npy                     # BOW matrix (n_docs × vocab_size)
│       │   ├── vocab.json                         # Vocabulary {word: index}
│       │   ├── vocab.txt                          # Vocabulary text
│       │   └── vocab_embeddings.npy               # Vocab embeddings (vocab_size × emb_dim)
│       └── embeddings/
│           ├── embeddings.npy                     # Document embeddings (n_docs × emb_dim)
│           └── metadata.json                      # Embedding metadata
│
└── models/                                        # Model experiments (training outputs)
    └── exp_{timestamp}_{exp_name}/
        ├── config.json                            # Training configuration
        ├── model/
        │   ├── etm_model_{ts}.pt                  # Model weights
        │   ├── theta_{ts}.npy                     # Doc-topic distribution (n_docs × K)
        │   ├── beta_{ts}.npy                      # Topic-word distribution (K × vocab_size)
        │   ├── topic_words_{ts}.json              # Topic word list
        │   ├── topic_embeddings_{ts}.npy          # Topic embeddings
        │   ├── training_history_{ts}.json         # Training history
        │   └── config_{ts}.json                   # Run configuration
        ├── evaluation/
        │   └── metrics_{ts}.json                  # Evaluation metrics (7 metrics)
        ├── topic_words/
        │   ├── topic_words_{ts}.json              # Topic words JSON
        │   └── topic_words_{ts}.txt               # Topic words text
        └── visualization/
            └── viz_{ts}/
                ├── README.md                      # Visualization summary report
                ├── global/                        # Global charts
                │   ├── topic_table.png
                │   ├── topic_network.png
                │   ├── doc_topic_umap.png
                │   ├── training_loss.png
                │   ├── metrics.png
                │   ├── topic_wordclouds.png
                │   ├── topic_similarity.png
                │   ├── pyldavis_interactive.html  # Interactive pyLDAvis
                │   └── ...
                └── topics/                        # Per-topic charts
                    ├── topic_1/word_importance.png
                    ├── topic_2/word_importance.png
                    └── ...
```

**Baseline models**:
```
result/baseline/{dataset}/
├── data/                                          # Data experiments
│   └── exp_{timestamp}_{exp_name}/
│       ├── config.json
│       ├── bow_matrix.npy
│       ├── vocab.json
│       ├── sbert_embeddings.npy                   # (ctm/dtm/bertopic)
│       ├── word2vec_embeddings.npy                # (etm)
│       └── time_slices.json                       # (dtm)
│
└── models/                                        # Model experiments
    ├── lda/exp_{timestamp}_{exp_name}/
    │   ├── theta_k20.npy
    │   ├── beta_k20.npy
    │   ├── topic_words_k20.json
    │   └── metrics_k20.json
    ├── hdp/exp_{timestamp}_{exp_name}/
    ├── stm/exp_{timestamp}_{exp_name}/
    ├── btm/exp_{timestamp}_{exp_name}/
    ├── nvdm/exp_{timestamp}_{exp_name}/
    ├── gsm/exp_{timestamp}_{exp_name}/
    ├── prodlda/exp_{timestamp}_{exp_name}/
    ├── ctm/exp_{timestamp}_{exp_name}/
    ├── etm/exp_{timestamp}_{exp_name}/
    ├── dtm/exp_{timestamp}_{exp_name}/
    └── bertopic/exp_{timestamp}_{exp_name}/
```

---

## Step 0: Setup

### `01_setup.sh`

Install Python dependencies and download pretrained data from HuggingFace.

```bash
bash scripts/01_setup.sh
```

**What it does**:
1. `pip install -r requirements.txt`
2. Install Agent dependencies (openai, python-dotenv)
3. Download pretrained embeddings and BOW from `CodeSoulco/THETA` (if not already present locally)

---

## Step 1: Data Cleaning

### `02_clean_data.sh`

Clean raw text data into a format suitable for topic modeling. Two modes:

- **CSV mode**: Row-by-row cleaning with user-specified column selection
- **Directory mode**: Convert docx/txt files into a single cleaned CSV

**Supported languages**: english, chinese, german, spanish

**Cleaning steps** (applied to text column only):
1. Remove URLs, emails, HTML tags
2. Collapse redaction markers (XXXX → `[REDACTED]`)
3. Remove special characters (keep basic punctuation)
4. Normalize whitespace
5. Lowercase
6. Remove short documents (< `--min_words` words)

#### Workflow: Preview → Clean

**Step 1: Preview columns** (recommended for CSV input):

```bash
# Inspect columns and get suggested usage
bash scripts/02_clean_data.sh --input data/FCPB/complaints_text_only.csv --preview
```

Output shows all columns with types, sample values, and a suggested `--text_column` / `--label_columns` command.

**Step 2: Clean with explicit column selection**:

```bash
# Text column only (no labels)
bash scripts/02_clean_data.sh \
    --input data/FCPB/complaints_text_only.csv \
    --language english \
    --text_column 'Consumer complaint narrative'

# Text + label column preserved as-is
bash scripts/02_clean_data.sh \
    --input data/hatespeech/hatespeech_text_only.csv \
    --language english \
    --text_column cleaned_content \
    --label_columns Label

# Multiple label columns
bash scripts/02_clean_data.sh \
    --input data/mental_health/mental_health_text_only.csv \
    --language english \
    --text_column clean_text \
    --label_columns 'subreddit_id'

# Keep ALL original columns, only clean the text column
bash scripts/02_clean_data.sh \
    --input raw.csv --language english \
    --text_column text --keep_all

# Directory mode (docx/txt files → CSV)
bash scripts/02_clean_data.sh \
    --input /root/autodl-tmp/data/edu_data/ \
    --language chinese
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--input` | ✓ | Input CSV file or directory (docx/txt files) | - |
| `--language` | ✓ (not for preview) | Data language: english, chinese, german, spanish | - |
| `--text_column` | ✓ (CSV mode) | Name of the text column to clean | - |
| `--label_columns` | | Comma-separated label/metadata columns to keep as-is | - |
| `--keep_all` | | Keep ALL original columns (only text column is cleaned) | false |
| `--preview` | | Show CSV columns and sample rows, then exit | false |
| `--output` | | Output CSV path | Auto-generated |
| `--min_words` | | Min words per document after cleaning | 3 |

**Output**: `data/{dataset}/{input_name}_cleaned.csv` (preserves original column structure)

---

## Step 2: Data Preparation

### `03_prepare_data.sh`

**All-in-one** script to generate BOW matrices and embeddings. Automatically determines required data types based on the target model.

### Supported Models and Data Requirements

| # | Model | Type | Data Requirements |
|---|-------|------|-------------------|
| 1 | **lda** | Traditional | BOW |
| 2 | **hdp** | Traditional | BOW |
| 3 | **stm** | Traditional | BOW |
| 4 | **btm** | Traditional | BOW |
| 5 | **nvdm** | Neural | BOW |
| 6 | **gsm** | Neural | BOW |
| 7 | **prodlda** | Neural | BOW |
| 8 | **ctm** | Neural | BOW + SBERT embeddings |
| 9 | **etm** | Neural | BOW + Word2Vec embeddings |
| 10 | **dtm** | Neural | BOW + SBERT + time slices |
| 11 | **bertopic** | Neural | SBERT + raw text |
| 12 | **theta** | THETA | BOW + Qwen embeddings |

> **Note**: Models 1–7 share BOW data and only need to be preprocessed once.

### Usage: Per-Model Complete Examples

```bash
# ==================================================
# BOW-only models (lda, hdp, stm, btm, nvdm, gsm, prodlda share the same data)
# ==================================================

# LDA — Latent Dirichlet Allocation
bash scripts/03_prepare_data.sh --dataset edu_data --model lda \
  --vocab_size 3500 --language chinese

# HDP — Hierarchical Dirichlet Process (can reuse LDA's data_exp)
bash scripts/03_prepare_data.sh --dataset edu_data --model hdp \
  --vocab_size 3500 --language chinese

# STM — Structural Topic Model (can reuse LDA's data_exp)
bash scripts/03_prepare_data.sh --dataset edu_data --model stm \
  --vocab_size 3500 --language chinese

# BTM — Biterm Topic Model (can reuse LDA's data_exp)
bash scripts/03_prepare_data.sh --dataset edu_data --model btm \
  --vocab_size 3500 --language chinese

# NVDM — Neural Variational Document Model (can reuse LDA's data_exp)
bash scripts/03_prepare_data.sh --dataset edu_data --model nvdm \
  --vocab_size 3500 --language chinese

# GSM — Gaussian Softmax Model (can reuse LDA's data_exp)
bash scripts/03_prepare_data.sh --dataset edu_data --model gsm \
  --vocab_size 3500 --language chinese

# ProdLDA — Product of Experts LDA (can reuse LDA's data_exp)
bash scripts/03_prepare_data.sh --dataset edu_data --model prodlda \
  --vocab_size 3500 --language chinese

# ==================================================
# Models requiring additional embeddings
# ==================================================

# CTM — Contextualized Topic Model (BOW + SBERT embeddings)
bash scripts/03_prepare_data.sh --dataset edu_data --model ctm \
  --vocab_size 3500 --language chinese

# ETM — Embedded Topic Model (BOW + Word2Vec embeddings)
bash scripts/03_prepare_data.sh --dataset edu_data --model etm \
  --vocab_size 3500 --language chinese

# DTM — Dynamic Topic Model (BOW + SBERT + time slices, requires --time_column)
bash scripts/03_prepare_data.sh --dataset edu_data --model dtm \
  --vocab_size 3500 --language chinese --time_column year

# BERTopic — BERT-based Topic Model (SBERT + raw text)
bash scripts/03_prepare_data.sh --dataset edu_data --model bertopic \
  --vocab_size 3500 --language chinese

# ==================================================
# THETA model (BOW + Qwen embeddings)
# ==================================================

# THETA zero_shot (fastest, no fine-tuning)
bash scripts/03_prepare_data.sh --dataset edu_data --model theta \
  --model_size 0.6B --mode zero_shot --vocab_size 3500 --language chinese

# THETA unsupervised (LoRA fine-tuned Qwen embeddings)
bash scripts/03_prepare_data.sh --dataset edu_data --model theta \
  --model_size 0.6B --mode unsupervised --vocab_size 3500 --language chinese \
  --emb_epochs 10 --emb_batch_size 8

# THETA supervised (requires label column)
bash scripts/03_prepare_data.sh --dataset edu_data --model theta \
  --model_size 0.6B --mode supervised --vocab_size 3500 --language chinese \
  --label_column province --emb_epochs 10 --emb_batch_size 8

# ==================================================
# Utility flags
# ==================================================

# Generate BOW only, skip embeddings
bash scripts/03_prepare_data.sh --dataset edu_data --model theta \
  --model_size 0.6B --mode zero_shot --vocab_size 3500 --language chinese --bow-only

# Check if data files already exist (no processing)
bash scripts/03_prepare_data.sh --dataset edu_data --model lda --check-only

# Clean raw data first, then prepare
bash scripts/03_prepare_data.sh --dataset edu_data --model lda \
  --vocab_size 3500 --language chinese --clean --raw-input /root/autodl-tmp/data/edu_data/edu_data_raw.csv
```

### Parameter Reference

**Required parameters**:

| Parameter | Description |
|-----------|-------------|
| `--dataset` | Dataset name (must exist in `/root/autodl-tmp/data/`) |
| `--model` | Target model: lda, hdp, stm, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta |

**Common optional parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--vocab_size` | Vocabulary size | 5000 |
| `--batch_size` | Embedding generation batch size | 32 |
| `--gpu` | GPU device ID | 0 |
| `--language` | Data language: english, chinese (controls BOW tokenization) | english |
| `--bow-only` | Only generate BOW, skip embeddings | false |
| `--check-only` | Only check if files exist | false |
| `--exp_name` | Experiment name tag | Auto-generated |
| `--clean` | Clean raw data first (use with `--raw-input`) | false |
| `--raw-input` | Raw input file path (use with `--clean`) | - |

**THETA-specific parameters** (`--model theta`):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_size` | Qwen model size: 0.6B, 4B, 8B | 0.6B |
| `--mode` | Embedding mode: zero_shot, unsupervised, supervised | zero_shot |
| `--label_column` | Label column name (required for supervised mode) | - |
| `--emb_epochs` | Embedding fine-tuning epochs | 10 |
| `--emb_lr` | Embedding fine-tuning learning rate | 2e-5 |
| `--emb_max_length` | Embedding max sequence length | 512 |
| `--emb_batch_size` | Embedding fine-tuning batch size (use smaller values for CausalLM) | 8 |

**DTM-specific parameters** (`--model dtm`):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--time_column` | Time column name in the CSV | year |

### THETA Embedding Modes

| Mode | Description | Training | VRAM | Speed |
|------|-------------|----------|------|-------|
| **zero_shot** | Use pretrained Qwen directly for embeddings | None | ~2GB | Fastest |
| **unsupervised** | LoRA fine-tuning (autoregressive LM loss) | Required | ~12GB | Medium |
| **supervised** | LoRA fine-tuning (classification loss, requires labels) | Required | ~8GB | Medium |

### THETA Data Preparation Pipeline (3 steps)

```
03_prepare_data.sh internally executes:

Step 1: BOW generation
  → prepare_data.py --bow-only
  → Output: exp_xxx/bow/ (bow_matrix.npy, vocab.json, vocab.txt)

Step 2: Embedding generation
  → 02_generate_embeddings.sh (calls embedding/main.py internally)
  → Output: exp_xxx/embeddings/ (embeddings.npy, metadata.json)

Step 3: Vocabulary embedding generation
  → VocabEmbedder
  → Output: exp_xxx/bow/vocab_embeddings.npy
```

### Output Paths

| Model Type | Output Path |
|------------|-------------|
| Baseline (lda, hdp, ...) | `result/baseline/{dataset}/data/exp_{timestamp}_{name}/` |
| THETA | `result/{model_size}/{dataset}/data/exp_{timestamp}_{name}/` |

---

## Appendix: Re-run Embeddings

### `02_generate_embeddings.sh` (sub-script of 03)

This script is normally called **internally by `03_prepare_data.sh`** and does not need to be run manually. Use it standalone only when:
- Embedding generation failed with OOM but BOW was already generated
- You want to regenerate embeddings with different parameters (batch_size, mode)
- You need to add embeddings to an existing experiment directory

```bash
# Zero-shot (fastest, no training)
bash scripts/02_generate_embeddings.sh \
  --dataset edu_data --mode zero_shot --model_size 0.6B

# Unsupervised (LoRA fine-tuning)
bash scripts/02_generate_embeddings.sh \
  --dataset edu_data --mode unsupervised --model_size 0.6B \
  --epochs 10 --batch_size 8 \
  --exp_dir /root/autodl-tmp/result/0.6B/edu_data/data/exp_xxx

# Supervised (requires label column)
bash scripts/02_generate_embeddings.sh \
  --dataset edu_data --mode supervised --model_size 0.6B \
  --epochs 10 --batch_size 8 --label_column province \
  --exp_dir /root/autodl-tmp/result/0.6B/edu_data/data/exp_xxx
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | - |
| `--mode` | zero_shot, unsupervised, supervised | zero_shot |
| `--model_size` | 0.6B, 4B, 8B | 0.6B |
| `--model_path` | Qwen model path | /root/autodl-tmp/qwen3_embedding_0.6B |
| `--max_length` | Max sequence length | 512 |
| `--batch_size` | Batch size (recommend ≤8 for unsupervised) | 16 |
| `--epochs` | Training epochs (supervised/unsupervised) | 10 |
| `--learning_rate` | Learning rate | 2e-5 |
| `--label_column` | Label column name (supervised mode) | - |
| `--exp_dir` | Save to a specific experiment directory | - |
| `--gpu` | GPU device ID | 0 |
| `--no_lora` | Disable LoRA, use full fine-tuning | false |

> **⚠️ VRAM Warning**: Unsupervised mode loads the full CausalLM model, consuming significantly more VRAM than zero_shot. Recommend `--batch_size 8`. If OOM, try `--batch_size 4`.

---

## Step 3: THETA Training

### `04_train_theta.sh`

Train the THETA model with integrated training + evaluation + visualization.

```bash
# ==================================================
# Basic usage
# ==================================================

# Zero-shot mode (minimal command)
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot \
  --num_topics 20

# Unsupervised mode
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode unsupervised \
  --num_topics 20

# Supervised mode (requires label column in data)
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode supervised \
  --num_topics 20

# Use a larger Qwen model
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 4B --mode zero_shot \
  --num_topics 20

# ==================================================
# Full parameters
# ==================================================

# Full parameter example
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot \
  --num_topics 20 --epochs 100 --batch_size 64 \
  --hidden_dim 512 --learning_rate 0.002 \
  --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 \
  --patience 10 --gpu 0 --language zh

# Custom KL annealing parameters
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot \
  --num_topics 20 --epochs 200 \
  --kl_start 0.1 --kl_end 0.8 --kl_warmup 40

# ==================================================
# Specify data experiment
# ==================================================

# Use --data_exp to specify an existing data experiment
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot \
  --data_exp exp_20260208_151906_vocab3500_theta_0.6B_zero_shot \
  --num_topics 20 --epochs 50 --language zh

# Custom experiment name
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot \
  --num_topics 20 --exp_name my_experiment

# ==================================================
# Skip training / visualization
# ==================================================

# Skip visualization (train + eval only, faster)
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot \
  --num_topics 20 --epochs 50 --skip-viz

# Skip training (eval + visualize existing model only)
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot \
  --skip-train --language zh
```

### Parameter Reference

**Required parameters**:

| Parameter | Description |
|-----------|-------------|
| `--dataset` | Dataset name (data must be prepared via 03_prepare_data.sh first) |

**Model parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_size` | Qwen model size: 0.6B, 4B, 8B | 0.6B |
| `--mode` | Embedding mode: zero_shot, unsupervised, supervised | zero_shot |
| `--data_exp` | Data experiment ID (auto-selects latest if not specified) | Auto-select |

**Training parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_topics` | Number of topics K | 20 |
| `--epochs` | Training epochs | 100 |
| `--batch_size` | Training batch size | 64 |
| `--hidden_dim` | Encoder hidden dimension | 512 |
| `--learning_rate` | Learning rate | 0.002 |
| `--patience` | Early stopping patience | 10 |

**KL annealing parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--kl_start` | KL annealing start weight | 0.0 |
| `--kl_end` | KL annealing end weight | 1.0 |
| `--kl_warmup` | KL annealing warmup epochs | 50 |

**Other parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gpu` | GPU device ID | 0 |
| `--language` | Visualization language: en (English charts), zh (Chinese charts) | en |
| `--skip-train` | Skip training, only evaluate existing model | false |
| `--skip-viz` | Skip visualization | false |
| `--exp_name` | Training experiment name tag | Auto-generated |

### KL Annealing Explained

KL Annealing gradually increases the KL divergence weight during VAE training to prevent posterior collapse:

```
KL weight = kl_start + (kl_end - kl_start) × min(1, epoch / kl_warmup)
```

- `kl_start=0.0, kl_end=1.0, kl_warmup=50`: linearly increases from 0 to 1 over the first 50 epochs

### Output Path

```
result/{model_size}/{dataset}/models/exp_{timestamp}_{exp_name}/
├── config.json
├── model/          # Model files
├── evaluation/     # Evaluation metrics
├── topic_words/    # Topic words
└── visualization/  # Visualization charts
```

---

## Step 4: Baseline Training

### `05_train_baseline.sh`

Train 11 baseline topic models for comparison with THETA.

### Supported Models

| Model | Type | Description | Model-Specific Parameters |
|-------|------|-------------|---------------------------|
| **lda** | Traditional | Latent Dirichlet Allocation | `--max_iter` |
| **hdp** | Traditional | Hierarchical Dirichlet Process (auto topic count) | `--max_topics`, `--alpha` |
| **stm** | Traditional | Structural Topic Model | `--max_iter` |
| **btm** | Traditional | Biterm Topic Model (best for short texts) | `--n_iter`, `--alpha`, `--beta` |
| **nvdm** | Neural | Neural Variational Document Model | `--epochs`, `--dropout` |
| **gsm** | Neural | Gaussian Softmax Model | `--epochs`, `--dropout` |
| **prodlda** | Neural | Product of Experts LDA | `--epochs`, `--dropout` |
| **ctm** | Neural | Contextualized Topic Model (requires SBERT) | `--epochs`, `--inference_type` |
| **etm** | Neural | Embedded Topic Model (requires Word2Vec) | `--epochs` |
| **dtm** | Neural | Dynamic Topic Model (requires timestamps) | `--epochs` |
| **bertopic** | Neural | BERT-based Topic Model (auto topic count) | - |

### Complete Per-Model Examples

```bash
# ==================================================
# 1. LDA — Latent Dirichlet Allocation
#    Type: Traditional | Data: BOW only
#    Specific params: --max_iter (max EM iterations)
# ==================================================

# Minimal
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models lda --num_topics 20

# Full parameters
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models lda \
  --num_topics 20 --max_iter 200 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_153424_vocab3500_lda \
  --exp_name lda_full

# ==================================================
# 2. HDP — Hierarchical Dirichlet Process
#    Type: Traditional | Data: BOW only
#    Note: Auto-determines topic count, --num_topics is IGNORED
#    Specific params: --max_topics, --alpha
# ==================================================

# Minimal (auto topic count)
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models hdp

# Full parameters
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models hdp \
  --max_topics 150 --alpha 1.0 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_153424_vocab3500_lda \
  --exp_name hdp_full

# ==================================================
# 3. STM — Structural Topic Model
#    Type: Traditional | Data: BOW only
#    Specific params: --max_iter
# ==================================================

# Minimal
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models stm --num_topics 20

# Full parameters
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models stm \
  --num_topics 20 --max_iter 200 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_153424_vocab3500_lda \
  --exp_name stm_full

# ==================================================
# 4. BTM — Biterm Topic Model
#    Type: Traditional | Data: BOW only
#    Note: Uses Gibbs sampling, very slow on long documents (samples max 50 words/doc)
#    Best suited for short texts (tweets, comments)
#    Specific params: --n_iter, --alpha, --beta
# ==================================================

# Minimal
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models btm --num_topics 20

# Full parameters
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models btm \
  --num_topics 20 --n_iter 100 --alpha 1.0 --beta 0.01 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_153424_vocab3500_lda \
  --exp_name btm_full

# ==================================================
# 5. NVDM — Neural Variational Document Model
#    Type: Neural | Data: BOW only
#    Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate, --dropout
# ==================================================

# Minimal
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models nvdm --num_topics 20

# Full parameters
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models nvdm \
  --num_topics 20 --epochs 200 --batch_size 128 \
  --hidden_dim 512 --learning_rate 0.002 --dropout 0.2 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_153424_vocab3500_lda \
  --exp_name nvdm_full

# ==================================================
# 6. GSM — Gaussian Softmax Model
#    Type: Neural | Data: BOW only
#    Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate, --dropout
# ==================================================

# Minimal
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models gsm --num_topics 20

# Full parameters
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models gsm \
  --num_topics 20 --epochs 200 --batch_size 128 \
  --hidden_dim 512 --learning_rate 0.002 --dropout 0.2 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_153424_vocab3500_lda \
  --exp_name gsm_full

# ==================================================
# 7. ProdLDA — Product of Experts LDA
#    Type: Neural | Data: BOW only
#    Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate, --dropout
# ==================================================

# Minimal
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models prodlda --num_topics 20

# Full parameters
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models prodlda \
  --num_topics 20 --epochs 200 --batch_size 128 \
  --hidden_dim 512 --learning_rate 0.002 --dropout 0.2 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_153424_vocab3500_lda \
  --exp_name prodlda_full

# ==================================================
# 8. CTM — Contextualized Topic Model
#    Type: Neural | Data: BOW + SBERT embeddings
#    Note: Requires SBERT data_exp (prepared with --model ctm)
#    Specific params: --epochs, --inference_type (zeroshot | combined)
# ==================================================

# Minimal (zeroshot inference, default)
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models ctm --num_topics 20

# Zeroshot inference (uses only SBERT embeddings for inference)
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models ctm \
  --num_topics 20 --epochs 100 --inference_type zeroshot \
  --batch_size 64 --hidden_dim 512 --learning_rate 0.002 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_154645_vocab3500_ctm \
  --exp_name ctm_zeroshot

# Combined inference (uses both BOW and SBERT)
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models ctm \
  --num_topics 20 --epochs 100 --inference_type combined \
  --batch_size 64 --hidden_dim 512 --learning_rate 0.002 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_154645_vocab3500_ctm \
  --exp_name ctm_combined

# ==================================================
# 9. ETM — Embedded Topic Model
#    Type: Neural | Data: BOW + Word2Vec embeddings
#    Note: Word2Vec embeddings are generated during BOW-only data prep
#    Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate
# ==================================================

# Minimal
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models etm --num_topics 20

# Full parameters
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models etm \
  --num_topics 20 --epochs 200 --batch_size 64 \
  --hidden_dim 512 --learning_rate 0.002 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_153424_vocab3500_lda \
  --exp_name etm_full

# ==================================================
# 10. DTM — Dynamic Topic Model
#     Type: Neural | Data: BOW + SBERT + time slices
#     Note: Requires data_exp prepared with --model dtm (includes time_slices.json)
#     Specific params: --epochs, --batch_size, --hidden_dim, --learning_rate
# ==================================================

# Minimal
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models dtm --num_topics 20

# Full parameters
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models dtm \
  --num_topics 20 --epochs 200 --batch_size 64 \
  --hidden_dim 512 --learning_rate 0.002 \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_171413_vocab3500_dtm \
  --exp_name dtm_full

# ==================================================
# 11. BERTopic — BERT-based Topic Model
#     Type: Neural | Data: SBERT + raw text
#     Note: Auto-determines topic count, --num_topics is IGNORED
#     Note: Requires SBERT data_exp (can reuse CTM's data_exp)
#     No model-specific training parameters
# ==================================================

# Minimal (auto topic count)
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models bertopic

# With visualization and explicit data_exp
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models bertopic \
  --gpu 0 --language zh --with-viz \
  --data_exp exp_20260208_154645_vocab3500_ctm \
  --exp_name bertopic_full

# ==================================================
# Batch training (multiple models at once)
# ==================================================

# Train all BOW-only models (share the same data_exp)
bash scripts/05_train_baseline.sh \
  --dataset edu_data \
  --models lda,hdp,stm,btm,nvdm,gsm,prodlda \
  --num_topics 20 --epochs 100 \
  --data_exp exp_20260208_153424_vocab3500_lda

# Train ETM separately (uses Word2Vec from BOW data_exp)
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models etm \
  --num_topics 20 --epochs 100 \
  --data_exp exp_20260208_153424_vocab3500_lda

# Train CTM + BERTopic (share SBERT data_exp)
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models ctm,bertopic \
  --num_topics 20 --epochs 100 \
  --data_exp exp_20260208_154645_vocab3500_ctm

# Train DTM separately (requires time_slices data_exp)
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models dtm \
  --num_topics 20 --epochs 100 \
  --data_exp exp_20260208_171413_vocab3500_dtm

# ==================================================
# Skip training / visualization
# ==================================================

# Skip training, only evaluate and visualize existing model
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models lda --num_topics 20 --skip-train

# Enable visualization (disabled by default, use --with-viz to enable)
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models lda --num_topics 20 \
  --with-viz --language zh
```

> **⚠️ Important notes**:
> - BTM uses Gibbs sampling and is very slow on long documents (samples max 50 words/doc). Best for short texts.
> - HDP and BERTopic auto-determine topic count; `--num_topics` is ignored for these models.
> - DTM requires a data experiment containing `time_slices.json` (prepared with `--model dtm`).
> - CTM and BERTopic require a data experiment containing SBERT embeddings.

### Parameter Reference

**Common parameters**:

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--dataset` | ✓ | Dataset name | - |
| `--models` | ✓ | Model list, comma-separated | - |
| `--num_topics` | | Number of topics (ignored for hdp/bertopic) | 20 |
| `--vocab_size` | | Vocabulary size | 5000 |
| `--epochs` | | Training epochs (neural models) | 100 |
| `--batch_size` | | Batch size | 64 |
| `--hidden_dim` | | Hidden layer dimension | 512 |
| `--learning_rate` | | Learning rate | 0.002 |
| `--gpu` | | GPU device ID | 0 |
| `--language` | | Visualization language: en, zh | en |
| `--skip-train` | | Skip training | false |
| `--skip-viz` | | Skip visualization (default: skipped) | true |
| `--with-viz` | | Enable visualization | false |
| `--data_exp` | | Data experiment ID | Auto-select latest |
| `--exp_name` | | Experiment name tag | Auto-generated |

**Model-specific parameters**:

| Parameter | Applicable Models | Description | Default |
|-----------|-------------------|-------------|---------|
| `--max_iter` | lda, stm | Max iterations (EM algorithm) | 100 |
| `--max_topics` | hdp | Max topic count | 150 |
| `--n_iter` | btm | Gibbs sampling iterations | 100 |
| `--alpha` | hdp, btm | Alpha prior | 1.0 |
| `--beta` | btm | Beta prior | 0.01 |
| `--inference_type` | ctm | Inference type: zeroshot, combined | zeroshot |
| `--dropout` | Neural models (nvdm, gsm, prodlda, ctm, etm, dtm) | Dropout rate | 0.2 |

### Output Path

```
result/baseline/{dataset}/models/{model}/exp_{timestamp}_{exp_name}/
├── theta_k{K}.npy          # Doc-topic distribution
├── beta_k{K}.npy           # Topic-word distribution
├── topic_words_k{K}.json   # Topic word list
├── metrics_k{K}.json       # Evaluation metrics
└── visualization/           # Visualization (if enabled)
```

---

## Step 5: Standalone Visualization

### `06_visualize.sh`

Generate visualizations for already-trained models without retraining.

```bash
# ==================================================
# THETA model visualization
# ==================================================

# Basic usage (auto-selects latest experiment)
bash scripts/06_visualize.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot --language zh

# Unsupervised mode
bash scripts/06_visualize.sh \
  --dataset edu_data --model_size 0.6B --mode unsupervised --language zh

# English charts + high DPI (for papers)
bash scripts/06_visualize.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot --language en --dpi 600

# ==================================================
# Baseline model visualization (all 11 models)
# ==================================================

# LDA
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model lda --num_topics 20 --language zh

# HDP (auto topic count, use actual K from training)
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model hdp --num_topics 150 --language zh

# STM
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model stm --num_topics 20 --language zh

# BTM
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model btm --num_topics 20 --language zh

# NVDM
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model nvdm --num_topics 20 --language zh

# GSM
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model gsm --num_topics 20 --language zh

# ProdLDA
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model prodlda --num_topics 20 --language zh

# CTM
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model ctm --num_topics 20 --language zh

# ETM
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model etm --num_topics 20 --language en

# DTM (includes topic evolution charts)
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model dtm --num_topics 20 --language zh

# BERTopic
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model bertopic --num_topics 20 --language zh

# ==================================================
# Advanced options
# ==================================================

# Specify a model experiment explicitly
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model ctm --model_exp exp_20260208_xxx --language zh

# High DPI output (for publication)
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model lda --num_topics 20 --language en --dpi 600
```

### Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | - |
| `--baseline` | Baseline model mode | false |
| `--model` | Baseline model name | - |
| `--model_exp` | Model experiment ID (auto-selects latest if not specified) | Auto-select |
| `--model_size` | THETA model size | 0.6B |
| `--mode` | THETA mode | zero_shot |
| `--language` | Visualization language: en, zh | en |
| `--dpi` | Image DPI | 300 |

### Generated Visualization Types

| Chart | Description | Filename (en) |
|-------|-------------|---------------|
| Topic Table | Top words per topic | topic_table.png |
| Topic Network | Inter-topic similarity network | topic_network.png |
| Document Clusters | UMAP document distribution | doc_topic_umap.png |
| Cluster Heatmap | Topic-document heatmap | cluster_heatmap.png |
| Topic Proportion | Document proportion per topic | topic_proportion.png |
| Training Loss | Loss curve | training_loss.png |
| Evaluation Metrics | 7-metric radar chart | metrics.png |
| Topic Coherence | Per-topic NPMI | topic_coherence.png |
| Topic Exclusivity | Per-topic exclusivity | topic_exclusivity.png |
| Word Clouds | All topic word clouds | topic_wordclouds.png |
| Topic Similarity | Inter-topic cosine similarity | topic_similarity.png |
| pyLDAvis | Interactive topic explorer | pyldavis_interactive.html |
| Per-topic Words | Per-topic word weights | topics/topic_N/word_importance.png |

---

## Step 6: Standalone Evaluation

### `07_evaluate.sh`

Compute evaluation metrics for already-trained models.

```bash
# ==================================================
# Evaluate baseline models (all 11)
# ==================================================

# LDA
bash scripts/07_evaluate.sh --dataset edu_data --model lda --num_topics 20

# HDP (topic count auto-determined; num_topics is used for file lookup)
bash scripts/07_evaluate.sh --dataset edu_data --model hdp --num_topics 150

# STM
bash scripts/07_evaluate.sh --dataset edu_data --model stm --num_topics 20

# BTM
bash scripts/07_evaluate.sh --dataset edu_data --model btm --num_topics 20

# NVDM
bash scripts/07_evaluate.sh --dataset edu_data --model nvdm --num_topics 20

# GSM
bash scripts/07_evaluate.sh --dataset edu_data --model gsm --num_topics 20

# ProdLDA
bash scripts/07_evaluate.sh --dataset edu_data --model prodlda --num_topics 20

# CTM
bash scripts/07_evaluate.sh --dataset edu_data --model ctm --num_topics 20

# ETM
bash scripts/07_evaluate.sh --dataset edu_data --model etm --num_topics 20

# DTM
bash scripts/07_evaluate.sh --dataset edu_data --model dtm --num_topics 20

# BERTopic
bash scripts/07_evaluate.sh --dataset edu_data --model bertopic --num_topics 20

# With custom vocab size
bash scripts/07_evaluate.sh --dataset edu_data --model lda --num_topics 20 --vocab_size 3500

# ==================================================
# Evaluate THETA models
# ==================================================

# Zero-shot THETA
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 0.6B --mode zero_shot

# Unsupervised THETA
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 0.6B --mode unsupervised

# Supervised THETA (4B model)
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 4B --mode supervised
```

### Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | - |
| `--model` | Model name (required): lda, hdp, stm, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta | - |
| `--num_topics` | Number of topics | 20 |
| `--vocab_size` | Vocabulary size | 5000 |
| `--baseline` | Baseline model mode | false |
| `--model_size` | THETA model size: 0.6B, 4B, 8B | 0.6B |
| `--mode` | THETA mode: zero_shot, unsupervised, supervised | zero_shot |

### Evaluation Metrics (7 metrics)

| Metric | Full Name | Direction | Description |
|--------|-----------|-----------|-------------|
| **TD** | Topic Diversity | ↑ Higher is better | Proportion of unique words across topics |
| **iRBO** | Inverse Rank-Biased Overlap | ↑ Higher is better | Rank-based topic diversity |
| **NPMI** | Normalized PMI | ↑ Higher is better | Normalized pointwise mutual information coherence |
| **C_V** | C_V Coherence | ↑ Higher is better | Sliding-window based coherence |
| **UMass** | UMass Coherence | → Closer to 0 is better | Document co-occurrence based coherence |
| **Exclusivity** | Topic Exclusivity | ↑ Higher is better | How exclusive words are to their topics |
| **PPL** | Perplexity | ↓ Lower is better | Model fit (lower = better generalization) |

---

## Step 7: Model Comparison

### `08_compare_models.sh`

Compare evaluation metrics across multiple models and generate comparison tables.

```bash
# Compare all baseline models
bash scripts/08_compare_models.sh \
  --dataset edu_data \
  --models lda,hdp,stm,btm,nvdm,gsm,prodlda,ctm,etm,dtm,bertopic \
  --num_topics 20

# Compare traditional models only
bash scripts/08_compare_models.sh \
  --dataset edu_data --models lda,hdp,stm,btm --num_topics 20

# Compare neural models only
bash scripts/08_compare_models.sh \
  --dataset edu_data --models nvdm,gsm,prodlda,ctm,etm,dtm --num_topics 20

# Compare specific models
bash scripts/08_compare_models.sh \
  --dataset edu_data --models lda,prodlda,ctm --num_topics 20

# Export to CSV file
bash scripts/08_compare_models.sh \
  --dataset edu_data \
  --models lda,hdp,nvdm,gsm,prodlda,ctm,etm,stm,dtm \
  --num_topics 20 --output comparison.csv
```

### Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | - |
| `--models` | Model list, comma-separated (required) | - |
| `--num_topics` | Number of topics | 20 |
| `--output` | Output CSV file path | None (terminal output only) |

**Example output**:
```
================================================================================
Model Comparison: edu_data (K=20)
================================================================================

Model              TD     iRBO     NPMI      C_V    UMass  Exclusivity        PPL
--------------------------------------------------------------------------------
lda            0.8500   0.7200   0.0512   0.4231  -2.1234       0.6543     123.45
prodlda        0.9200   0.8100   0.0634   0.4567  -1.8765       0.7234      98.76
ctm            0.8800   0.7800   0.0589   0.4412  -1.9876       0.6987     105.32
--------------------------------------------------------------------------------

Best Models:
  - Best TD (Topic Diversity): prodlda (0.9200)
  - Best NPMI (Coherence):     prodlda (0.0634)
  - Best PPL (Perplexity):     prodlda (98.76)
```

---

## Utility Scripts

### `09_download_from_hf.sh`

Download pretrained data (embeddings, BOW, LoRA weights) from HuggingFace.

```bash
bash scripts/09_download_from_hf.sh
```

### `10_quick_start_english.sh`

Quick start for English datasets (one-click data preparation + training).

```bash
bash scripts/10_quick_start_english.sh my_dataset
```

### `11_quick_start_chinese.sh`

Quick start for Chinese datasets (one-click data preparation + training, Chinese chart labels).

```bash
bash scripts/11_quick_start_chinese.sh my_chinese_dataset
```

### `12_train_multi_gpu.sh`

Multi-GPU distributed training (PyTorch DDP).

```bash
bash scripts/12_train_multi_gpu.sh \
  --dataset edu_data --num_gpus 2 --model_size 0.6B --mode zero_shot
```

### `13_test_agent.sh`

Test LLM Agent connectivity and functionality.

```bash
bash scripts/13_test_agent.sh
```

### `14_start_agent_api.sh`

Start the AI Agent API server (FastAPI) with LangChain-based tool-calling agent.

```bash
bash scripts/14_start_agent_api.sh --port 8000
```

API endpoints:
- `POST /api/agent/chat` — Chat with LangChain agent (auto tool-calling)
- `POST /api/agent/chat/stream` — Streaming chat (SSE)
- `GET /api/agent/tools` — List available tools
- `GET /api/agent/sessions` — List active sessions
- `POST /chat` — Legacy simple Q&A
- `GET /docs` — OpenAPI documentation

---

## End-to-End Example: edu_data

The following demonstrates the complete pipeline from data cleaning to model comparison using `edu_data` (823 Chinese education policy documents).

### 1. Setup

```bash
bash scripts/01_setup.sh
```

### 2. Data Cleaning (if raw data is not yet cleaned)

```bash
# Preview columns first
bash scripts/02_clean_data.sh --input /root/autodl-tmp/data/edu_data/edu_data_raw.csv --preview

# Clean with explicit column selection (directory mode for docx/txt)
bash scripts/02_clean_data.sh --input /root/autodl-tmp/data/edu_data/ --language chinese

# Clean CSV with text column specified
bash scripts/02_clean_data.sh \
    --input /root/autodl-tmp/data/edu_data/edu_data_raw.csv \
    --language chinese --text_column cleaned_content
# Output: data/edu_data/edu_data_raw_cleaned.csv
```

### 3. Data Preparation — Baseline Models

```bash
# BOW-only models (lda, hdp, stm, btm, nvdm, gsm, prodlda share the same data)
bash scripts/03_prepare_data.sh \
  --dataset edu_data --model lda --vocab_size 3500 --language chinese
# Output: result/baseline/edu_data/data/exp_xxx/

# CTM (additionally requires SBERT embeddings)
bash scripts/03_prepare_data.sh \
  --dataset edu_data --model ctm --vocab_size 3500 --language chinese

# ETM (additionally requires Word2Vec embeddings)
bash scripts/03_prepare_data.sh \
  --dataset edu_data --model etm --vocab_size 3500 --language chinese

# DTM (additionally requires SBERT + time slices)
bash scripts/03_prepare_data.sh \
  --dataset edu_data --model dtm --vocab_size 3500 --language chinese --time_column year

# BERTopic (SBERT + raw text)
bash scripts/03_prepare_data.sh \
  --dataset edu_data --model bertopic --vocab_size 3500 --language chinese
```

### 4. Data Preparation — THETA Model

```bash
# Zero-shot (fastest, recommended for initial testing)
bash scripts/03_prepare_data.sh \
  --dataset edu_data --model theta --model_size 0.6B --mode zero_shot \
  --vocab_size 3500 --language chinese
# Output: result/0.6B/edu_data/data/exp_xxx_vocab3500_theta_0.6B_zero_shot/

# Unsupervised (LoRA fine-tuning, potentially better results)
bash scripts/03_prepare_data.sh \
  --dataset edu_data --model theta --model_size 0.6B --mode unsupervised \
  --vocab_size 3500 --language chinese --emb_epochs 10 --emb_batch_size 8
# Output: result/0.6B/edu_data/data/exp_xxx_vocab3500_theta_0.6B_unsupervised/
```

### 5. Train Baseline Models

```bash
# Train all BOW-only models at once
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models lda,hdp,stm,btm,nvdm,gsm,prodlda \
  --num_topics 20 --epochs 100

# Train CTM
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models ctm --num_topics 20 --epochs 50

# Train ETM
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models etm --num_topics 20 --epochs 50

# Train DTM
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models dtm --num_topics 20 --epochs 50

# Train BERTopic
bash scripts/05_train_baseline.sh \
  --dataset edu_data --models bertopic
```

### 6. Train THETA Model

```bash
# Zero-shot THETA (Chinese visualization)
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot \
  --num_topics 20 --epochs 100 --language zh

# Unsupervised THETA
bash scripts/04_train_theta.sh \
  --dataset edu_data --model_size 0.6B --mode unsupervised \
  --num_topics 20 --epochs 100 --language zh
```

### 7. Standalone Visualization (optional, already generated during training)

```bash
# THETA visualization
bash scripts/06_visualize.sh \
  --dataset edu_data --model_size 0.6B --mode zero_shot --language zh

# Baseline visualization
bash scripts/06_visualize.sh \
  --baseline --dataset edu_data --model lda --num_topics 20 --language zh
```

### 8. Model Comparison

```bash
bash scripts/08_compare_models.sh \
  --dataset edu_data \
  --models lda,hdp,stm,btm,nvdm,gsm,prodlda,ctm,etm \
  --num_topics 20
```

### Final Result Directory

```
result/
├── 0.6B/edu_data/                          # THETA results
│   ├── data/
│   │   ├── exp_xxx_vocab3500_theta_0.6B_zero_shot/
│   │   │   ├── bow/ (bow_matrix.npy, vocab.json, vocab_embeddings.npy)
│   │   │   └── embeddings/ (embeddings.npy)
│   │   └── exp_xxx_vocab3500_theta_0.6B_unsupervised/
│   │       ├── bow/
│   │       └── embeddings/
│   └── models/
│       ├── exp_xxx_k20_e100_zero_shot/
│       │   ├── model/ (etm_model.pt, theta.npy, beta.npy, ...)
│       │   ├── evaluation/ (metrics.json)
│       │   ├── topic_words/ (topic_words.json, topic_words.txt)
│       │   └── visualization/viz_xxx/ (30+ charts)
│       └── exp_xxx_k20_e100_unsupervised/
│
└── baseline/edu_data/                      # Baseline results
    ├── data/
    │   ├── exp_xxx_vocab3500/              # Shared by BOW-only models
    │   ├── exp_xxx_ctm_vocab3500/          # CTM-specific
    │   ├── exp_xxx_etm_vocab3500/          # ETM-specific
    │   ├── exp_xxx_dtm_vocab3500/          # DTM-specific
    │   └── exp_xxx_bertopic_vocab3500/     # BERTopic-specific
    └── models/
        ├── lda/exp_xxx/ (theta_k20.npy, beta_k20.npy, metrics_k20.json)
        ├── hdp/exp_xxx/
        ├── stm/exp_xxx/
        ├── btm/exp_xxx/
        ├── nvdm/exp_xxx/
        ├── gsm/exp_xxx/
        ├── prodlda/exp_xxx/
        ├── ctm/exp_xxx/
        ├── etm/exp_xxx/
        ├── dtm/exp_xxx/
        └── bertopic/exp_xxx/
```

---

## FAQ

### Q: CUDA out of memory

**Cause**: Insufficient GPU VRAM.

**Solutions**:
- Embedding generation (unsupervised/supervised): reduce `--batch_size` (recommend 4–8)
- THETA training: reduce `--batch_size` (recommend 32–64)
- Check for other processes using the GPU: `nvidia-smi`
- Kill zombie processes: `kill -9 <PID>`

### Q: EMB shows ✗ (embeddings not generated)

**Cause**: Embedding generation failed (usually OOM) but the script did not exit with an error.

**Solution**:
```bash
# Regenerate with a smaller batch_size
bash scripts/02_generate_embeddings.sh \
  --dataset edu_data --mode unsupervised --model_size 0.6B \
  --batch_size 4 --gpu 0 \
  --exp_dir /root/autodl-tmp/result/0.6B/edu_data/data/exp_xxx
```

### Q: How to choose an embedding mode?

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| Quick testing | zero_shot | No training needed, completes in seconds |
| Unlabeled data | unsupervised | LoRA fine-tuning adapts to the domain |
| Labeled data | supervised | Leverages label information to enhance embeddings |
| Large datasets | zero_shot | Avoids lengthy fine-tuning |

### Q: How to choose the number of topics K?

- Small datasets (<1000 docs): K = 5–15
- Medium datasets (1000–10000): K = 10–30
- Large datasets (>10000): K = 20–50
- Use `hdp` or `bertopic` to auto-determine topic count as a reference

### Q: What does the visualization `--language` parameter do?

- `en`: Chart titles, axes, and legends in English
- `zh`: Chart titles, axes, and legends in Chinese (e.g., "主题表", "训练损失图")
- Only affects visualization; does not affect model training or evaluation

### Q: What is the difference between BOW `--language` and visualization `--language`?

| Parameter | Script | Values | Purpose |
|-----------|--------|--------|---------|
| `--language` in `03_prepare_data.sh` | BOW generation | english, chinese | Controls tokenization and stopword filtering |
| `--language` in `04_train_theta.sh` | Visualization | en, zh | Controls chart label language |
| `--language` in `05_train_baseline.sh` | Visualization | en, zh | Controls chart label language |

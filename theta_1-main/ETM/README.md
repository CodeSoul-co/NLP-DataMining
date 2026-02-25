# ETM Topic Model Pipeline

Unified framework for topic model training, evaluation, and visualization.

## Directory Structure

### Result Directory Organization

All results are now organized in a standardized structure:

```
/root/autodl-tmp/result/baseline/{dataset}/{model}/
├── bow/                    # BOW data and vocabulary
│   ├── bow_matrix.npy
│   ├── vocab.json
│   ├── vocab.txt
│   └── vocab_embeddings.npy (optional)
├── model/                  # Model parameter files
│   ├── theta_k{K}.npy
│   ├── beta_k{K}.npy
│   ├── beta_over_time_k{K}.npy (DTM only)
│   ├── model_k{K}.pt
│   └── training_history_k{K}.json
├── evaluation/             # Evaluation results
│   ├── metrics_k{K}.json
│   └── metrics_k{K}.csv
├── topicwords/             # Topic words related
│   ├── topic_words_k{K}.json
│   └── topic_evolution_k{K}.json (DTM only)
└── visualization_k{K}_{lang}_{timestamp}/  # Visualization results
```

### Using ResultManager

The `ResultManager` class provides a unified interface for saving and loading results:

```python
from utils.result_manager import ResultManager

# Initialize manager
manager = ResultManager(
    result_dir='/root/autodl-tmp/result/baseline',
    dataset='edu_data',
    model='dtm',
    num_topics=20
)

# Save all results
manager.save_all(
    theta=theta,
    beta=beta,
    vocab=vocab,
    topic_words=topic_words,
    metrics=metrics,
    beta_over_time=beta_over_time,  # DTM only
    topic_evolution=topic_evolution  # DTM only
)

# Load all results
data = manager.load_all(num_topics=20)
```

### Migrating Old Results

If you have results in the old flat directory structure, use the migration utility:

```bash
# Migrate old structure to new structure
python -m utils.result_manager --action migrate --dataset edu_data --model dtm

# Or in Python
from utils.result_manager import migrate_baseline_results
migrate_baseline_results(dataset='edu_data', model='dtm')
```

## Supported Models

| Model | Description | Embedding |
|-------|-------------|-----------|
| **THETA** | Qwen Embedding-based Topic Model (Our Method) | Qwen 0.6B/4B/8B |
| **LDA** | Latent Dirichlet Allocation (sklearn) | None |
| **ETM** | Embedded Topic Model (Original) | Word2Vec |
| **CTM** | Contextualized Topic Model | SBERT |
| **DTM** | Dynamic Topic Model (Temporal) | SBERT |

## Quick Start

### Unified Entry Point: `run_pipeline.py`

```bash
# THETA model (requires model size and mode)
python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --mode zero_shot
python run_pipeline.py --dataset socialTwitter --models theta --model_size 4B --mode supervised

# Baseline models (no model_size needed)
python run_pipeline.py --dataset socialTwitter --models lda
python run_pipeline.py --dataset socialTwitter --models lda,etm,ctm

# DTM model (requires timestamp data)
python run_pipeline.py --dataset edu_data --models dtm --num_topics 20 --epochs 100

# Check if data files exist (pre-run check)
python run_pipeline.py --dataset socialTwitter --models theta --model_size 4B --check-only

# Skip training, only evaluate and visualize
python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --skip-train

# Only train, skip evaluation and visualization
python run_pipeline.py --dataset socialTwitter --models lda --skip-eval --skip-viz
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name | Required |
| `--models` | Model list (comma-separated) | Required |
| `--model_size` | Qwen model size (THETA only) | 0.6B |
| `--mode` | THETA mode | zero_shot |
| `--num_topics` | Number of topics | 20 |
| `--epochs` | Training epochs | 100 |
| `--batch_size` | Batch size | 64 |
| `--hidden_dim` | Encoder hidden dimension | 512 |
| `--learning_rate` | Learning rate | 0.002 |
| `--kl_start` | KL annealing start weight | 0.0 |
| `--kl_end` | KL annealing end weight | 1.0 |
| `--kl_warmup` | KL warmup epochs | 50 |
| `--patience` | Early stopping patience | 10 |
| `--no_early_stopping` | Disable early stopping | False |
| `--skip-train` | Skip training | False |
| `--skip-eval` | Skip evaluation | False |
| `--skip-viz` | Skip visualization | False |
| `--check-only` | Only check data files | False |
| `--gpu` | GPU ID | 0 |
| `--language` | Visualization language (en/zh) | en |

### prepare_data.py Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name | Required |
| `--model` | Target model type (theta/baseline/dtm) | Required |
| `--model_size` | Qwen model size (THETA only) | 0.6B |
| `--mode` | THETA mode | zero_shot |
| `--vocab_size` | Vocabulary size | 5000 |
| `--batch_size` | Batch size | 32 |
| `--max_length` | Embedding model max input length | 512 |
| `--clean` | Perform data cleaning first | False |
| `--raw-input` | Raw data input path | None |
| `--language` | Language for cleaning (english/chinese) | english |

### Qwen Model Sizes

| Model | Parameters | Embedding Dim | Description |
|-------|------------|---------------|-------------|
| 0.6B | 600M | 1024 | Default, fastest |
| 4B | 4B | 2560 | Medium |
| 8B | 8B | 4096 | Largest, best performance |

### Supported Datasets

- `socialTwitter`
- `hatespeech`
- `mental_health`
- `FCPB`
- `germanCoal`
- `edu_data` (Chinese education policy documents with timestamps, suitable for DTM)

## New Dataset Workflow

### Complete Data Flow

```
Raw Data → [dataclean] → Cleaned CSV → [prepare_data] → embedding/BOW → [run_pipeline] → Train/Evaluate/Visualize
```

### Option 1: Already Have Cleaned CSV

If you already have a cleaned CSV file:

```bash
# 1. Place CSV file
mkdir -p data/my_dataset
cp your_cleaned_data.csv data/my_dataset/my_dataset_cleaned.csv

# 2. Generate preprocessed data
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B

# 3. Run training
python run_pipeline.py --dataset my_dataset --models theta --model_size 0.6B
```

### Option 2: Start from Raw Data (Including Cleaning)

If you have raw CSV/TXT files that need cleaning:

```bash
# One step: clean + preprocess
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B \
    --clean --raw-input /path/to/raw_data.csv --language english

# Chinese data
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B \
    --clean --raw-input /path/to/raw_data.csv --language chinese
```

### Option 3: Step-by-Step Execution

```bash
# Step 1: Data cleaning (optional, if raw data needs cleaning)
python prepare_data.py --dataset my_dataset --model baseline \
    --clean --raw-input /path/to/raw_data.csv

# Step 2: Generate preprocessed data
# THETA model
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B --mode zero_shot

# Baseline models
python prepare_data.py --dataset my_dataset --model baseline

# Step 3: Check if data is ready
python prepare_data.py --dataset my_dataset --model theta --model_size 0.6B --check-only

# Step 4: Run training
python run_pipeline.py --dataset my_dataset --models theta --model_size 0.6B
```

### CSV File Requirements

- **Text column** (required): one of `cleaned_content`, `clean_text`, `text`, `content`
- **Label column** (optional): `label`, `category`

### Required Preprocessed Data for Each Model

| Model | Required Data | Generation Command |
|-------|---------------|--------------------|
| **THETA** | Qwen embedding + BOW + vocab_embedding | `--model theta` |
| **LDA** | BOW | `--model baseline` |
| **ETM** | BOW + Word2Vec (auto-generated) | `--model baseline` |
| **CTM** | BOW + SBERT embedding | `--model baseline` |
| **DTM** | BOW + SBERT embedding + time slice info | `--model dtm` |

## DTM Model Guide

DTM (Dynamic Topic Model) is used to analyze topic evolution over time and requires timestamped data.

### DTM Data Requirements

CSV file must contain:
- **Text column**: `cleaned_content`, `text`, etc.
- **Time column**: `year`, `timestamp`, `date`, etc. (for time slice division)

### DTM Usage Examples

```bash
# 1. Process data from docx directory (automatically uses dataclean module)
python prepare_data.py --dataset edu_data --model dtm \
    --clean --raw-input /path/to/docx_directory --language chinese

# 2. Prepare DTM data from existing CSV
python prepare_data.py --dataset my_data --model dtm --time_column year

# 3. Train DTM model
python run_pipeline.py --dataset edu_data --models dtm --num_topics 20 --epochs 100
```

### DTM Output Files

```
result/baseline/{dataset}/dtm/
├── theta_k20.npy              # Document-topic distribution
├── beta_k20.npy               # Topic-word distribution (last time slice)
├── beta_over_time_k20.npy     # Topic-word distribution for all time slices
├── topic_words_k20.json       # Topic words
├── topic_evolution_k20.json   # Topic word evolution over time
├── training_history_k20.json  # Training history (loss and perplexity)
├── metrics_k20.json           # Evaluation metrics
└── visualization/             # Visualization charts
```

## Project Structure

```
ETM/
├── run_pipeline.py      # Unified entry script ⭐
├── prepare_data.py      # Data preprocessing script ⭐
├── main.py              # THETA model main entry
├── config.py            # Configuration management
│
├── model/               # Model definitions
│   ├── baseline_trainer.py  # Baseline trainer (LDA/ETM/CTM/DTM)
│   ├── etm.py              # THETA model
│   ├── lda.py              # LDA model
│   ├── ctm.py              # CTM model
│   ├── dtm.py              # DTM model (Dynamic Topic Model)
│   └── etm_original.py     # Original ETM model
│
├── evaluation/          # Evaluation module
│   ├── unified_evaluator.py  # Unified evaluator ⭐
│   └── topic_metrics.py      # Evaluation metrics
│
├── visualization/       # Visualization module
│   ├── run_visualization.py
│   └── visualization_generator.py
│
├── bow/                 # BOW generation
├── data/                # Data loading
├── dataclean/           # Data cleaning
└── preprocessing/       # Preprocessing
```

## Data Flow

```
Data Processing → Model Training → Evaluation → Visualization → Result Saving
```

### Result Storage Locations

- **THETA**: `result/0.6B/{dataset}/{mode}/`
- **Baseline**: `result/baseline/{dataset}/{model}/`

### Evaluation Metrics (7 total)

1. **TD** - Topic Diversity
2. **iRBO** - Inverse Rank-Biased Overlap
3. **NPMI** - Normalized PMI Coherence
4. **C_V** - C_V Coherence
5. **UMass** - UMass Coherence
6. **Exclusivity** - Topic Exclusivity
7. **PPL** - Perplexity

## Examples

### Train LDA and View Results

```bash
# Train
python run_pipeline.py --dataset socialTwitter --models lda

# Result location
ls result/baseline/socialTwitter/lda/
# theta_k20.npy, beta_k20.npy, topic_words_k20.json, metrics_k20.json
# visualization/global/*.png
```

### Compare Multiple Models

```bash
# Train LDA, ETM, CTM simultaneously
python run_pipeline.py --dataset socialTwitter --models lda,etm,ctm

# View evaluation results
cat result/baseline/socialTwitter/lda/metrics_k20.json
cat result/baseline/socialTwitter/etm/metrics_k20.json
cat result/baseline/socialTwitter/ctm_zeroshot/metrics_k20.json
```

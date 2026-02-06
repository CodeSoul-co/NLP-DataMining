# prepare_data.py

Data preprocessing script for generating embeddings and bag-of-words representations.

---

## Basic Usage

```bash
python prepare_data.py --dataset DATASET --model MODEL [OPTIONS]
```

---

## Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--dataset` | string | Dataset name (must match directory name in `/root/autodl-tmp/data/`) |
| `--model` | string | Model type: `theta`, `baseline`, or `dtm` |

## Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_size` | string | `0.6B` | Qwen model size: `0.6B`, `4B`, or `8B` (THETA only) |
| `--mode` | string | `zero_shot` | Training mode: `zero_shot`, `supervised`, or `unsupervised` (THETA only) |

## Data Processing

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--vocab_size` | int | `5000` | 1000-20000 | Vocabulary size for BOW representation |
| `--batch_size` | int | `32` | 8-128 | Batch size for embedding generation |
| `--max_length` | int | `512` | 128-2048 | Maximum sequence length for embeddings |

## GPU Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gpu` | int | `0` | GPU device ID (0, 1, 2, ...) |

## Data Cleaning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--clean` | flag | False | Clean data before preprocessing |
| `--raw-input` | string | None | Path to raw CSV file (requires `--clean`) |
| `--language` | string | `english` | Cleaning language: `english` or `chinese` |

## Utility Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--bow-only` | flag | False | Generate BOW only, skip embeddings |
| `--check-only` | flag | False | Check if preprocessed files exist |
| `--time_column` | string | `year` | Time column name for DTM (DTM only) |

---

## Examples

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

---

## Output Files

Preprocessed data is saved to:
```
/root/autodl-tmp/result/{model_size}/{dataset}/bow/
```

Generated files:
- `qwen_embeddings_{mode}.npy`: Document embeddings
- `vocab.pkl`: Vocabulary dictionary
- `doc_indices.npy`: Document-term indices
- `bow_matrix.npz`: Sparse BOW matrix

# Data Preprocessing

Preprocessing converts cleaned text into numerical representations required for training. This stage generates embeddings using Qwen models and constructs bag-of-words representations.

---

## THETA Model Preprocessing

### Basic Preprocessing

For a dataset named `my_dataset` with a cleaned CSV file:

```bash
cd /root/autodl-tmp/ETM

python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --gpu 0
```

This command:
1. Loads the CSV from `/root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv`
2. Generates Qwen embeddings (dimension 1024 for 0.6B model)
3. Constructs bag-of-words with vocabulary size 5000
4. Saves output to `/root/autodl-tmp/result/0.6B/my_dataset/bow/`

### Model Size Selection

**0.6B Model - Default choice for most use cases**

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

Processing speed: ~1000 documents per minute on single GPU
Memory requirement: 4GB VRAM

**4B Model - Better quality at moderate cost**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 4B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 16 \
    --gpu 0
```

Processing speed: ~400 documents per minute
Memory requirement: 12GB VRAM
Batch size reduced to 16 due to larger embeddings (dimension 2560)

**8B Model - Highest quality**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 8B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 8 \
    --gpu 0
```

Processing speed: ~200 documents per minute
Memory requirement: 24GB VRAM
Batch size reduced to 8 due to large embeddings (dimension 4096)

### Training Mode Selection

**zero_shot mode - Standard unsupervised learning**

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

Use when: No labels available or labels should be ignored

**supervised mode - Label-guided learning**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode supervised \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

Use when: CSV contains `label` or `category` column
The model will incorporate label information during training

**unsupervised mode - Explicit unsupervised mode**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode unsupervised \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

Use when: Comparing with zero_shot mode on labeled data while ignoring labels

### Vocabulary Configuration

Vocabulary size affects model capacity and training speed. Larger vocabularies capture more terms but increase computation.

| Vocabulary Size | Appropriate For |
|----------------|-----------------|
| 3000-5000 | Small corpora, domain-specific text, faster training |
| 5000-8000 | General purpose, default setting |
| 8000-15000 | Large diverse corpora, capturing rare terms |

### Sequence Length Configuration

The `max_length` parameter controls input truncation for embedding generation.

| Max Length | Appropriate For |
|-----------|-----------------|
| 256 | Short documents (tweets, reviews), faster processing |
| 512 | Medium documents (news articles), default setting |
| 1024 | Long documents (papers, reports), requires more VRAM |

### Combined Cleaning and Preprocessing

Process raw data in a single step:

**English data:**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --clean \
    --raw-input /root/autodl-tmp/data/my_dataset/raw_data.csv \
    --language english \
    --gpu 0
```

**Chinese data:**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --clean \
    --raw-input /root/autodl-tmp/data/my_dataset/raw_data.csv \
    --language chinese \
    --gpu 0
```

The `--clean` flag triggers automatic cleaning before preprocessing. The cleaned CSV is saved as `{dataset}_cleaned.csv` in the dataset directory.

### Verifying Preprocessed Data

Check that all required files were generated:

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --check-only
```

Expected output:
```
Checking preprocessed files for dataset: my_dataset
✓ BOW data: /root/autodl-tmp/result/0.6B/my_dataset/bow/
✓ Embeddings: qwen_embeddings_zeroshot.npy (1024 dims)
✓ Vocabulary: vocab.pkl (5000 words)
✓ Document indices: doc_indices.npy
All required files present.
```

---

## Baseline Model Preprocessing

Baseline models (LDA, ETM, CTM) use different preprocessing pipelines that do not require Qwen embeddings.

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model baseline \
    --vocab_size 5000
```

This generates:
- Bag-of-words representations
- TF-IDF vectors (for CTM)
- Word2Vec embeddings (for ETM)
- Document-term matrix (for LDA)

Output location: `/root/autodl-tmp/result/baseline/my_dataset/bow/`

---

## DTM Model Preprocessing

DTM requires temporal information in the CSV. Specify the time column name:

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model dtm \
    --vocab_size 5000 \
    --time_column year
```

The time column can be named `year`, `timestamp`, or `date`. Documents are automatically grouped by time slice for temporal modeling.

# Quick Start

This tutorial demonstrates how to train a THETA model on your dataset in under 5 minutes.

---

## Step 1: Prepare Your Data

Create a CSV file with your text data. The CSV must contain a column with text content.

**Example CSV format:**

```csv
text
"First document discussing climate change and global warming."
"Second document about renewable energy sources."
"Third document on environmental policy and regulations."
```

**Required columns:**

| Column Name | Type | Required | Description |
|------------|------|----------|-------------|
| text / content / cleaned_content / clean_text | string | Yes | Text content for topic modeling |
| label / category | string/int | No | Labels for supervised mode |
| year / timestamp / date | int/string | No | Timestamp for DTM model |

Save your CSV file to the data directory:

```bash
mkdir -p /root/autodl-tmp/data/my_dataset
cp your_data.csv /root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv
```

Note: The CSV filename must follow the pattern `{dataset_name}_cleaned.csv`.

---

## Step 2: Preprocess Data

Generate embeddings and bag-of-words representations:

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

**What this does:**
1. Loads your CSV file
2. Generates Qwen embeddings for all documents
3. Creates bag-of-words representations
4. Builds vocabulary
5. Saves preprocessed data to `/root/autodl-tmp/result/0.6B/my_dataset/bow/`

**Expected output:**
```
Loading dataset: my_dataset
Processing 1000 documents...
Generating embeddings: 100%|████████| 32/32 [00:45<00:00, 1.41s/batch]
Building vocabulary (size=5000)...
Saving preprocessed data...
Done! Files saved to /root/autodl-tmp/result/0.6B/my_dataset/bow/
```

Verify that data files were created:

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --check-only
```

---

## Step 3: Train the Model

Train a THETA model with 20 topics:

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
    --gpu 0 \
    --language en
```

**Training parameters explained:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--num_topics` | 20 | Number of topics to discover |
| `--epochs` | 100 | Maximum training epochs |
| `--batch_size` | 64 | Batch size for training |
| `--hidden_dim` | 512 | Hidden dimension of encoder |
| `--learning_rate` | 0.002 | Learning rate for optimizer |
| `--kl_start` | 0.0 | Initial KL annealing weight |
| `--kl_end` | 1.0 | Final KL annealing weight |
| `--kl_warmup` | 50 | Epochs for KL warmup |
| `--patience` | 10 | Early stopping patience |

**Training progress:**
```
Epoch 1/100: Loss=245.32, ELBO=-243.12, KL=2.20
Epoch 10/100: Loss=156.78, ELBO=-154.56, KL=2.22
Epoch 20/100: Loss=142.35, ELBO=-139.87, KL=2.48
...
Epoch 65/100: Loss=128.45, ELBO=-125.23, KL=3.22
Early stopping triggered at epoch 65
Training completed in 23.5 minutes
```

The training process automatically:
1. Trains the model
2. Evaluates on multiple metrics
3. Generates visualizations
4. Saves all results

---

## Step 4: View Results

After training, results are saved in:

```
/root/autodl-tmp/result/0.6B/my_dataset/zero_shot/
├── checkpoints/
│   └── best_model.pt
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

**View evaluation metrics:**

```bash
cat /root/autodl-tmp/result/0.6B/my_dataset/zero_shot/metrics/evaluation_results.json
```

Example output:
```json
{
  "TD": 0.89,
  "iRBO": 0.76,
  "NPMI": 0.42,
  "C_V": 0.65,
  "UMass": -2.34,
  "Exclusivity": 0.82,
  "PPL": 145.23
}
```

**View visualizations:**

Open the visualization files in your browser or image viewer:
- `topic_words_bars.png`: Bar charts showing top words for each topic
- `topic_similarity.png`: Heatmap of topic similarities
- `doc_topic_umap.png`: UMAP projection of documents in topic space
- `pyldavis.html`: Interactive visualization (open in browser)

---

## What's Next?

- [User Guide](../user-guide/data-preparation.md) - Complete workflow documentation
- [Advanced Usage](../advanced/custom-datasets.md) - Advanced features
- [Examples](../examples/english-dataset.md) - Real-world use cases

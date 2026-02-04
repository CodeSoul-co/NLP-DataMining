# User Guide

This guide covers the complete workflow for using THETA, from preparing your data to analyzing results.

---

## Data Preparation

### Data Format Requirements

THETA accepts CSV files with specific column requirements. The preprocessing pipeline recognizes several standard column names for text content.

**Accepted text column names:**
- `text`
- `content`
- `cleaned_content`
- `clean_text`

**Optional columns:**
- `label` or `category` - Required for supervised mode
- `year`, `timestamp`, or `date` - Required for DTM (temporal analysis)

Example CSV structure:

```csv
text,label,year
"Document about renewable energy and solar panels.",Environment,2020
"Article discussing machine learning applications.",Technology,2021
"Policy paper on healthcare reform.",Healthcare,2022
```

### Data Cleaning

Raw text often contains noise that degrades topic quality. The data cleaning module handles common issues in both English and Chinese text.

#### English Data Cleaning

```bash
cd /root/autodl-tmp/ETM

python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language english
```

The cleaning process removes:
- HTML tags and markup
- URLs and email addresses
- Special characters and symbols
- Extra whitespace
- Non-printable characters

#### Chinese Data Cleaning

Chinese text requires specialized processing for proper segmentation and cleaning.

```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language chinese
```

Additional steps for Chinese:
- Removes traditional punctuation marks
- Handles full-width and half-width characters
- Preserves Chinese word boundaries

#### Batch Cleaning

Process multiple files in a directory:

```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw/ \
    --output /root/autodl-tmp/data/cleaned/ \
    --language english
```

All CSV files in the input directory will be processed and saved to the output directory with the same filenames.

---

## Data Preprocessing

Preprocessing converts cleaned text into numerical representations required for training. This stage generates embeddings using Qwen models and constructs bag-of-words representations.

### THETA Model Preprocessing

#### Basic Preprocessing

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

#### Model Size Selection

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

#### Training Mode Selection

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

#### Vocabulary Configuration

Vocabulary size affects model capacity and training speed. Larger vocabularies capture more terms but increase computation.

**Small vocabulary (3000-5000):**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 3000 \
    --batch_size 32 \
    --gpu 0
```

Appropriate for: Small corpora, domain-specific text, faster training

**Standard vocabulary (5000-8000):**
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

Appropriate for: General purpose, default setting

**Large vocabulary (8000-15000):**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 10000 \
    --batch_size 32 \
    --gpu 0
```

Appropriate for: Large diverse corpora, capturing rare terms

#### Sequence Length Configuration

The `max_length` parameter controls input truncation for embedding generation.

**Short sequences (256):**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --max_length 256 \
    --gpu 0
```

Appropriate for: Short documents (tweets, reviews), faster processing

**Standard sequences (512):**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --max_length 512 \
    --gpu 0
```

Appropriate for: Medium documents (news articles), default setting

**Long sequences (1024):**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --max_length 1024 \
    --gpu 0
```

Appropriate for: Long documents (papers, reports), requires more VRAM

#### Combined Cleaning and Preprocessing

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

#### Verifying Preprocessed Data

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

### Baseline Model Preprocessing

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

### DTM Model Preprocessing

DTM requires temporal information in the CSV. Specify the time column name:

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model dtm \
    --vocab_size 5000 \
    --time_column year
```

The time column can be named `year`, `timestamp`, or `date`. Documents are automatically grouped by time slice for temporal modeling.

---

## Training Models

### THETA Model Training

#### Basic Training

Train a THETA model with default hyperparameters:

```bash
cd /root/autodl-tmp/ETM

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

Training typically completes in 20-40 minutes depending on dataset size and hardware.

#### Topic Number Selection

The number of topics is a key hyperparameter that affects granularity:

**Few topics (10-15):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 10 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

Appropriate for: Small corpora, broad categories, high-level overview

**Standard topics (20-30):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

Appropriate for: Medium corpora, balanced granularity, default choice

**Many topics (40-100):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 50 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

Appropriate for: Large diverse corpora, fine-grained analysis, detailed taxonomy

#### Learning Rate Tuning

Learning rate affects convergence speed and final quality:

**Conservative learning rate (0.001):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --gpu 0 \
    --language en
```

Use when: Training is unstable, loss oscillates, need slower convergence

**Standard learning rate (0.002):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en
```

Use when: Default choice for most datasets

**Aggressive learning rate (0.005):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.005 \
    --gpu 0 \
    --language en
```

Use when: Training is too slow, need faster convergence

#### KL Annealing Configuration

KL annealing gradually increases the KL divergence weight during training to prevent posterior collapse.

**Standard KL annealing:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 50 \
    --gpu 0 \
    --language en
```

Weight increases linearly from 0.0 to 1.0 over 50 epochs.

**Slow KL annealing:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 80 \
    --gpu 0 \
    --language en
```

Longer warmup period helps prevent early posterior collapse.

**Partial KL annealing:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --kl_start 0.1 \
    --kl_end 0.9 \
    --kl_warmup 30 \
    --gpu 0 \
    --language en
```

Starts with non-zero weight and stops before full weight.

#### Hidden Dimension Configuration

Hidden dimension controls the capacity of the variational encoder:

**Small hidden dimension (256):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 256 \
    --gpu 0 \
    --language en
```

Use when: Small datasets, faster training, limited VRAM

**Standard hidden dimension (512):**
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
    --gpu 0 \
    --language en
```

Use when: Default choice for most datasets

**Large hidden dimension (768 or 1024):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 768 \
    --gpu 0 \
    --language en
```

Use when: Large complex datasets, sufficient VRAM available

#### Early Stopping

Early stopping prevents overfitting by monitoring validation performance:

**With early stopping (default):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --patience 10 \
    --gpu 0 \
    --language en
```

Training stops if no improvement after 10 epochs.

**Without early stopping:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 200 \
    --batch_size 64 \
    --no_early_stopping \
    --gpu 0 \
    --language en
```

Trains for all specified epochs regardless of performance.

#### Chinese Data Training

Training Chinese data requires setting the language parameter:

```bash
python run_pipeline.py \
    --dataset chinese_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language zh
```

The language parameter affects visualization rendering (fonts, layout) but does not change the training algorithm.

#### Supervised Training

For datasets with labels:

```bash
python run_pipeline.py \
    --dataset labeled_dataset \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

The model incorporates label information to guide topic discovery.

### Baseline Model Training

#### LDA Training

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

LDA uses Gibbs sampling and does not utilize GPU acceleration. Training completes faster than neural models.

#### ETM Training

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models etm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en
```

ETM uses Word2Vec embeddings (300 dimensions) instead of Qwen embeddings.

#### CTM Training

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models ctm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en
```

CTM uses SBERT embeddings (768 dimensions). Training time is between LDA and THETA.

#### DTM Training

```bash
python run_pipeline.py \
    --dataset temporal_dataset \
    --models dtm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en
```

DTM models topic evolution across time slices defined by the time column in preprocessing.

#### Training Multiple Models

Compare multiple models simultaneously:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

Models train sequentially. Results are saved in separate directories for comparison.

---

## Evaluation

Training automatically runs evaluation using multiple metrics. Results are saved in JSON format.

### Evaluation Metrics

**Topic Diversity (TD)**
- Range: 0-1
- Higher is better
- Measures uniqueness of topics
- Computed as percentage of unique words in top-N words across all topics

**Inverse Rank-Biased Overlap (iRBO)**
- Range: 0-1
- Higher is better
- Measures topic distinctiveness
- Lower values indicate redundant or overlapping topics

**Normalized PMI (NPMI)**
- Range: -1 to 1
- Higher is better
- Measures semantic coherence of topic words
- Based on pointwise mutual information in external corpus

**C_V Coherence**
- Range: 0-1
- Higher is better
- Alternative coherence measure
- Based on sliding window co-occurrence

**UMass Coherence**
- Range: Negative values
- Closer to 0 is better
- Classic coherence metric
- Based on document co-occurrence

**Exclusivity**
- Range: 0-1
- Higher is better
- Measures topic specificity
- Computed using FREX score

**Perplexity (PPL)**
- Range: Positive values
- Lower is better
- Measures model fit on held-out data
- Standard evaluation for probabilistic models

### Viewing Evaluation Results

Results are saved in the metrics directory:

```bash
cat /root/autodl-tmp/result/0.6B/my_dataset/zero_shot/metrics/evaluation_results.json
```

Example output:
```json
{
  "TD": 0.891,
  "iRBO": 0.762,
  "NPMI": 0.418,
  "C_V": 0.654,
  "UMass": -2.341,
  "Exclusivity": 0.823,
  "PPL": 145.23,
  "training_time": 1425.6,
  "num_topics": 20,
  "num_documents": 5000
}
```

### Running Evaluation Separately

Skip training and only evaluate existing models:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --skip-train \
    --gpu 0 \
    --language en
```

This loads the trained model from checkpoints and recomputes all metrics.

### Comparing Multiple Models

Evaluate all baseline models:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --skip-train \
    --gpu 0 \
    --language en
```

Results for each model are saved separately. Use the metrics files to construct comparison tables.

---

## Visualization

Training automatically generates visualizations. Additional visualizations can be created separately.

### Visualization Outputs

**topic_words_bars.png**
Bar charts showing top-10 words for each topic with probability weights.

**topic_similarity.png**
Heatmap showing cosine similarity between topic-word distributions.

**doc_topic_umap.png**
UMAP projection of documents in topic space. Points are colored by dominant topic.

**topic_wordclouds.png**
Word clouds for each topic sized by word probability.

**metrics.png**
Bar charts comparing evaluation metrics.

**pyldavis.html**
Interactive visualization using pyLDAvis library. Open in web browser.

### Generating Visualizations Separately

Generate visualizations for THETA models:

```bash
cd /root/autodl-tmp/ETM

python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 300
```

Generate visualizations for baseline models:

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

Replace `lda` with `etm`, `ctm`, or `dtm` for other baseline models.

### Customizing Visualization

**Higher resolution:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 600
```

**Chinese language visualizations:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset chinese_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language zh \
    --dpi 300
```

Chinese visualizations use appropriate fonts and handle character rendering correctly.

### Skipping Visualization During Training

Skip automatic visualization to save time:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --skip-viz \
    --gpu 0 \
    --language en
```

Visualizations can be generated later using the separate visualization command.

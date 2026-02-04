# Getting Started

This guide will help you install THETA and train your first topic model.

---

## Installation

### System Requirements

THETA requires the following system specifications:

**Operating System:**
- Linux (Ubuntu 18.04 or later recommended)
- macOS 10.14 or later
- Windows 10/11 with WSL2

**Hardware Requirements:**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.9+ |
| RAM | 8GB | 16GB+ |
| GPU Memory | 4GB (0.6B model) | 12GB+ (4B model) |
| CUDA | 11.8+ | 12.1+ |
| Storage | 20GB | 50GB+ |

**Model-Specific GPU Requirements:**

| Model Size | Parameters | Embedding Dim | VRAM Required | Use Case |
|-----------|-----------|---------------|---------------|----------|
| 0.6B | 600M | 1024 | ~4GB | Quick experiments, limited resources |
| 4B | 4B | 2560 | ~12GB | Balanced performance and speed |
| 8B | 8B | 4096 | ~24GB | Best quality results |

---

### Installation Steps

#### Step 1: Clone the Repository

```bash
git clone https://github.com/CodeSoul-co/THETA.git
cd THETA
```

#### Step 2: Create Virtual Environment

Using conda (recommended):

```bash
conda create -n theta python=3.9
conda activate theta
```

Using venv:

```bash
python -m venv theta-env
source theta-env/bin/activate  # On Linux/macOS
# theta-env\Scripts\activate   # On Windows
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The installation includes the following key packages:
- PyTorch (with CUDA support)
- Transformers
- Sentence-Transformers
- Gensim
- scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn
- UMAP-learn

#### Step 4: Download Embedding Models

Download the Qwen3-Embedding models:

```bash
# For 0.6B model (recommended for first-time users)
python scripts/download_models.py --model 0.6B

# For 4B model
python scripts/download_models.py --model 4B

# For 8B model
python scripts/download_models.py --model 8B
```

Models will be downloaded to `/root/embedding_models/` by default.

---

### Verify Installation

Check that PyTorch and CUDA are properly installed:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch version: 2.0.1+cu118
CUDA available: True
CUDA version: 11.8
```

Check THETA installation:

```bash
python -c "from src.model import etm; print('THETA installed successfully')"
```

---

## Quick Start

This tutorial demonstrates how to train a THETA model on your dataset in under 5 minutes.

### Step 1: Prepare Your Data

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

### Step 2: Preprocess Data

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

### Step 3: Train the Model

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

### Step 4: View Results

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

## Project Overview

Understanding THETA's architecture and workflow will help you use it effectively.

### Architecture Overview

THETA consists of three main components:

1. **Embedding Module**: Generates contextual embeddings using Qwen3-Embedding
2. **Topic Model**: Neural variational inference for topic discovery
3. **Evaluation & Visualization**: Comprehensive assessment and presentation

**Data flow:**

```
Raw Text → Data Cleaning → Preprocessing → Training → Evaluation → Visualization
              ↓              ↓               ↓           ↓            ↓
         Cleaned CSV    Embeddings+BOW   Model Ckpt  Metrics     Figures
```

---

### Supported Models

THETA supports multiple topic modeling approaches:

#### THETA Model (Our Method)

**Architecture:**
- Variational autoencoder with Qwen3-Embedding
- Neural encoder for topic distribution inference
- Reconstruction via topic-word distributions

**Training Modes:**

| Mode | Description | Use Case | Requirements |
|------|-------------|----------|--------------|
| zero_shot | Unsupervised learning | No labels available | Text column only |
| supervised | Label-guided learning | Labels available | Text + label columns |
| unsupervised | Unsupervised (ignores labels) | Compare with zero_shot | Text column only |

**Model Sizes:**

All three sizes share the same architecture but differ in embedding quality:
- **0.6B**: Fastest, suitable for development and testing
- **4B**: Balanced performance for production use
- **8B**: Best quality for research and high-stakes applications

#### Baseline Models

**LDA (Latent Dirichlet Allocation)**
- Classic probabilistic topic model
- No neural components
- Fast and interpretable

**ETM (Embedded Topic Model)**
- Uses Word2Vec embeddings
- Neural topic model
- Better than LDA, faster than THETA

**CTM (Contextualized Topic Model)**
- Uses SBERT embeddings
- Contextualized representations
- Good balance of quality and speed

**DTM (Dynamic Topic Model)**
- Temporal topic modeling
- Tracks topic evolution over time
- Requires timestamp information

---

### Directory Structure

THETA organizes files in the following structure:

#### Project Directory

```
/root/autodl-tmp/ETM/
├── main.py                   # THETA training script
├── run_pipeline.py           # Unified entry point
├── prepare_data.py           # Data preprocessing
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── dataclean/               # Data cleaning module
│   └── main.py
├── src/
│   ├── bow/                 # BOW generation
│   ├── model/               # Model definitions
│   │   ├── etm.py          # THETA/ETM model
│   │   ├── lda.py          # LDA model
│   │   ├── ctm.py          # CTM model
│   │   └── baseline_trainer.py
│   ├── evaluation/          # Evaluation metrics
│   │   ├── topic_metrics.py
│   │   └── unified_evaluator.py
│   ├── visualization/       # Visualization
│   │   ├── run_visualization.py
│   │   ├── topic_visualizer.py
│   │   └── visualization_generator.py
│   └── utils/               # Utilities
│       └── result_manager.py
└── scripts/
    └── download_models.py
```

#### Data Directory

```
/root/autodl-tmp/data/
└── {dataset_name}/
    └── {dataset_name}_cleaned.csv
```

#### Results Directory

```
/root/autodl-tmp/result/
├── 0.6B/                    # THETA 0.6B results
│   └── {dataset_name}/
│       ├── bow/             # Shared by all modes
│       ├── zero_shot/       # Zero-shot results
│       │   ├── checkpoints/
│       │   ├── metrics/
│       │   └── visualizations/
│       ├── supervised/      # Supervised results
│       └── unsupervised/    # Unsupervised results
├── 4B/                      # THETA 4B results
├── 8B/                      # THETA 8B results
└── baseline/                # Baseline results
    └── {dataset_name}/
        ├── bow/
        ├── lda/
        │   └── K20/        # 20 topics
        ├── etm/
        ├── ctm/
        └── dtm/
```

#### Embedding Models Directory

```
/root/embedding_models/
├── qwen3_embedding_0.6B/
├── qwen3_embedding_4B/
└── qwen3_embedding_8B/
```

---

### Workflow Summary

The typical THETA workflow consists of four stages:

**Stage 1: Data Preparation**
1. Collect raw text data
2. Clean and format as CSV
3. Ensure proper column names

**Stage 2: Preprocessing**
1. Run `prepare_data.py` to generate embeddings
2. Create bag-of-words representations
3. Build vocabulary
4. Save preprocessed files

**Stage 3: Training**
1. Run `run_pipeline.py` to train model
2. Model trains with early stopping
3. Automatic evaluation on multiple metrics
4. Automatic visualization generation

**Stage 4: Analysis**
1. Review evaluation metrics
2. Examine visualizations
3. Analyze discovered topics
4. Compare with baseline models

---

### Next Steps

Now that you have successfully installed THETA and trained your first model, you can:

- Explore the **User Guide** for detailed documentation on each component
- Try different **training modes** (supervised, unsupervised)
- Experiment with **different model sizes** (4B, 8B)
- Learn about **hyperparameter tuning** in the Advanced Usage section
- Compare THETA with **baseline models** (LDA, ETM, CTM)
- Process **Chinese text data** with specialized pipelines
- Analyze **temporal dynamics** with DTM

Continue to the User Guide to learn more about each stage of the workflow.

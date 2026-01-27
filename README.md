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
- Generative topic models
  - ETM (Embedded Topic Model) for coherent and interpretable topics
  - DTM (Dynamic Topic Model) for topic evolution over time
- Scientific validation via 7 intrinsic metrics (PPL, TD, iRBO, NPMI, C_V, UMass, Exclusivity)
- Comprehensive visualization with bilingual support (English/Chinese)

THETA aims to move topic modeling from "clustering with pretty plots" to a reproducible, validated scientific workflow.

---

## Key Features

- **Hybrid embedding topic analysis**: Zero-shot / Supervised / Unsupervised modes
- **Multiple Qwen model sizes**: 0.6B (1024-dim), 4B (2560-dim), 8B (4096-dim)
- **Baseline models**: LDA, ETM, CTM, DTM for comparison
- **Data governance**: Domain-aware cleaning for multiple languages (English, Chinese, German, Spanish)
- **Unified evaluation**: 7 metrics with JSON/CSV export
- **Rich visualization**: 20+ chart types with bilingual labels

---

## Project Structure

```
/root/autodl-tmp/
├── ETM/                          # Main codebase
│   ├── run_pipeline.py           # Unified entry point
│   ├── prepare_data.py           # Data preprocessing
│   ├── config.py                 # Configuration management
│   ├── dataclean/                # Data cleaning module
│   ├── model/                    # Model implementations
│   ├── evaluation/               # Evaluation metrics
│   ├── visualization/            # Visualization tools
│   └── utils/                    # Utilities (ResultManager, etc.)
├── data/                         # Raw datasets
├── result/                       # Training results
│   ├── 0.6B/                     # THETA 0.6B results
│   ├── 4B/                       # THETA 4B results
│   ├── 8B/                       # THETA 8B results
│   └── baseline/                 # Baseline model results
└── embedding_models/             # Qwen embedding models
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
cd THETA/ETM

pip install -r requirements.txt
```

---

## Quickstart

### A) THETA Model (Zero-shot Mode)

```bash
# 1) Prepare data (generate Qwen embeddings + BOW)
python prepare_data.py --dataset socialTwitter --model theta --model_size 0.6B --mode zero_shot

# 2) Run complete pipeline (train + evaluate + visualize)
python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --mode zero_shot

# 3) Skip training, only evaluate and visualize existing results
python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --skip-train
```

### B) THETA Model (Supervised Mode with 4B/8B)

```bash
# Use larger model with supervised fine-tuning
python run_pipeline.py --dataset hatespeech --models theta --model_size 4B --mode supervised

# Use 8B model for best quality
python run_pipeline.py --dataset mental_health --models theta --model_size 8B --mode supervised
```

### C) Baseline Models (LDA, ETM, CTM)

```bash
# Train single baseline model
python run_pipeline.py --dataset socialTwitter --models lda

# Train multiple baseline models
python run_pipeline.py --dataset socialTwitter --models lda,etm,ctm

# With custom parameters
python run_pipeline.py --dataset hatespeech --models etm --num_topics 30 --epochs 200
```

### D) Dynamic Topic Model (DTM)

```bash
# Prepare data with time slices (requires timestamp column)
python prepare_data.py --dataset edu_data --model dtm --time_column year

# Train DTM model
python run_pipeline.py --dataset edu_data --models dtm --num_topics 20 --epochs 100

# Generate visualization in Chinese
python run_pipeline.py --dataset edu_data --models dtm --skip-train --language zh
```
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

## Contact

Please contact us if you have any questions:
- duanzhenke@code-soul.com
- panjiqun@code-soul.com
- lixin@code-soul.com

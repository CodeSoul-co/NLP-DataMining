# Project Overview

Understanding THETA's architecture and workflow will help you use it effectively.

---

## Architecture Overview

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

## Supported Models

THETA supports multiple topic modeling approaches:

### THETA Model (Our Method)

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

### Baseline Models

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

## Directory Structure

THETA organizes files in the following structure:

### Project Directory

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

### Data Directory

```
/root/autodl-tmp/data/
└── {dataset_name}/
    └── {dataset_name}_cleaned.csv
```

### Results Directory

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

### Embedding Models Directory

```
/root/embedding_models/
├── qwen3_embedding_0.6B/
├── qwen3_embedding_4B/
└── qwen3_embedding_8B/
```

---

## Workflow Summary

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

## Next Steps

Now that you understand the architecture, you can:

- Explore the **[User Guide](../user-guide/data-preparation.md)** for detailed documentation on each component
- Try different **training modes** (supervised, unsupervised)
- Experiment with **different model sizes** (4B, 8B)
- Learn about **[hyperparameter tuning](../advanced/hyperparameters.md)** in the Advanced Usage section
- Compare THETA with **[baseline models](../models/baselines.md)** (LDA, ETM, CTM)
- Process **[Chinese text data](../advanced/chinese-data.md)** with specialized pipelines

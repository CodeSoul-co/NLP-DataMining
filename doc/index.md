# THETA Topic Model

**Advanced Topic Modeling with Qwen Embeddings**

---

THETA is a state-of-the-art topic modeling framework that leverages **Qwen3-Embedding** models to achieve superior performance in topic discovery and analysis. Designed as an improvement over traditional topic models like LDA and ETM, THETA combines the power of large language model embeddings with advanced neural topic modeling architectures.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Install THETA and train your first topic model in minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Complete workflow from data preparation to result analysis

    [:octicons-arrow-right-24: User Guide](user-guide/data-preparation.md)

-   :material-brain:{ .lg .middle } **Models**

    ---

    Architecture details of THETA and baseline models

    [:octicons-arrow-right-24: Models](models/theta.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete parameter documentation for all CLI tools

    [:octicons-arrow-right-24: API Reference](api/prepare-data.md)

</div>

---

## Key Features

| Feature | Description |
|---------|-------------|
| :material-chip: **Powerful Embeddings** | Built on Qwen3-Embedding (0.6B / 4B / 8B) for superior semantic understanding |
| :material-tune: **Flexible Training** | Zero-shot, supervised, and unsupervised modes |
| :material-chart-box: **Rich Visualizations** | Topic distributions, heatmaps, UMAP projections, pyLDAvis |
| :material-translate: **Multilingual** | Full support for English and Chinese data |
| :material-cog: **Extensible** | Easy customization with new datasets and configurations |
| :material-speedometer: **Comprehensive Evaluation** | TD, TC, NPMI, and more metrics |

---

## Model Comparison

| Model | Embedding | Type | Characteristics |
|-------|-----------|------|----------------|
| **THETA** | Qwen3-Embedding | Neural | Our method — best performance |
| LDA | — | Probabilistic | Classic generative model |
| ETM | Word2Vec | Neural | Embedded topic model |
| CTM | SBERT | Neural | Contextualized model |
| DTM | SBERT | Neural | Dynamic temporal model |

---

## Quick Example

```bash
# 1. Preprocess data
python prepare_data.py \
    --dataset 20ng \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --gpu 0

# 2. Train model
python run_pipeline.py \
    --dataset 20ng \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --gpu 0
```

---

## Citation

If you use THETA in your research, please cite:

```bibtex
@article{theta2025,
  title={THETA: Advanced Topic Modeling with Qwen Embeddings},
  author={CodeSoul},
  year={2025}
}
```

---

## Links

- [:fontawesome-brands-github: GitHub Repository](https://github.com/CodeSoul-co/THETA)
- [:material-web: Website](https://theta.code-soul.com)

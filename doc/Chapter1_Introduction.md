# THETA Topic Model Documentation

**Version:** 2026-02-03  
**Website:** https://theta.code-soul.com  
**GitHub:** https://github.com/CodeSoul-co/THETA

---

## Welcome to THETA

THETA is a state-of-the-art topic modeling framework that leverages Qwen3-Embedding models to achieve superior performance in topic discovery and analysis. Designed as an improvement over traditional topic models like LDA and ETM, THETA combines the power of large language model embeddings with advanced neural topic modeling architectures.

---

##  Key Features

###  Powerful Embeddings
Built on Qwen3-Embedding models (0.6B/4B/8B parameters) for superior semantic understanding and topic quality.

###  Flexible Training Modes
- **Zero-shot**: Unsupervised topic discovery without labels
- **Supervised**: Leverage labeled data for guided topic learning
- **Unsupervised**: Pure unsupervised learning while ignoring labels

###  Rich Visualizations
Generate comprehensive visualizations including:
- Topic word distributions
- Topic similarity heatmaps
- Document-topic UMAP projections
- Interactive pyLDAvis visualizations
- Word clouds for each topic

###  Multilingual Support
Native support for both English and Chinese text processing with specialized data cleaning and visualization pipelines.

###  Multiple Model Architectures
Compare THETA against classic baselines:
- **LDA** (Latent Dirichlet Allocation)
- **ETM** (Embedded Topic Model with Word2Vec)
- **CTM** (Contextualized Topic Model with SBERT)
- **DTM** (Dynamic Topic Model for temporal analysis)

###  Comprehensive Evaluation
Built-in evaluation metrics including:
- Topic Diversity (TD)
- Inverse Rank-Biased Overlap (iRBO)
- Normalized PMI Coherence (NPMI)
- C_V Coherence
- UMass Coherence
- Topic Exclusivity
- Perplexity (PPL)

---

##  Quick Links

- **[Installation Guide](getting-started/installation.md)** - Get started in minutes
- **[Quick Start Tutorial](getting-started/quickstart.md)** - Train your first model in 5 minutes
- **[User Guide](user-guide/data-preparation.md)** - Complete workflow documentation
- **[API Reference](api/prepare-data.md)** - Detailed parameter documentation
- **[Examples](examples/english-dataset.md)** - Real-world use cases

---

##  Model Comparison

| Model | Embedding | Parameters | Characteristics | Best For |
|-------|-----------|------------|----------------|----------|
| **THETA (Ours)** | Qwen3-Embedding | 0.6B/4B/8B | Contextual understanding, best performance | High-quality topic modeling |
| **LDA** | None | - | Classic probabilistic model | Baseline comparison, interpretability |
| **ETM** | Word2Vec | 100-300 | Embedded representations | Moderate performance |
| **CTM** | SBERT | 110M-340M | Contextualized embeddings | Better than LDA, faster than THETA |
| **DTM** | SBERT | 110M-340M | Temporal dynamics | Time-series topic evolution |

---

##  Why Choose THETA?

### Superior Topic Quality
THETA consistently outperforms traditional topic models across multiple evaluation metrics thanks to Qwen3-Embedding's powerful semantic representations.

### Flexible Scale
Choose from three model sizes based on your needs:
- **0.6B**: Fast experiments with ~4GB VRAM
- **4B**: Balanced performance with ~12GB VRAM  
- **8B**: Best quality with ~24GB VRAM

### Production-Ready
- Complete command-line interface
- Comprehensive evaluation and visualization
- Support for large-scale datasets
- Easy integration with existing pipelines

### Open Research
Built on solid research foundations with transparent implementation and reproducible results.

---

##  Quick Start Example

```bash
# 1. Install THETA
git clone https://github.com/CodeSoul-co/THETA.git
cd THETA
pip install -r requirements.txt

# 2. Prepare your data
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot

# 3. Train the model
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --num_topics 20 \
    --epochs 100

# 4. View results
# Results saved in /root/autodl-tmp/result/0.6B/my_dataset/
```

---

##  Documentation Structure

This documentation is organized into the following sections:

### For Beginners
1. **[Getting Started](getting-started/installation.md)** - Installation and setup
2. **[Quick Start](getting-started/quickstart.md)** - 5-minute tutorial
3. **[Basic Examples](examples/english-dataset.md)** - Learn by example

### For Regular Users
4. **[User Guide](user-guide/data-preparation.md)** - Complete workflow
   - Data Preparation
   - Data Preprocessing
   - Training Models
   - Evaluation
   - Visualization

### For Advanced Users
5. **[Advanced Usage](advanced/new-datasets.md)** - Advanced features
6. **[API Reference](api/prepare-data.md)** - Detailed parameters
7. **[Models](models/theta.md)** - Model architecture details

### For Troubleshooting
8. **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
9. **[FAQ](appendix/faq.md)** - Frequently asked questions

---

##  Use Cases

### Academic Research
- Topic modeling for scientific literature
- Social media content analysis
- News article categorization
- Historical text analysis

### Industry Applications
- Customer feedback analysis
- Content recommendation systems
- Document organization
- Trend detection

### Temporal Analysis
- Topic evolution over time
- Emerging topic detection
- Historical discourse analysis

---

##  Citation

If you use THETA in your research, please cite:

```bibtex
@article{theta2024,
  title={THETA: Advanced Topic Modeling with Qwen Embeddings},
  author={CodeSoul Team},
  journal={arXiv preprint},
  year={2024},
  url={https://github.com/CodeSoul-co/THETA}
}
```

---

##  Contributing

We welcome contributions! Please see our [Contributing Guide](developer/contributing.md) for details on:
- Code style guidelines
- Development setup
- Pull request process
- Issue reporting

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/CodeSoul-co/THETA/blob/main/LICENSE) file for details.

---

##  Links

- **GitHub Repository**: [https://github.com/CodeSoul-co/THETA](https://github.com/CodeSoul-co/THETA)
- **Documentation**: [https://theta.code-soul.com](https://theta.code-soul.com)
- **Issues**: [https://github.com/CodeSoul-co/THETA/issues](https://github.com/CodeSoul-co/THETA/issues)
- **Discussions**: [https://github.com/CodeSoul-co/THETA/discussions](https://github.com/CodeSoul-co/THETA/discussions)

---

##  Community

- **GitHub Discussions**: Ask questions and share ideas
- **Issue Tracker**: Report bugs and request features
- **Email**: support@theta.code-soul.com

---

##  Version History

- **2026-02-03**: Initial release
  - THETA model with Qwen3-Embedding
  - Support for 0.6B/4B/8B models
  - Baseline models (LDA/ETM/CTM/DTM)
  - Comprehensive evaluation and visualization
  - English and Chinese language support

---

##  What's Next?

Ready to get started? Follow our [Installation Guide](getting-started/installation.md) to set up THETA, or jump straight to the [Quick Start Tutorial](getting-started/quickstart.md) to train your first topic model!

---

**Last Updated**: February 4, 2026  
**Document Version**: 1.0.0

# Appendix

Reference materials and supplementary information.

---

## Complete Parameter Reference

### prepare_data.py

| Parameter | Type | Default | Range/Options | Required | Description |
|-----------|------|---------|--------------|----------|-------------|
| `--dataset` | string | - | - | Yes | Dataset name |
| `--model` | string | - | theta/baseline/dtm | Yes | Model type |
| `--model_size` | string | 0.6B | 0.6B/4B/8B | No | Qwen model size |
| `--mode` | string | zero_shot | zero_shot/supervised/unsupervised | No | Training mode |
| `--vocab_size` | int | 5000 | 1000-20000 | No | Vocabulary size |
| `--batch_size` | int | 32 | 8-128 | No | Batch size |
| `--max_length` | int | 512 | 128-2048 | No | Max sequence length |
| `--gpu` | int | 0 | 0-7 | No | GPU device ID |
| `--clean` | flag | False | - | No | Clean data first |
| `--raw-input` | string | None | filepath | No | Raw CSV path |
| `--language` | string | english | english/chinese | No | Cleaning language |
| `--bow-only` | flag | False | - | No | BOW only |
| `--check-only` | flag | False | - | No | Check files only |
| `--time_column` | string | year | column name | No | Time column (DTM) |

### run_pipeline.py

| Parameter | Type | Default | Range/Options | Required | Description |
|-----------|------|---------|--------------|----------|-------------|
| `--dataset` | string | - | - | Yes | Dataset name |
| `--models` | string | - | theta,lda,etm,ctm,dtm | Yes | Model list |
| `--model_size` | string | 0.6B | 0.6B/4B/8B | No | Qwen model size |
| `--mode` | string | zero_shot | zero_shot/supervised/unsupervised | No | Training mode |
| `--num_topics` | int | 20 | 5-100 | No | Number of topics |
| `--epochs` | int | 100 | 10-500 | No | Training epochs |
| `--batch_size` | int | 64 | 8-512 | No | Batch size |
| `--hidden_dim` | int | 512 | 128-1024 | No | Hidden dimension |
| `--learning_rate` | float | 0.002 | 0.00001-0.1 | No | Learning rate |
| `--kl_start` | float | 0.0 | 0.0-1.0 | No | KL start weight |
| `--kl_end` | float | 1.0 | 0.0-1.0 | No | KL end weight |
| `--kl_warmup` | int | 50 | 0-200 | No | KL warmup epochs |
| `--patience` | int | 10 | 1-50 | No | Early stopping patience |
| `--no_early_stopping` | flag | False | - | No | Disable early stopping |
| `--gpu` | int | 0 | 0-7 | No | GPU device ID |
| `--language` | string | en | en/zh | No | Visualization language |
| `--skip-train` | flag | False | - | No | Skip training |
| `--skip-eval` | flag | False | - | No | Skip evaluation |
| `--skip-viz` | flag | False | - | No | Skip visualization |

### visualization.run_visualization

| Parameter | Type | Default | Range/Options | Required | Description |
|-----------|------|---------|--------------|----------|-------------|
| `--result_dir` | string | - | directory | Yes | Results directory |
| `--dataset` | string | - | - | Yes | Dataset name |
| `--mode` | string | zero_shot | zero_shot/supervised/unsupervised | No | THETA mode |
| `--model_size` | string | 0.6B | 0.6B/4B/8B | No | Model size |
| `--baseline` | flag | False | - | No | Baseline flag |
| `--model` | string | None | lda/etm/ctm/dtm | No | Baseline model |
| `--num_topics` | int | 20 | 5-100 | No | Number of topics |
| `--language` | string | en | en/zh | No | Language |
| `--dpi` | int | 300 | 72-1200 | No | Image resolution |

---

## Directory Structure

```
/root/autodl-tmp/
├── ETM/
│   ├── main.py
│   ├── run_pipeline.py
│   ├── prepare_data.py
│   └── src/
├── data/
│   └── {dataset}/
│       └── {dataset}_cleaned.csv
├── result/
│   ├── 0.6B/
│   ├── 4B/
│   ├── 8B/
│   └── baseline/
└── embedding_models/
```

---

## Hardware Requirements

| Setup | CPU | RAM | GPU | CUDA | Storage |
|-------|-----|-----|-----|------|---------|
| Minimum | 4 cores | 8GB | 4GB VRAM | 11.8+ | 20GB |
| Recommended | 8 cores | 16GB | 12GB VRAM | 12.1+ | 50GB SSD |
| High-Performance | 16+ cores | 32GB+ | A100 40GB | 12.1+ | 200GB NVMe |

---

## FAQ

**Q: What makes THETA different?**  
A: THETA uses Qwen embeddings and neural variational inference for better semantic understanding than LDA or ETM.

**Q: Which model size to use?**  
A: 0.6B for prototyping, 4B for production, 8B for maximum quality.

**Q: Minimum dataset size?**  
A: 500+ documents with 50+ words average recommended.

**Q: Training time?**  
A: 5K docs with 0.6B on V100: ~25 min. 4B: ~50 min.

**Q: GPU required?**  
A: Yes. GPU required for preprocessing and training.

---

## Citation

```bibtex
@article{theta2024,
  title={THETA: Advanced Topic Modeling with Qwen Embeddings},
  author={CodeSoul Team},
  year={2024},
  url={https://github.com/CodeSoul-co/THETA}
}
```

---

## Contact

- Website: [https://theta.code-soul.com](https://theta.code-soul.com)
- GitHub: [https://github.com/CodeSoul-co/THETA](https://github.com/CodeSoul-co/THETA)
- Email: support@theta.code-soul.com

---

**Document Version**: 1.0.0  
**Last Updated**: February 6, 2026

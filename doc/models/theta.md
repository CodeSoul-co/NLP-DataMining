# THETA Model

THETA is a neural topic model that combines variational autoencoders with Qwen3-Embedding representations.

---

## Architecture

The model consists of three main components:

**Encoder Network**
- Input: Qwen embeddings (dimension 1024/2560/4096 depending on model size)
- Architecture: Multi-layer perceptron with configurable hidden dimension
- Output: Parameters of variational posterior q(θ|x)
  - Mean μ ∈ R^K (K = number of topics)
  - Log-variance log σ^2 ∈ R^K

**Reparameterization**
- Sample topic distribution using reparameterization trick
- θ = μ + σ ⊙ ε, where ε ~ N(0, I)
- Enables gradient-based training through stochastic sampling

**Decoder Network**
- Topic-word matrix β ∈ R^(K×V) (V = vocabulary size)
- Reconstruction: p(w|θ) = softmax(θ^T β)
- Loss: Negative ELBO = -E_q[log p(w|θ)] + KL[q(θ|x) || p(θ)]

---

## Training Objective

The model maximizes the evidence lower bound (ELBO):

```
ELBO = E_q(θ|x)[log p(w|θ)] - KL[q(θ|x) || p(θ)]
```

Components:
- Reconstruction term: Expected log-likelihood of observed words
- KL divergence: Regularization toward prior p(θ) = Dir(α)

KL annealing is applied to prevent posterior collapse:
```
Loss = -Reconstruction + β_t * KL
```
where β_t increases from 0 to 1 during warmup period.

---

## Model Specifications

**0.6B Model**

| Property | Value |
|----------|-------|
| Parameters | 600M |
| Embedding Dimension | 1024 |
| VRAM Requirement | ~4GB |
| Processing Speed | ~1000 docs/min |
| Batch Size (preprocessing) | 32 |
| Batch Size (training) | 64 |

Characteristics:
- Fastest processing speed
- Suitable for development and iteration
- Good quality on most datasets
- Recommended starting point

**4B Model**

| Property | Value |
|----------|-------|
| Parameters | 4B |
| Embedding Dimension | 2560 |
| VRAM Requirement | ~12GB |
| Processing Speed | ~400 docs/min |
| Batch Size (preprocessing) | 16 |
| Batch Size (training) | 32 |

Characteristics:
- Balanced performance and cost
- Better semantic understanding than 0.6B
- Suitable for production deployments
- Recommended for final results

**8B Model**

| Property | Value |
|----------|-------|
| Parameters | 8B |
| Embedding Dimension | 4096 |
| VRAM Requirement | ~24GB |
| Processing Speed | ~200 docs/min |
| Batch Size (preprocessing) | 8 |
| Batch Size (training) | 16 |

Characteristics:
- Highest quality embeddings
- Best performance on all metrics
- Requires high-end GPU (A100, H100)
- Recommended for research and critical applications

---

## Training Modes

**zero_shot Mode**

Standard unsupervised topic modeling:
- No label information used
- Topics emerge purely from text patterns
- Default choice when labels are unavailable

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20
```

**supervised Mode**

Label-guided topic discovery:
- Incorporates label information during training
- Topics align with provided categories
- Requires label column in CSV

The model adds a classification objective:
```
Loss = -ELBO + λ * CrossEntropy(y_pred, y_true)
```

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 20
```

**unsupervised Mode**

Explicit unsupervised learning:
- Similar to zero_shot but explicitly ignores labels if present
- Used for comparison experiments on labeled data
- Useful for ablation studies

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode unsupervised \
    --num_topics 20
```

---

## Hyperparameter Guidelines

**Number of Topics**

Selection depends on corpus characteristics:
- Small corpus (< 1K documents): 10-20 topics
- Medium corpus (1K-10K documents): 20-50 topics
- Large corpus (> 10K documents): 50-100 topics

**Hidden Dimension**

Controls encoder capacity:
- 256: Minimal capacity, faster training
- 512: Default choice, works for most cases
- 768-1024: Higher capacity for complex corpora

**Learning Rate**

Affects convergence speed and stability:
- 0.001: Conservative, stable convergence
- 0.002: Default, balanced performance
- 0.005: Aggressive, faster but less stable

**KL Annealing**

Standard schedule:
- Start: 0.0 (no KL penalty)
- End: 1.0 (full KL penalty)
- Warmup: 50 epochs (gradual increase)

---

## Implementation Details

### Checkpoint Management

Model checkpoints are saved during training:
- `best_model.pt`: Best model by validation loss
- `last_model.pt`: Final epoch model
- `training_history.json`: Loss curves and metrics

Load checkpoint for inference:
```python
from src.model import etm
model = etm.THETA(num_topics=20, vocab_size=5000)
model.load_state_dict(torch.load('best_model.pt'))
```

### Memory Management

GPU memory usage scales with:
- Batch size (linear scaling)
- Embedding dimension (linear scaling)
- Vocabulary size (linear scaling)
- Hidden dimension (linear scaling)

Reduce memory usage by:
- Decreasing batch size
- Using smaller model (0.6B instead of 4B)
- Reducing vocabulary size
- Reducing hidden dimension

### Reproducibility

Set random seeds for reproducible results:
```python
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
```

Deterministic operations:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: Some operations are non-deterministic on GPU even with seeding.

# Models

This section describes the architecture and characteristics of THETA and baseline topic models.

---

## THETA Model

THETA is a neural topic model that combines variational autoencoders with Qwen3-Embedding representations.

### Architecture

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

### Training Objective

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

### Model Specifications

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

### Training Modes

**zero_shot Mode**

Standard unsupervised topic modeling:
- No label information used
- Topics emerge purely from text patterns
- Default choice when labels are unavailable

Training:
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

Training:
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

Training:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode unsupervised \
    --num_topics 20
```

### Hyperparameter Guidelines

**Number of Topics**

Selection depends on corpus characteristics:
- Small corpus (< 1K documents): 10-20 topics
- Medium corpus (1K-10K documents): 20-50 topics
- Large corpus (> 10K documents): 50-100 topics

Evaluation metrics help determine optimal K:
- High topic diversity indicates good separation
- Low perplexity indicates good fit
- High coherence indicates meaningful topics

**Hidden Dimension**

Controls encoder capacity:
- 256: Minimal capacity, faster training
- 512: Default choice, works for most cases
- 768-1024: Higher capacity for complex corpora

Larger hidden dimensions require more data to avoid overfitting.

**Learning Rate**

Affects convergence speed and stability:
- 0.001: Conservative, stable convergence
- 0.002: Default, balanced performance
- 0.005: Aggressive, faster but less stable

Monitor training loss to adjust if needed.

**KL Annealing**

Standard schedule:
- Start: 0.0 (no KL penalty)
- End: 1.0 (full KL penalty)
- Warmup: 50 epochs (gradual increase)

Adjust warmup period based on dataset:
- Short warmup (30 epochs): Small datasets
- Standard warmup (50 epochs): Default
- Long warmup (80 epochs): Large or complex datasets

---

## Baseline Models

### LDA (Latent Dirichlet Allocation)

Classic probabilistic topic model using Dirichlet-multinomial distributions.

**Architecture:**
- No neural components
- Dirichlet priors on topic and word distributions
- Inference via Gibbs sampling or variational inference

**Characteristics:**
- Fast training (no GPU required)
- Highly interpretable
- Strong baseline for comparison
- No embeddings required

**Strengths:**
- Well-established theoretical foundation
- Interpretable probabilistic framework
- Efficient on CPU
- Deterministic given random seed

**Limitations:**
- No semantic word relationships
- Bag-of-words assumption
- Performance plateaus on large vocabularies

**Training:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda \
    --num_topics 20 \
    --epochs 100
```

**Key Parameters:**
- `num_topics`: Number of topics K
- `epochs`: Number of Gibbs sampling iterations
- Alpha and Beta priors (fixed internally)

### ETM (Embedded Topic Model)

Neural topic model using Word2Vec embeddings.

**Architecture:**
- VAE framework similar to THETA
- Word2Vec embeddings (300 dimensions)
- Topic embeddings in same space as word embeddings

**Characteristics:**
- Incorporates word semantics via embeddings
- Neural variational inference
- Moderate computational cost
- Better than LDA, faster than THETA

**Strengths:**
- Captures semantic relationships
- More coherent topics than LDA
- Efficient training with GPU
- Original ETM implementation

**Limitations:**
- Word2Vec limited to static embeddings
- 300-dimensional embeddings less expressive than Qwen
- Requires pre-trained Word2Vec model

**Training:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models etm \
    --num_topics 20 \
    --epochs 100 \
    --hidden_dim 512 \
    --learning_rate 0.002
```

**Key Parameters:**
- Same as THETA except model_size and mode
- Uses Word2Vec instead of Qwen embeddings

### CTM (Contextualized Topic Model)

Neural topic model using SBERT contextualized embeddings.

**Architecture:**
- VAE framework with SBERT encoder
- SBERT embeddings (768 dimensions)
- Contextualized document representations

**Characteristics:**
- Leverages pre-trained SBERT
- Contextualized understanding
- Performance between ETM and THETA
- Good balance of quality and speed

**Strengths:**
- Contextualized semantic representations
- Better than ETM on most metrics
- Faster than large THETA models
- Widely used in recent research

**Limitations:**
- SBERT embeddings fixed at 768 dimensions
- Less powerful than Qwen 4B/8B models
- Requires SBERT model download

**Training:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models ctm \
    --num_topics 20 \
    --epochs 100 \
    --hidden_dim 512 \
    --learning_rate 0.002
```

**Key Parameters:**
- Same training parameters as ETM
- Uses SBERT embeddings automatically

### DTM (Dynamic Topic Model)

Temporal extension of CTM for tracking topic evolution over time.

**Architecture:**
- Based on CTM architecture
- Additional temporal dynamics layer
- Models topic transitions between time slices

**Characteristics:**
- Tracks topic evolution across time
- Requires temporal metadata
- Suitable for longitudinal studies
- More complex than static models

**Strengths:**
- Captures temporal dynamics
- Reveals emerging and declining topics
- Useful for trend analysis
- Handles variable time slice sizes

**Limitations:**
- Requires time column in data
- More parameters to estimate
- Longer training time
- Needs sufficient documents per time slice

**Training:**
```bash
python run_pipeline.py \
    --dataset temporal_dataset \
    --models dtm \
    --num_topics 20 \
    --epochs 100 \
    --hidden_dim 512 \
    --learning_rate 0.002
```

Data requirements:
- CSV must contain time column (year, timestamp, or date)
- Preprocessing must specify time column via `--time_column`

**Key Parameters:**
- Standard neural model parameters
- Time slices determined automatically from data

---

## Model Comparison

### Performance Comparison

Typical performance on benchmark datasets:

| Model | TD | NPMI | C_V | PPL | Speed | VRAM |
|-------|-------|-------|-------|------|-------|------|
| LDA | 0.75 | 0.25 | 0.45 | 180 | Fast | 0GB |
| ETM | 0.82 | 0.32 | 0.52 | 165 | Medium | 4GB |
| CTM | 0.85 | 0.38 | 0.58 | 155 | Medium | 6GB |
| THETA-0.6B | 0.88 | 0.42 | 0.64 | 145 | Medium | 8GB |
| THETA-4B | 0.91 | 0.48 | 0.69 | 138 | Slow | 16GB |
| THETA-8B | 0.93 | 0.52 | 0.72 | 132 | Slowest | 28GB |

Values are approximate and vary by dataset. Higher is better for TD, NPMI, C_V. Lower is better for PPL.

### Selection Guidelines

**Use LDA when:**
- Need fast baseline results
- Interpretability is critical
- No GPU available
- Computing topic distributions for new documents frequently

**Use ETM when:**
- Want better performance than LDA
- Have GPU available
- Need moderate computational budget
- Comparing against original ETM papers

**Use CTM when:**
- Need contextualized understanding
- Want good balance of quality and speed
- Following recent topic modeling literature
- Working with standard-size corpora

**Use DTM when:**
- Analyzing temporal dynamics
- Have time-stamped documents
- Studying topic evolution
- Investigating emerging trends

**Use THETA-0.6B when:**
- Need better quality than CTM
- Have 8-12GB VRAM available
- Rapid experimentation required
- Cost-quality tradeoff favors 0.6B

**Use THETA-4B when:**
- Need high-quality results
- Have 16-20GB VRAM available
- Production deployment
- Publishing results in papers

**Use THETA-8B when:**
- Need highest possible quality
- Have 24-32GB VRAM available
- Critical applications
- Research with challenging datasets

### Computational Requirements

Training time comparison on 10K document corpus:

| Model | CPU Time | GPU Time | VRAM | Storage |
|-------|----------|----------|------|---------|
| LDA | 15 min | N/A | 0GB | 100MB |
| ETM | N/A | 20 min | 4GB | 500MB |
| CTM | N/A | 25 min | 6GB | 800MB |
| THETA-0.6B | N/A | 30 min | 8GB | 2GB |
| THETA-4B | N/A | 50 min | 16GB | 6GB |
| THETA-8B | N/A | 90 min | 28GB | 12GB |

Times assume single GPU (V100 or A100). Storage includes preprocessed data and model checkpoints.

### Embedding Comparison

Different models use different embedding approaches:

| Model | Embedding | Dimension | Contextual | Pre-trained |
|-------|-----------|-----------|------------|-------------|
| LDA | None | N/A | No | N/A |
| ETM | Word2Vec | 300 | No | Yes |
| CTM | SBERT | 768 | Yes | Yes |
| THETA-0.6B | Qwen3 | 1024 | Yes | Yes |
| THETA-4B | Qwen3 | 2560 | Yes | Yes |
| THETA-8B | Qwen3 | 4096 | Yes | Yes |

Contextual embeddings (SBERT, Qwen) capture word meaning in context. Static embeddings (Word2Vec) use fixed representations.

### Topic Quality Factors

Several factors influence topic quality:

**Semantic Understanding**
- Qwen embeddings provide deepest semantic understanding
- SBERT captures context better than Word2Vec
- Word2Vec captures basic semantic similarity
- LDA uses only co-occurrence patterns

**Corpus Size**
- Large corpora benefit more from powerful embeddings
- Small corpora may see diminishing returns from 8B vs 4B
- Very small corpora (< 500 docs) may favor simpler models

**Domain Specificity**
- Technical domains benefit from powerful embeddings
- General domains work well with all models
- Specialized vocabularies may need larger models

**Document Length**
- Long documents benefit from contextualized embeddings
- Short documents (tweets) may not fully leverage 8B model
- Very short text may favor CTM or simpler models

---

## Model Selection Workflow

### Step 1: Determine Requirements

Consider the following:
- Dataset size (number of documents)
- Available computational resources (GPU memory)
- Time constraints (processing deadline)
- Quality requirements (research vs prototyping)
- Budget (compute cost)

### Step 2: Choose Initial Model

Default recommendations:
- Prototyping: THETA-0.6B or CTM
- Production: THETA-4B
- Research: THETA-8B
- Quick baseline: LDA
- Temporal analysis: DTM

### Step 3: Evaluate and Compare

Train multiple models:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm,theta \
    --model_size 0.6B \
    --num_topics 20
```

Compare evaluation metrics to determine best model for your use case.

### Step 4: Refine Hyperparameters

Once model type is selected, tune hyperparameters:
- Number of topics
- Learning rate
- Hidden dimension
- KL annealing schedule

### Step 5: Scale Up If Needed

If quality is insufficient:
- THETA-0.6B → THETA-4B: Significant quality improvement
- THETA-4B → THETA-8B: Marginal quality improvement
- Consider collecting more data before scaling model size

---

## Implementation Details

### Training Process

All neural models (THETA, ETM, CTM) follow similar training procedure:

1. Load preprocessed embeddings and BOW data
2. Initialize encoder and decoder networks
3. Train for specified number of epochs
4. Apply early stopping based on validation loss
5. Save best model checkpoint
6. Compute evaluation metrics
7. Generate visualizations

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

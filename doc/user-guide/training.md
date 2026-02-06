# Training Models

This guide covers training THETA and baseline models with various configurations.

---

## THETA Model Training

### Basic Training

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

### Topic Number Selection

The number of topics is a key hyperparameter that affects granularity:

| Topics | Appropriate For |
|--------|----------------|
| 10-15 | Small corpora, broad categories, high-level overview |
| 20-30 | Medium corpora, balanced granularity, default choice |
| 40-100 | Large diverse corpora, fine-grained analysis |

### Learning Rate Tuning

| Learning Rate | Use When |
|--------------|----------|
| 0.001 | Training is unstable, loss oscillates |
| 0.002 | Default choice for most datasets |
| 0.005 | Training is too slow, need faster convergence |

### KL Annealing Configuration

KL annealing gradually increases the KL divergence weight during training to prevent posterior collapse.

**Standard KL annealing:**
Weight increases linearly from 0.0 to 1.0 over 50 epochs.

**Slow KL annealing:**
`--kl_warmup 80` — Longer warmup period helps prevent early posterior collapse.

**Partial KL annealing:**
`--kl_start 0.1 --kl_end 0.9 --kl_warmup 30` — Starts with non-zero weight and stops before full weight.

### Hidden Dimension Configuration

| Hidden Dim | Use When |
|-----------|----------|
| 256 | Small datasets, faster training, limited VRAM |
| 512 | Default choice for most datasets |
| 768-1024 | Large complex datasets, sufficient VRAM available |

### Early Stopping

Early stopping prevents overfitting by monitoring validation performance:

- **Default**: `--patience 10` — Stops if no improvement after 10 epochs
- **Disabled**: `--no_early_stopping` — Trains for all specified epochs

### Chinese Data Training

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

### Supervised Training

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

---

## Baseline Model Training

### LDA Training

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

LDA uses Gibbs sampling and does not utilize GPU acceleration.

### ETM Training

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

ETM uses Word2Vec embeddings (300 dimensions).

### CTM Training

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

CTM uses SBERT embeddings (768 dimensions).

### DTM Training

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

### Training Multiple Models

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

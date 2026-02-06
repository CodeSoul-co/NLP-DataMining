# Baseline Models

THETA includes several baseline models for comparison.

---

## LDA (Latent Dirichlet Allocation)

Classic probabilistic topic model using Dirichlet-multinomial distributions.

**Architecture:**
- No neural components
- Dirichlet priors on topic and word distributions
- Inference via Gibbs sampling or variational inference

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

---

## ETM (Embedded Topic Model)

Neural topic model using Word2Vec embeddings.

**Architecture:**
- VAE framework similar to THETA
- Word2Vec embeddings (300 dimensions)
- Topic embeddings in same space as word embeddings

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

---

## CTM (Contextualized Topic Model)

Neural topic model using SBERT contextualized embeddings.

**Architecture:**
- VAE framework with SBERT encoder
- SBERT embeddings (768 dimensions)
- Contextualized document representations

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

---

## DTM (Dynamic Topic Model)

Temporal extension of CTM for tracking topic evolution over time.

**Architecture:**
- Based on CTM architecture
- Additional temporal dynamics layer
- Models topic transitions between time slices

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

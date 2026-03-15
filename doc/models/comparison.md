# Model Comparison

Comprehensive comparison of all models supported by THETA.

---

## Performance Comparison

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

---

## Selection Guidelines

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

**Use THETA-4B when:**
- Need high-quality results
- Have 16-20GB VRAM available
- Production deployment

**Use THETA-8B when:**
- Need highest possible quality
- Have 24-32GB VRAM available
- Critical applications

---

## Computational Requirements

Training time comparison on 10K document corpus:

| Model | CPU Time | GPU Time | VRAM | Storage |
|-------|----------|----------|------|---------|
| LDA | 15 min | N/A | 0GB | 100MB |
| ETM | N/A | 20 min | 4GB | 500MB |
| CTM | N/A | 25 min | 6GB | 800MB |
| THETA-0.6B | N/A | 30 min | 8GB | 2GB |
| THETA-4B | N/A | 50 min | 16GB | 6GB |
| THETA-8B | N/A | 90 min | 28GB | 12GB |

Times assume single GPU (V100 or A100).

---

## Embedding Comparison

| Model | Embedding | Dimension | Contextual | Pre-trained |
|-------|-----------|-----------|------------|-------------|
| LDA | None | N/A | No | N/A |
| ETM | Word2Vec | 300 | No | Yes |
| CTM | SBERT | 768 | Yes | Yes |
| THETA-0.6B | Qwen3 | 1024 | Yes | Yes |
| THETA-4B | Qwen3 | 2560 | Yes | Yes |
| THETA-8B | Qwen3 | 4096 | Yes | Yes |

---

## Model Selection Workflow

### Step 1: Determine Requirements

Consider:
- Dataset size (number of documents)
- Available computational resources (GPU memory)
- Time constraints
- Quality requirements (research vs prototyping)

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

### Step 4: Scale Up If Needed

- THETA-0.6B → THETA-4B: Significant quality improvement
- THETA-4B → THETA-8B: Marginal quality improvement
- Consider collecting more data before scaling model size

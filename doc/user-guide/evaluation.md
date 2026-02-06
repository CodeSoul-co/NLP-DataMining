# Evaluation

Training automatically runs evaluation using multiple metrics. Results are saved in JSON format.

---

## Evaluation Metrics

**Topic Diversity (TD)**
- Range: 0-1
- Higher is better
- Measures uniqueness of topics
- Computed as percentage of unique words in top-N words across all topics

**Inverse Rank-Biased Overlap (iRBO)**
- Range: 0-1
- Higher is better
- Measures topic distinctiveness
- Lower values indicate redundant or overlapping topics

**Normalized PMI (NPMI)**
- Range: -1 to 1
- Higher is better
- Measures semantic coherence of topic words
- Based on pointwise mutual information in external corpus

**C_V Coherence**
- Range: 0-1
- Higher is better
- Alternative coherence measure
- Based on sliding window co-occurrence

**UMass Coherence**
- Range: Negative values
- Closer to 0 is better
- Classic coherence metric
- Based on document co-occurrence

**Exclusivity**
- Range: 0-1
- Higher is better
- Measures topic specificity
- Computed using FREX score

**Perplexity (PPL)**
- Range: Positive values
- Lower is better
- Measures model fit on held-out data
- Standard evaluation for probabilistic models

---

## Viewing Evaluation Results

Results are saved in the metrics directory:

```bash
cat /root/autodl-tmp/result/0.6B/my_dataset/zero_shot/metrics/evaluation_results.json
```

Example output:
```json
{
  "TD": 0.891,
  "iRBO": 0.762,
  "NPMI": 0.418,
  "C_V": 0.654,
  "UMass": -2.341,
  "Exclusivity": 0.823,
  "PPL": 145.23,
  "training_time": 1425.6,
  "num_topics": 20,
  "num_documents": 5000
}
```

---

## Running Evaluation Separately

Skip training and only evaluate existing models:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --skip-train \
    --gpu 0 \
    --language en
```

This loads the trained model from checkpoints and recomputes all metrics.

---

## Comparing Multiple Models

Evaluate all baseline models:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --skip-train \
    --gpu 0 \
    --language en
```

Results for each model are saved separately. Use the metrics files to construct comparison tables.

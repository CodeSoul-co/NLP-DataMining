# Chinese Data Processing

Specialized guide for processing Chinese text with THETA.

---

## Specialized Preprocessing

Chinese text requires different handling than English:

**Data cleaning:**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/chinese_corpus/raw_data.csv \
    --output /root/autodl-tmp/data/chinese_corpus/chinese_corpus_cleaned.csv \
    --language chinese
```

Cleaning operations for Chinese:
- Remove HTML entities
- Normalize full-width and half-width characters
- Handle Chinese punctuation
- Preserve Chinese word boundaries
- Convert traditional to simplified (optional)

**Preprocessing:**
```bash
python prepare_data.py \
    --dataset chinese_corpus \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

Qwen models handle Chinese tokenization internally.

**Training:**
```bash
python run_pipeline.py \
    --dataset chinese_corpus \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language zh
```

The `--language zh` setting ensures Chinese fonts in visualizations.

---

## Chinese Visualization

Chinese visualizations require proper font configuration:

```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset chinese_corpus \
    --mode zero_shot \
    --model_size 0.6B \
    --language zh \
    --dpi 300
```

The visualization module automatically:
- Selects Chinese-compatible fonts
- Handles character encoding
- Adjusts layout for Chinese text
- Renders word clouds with Chinese characters

---

## Chinese-English Mixed Data

For datasets containing both languages:

1. Clean as Chinese (preserves both languages)
2. Preprocess normally (Qwen handles multilingual)
3. Train with appropriate language setting
4. Visualizations may show mixed text

Primary language should be specified in `--language` parameter based on majority content.

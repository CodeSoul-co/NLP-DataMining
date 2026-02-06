# visualization.run_visualization

Separate visualization generation tool.

---

## Basic Usage

```bash
python -m visualization.run_visualization --result_dir DIR --dataset DATASET [OPTIONS]
```

---

## Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--result_dir` | string | Results directory path |
| `--dataset` | string | Dataset name |

## THETA Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mode` | string | `zero_shot` | Training mode (for THETA models) |
| `--model_size` | string | `0.6B` | Qwen model size (for THETA models) |

## Baseline Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--baseline` | flag | False | Indicates baseline model |
| `--model` | string | None | Baseline model name: `lda`, `etm`, `ctm`, or `dtm` |
| `--num_topics` | int | `20` | Number of topics (for baseline models) |

## Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--language` | string | `en` | Visualization language: `en` or `zh` |
| `--dpi` | int | `300` | Image resolution (dots per inch) |

---

## Examples

**THETA model visualization:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 300
```

**LDA model visualization:**
```bash
python -m visualization.run_visualization \
    --baseline \
    --result_dir /root/autodl-tmp/result/baseline \
    --dataset my_dataset \
    --model lda \
    --num_topics 20 \
    --language en \
    --dpi 300
```

**High-resolution visualization:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 600
```

**Chinese visualization:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset chinese_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language zh \
    --dpi 300
```

---

## Output Files

Visualizations are saved to the same directory as the model results:
- `topic_words_bars.png`: Bar charts of topic words
- `topic_similarity.png`: Topic similarity heatmap
- `doc_topic_umap.png`: Document-topic UMAP projection
- `topic_wordclouds.png`: Word clouds for each topic
- `metrics.png`: Evaluation metrics comparison
- `pyldavis.html`: Interactive visualization

---

## dataclean.main

Data cleaning module for preprocessing raw text.

### Basic Usage

```bash
python -m dataclean.main --input INPUT --output OUTPUT --language LANGUAGE
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--input` | string | Input CSV file path or directory |
| `--output` | string | Output CSV file path or directory |
| `--language` | string | Language: `english` or `chinese` |

### Examples

**Clean single file (English):**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language english
```

**Clean single file (Chinese):**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language chinese
```

**Clean directory:**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw/ \
    --output /root/autodl-tmp/data/cleaned/ \
    --language english
```

### Cleaning Operations

**English cleaning:**
- Remove HTML tags and entities
- Remove URLs and email addresses
- Remove special characters (except basic punctuation)
- Normalize whitespace
- Remove non-ASCII characters (optional)
- Lowercase text (optional)

**Chinese cleaning:**
- Remove HTML tags and entities
- Remove URLs and email addresses
- Normalize full-width and half-width characters
- Handle Chinese punctuation
- Remove non-Chinese characters (optional)
- Preserve word boundaries

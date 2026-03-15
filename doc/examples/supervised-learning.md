# Supervised Learning Example

This example demonstrates supervised topic modeling and temporal analysis.

---

## Example 1: Supervised Topic Classification

### Dataset Description

- Domain: Customer reviews
- Size: 3000 documents
- Language: English
- Labels: 5 product categories
- Goal: Discover category-aligned topics

### Step 1: Prepare Labeled Data

CSV format with labels:
```csv
text,label
"Great laptop with fast processor and long battery life",Electronics
"Comfortable running shoes with good arch support",Sports
"Delicious coffee beans with rich aroma",Food
```

### Step 2: Preprocess in Supervised Mode

```bash
python prepare_data.py \
    --dataset reviews \
    --model theta \
    --model_size 0.6B \
    --mode supervised \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

### Step 3: Train with Supervision

```bash
python run_pipeline.py \
    --dataset reviews \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 15 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

### Step 4: Compare Modes

Train both supervised and zero-shot for comparison:

```bash
# Zero-shot (ignores labels)
python run_pipeline.py \
    --dataset reviews \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 15 \
    --gpu 0

# Supervised (uses labels)
python run_pipeline.py \
    --dataset reviews \
    --models theta \
    --model_size 0.6B \
    --mode supervised \
    --num_topics 15 \
    --gpu 0
```

Results comparison:

| Mode | TD | NPMI | Label Alignment |
|------|-----|------|----------------|
| Zero-shot | 0.85 | 0.41 | 0.62 |
| Supervised | 0.83 | 0.38 | 0.89 |

Supervised mode achieves better label alignment with slight reduction in diversity.

### Step 5: Topic-Label Analysis

```python
import json
import numpy as np

# Load results
with open('result/0.6B/reviews/supervised/metrics/evaluation_results.json') as f:
    results = json.load(f)

# Analyze topic-label correspondence
# Topics 0-2: Electronics
# Topics 3-5: Sports
# Topics 6-8: Food
# Topics 9-11: Books
# Topics 12-14: Clothing
```

---

## Example 2: Temporal Topic Evolution

### Dataset Description

- Domain: Academic papers
- Size: 10000 documents
- Language: English
- Time range: 2015-2023
- Field: Machine learning

### Step 1: Prepare Temporal Data

CSV with year column:
```csv
text,year
"Deep learning approaches for image recognition...",2015
"Transformer architectures for natural language...",2019
"Large language models and emergent capabilities...",2023
```

### Step 2: Preprocess with Time Information

```bash
python prepare_data.py \
    --dataset ml_papers \
    --model dtm \
    --vocab_size 8000 \
    --time_column year
```

### Step 3: Train DTM Model

```bash
python run_pipeline.py \
    --dataset ml_papers \
    --models dtm \
    --num_topics 30 \
    --epochs 150 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language en
```

### Step 4: Analyze Topic Evolution

DTM tracks topic changes over time:

**Topic 5: Deep Learning (2015-2018)**
- 2015: convolutional, neural, network, classification
- 2016: deep, learning, layers, training
- 2017: residual, connections, skip, depth
- 2018: architectures, design, efficient, mobile

**Topic 12: Attention Mechanisms (2017-2020)**
- 2017: attention, mechanism, sequence, encoder
- 2018: self-attention, multi-head, transformer
- 2019: bert, pre-training, fine-tuning, downstream
- 2020: scaling, models, parameters, performance

**Topic 18: Large Language Models (2020-2023)**
- 2020: gpt, generation, language, model
- 2021: prompting, few-shot, in-context, learning
- 2022: instruction, tuning, alignment, human
- 2023: emergent, capabilities, scaling, laws

### Step 5: Visualize Trends

```bash
python -m visualization.run_visualization \
    --baseline \
    --result_dir /root/autodl-tmp/result/baseline \
    --dataset ml_papers \
    --model dtm \
    --num_topics 30 \
    --language en \
    --dpi 300
```

Visualizations show:
- Topic birth and death
- Word probability changes over time
- Topic intensity trends

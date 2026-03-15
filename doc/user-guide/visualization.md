# Visualization

Training automatically generates visualizations. Additional visualizations can be created separately.

---

## Visualization Outputs

**topic_words_bars.png**
Bar charts showing top-10 words for each topic with probability weights.

**topic_similarity.png**
Heatmap showing cosine similarity between topic-word distributions.

**doc_topic_umap.png**
UMAP projection of documents in topic space. Points are colored by dominant topic.

**topic_wordclouds.png**
Word clouds for each topic sized by word probability.

**metrics.png**
Bar charts comparing evaluation metrics.

**pyldavis.html**
Interactive visualization using pyLDAvis library. Open in web browser.

---

## Generating Visualizations Separately

Generate visualizations for THETA models:

```bash
cd /root/autodl-tmp/ETM

python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 300
```

Generate visualizations for baseline models:

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

Replace `lda` with `etm`, `ctm`, or `dtm` for other baseline models.

---

## Customizing Visualization

**Higher resolution:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 600
```

**Chinese language visualizations:**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset chinese_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language zh \
    --dpi 300
```

Chinese visualizations use appropriate fonts and handle character rendering correctly.

---

## Skipping Visualization During Training

Skip automatic visualization to save time:

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --skip-viz \
    --gpu 0 \
    --language en
```

Visualizations can be generated later using the separate visualization command.

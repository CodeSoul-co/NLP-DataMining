# 可视化

**[English](visualization.md)** | **[中文](visualization.zh.md)**

---

训练过程会自动生成可视化。额外的可视化可以单独创建。

---

## 可视化输出

**topic_words_bars.png**
条形图，显示每个主题的前10个词及其概率权重。

**topic_similarity.png**
热图，显示主题-词分布之间的余弦相似度。

**doc_topic_umap.png**
文档在主题空间中的UMAP投影。点按主导主题着色。

**topic_wordclouds.png**
每个主题的词云，大小按词概率缩放。

**metrics.png**
比较评估指标的条形图。

**pyldavis.html**
使用pyLDAvis库的交互式可视化。在网页浏览器中打开。

---

## 单独生成可视化

为THETA模型生成可视化：

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

为基线模型生成可视化：

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

将 `lda` 替换为 `etm`、`ctm` 或 `dtm` 可用于其他基线模型。

---

## 自定义可视化

**更高分辨率：**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 600
```

**中文可视化：**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset chinese_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language zh \
    --dpi 300
```

中文可视化使用适当的字体并正确处理字符渲染。

---

## 训练期间跳过可视化

跳过自动可视化以节省时间：

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

之后可以使用单独的可视化命令生成可视化。
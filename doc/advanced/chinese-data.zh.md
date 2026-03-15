# 中文数据处理

**[English](chinese-data.md)** | **[中文](chinese-data.zh.md)**

---

THETA处理中文文本的专门指南。

---

## 专门预处理

中文文本需要与英文不同的处理方式：

**数据清洗：**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/chinese_corpus/raw_data.csv \
    --output /root/autodl-tmp/data/chinese_corpus/chinese_corpus_cleaned.csv \
    --language chinese
```

中文清洗操作：
- 移除HTML实体
- 规范化全角和半角字符
- 处理中文标点符号
- 保留中文词边界
- 繁体转简体（可选）

**预处理：**
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

通义千问模型内部处理中文分词。

**训练：**
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

`--language zh`设置确保可视化中的中文字体。

---

## 中文可视化

中文可视化需要正确的字体配置：

```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset chinese_corpus \
    --mode zero_shot \
    --model_size 0.6B \
    --language zh \
    --dpi 300
```

可视化模块会自动：
- 选择兼容中文的字体
- 处理字符编码
- 调整中文文本布局
- 使用中文字符渲染词云

---

## 中英文混合数据

对于包含两种语言的数据集：

1. 按中文清洗（保留两种语言）
2. 正常预处理（通义千问处理多语言）
3. 使用适当的语言设置进行训练
4. 可视化可能显示混合文本

应根据主要内容语言在`--language`参数中指定主要语言。
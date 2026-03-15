# 快速入门

本教程演示如何在5分钟内对您的数据集训练THETA模型。

---

## 步骤1：准备数据

创建一个包含文本数据的CSV文件。CSV必须包含一个文本内容的列。

**示例CSV格式：**

```csv
text
"第一篇关于气候变化和全球变暖的文档。"
"第二篇关于可再生能源的文档。"
"第三篇关于环境政策和法规的文档。"
```

**必需的列：**

| 列名 | 类型 | 必需 | 描述 |
|------------|------|----------|-------------|
| text / content / cleaned_content / clean_text | 字符串 | 是 | 用于主题建模的文本内容 |
| label / category | 字符串/整数 | 否 | 有监督模式的标签 |
| year / timestamp / date | 整数/字符串 | 否 | DTM模型的时间戳 |

将您的CSV文件保存到数据目录：

```bash
mkdir -p /root/autodl-tmp/data/my_dataset
cp your_data.csv /root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv
```

注意：CSV文件名必须遵循 `{数据集名称}_cleaned.csv` 的格式。

---

## 步骤2：预处理数据

生成嵌入和词袋表示：

```bash
cd /root/autodl-tmp/ETM

python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --gpu 0
```

**此步骤的作用：**
1. 加载您的CSV文件
2. 为所有文档生成通义千问嵌入
3. 创建词袋表示
4. 构建词汇表
5. 将预处理数据保存到 `/root/autodl-tmp/result/0.6B/my_dataset/bow/`

**预期输出：**
```
加载数据集：my_dataset
处理1000篇文档...
生成嵌入：100%|████████| 32/32 [00:45<00:00, 1.41秒/批]
构建词汇表（大小=5000）...
保存预处理数据...
完成！文件保存到 /root/autodl-tmp/result/0.6B/my_dataset/bow/
```

验证数据文件是否已创建：

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --check-only
```

---

## 步骤3：训练模型

训练一个包含20个主题的THETA模型：

```bash
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

**训练参数说明：**

| 参数 | 值 | 描述 |
|-----------|-------|-------------|
| `--num_topics` | 20 | 要发现的主题数量 |
| `--epochs` | 100 | 最大训练轮数 |
| `--batch_size` | 64 | 训练的批大小 |
| `--hidden_dim` | 512 | 编码器的隐藏层维度 |
| `--learning_rate` | 0.002 | 优化器的学习率 |
| `--kl_start` | 0.0 | KL退火初始权重 |
| `--kl_end` | 1.0 | KL退火最终权重 |
| `--kl_warmup` | 50 | KL退火的预热轮数 |
| `--patience` | 10 | 早停耐心值 |

**训练进度：**
```
第1/100轮：损失=245.32，ELBO=-243.12，KL=2.20
第10/100轮：损失=156.78，ELBO=-154.56，KL=2.22
第20/100轮：损失=142.35，ELBO=-139.87，KL=2.48
...
第65/100轮：损失=128.45，ELBO=-125.23，KL=3.22
在第65轮触发早停
训练完成，用时23.5分钟
```

训练过程会自动：
1. 训练模型
2. 在多指标上评估
3. 生成可视化
4. 保存所有结果

---

## 步骤4：查看结果

训练后，结果保存在：

```
/root/autodl-tmp/result/0.6B/my_dataset/zero_shot/
├── checkpoints/
│   └── best_model.pt
├── metrics/
│   └── evaluation_results.json
└── visualizations/
    ├── topic_words_bars.png
    ├── topic_similarity.png
    ├── doc_topic_umap.png
    ├── topic_wordclouds.png
    ├── metrics.png
    └── pyldavis.html
```

**查看评估指标：**

```bash
cat /root/autodl-tmp/result/0.6B/my_dataset/zero_shot/metrics/evaluation_results.json
```

示例输出：
```json
{
  "TD": 0.89,
  "iRBO": 0.76,
  "NPMI": 0.42,
  "C_V": 0.65,
  "UMass": -2.34,
  "Exclusivity": 0.82,
  "PPL": 145.23
}
```

**查看可视化：**

在浏览器或图像查看器中打开可视化文件：
- `topic_words_bars.png`：显示每个主题前几个词的条形图
- `topic_similarity.png`：主题相似性热图
- `doc_topic_umap.png`：文档在主题空间中的UMAP投影
- `pyldavis.html`：交互式可视化（在浏览器中打开）

---

## 下一步？

- [用户指南](../user-guide/data-preparation.md) - 完整工作流程文档
- [高级用法](../advanced/custom-datasets.md) - 高级功能
- [示例](../examples/english-dataset.md) - 实际使用案例
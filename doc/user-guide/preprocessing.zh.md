# 数据预处理

**[English](preprocessing.md)** | **[中文](preprocessing.zh.md)**

---

预处理将清洗后的文本转换为训练所需的数值表示。此阶段使用Qwen模型生成嵌入并构建词袋表示。

---

## THETA模型预处理

### 基本预处理

对于名为 `my_dataset` 且带有清洗后CSV文件的数据集：

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

此命令：
1. 从 `/root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv` 加载CSV
2. 生成Qwen嵌入（0.6B模型维度为1024）
3. 构建词汇表大小为5000的词袋
4. 将输出保存到 `/root/autodl-tmp/result/0.6B/my_dataset/bow/`

### 模型规模选择

**0.6B模型 - 大多数用例的默认选择**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

处理速度：单GPU上约1000篇文档/分钟
内存需求：4GB显存

**4B模型 - 适中成本下更高质量**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 4B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 16 \
    --gpu 0
```

处理速度：约400篇文档/分钟
内存需求：12GB显存
由于嵌入更大（维度2560），批大小减小到16

**8B模型 - 最高质量**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 8B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 8 \
    --gpu 0
```

处理速度：约200篇文档/分钟
内存需求：24GB显存
由于嵌入很大（维度4096），批大小减小到8

### 训练模式选择

**zero_shot模式 - 标准无监督学习**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

适用场景：无可用标签或应忽略标签

**supervised模式 - 标签引导学习**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode supervised \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

适用场景：CSV包含 `label` 或 `category` 列
模型将在训练过程中融入标签信息

**unsupervised模式 - 显式无监督模式**

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode unsupervised \
    --vocab_size 5000 \
    --batch_size 32 \
    --gpu 0
```

适用场景：在带标签数据上与zero_shot模式比较，同时忽略标签

### 词汇表配置

词汇表大小影响模型容量和训练速度。更大的词汇表能捕捉更多词汇，但会增加计算量。

| 词汇表大小 | 适用场景 |
|----------------|-----------------|
| 3000-5000 | 小型语料库、领域特定文本、训练更快 |
| 5000-8000 | 通用目的，默认设置 |
| 8000-15000 | 大型多样化语料库，捕捉罕见词 |

### 序列长度配置

`max_length` 参数控制嵌入生成的输入截断。

| 最大长度 | 适用场景 |
|-----------|-----------------|
| 256 | 短文档（推文、评论），处理更快 |
| 512 | 中等长度文档（新闻文章），默认设置 |
| 1024 | 长文档（论文、报告），需要更多显存 |

### 组合清洗和预处理

单步处理原始数据：

**英文数据：**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --clean \
    --raw-input /root/autodl-tmp/data/my_dataset/raw_data.csv \
    --language english \
    --gpu 0
```

**中文数据：**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --clean \
    --raw-input /root/autodl-tmp/data/my_dataset/raw_data.csv \
    --language chinese \
    --gpu 0
```

`--clean` 标志触发预处理前的自动清洗。清洗后的CSV保存为数据集目录下的 `{dataset}_cleaned.csv`。

### 验证预处理数据

检查是否生成了所有必需文件：

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --check-only
```

预期输出：
```
检查数据集预处理文件：my_dataset
✓ BOW数据：/root/autodl-tmp/result/0.6B/my_dataset/bow/
✓ 嵌入：qwen_embeddings_zeroshot.npy（1024维）
✓ 词汇表：vocab.pkl（5000词）
✓ 文档索引：doc_indices.npy
所有必需文件都存在。
```

---

## 基线模型预处理

基线模型（LDA、ETM、CTM）使用不同的预处理流程，不需要Qwen嵌入。

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model baseline \
    --vocab_size 5000
```

这会生成：
- 词袋表示
- TF-IDF向量（用于CTM）
- Word2Vec嵌入（用于ETM）
- 文档-词项矩阵（用于LDA）

输出位置：`/root/autodl-tmp/result/baseline/my_dataset/bow/`

---

## DTM模型预处理

DTM需要CSV中的时间信息。指定时间列名：

```bash
python prepare_data.py \
    --dataset my_dataset \
    --model dtm \
    --vocab_size 5000 \
    --time_column year
```

时间列可以命名为 `year`、`timestamp` 或 `date`。文档会自动按时间切片分组用于时序建模。
# 英文数据集示例

**[English](english-dataset.md)** | **[中文](english-dataset.zh.md)**

---

展示使用THETA处理英文数据的完整教程。

---

## 示例1：英文新闻数据集

本示例演示分析新闻文章的完整工作流程。

### 数据集描述

- 领域：新闻文章
- 规模：5000篇文档
- 语言：英文
- 来源：在线新闻聚合器
- 时间范围：2020-2023年

### 步骤1：准备数据

创建数据集目录并放置CSV文件：

```bash
mkdir -p /root/autodl-tmp/data/news_corpus
```

CSV格式：
```csv
text
"美联储加息以应对通货膨胀..."
"气候峰会达成历史性排放协议..."
"科技公司因经济不确定性宣布裁员..."
```

保存为 `/root/autodl-tmp/data/news_corpus/news_corpus_cleaned.csv`

### 步骤2：预处理

```bash
cd /root/autodl-tmp/ETM

python prepare_data.py \
    --dataset news_corpus \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --batch_size 32 \
    --max_length 512 \
    --gpu 0
```

处理时间：在V100 GPU上约5分钟

### 步骤3：训练模型

使用25个主题训练以捕捉多样的新闻类别：

```bash
python run_pipeline.py \
    --dataset news_corpus \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 25 \
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

训练时间：约25分钟

### 步骤4：分析结果

```bash
cat /root/autodl-tmp/result/0.6B/news_corpus/zero_shot/metrics/evaluation_results.json
```

示例输出：
```json
{
  "TD": 0.87,
  "iRBO": 0.73,
  "NPMI": 0.39,
  "C_V": 0.62,
  "UMass": -2.56,
  "Exclusivity": 0.81,
  "PPL": 152.34
}
```

### 步骤5：与基线模型比较

```bash
python prepare_data.py \
    --dataset news_corpus \
    --model baseline \
    --vocab_size 5000

python run_pipeline.py \
    --dataset news_corpus \
    --models lda,etm,ctm \
    --num_topics 25 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

比较结果：

| 模型 | TD | NPMI | C_V | PPL |
|-------|-----|------|-----|-----|
| LDA | 0.72 | 0.24 | 0.48 | 185.2 |
| ETM | 0.79 | 0.31 | 0.55 | 168.5 |
| CTM | 0.83 | 0.36 | 0.59 | 158.7 |
| THETA | 0.87 | 0.39 | 0.62 | 152.3 |

---

## 示例2：使用4B模型的大规模数据集

本示例演示扩展到更大模型和数据集的用法。

### 数据集描述

- 领域：维基百科文章
- 规模：50000篇文档
- 语言：英文
- 复杂度：多样的主题和词汇

### 步骤1：使用4B模型预处理

```bash
python prepare_data.py \
    --dataset wikipedia \
    --model theta \
    --model_size 4B \
    --mode zero_shot \
    --vocab_size 10000 \
    --batch_size 16 \
    --max_length 512 \
    --gpu 0
```

处理时间：在A100 GPU上约4小时

### 步骤2：使用增强容量训练

```bash
python run_pipeline.py \
    --dataset wikipedia \
    --models theta \
    --model_size 4B \
    --mode zero_shot \
    --num_topics 50 \
    --epochs 150 \
    --batch_size 32 \
    --hidden_dim 768 \
    --learning_rate 0.001 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 80 \
    --patience 15 \
    --gpu 0 \
    --language en
```

### 步骤3：比较模型规模

| 模型 | TD | NPMI | C_V | PPL | 时间 |
|-------|-----|------|-----|-----|------|
| 0.6B | 0.89 | 0.43 | 0.66 | 138.5 | 90分钟 |
| 4B | 0.92 | 0.49 | 0.71 | 128.2 | 180分钟 |

4B模型以2倍成本提供显著的质量提升。

---

## 示例3：多模型比较

### 数据集描述

- 领域：电影评论
- 规模：4000篇文档
- 语言：英文

### 步骤1：为所有模型预处理

```bash
python prepare_data.py \
    --dataset movie_reviews \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --gpu 0

python prepare_data.py \
    --dataset movie_reviews \
    --model baseline \
    --vocab_size 5000
```

### 步骤2：训练所有模型

```bash
python run_pipeline.py \
    --dataset movie_reviews \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en

python run_pipeline.py \
    --dataset movie_reviews \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --gpu 0 \
    --language en
```

### 步骤3：比较表格

| 模型 | TD | iRBO | NPMI | C_V | Exclusivity | PPL | 时间 |
|-------|-----|------|------|-----|------------|-----|------|
| LDA | 0.74 | 0.68 | 0.26 | 0.49 | 0.76 | 178.3 | 12分钟 |
| ETM | 0.81 | 0.71 | 0.33 | 0.56 | 0.79 | 163.5 | 18分钟 |
| CTM | 0.84 | 0.74 | 0.37 | 0.60 | 0.82 | 154.2 | 22分钟 |
| THETA | 0.88 | 0.77 | 0.41 | 0.64 | 0.85 | 147.8 | 26分钟 |

---

## 示例4：超参数网格搜索

### 设置

数据集：2000篇新闻文章
目标：寻找主题质量的最佳超参数

### 网格搜索脚本

```bash
#!/bin/bash

topics=(15 20 25 30)
learning_rates=(0.001 0.002 0.005)
hidden_dims=(256 512 768)

for K in "${topics[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for hd in "${hidden_dims[@]}"; do
            echo "训练 K=$K, lr=$lr, hd=$hd"
            
            python run_pipeline.py \
                --dataset news \
                --models theta \
                --model_size 0.6B \
                --mode zero_shot \
                --num_topics $K \
                --learning_rate $lr \
                --hidden_dim $hd \
                --epochs 100 \
                --batch_size 64 \
                --gpu 0 \
                --language en
            
            mkdir -p results_grid/K${K}_lr${lr}_hd${hd}
            cp -r result/0.6B/news/zero_shot/* results_grid/K${K}_lr${lr}_hd${hd}/
        done
    done
done
```

### 最佳配置

分析显示最佳设置：
- 主题数量：20
- 学习率：0.002
- 隐藏层维度：512

---

## 最佳实践总结

### 数据准备
1. 预处理前彻底清洗数据
2. 确保CSV符合命名规则
3. 通过探索性分析验证数据质量

### 模型选择
1. 原型开发从THETA 0.6B开始
2. 与CTM基线进行比较
3. 如需生产环境，扩展到4B

### 超参数调优
1. 从默认参数开始
2. 根据语料库调整主题数量
3. 训练不稳定时调整学习率

### 评估
1. 查看多个指标，而非单一指标
2. 检查可视化进行定性评估
3. 与基线模型进行比较
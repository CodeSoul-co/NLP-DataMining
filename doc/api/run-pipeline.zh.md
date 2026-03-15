# run_pipeline.py

**[English](run-pipeline.md)** | **[中文](run-pipeline.zh.md)**

---

统一的训练、评估和可视化流程。

---

## 基本用法

```bash
python run_pipeline.py --dataset 数据集名称 --models 模型列表 [选项]
```

---

## 必需参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `--dataset` | 字符串 | 数据集名称 |
| `--models` | 字符串 | 逗号分隔的模型列表：`theta`、`lda`、`etm`、`ctm`、`dtm` |

## 模型配置（THETA）

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--model_size` | 字符串 | `0.6B` | 通义千问模型规模：`0.6B`、`4B` 或 `8B` |
| `--mode` | 字符串 | `zero_shot` | 训练模式：`zero_shot`、`supervised` 或 `unsupervised` |

## 主题模型参数

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|-----------|------|---------|-------|-------------|
| `--num_topics` | 整数 | `20` | 5-100 | 要发现的主题数量 |
| `--epochs` | 整数 | `100` | 10-500 | 最大训练轮数 |
| `--batch_size` | 整数 | `64` | 8-512 | 训练批处理大小 |

## 神经网络架构

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|-----------|------|---------|-------|-------------|
| `--hidden_dim` | 整数 | `512` | 128-1024 | 编码器隐藏层维度 |

## 优化

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|-----------|------|---------|-------|-------------|
| `--learning_rate` | 浮点数 | `0.002` | 0.00001-0.1 | 优化器的学习率 |

## KL退火

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|-----------|------|---------|-------|-------------|
| `--kl_start` | 浮点数 | `0.0` | 0.0-1.0 | KL散度初始权重 |
| `--kl_end` | 浮点数 | `1.0` | 0.0-1.0 | KL散度最终权重 |
| `--kl_warmup` | 整数 | `50` | 0-200 | KL退火的预热轮数 |

## 早停

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|-----------|------|---------|-------|-------------|
| `--patience` | 整数 | `10` | 1-50 | 早停前等待的轮数 |
| `--no_early_stopping` | 标志 | False | 不适用 | 禁用早停 |

## 硬件配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--gpu` | 整数 | `0` | GPU设备ID |

## 输出配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--language` | 字符串 | `en` | 可视化语言：`en` 或 `zh` |

## 流程控制

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--skip-train` | 标志 | False | 跳过训练，仅评估 |
| `--skip-eval` | 标志 | False | 跳过评估 |
| `--skip-viz` | 标志 | False | 跳过可视化 |
| `--check-only` | 标志 | False | 仅检查数据文件 |
| `--prepare` | 标志 | False | 训练前运行预处理 |

---

## 示例

**基本THETA训练：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --gpu 0
```

**多个基线模型：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --gpu 0
```

**自定义超参数：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 30 \
    --epochs 150 \
    --batch_size 32 \
    --hidden_dim 768 \
    --learning_rate 0.001 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 80 \
    --patience 15 \
    --gpu 0
```

**评估现有模型：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --skip-train \
    --gpu 0
```

---

## 输出文件

**THETA模型：**
```
/root/autodl-tmp/result/{model_size}/{dataset}/{mode}/
├── checkpoints/
│   ├── best_model.pt
│   └── training_history.json
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

**基线模型：**
```
/root/autodl-tmp/result/baseline/{dataset}/{model}/K{num_topics}/
├── checkpoints/
├── metrics/
└── visualizations/
```

---

## 返回码

| 退出码 | 含义 |
|-----------|---------|
| 0 | 成功 |
| 1 | 一般错误 |
| 2 | 文件未找到 |
| 3 | 无效参数 |
| 4 | CUDA内存不足 |
| 5 | 收敛失败 |
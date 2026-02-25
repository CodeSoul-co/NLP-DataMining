# THETA 主题模型技术说明文档

本文档详细说明 THETA 主题模型项目的完整使用流程，包括数据处理、模型训练、评估和可视化的所有命令和参数。

**所有命令均为完整可执行命令，可直接复制使用。**

---

## 目录

1. [项目概述](#1-项目概述)
2. [支持的模型](#2-支持的模型)
3. [新数据集使用指南](#3-新数据集使用指南)
4. [数据清洗命令](#4-数据清洗命令)
5. [数据预处理命令](#5-数据预处理命令)
6. [THETA模型训练命令](#6-theta模型训练命令)
7. [Baseline模型训练命令](#7-baseline模型训练命令)
8. [评估和可视化命令](#8-评估和可视化命令)
9. [目录结构](#9-目录结构)
10. [参数参考表](#10-参数参考表)

---

## 1. 项目概述

THETA 是一个基于 Qwen Embedding 的主题模型，在 ETM (Embedded Topic Model) 基础上改进设计。项目支持多种主题模型的训练和对比：

- **THETA** (我们的方法): 基于 Qwen3-Embedding 的主题模型
- **LDA**: Latent Dirichlet Allocation
- **ETM**: Embedded Topic Model (原始版本，使用 Word2Vec)
- **CTM**: Contextualized Topic Model (使用 SBERT)
- **DTM**: Dynamic Topic Model (时序主题模型)

---

## 2. 支持的模型

### 2.1 THETA 模型

| 特性 | 说明 |
|------|------|
| Embedding 模型 | Qwen3-Embedding (0.6B / 4B / 8B) |
| 训练模式 | zero_shot / supervised / unsupervised |
| 输入要求 | 清洗后的 CSV + Qwen Embedding + BOW |

**Qwen 模型规格：**

| 模型 | 参数量 | Embedding 维度 | 显存需求 | 推荐场景 |
|------|--------|----------------|----------|----------|
| 0.6B | 600M | 1024 | ~4GB | 快速实验，资源有限 |
| 4B | 4B | 2560 | ~12GB | 平衡效果与速度 |
| 8B | 8B | 4096 | ~24GB | 最佳效果 |

**训练模式说明：**

| 模式 | 说明 | CSV 要求 |
|------|------|----------|
| zero_shot | 无监督，不使用标签 | 只需 text 列 |
| supervised | 有监督，使用标签信息 | 需要 text + label 列 |
| unsupervised | 无监督，忽略标签 | 只需 text 列 |

### 2.2 Baseline 模型

| 模型 | Embedding | 特点 |
|------|-----------|------|
| LDA | 无 | 经典概率主题模型 |
| ETM | Word2Vec | 嵌入式主题模型 |
| CTM | SBERT | 上下文化主题模型 |
| DTM | SBERT | 动态时序主题模型 |

---

### 2.3 数据流程

```
原始数据 → [dataclean] → 清洗后CSV → [prepare_data] → Embedding/BOW → [run_pipeline] → 训练/评估/可视化
```

### 2.4 数据格式要求

**输入 CSV 文件要求：**

| 列名 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `text` / `content` / `cleaned_content` / `clean_text` | string | ✅ | 文本内容 |
| `label` / `category` | string/int | ❌ | 标签 (supervised 模式需要) |
| `year` / `timestamp` / `date` | int/string | ❌ | 时间戳 (DTM 模型需要) |

---

## 3. 新数据集使用指南

新数据集存放在 `/root/autodl-tmp/data/` 目录下，数据集名称可以是任意名称。

### 3.1 新数据集目录结构

```
/root/autodl-tmp/data/
└── {your_dataset_name}/                    # 任意数据集名称
    └── {your_dataset_name}_cleaned.csv     # 清洗后的 CSV 文件
```

### 3.2 新数据集完整流程 - THETA 模型 (英文数据)

假设新数据集名称为 `my_new_dataset`：

#### Step 1: 创建数据集目录

```bash
mkdir -p /root/autodl-tmp/data/my_new_dataset
```

#### Step 2: 放置清洗后的 CSV 文件

```bash
cp /path/to/your_cleaned_data.csv /root/autodl-tmp/data/my_new_dataset/my_new_dataset_cleaned.csv
```

#### Step 3: 数据预处理 (生成 Embedding 和 BOW)

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset my_new_dataset --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

#### Step 4: 检查数据文件是否生成成功

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset my_new_dataset --model theta --model_size 0.6B --mode zero_shot --check-only
```

#### Step 5: 训练 THETA 模型 (包含评估)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset my_new_dataset --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### Step 6: 单独运行可视化 (可选，训练时已自动生成)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/0.6B --dataset my_new_dataset --mode zero_shot --model_size 0.6B --language en --dpi 300
```

#### Step 7: 查看评估结果

评估结果保存在:
- 评估指标: `/root/autodl-tmp/result/0.6B/my_new_dataset/zero_shot/metrics/evaluation_results.json`
- 可视化图表: `/root/autodl-tmp/result/0.6B/my_new_dataset/zero_shot/visualizations/`

### 3.3 新数据集完整流程 - THETA 模型 (中文数据)

假设新数据集名称为 `chinese_dataset`：

#### Step 1: 创建数据集目录

```bash
mkdir -p /root/autodl-tmp/data/chinese_dataset
```

#### Step 2: 放置清洗后的 CSV 文件

```bash
cp /path/to/your_cleaned_data.csv /root/autodl-tmp/data/chinese_dataset/chinese_dataset_cleaned.csv
```

#### Step 3: 数据预处理 (生成 Embedding 和 BOW)

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset chinese_dataset --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

#### Step 4: 检查数据文件是否生成成功

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset chinese_dataset --model theta --model_size 0.6B --mode zero_shot --check-only
```

#### Step 5: 训练 THETA 模型 (包含评估)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset chinese_dataset --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language zh
```

#### Step 6: 单独运行可视化 (可选，训练时已自动生成)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/0.6B --dataset chinese_dataset --mode zero_shot --model_size 0.6B --language zh --dpi 300
```

#### Step 7: 查看评估结果

评估结果保存在:
- 评估指标: `/root/autodl-tmp/result/0.6B/chinese_dataset/zero_shot/metrics/evaluation_results.json`
- 可视化图表: `/root/autodl-tmp/result/0.6B/chinese_dataset/zero_shot/visualizations/`

### 3.4 新数据集完整流程 - 从原始数据清洗开始 (英文)

假设新数据集名称为 `raw_english_data`：

#### Step 1: 创建数据集目录

```bash
mkdir -p /root/autodl-tmp/data/raw_english_data
```

#### Step 2: 放置原始数据文件

```bash
cp /path/to/your_raw_data.csv /root/autodl-tmp/data/raw_english_data/raw_data.csv
```

#### Step 3: 清洗 + 预处理一步完成

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset raw_english_data --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --clean --raw-input /root/autodl-tmp/data/raw_english_data/raw_data.csv --language english --gpu 0
```

#### Step 4: 训练 THETA 模型 (包含评估)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset raw_english_data --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### Step 5: 单独运行可视化 (可选)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/0.6B --dataset raw_english_data --mode zero_shot --model_size 0.6B --language en --dpi 300
```

#### Step 6: 查看评估结果

评估结果保存在:
- 评估指标: `/root/autodl-tmp/result/0.6B/raw_english_data/zero_shot/metrics/evaluation_results.json`
- 可视化图表: `/root/autodl-tmp/result/0.6B/raw_english_data/zero_shot/visualizations/`

### 3.5 新数据集完整流程 - 从原始数据清洗开始 (中文)

假设新数据集名称为 `raw_chinese_data`：

#### Step 1: 创建数据集目录

```bash
mkdir -p /root/autodl-tmp/data/raw_chinese_data
```

#### Step 2: 放置原始数据文件

```bash
cp /path/to/your_raw_data.csv /root/autodl-tmp/data/raw_chinese_data/raw_data.csv
```

#### Step 3: 清洗 + 预处理一步完成

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset raw_chinese_data --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --clean --raw-input /root/autodl-tmp/data/raw_chinese_data/raw_data.csv --language chinese --gpu 0
```

#### Step 4: 训练 THETA 模型 (包含评估)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset raw_chinese_data --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language zh
```

#### Step 5: 单独运行可视化 (可选)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/0.6B --dataset raw_chinese_data --mode zero_shot --model_size 0.6B --language zh --dpi 300
```

#### Step 6: 查看评估结果

评估结果保存在:
- 评估指标: `/root/autodl-tmp/result/0.6B/raw_chinese_data/zero_shot/metrics/evaluation_results.json`
- 可视化图表: `/root/autodl-tmp/result/0.6B/raw_chinese_data/zero_shot/visualizations/`

### 3.6 新数据集完整流程 - Baseline 模型 (LDA/ETM/CTM)

假设新数据集名称为 `my_baseline_data`：

#### Step 1: 创建数据集目录并放置 CSV 文件

```bash
mkdir -p /root/autodl-tmp/data/my_baseline_data
cp /path/to/your_cleaned_data.csv /root/autodl-tmp/data/my_baseline_data/my_baseline_data_cleaned.csv
```

#### Step 2: 数据预处理

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset my_baseline_data --model baseline --vocab_size 5000
```

#### Step 3: 训练 LDA 模型

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset my_baseline_data --models lda --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### Step 4: 训练 ETM 模型

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset my_baseline_data --models etm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### Step 5: 训练 CTM 模型

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset my_baseline_data --models ctm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### Step 6: 同时训练所有 Baseline 模型

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset my_baseline_data --models lda,etm,ctm --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### Step 7: 单独运行可视化 (可选)

```bash
# LDA 可视化
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset my_baseline_data --model lda --num_topics 20 --language en --dpi 300

# ETM 可视化
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset my_baseline_data --model etm --num_topics 20 --language en --dpi 300

# CTM 可视化
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset my_baseline_data --model ctm --num_topics 20 --language en --dpi 300
```

#### Step 8: 查看评估结果

评估结果保存在:
- LDA: `/root/autodl-tmp/result/baseline/my_baseline_data/lda/K20/`
- ETM: `/root/autodl-tmp/result/baseline/my_baseline_data/etm/K20/`
- CTM: `/root/autodl-tmp/result/baseline/my_baseline_data/ctm/K20/`

### 3.7 新数据集完整流程 - DTM 模型 (需要时间戳)

假设新数据集名称为 `my_temporal_data`，CSV 文件包含 `year` 列：

#### Step 1: 创建数据集目录并放置 CSV 文件

```bash
mkdir -p /root/autodl-tmp/data/my_temporal_data
cp /path/to/your_temporal_data.csv /root/autodl-tmp/data/my_temporal_data/my_temporal_data_cleaned.csv
```

#### Step 2: 数据预处理 (指定时间列)

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset my_temporal_data --model dtm --vocab_size 5000 --time_column year
```

#### Step 3: 训练 DTM 模型 (包含评估)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset my_temporal_data --models dtm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### Step 4: 单独运行可视化 (可选)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset my_temporal_data --model dtm --num_topics 20 --language en --dpi 300
```

#### Step 5: 查看评估结果

评估结果保存在:
- DTM: `/root/autodl-tmp/result/baseline/my_temporal_data/dtm/K20/`

### 3.8 新数据集 - 使用 4B 模型

假设新数据集名称为 `large_dataset`：

#### Step 1: 数据预处理 (使用 4B 模型)

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset large_dataset --model theta --model_size 4B --mode zero_shot --vocab_size 5000 --batch_size 16 --max_length 512 --gpu 0
```

#### Step 2: 训练 THETA 模型 (包含评估)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset large_dataset --models theta --model_size 4B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 32 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### Step 3: 单独运行可视化 (可选)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/4B --dataset large_dataset --mode zero_shot --model_size 4B --language en --dpi 300
```

#### Step 4: 查看评估结果

评估结果保存在:
- 评估指标: `/root/autodl-tmp/result/4B/large_dataset/zero_shot/metrics/evaluation_results.json`
- 可视化图表: `/root/autodl-tmp/result/4B/large_dataset/zero_shot/visualizations/`

### 3.9 新数据集 - 使用 8B 模型

假设新数据集名称为 `premium_dataset`：

#### Step 1: 数据预处理 (使用 8B 模型)

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset premium_dataset --model theta --model_size 8B --mode zero_shot --vocab_size 5000 --batch_size 8 --max_length 512 --gpu 0
```

#### Step 2: 训练 THETA 模型 (包含评估)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset premium_dataset --models theta --model_size 8B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 16 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### Step 3: 单独运行可视化 (可选)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/8B --dataset premium_dataset --mode zero_shot --model_size 8B --language en --dpi 300
```

#### Step 4: 查看评估结果

评估结果保存在:
- 评估指标: `/root/autodl-tmp/result/8B/premium_dataset/zero_shot/metrics/evaluation_results.json`
- 可视化图表: `/root/autodl-tmp/result/8B/premium_dataset/zero_shot/visualizations/`

### 3.10 新数据集 - supervised 模式 (需要 label 列)

假设新数据集名称为 `labeled_dataset`，CSV 文件包含 `label` 列：

#### Step 1: 数据预处理 (supervised 模式)

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset labeled_dataset --model theta --model_size 0.6B --mode supervised --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

#### Step 2: 训练 THETA 模型 (包含评估)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset labeled_dataset --models theta --model_size 0.6B --mode supervised --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### Step 3: 单独运行可视化 (可选)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/0.6B --dataset labeled_dataset --mode supervised --model_size 0.6B --language en --dpi 300
```

#### Step 4: 查看评估结果

评估结果保存在:
- 评估指标: `/root/autodl-tmp/result/0.6B/labeled_dataset/supervised/metrics/evaluation_results.json`
- 可视化图表: `/root/autodl-tmp/result/0.6B/labeled_dataset/supervised/visualizations/`

---

## 4. 数据清洗命令

### 4.1 使用 dataclean 模块清洗英文数据

```bash
cd /root/autodl-tmp/ETM
python -m dataclean.main --input /root/autodl-tmp/data/hatespeech/raw_data.csv --output /root/autodl-tmp/data/hatespeech/hatespeech_cleaned.csv --language english
```

### 4.2 使用 dataclean 模块清洗中文数据

```bash
cd /root/autodl-tmp/ETM
python -m dataclean.main --input /root/autodl-tmp/data/edu_data/raw_data.csv --output /root/autodl-tmp/data/edu_data/edu_data_cleaned.csv --language chinese
```

### 4.3 清洗整个目录的文件

```bash
cd /root/autodl-tmp/ETM
python -m dataclean.main --input /root/autodl-tmp/data/my_dataset/raw/ --output /root/autodl-tmp/data/my_dataset/cleaned/ --language english
```

---

## 5. 数据预处理命令

### 5.1 THETA 模型预处理 - hatespeech 数据集

#### 5.1.1 使用 0.6B 模型 + zero_shot 模式

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

#### 5.1.2 使用 0.6B 模型 + supervised 模式

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 0.6B --mode supervised --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

#### 5.1.3 使用 0.6B 模型 + unsupervised 模式

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 0.6B --mode unsupervised --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

#### 5.1.4 使用 4B 模型 + zero_shot 模式

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 4B --mode zero_shot --vocab_size 5000 --batch_size 16 --max_length 512 --gpu 0
```

#### 5.1.5 使用 8B 模型 + zero_shot 模式

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 8B --mode zero_shot --vocab_size 5000 --batch_size 8 --max_length 512 --gpu 0
```

### 5.2 THETA 模型预处理 - mental_health 数据集

#### 5.2.1 使用 0.6B 模型 + zero_shot 模式

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset mental_health --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

#### 5.2.2 使用 0.6B 模型 + supervised 模式

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset mental_health --model theta --model_size 0.6B --mode supervised --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

### 5.3 THETA 模型预处理 - socialTwitter 数据集

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset socialTwitter --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

### 5.4 THETA 模型预处理 - FCPB 数据集

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset FCPB --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

### 5.5 THETA 模型预处理 - germanCoal 数据集

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset germanCoal --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0
```

### 5.6 THETA 模型预处理 - 自定义词表大小

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 0.6B --mode zero_shot --vocab_size 8000 --batch_size 32 --max_length 512 --gpu 0
```

### 5.7 THETA 模型预处理 - 自定义 max_length

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 1024 --gpu 0
```

### 5.8 THETA 模型预处理 - 只生成 BOW

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --bow-only
```

### 5.9 THETA 模型预处理 - 检查数据文件是否存在

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 0.6B --mode zero_shot --check-only
```

### 5.10 THETA 模型预处理 - 从原始数据清洗并预处理 (英文)

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --clean --raw-input /root/autodl-tmp/data/hatespeech/raw_data.csv --language english --gpu 0
```

### 5.11 THETA 模型预处理 - 从原始数据清洗并预处理 (中文)

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset edu_data --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --clean --raw-input /root/autodl-tmp/data/edu_data/raw_data.csv --language chinese --gpu 0
```

### 5.12 Baseline 模型预处理 - hatespeech 数据集

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset hatespeech --model baseline --vocab_size 5000
```

### 5.13 Baseline 模型预处理 - mental_health 数据集

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset mental_health --model baseline --vocab_size 5000
```

### 5.14 Baseline 模型预处理 - socialTwitter 数据集

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset socialTwitter --model baseline --vocab_size 5000
```

### 5.15 DTM 模型预处理 - edu_data 数据集

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset edu_data --model dtm --vocab_size 5000 --time_column year
```

### 5.16 DTM 模型预处理 - 自定义时间列

```bash
cd /root/autodl-tmp/ETM
python prepare_data.py --dataset edu_data --model dtm --vocab_size 5000 --time_column timestamp
```

---

## 6. THETA 模型训练命令

### 6.1 THETA 训练 - hatespeech 数据集

#### 6.1.1 使用 0.6B 模型 + zero_shot 模式 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### 6.1.2 使用 0.6B 模型 + supervised 模式 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode supervised --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### 6.1.3 使用 0.6B 模型 + unsupervised 模式 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode unsupervised --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### 6.1.4 使用 4B 模型 + zero_shot 模式 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 4B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 32 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### 6.1.5 使用 8B 模型 + zero_shot 模式 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 8B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 16 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### 6.1.6 使用 0.6B 模型 + zero_shot 模式 + 30个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 30 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### 6.1.7 使用 0.6B 模型 + zero_shot 模式 + 50个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 50 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

### 6.2 THETA 训练 - mental_health 数据集

#### 6.2.1 使用 0.6B 模型 + zero_shot 模式 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset mental_health --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

#### 6.2.2 使用 0.6B 模型 + supervised 模式 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset mental_health --models theta --model_size 0.6B --mode supervised --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

### 6.3 THETA 训练 - socialTwitter 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

### 6.4 THETA 训练 - FCPB 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset FCPB --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

### 6.5 THETA 训练 - germanCoal 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset germanCoal --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

### 6.6 THETA 训练 - edu_data 数据集 (中文)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset edu_data --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language zh
```

### 6.7 THETA 训练 - 自定义学习率

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.001 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

### 6.8 THETA 训练 - 自定义隐藏层维度

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 768 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language en
```

### 6.9 THETA 训练 - 自定义 KL 退火参数

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.1 --kl_end 0.9 --kl_warmup 30 --patience 10 --gpu 0 --language en
```

### 6.10 THETA 训练 - 禁用早停

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 200 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --no_early_stopping --gpu 0 --language en
```

### 6.11 THETA 训练 - 跳过训练只评估

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --skip-train --gpu 0 --language en
```

### 6.12 THETA 训练 - 跳过可视化

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --skip-viz --gpu 0 --language en
```

### 6.13 THETA 训练 - 只检查数据文件

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --check-only
```

---

## 7. Baseline 模型训练命令

### 7.1 LDA 模型训练

#### 7.1.1 LDA - hatespeech 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models lda --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### 7.1.2 LDA - mental_health 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset mental_health --models lda --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### 7.1.3 LDA - socialTwitter 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset socialTwitter --models lda --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### 7.1.4 LDA - FCPB 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset FCPB --models lda --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### 7.1.5 LDA - germanCoal 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset germanCoal --models lda --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### 7.1.6 LDA - edu_data 数据集 + 20个主题 (中文)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset edu_data --models lda --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language zh
```

#### 7.1.7 LDA - hatespeech 数据集 + 30个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models lda --num_topics 30 --epochs 100 --batch_size 64 --gpu 0 --language en
```

### 7.2 ETM 模型训练

#### 7.2.1 ETM - hatespeech 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models etm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### 7.2.2 ETM - mental_health 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset mental_health --models etm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### 7.2.3 ETM - socialTwitter 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset socialTwitter --models etm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### 7.2.4 ETM - FCPB 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset FCPB --models etm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### 7.2.5 ETM - edu_data 数据集 + 20个主题 (中文)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset edu_data --models etm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language zh
```

### 7.3 CTM 模型训练

#### 7.3.1 CTM - hatespeech 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models ctm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### 7.3.2 CTM - mental_health 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset mental_health --models ctm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### 7.3.3 CTM - socialTwitter 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset socialTwitter --models ctm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### 7.3.4 CTM - FCPB 数据集 + 20个主题

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset FCPB --models ctm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language en
```

#### 7.3.5 CTM - edu_data 数据集 + 20个主题 (中文)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset edu_data --models ctm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language zh
```

### 7.4 DTM 模型训练

#### 7.4.1 DTM - edu_data 数据集 + 20个主题 (中文)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset edu_data --models dtm --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language zh
```

#### 7.4.2 DTM - edu_data 数据集 + 15个主题 (中文)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset edu_data --models dtm --num_topics 15 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language zh
```

#### 7.4.3 DTM - edu_data 数据集 + 30个主题 (中文)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset edu_data --models dtm --num_topics 30 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --gpu 0 --language zh
```

### 7.5 多模型同时训练

#### 7.5.1 同时训练 LDA + ETM + CTM - hatespeech 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models lda,etm,ctm --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### 7.5.2 同时训练 LDA + ETM + CTM - mental_health 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset mental_health --models lda,etm,ctm --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### 7.5.3 同时训练 LDA + ETM + CTM - socialTwitter 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset socialTwitter --models lda,etm,ctm --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language en
```

#### 7.5.4 同时训练所有 Baseline 模型 - edu_data 数据集 (中文)

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset edu_data --models lda,etm,ctm,dtm --num_topics 20 --epochs 100 --batch_size 64 --gpu 0 --language zh
```

---

## 8. 评估和可视化命令

### 8.1 支持的评估指标

| 指标 | 全称 | 范围 | 说明 |
|------|------|------|------|
| **TD** | Topic Diversity | 0-1 | 主题多样性，越高越好 |
| **iRBO** | Inverse Rank-Biased Overlap | 0-1 | 逆 RBO，越高越好 |
| **NPMI** | Normalized PMI Coherence | -1 to 1 | 标准化 PMI 一致性，越高越好 |
| **C_V** | C_V Coherence | 0-1 | C_V 一致性，越高越好 |
| **UMass** | UMass Coherence | 负数 | UMass 一致性，越接近 0 越好 |
| **Exclusivity** | Topic Exclusivity | 0-1 | 主题排他性，越高越好 |
| **PPL** | Perplexity | 正数 | 困惑度，越低越好 |

### 8.2 可视化输出文件

| 可视化 | 文件名 | 说明 |
|--------|--------|------|
| 主题词条形图 | topic_words_bars.png | 每个主题的 Top-N 词 |
| 主题相似度热力图 | topic_similarity.png | 主题间相似度矩阵 |
| 文档-主题 UMAP | doc_topic_umap.png | 文档在主题空间的分布 |
| 主题词云 | topic_wordclouds.png | 每个主题的词云 |
| 评估指标图 | metrics.png | 各指标的可视化 |
| pyLDAvis | pyldavis.html | 交互式主题可视化 |

### 8.3 单独运行评估命令

#### 8.3.1 评估 THETA 模型 - hatespeech 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --skip-train --gpu 0 --language en
```

#### 8.3.2 评估 THETA 模型 - mental_health 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset mental_health --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --skip-train --gpu 0 --language en
```

#### 8.3.3 评估 LDA 模型 - hatespeech 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models lda --num_topics 20 --skip-train --gpu 0 --language en
```

#### 8.3.4 评估多个 Baseline 模型 - hatespeech 数据集

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py --dataset hatespeech --models lda,etm,ctm --num_topics 20 --skip-train --gpu 0 --language en
```

### 8.4 单独运行可视化命令

#### 8.4.1 THETA 模型可视化 - hatespeech 数据集 (英文)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/0.6B --dataset hatespeech --mode zero_shot --model_size 0.6B --language en --dpi 300
```

#### 8.4.2 THETA 模型可视化 - mental_health 数据集 (英文)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/0.6B --dataset mental_health --mode zero_shot --model_size 0.6B --language en --dpi 300
```

#### 8.4.3 THETA 模型可视化 - edu_data 数据集 (中文)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --result_dir /root/autodl-tmp/result/0.6B --dataset edu_data --mode zero_shot --model_size 0.6B --language zh --dpi 300
```

#### 8.4.4 LDA 模型可视化 - hatespeech 数据集 (英文)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset hatespeech --model lda --num_topics 20 --language en --dpi 300
```

#### 8.4.5 ETM 模型可视化 - hatespeech 数据集 (英文)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset hatespeech --model etm --num_topics 20 --language en --dpi 300
```

#### 8.4.6 CTM 模型可视化 - hatespeech 数据集 (英文)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset hatespeech --model ctm --num_topics 20 --language en --dpi 300
```

#### 8.4.7 DTM 模型可视化 - edu_data 数据集 (中文)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset edu_data --model dtm --num_topics 20 --language zh --dpi 300
```

#### 8.4.8 LDA 模型可视化 - mental_health 数据集 (英文)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset mental_health --model lda --num_topics 20 --language en --dpi 300
```

#### 8.4.9 CTM 模型可视化 - socialTwitter 数据集 (英文)

```bash
cd /root/autodl-tmp/ETM
python -m visualization.run_visualization --baseline --result_dir /root/autodl-tmp/result/baseline --dataset socialTwitter --model ctm --num_topics 20 --language en --dpi 300
```

---

## 9. 目录结构

### 9.1 项目目录

```
/root/autodl-tmp/ETM/
├── main.py                     # THETA 主训练脚本
├── run_pipeline.py             # 统一入口
├── prepare_data.py             # 数据预处理
├── config.py                   # 配置管理
├── requirements.txt            # 依赖
├── README.md                   # 简要说明
├── TECHNICAL_GUIDE.md          # 本文档
├── dataclean/                  # 数据清洗模块
│   ├── main.py
│   └── src/
├── bow/                        # BOW 生成模块
├── model/                      # 模型定义
│   ├── etm.py                  # THETA/ETM 模型
│   ├── lda.py                  # LDA 模型
│   ├── ctm.py                  # CTM 模型
│   ├── baseline_trainer.py     # Baseline 训练器
│   └── vocab_embedder.py       # 词向量生成
├── evaluation/                 # 评估模块
│   ├── topic_metrics.py        # 评估指标
│   └── unified_evaluator.py    # 统一评估器
├── visualization/              # 可视化模块
│   ├── run_visualization.py
│   ├── topic_visualizer.py
│   └── visualization_generator.py
└── utils/                      # 工具模块
    └── result_manager.py       # 结果管理
```

### 9.2 数据目录

```
/root/autodl-tmp/data/
└── {dataset}/
    └── {dataset}_cleaned.csv   # 清洗后的数据
```

### 9.3 结果目录

```
/root/autodl-tmp/result/
├── 0.6B/                       # THETA 0.6B 结果
│   └── {dataset}/
│       ├── bow/                # BOW 数据 (所有 mode 共享)
│       ├── zero_shot/          # zero_shot 模式结果
│       ├── supervised/         # supervised 模式结果
│       └── unsupervised/       # unsupervised 模式结果
├── 4B/                         # THETA 4B 结果
├── 8B/                         # THETA 8B 结果
└── baseline/                   # Baseline 模型结果
    └── {dataset}/
        ├── bow/                # BOW 数据
        ├── lda/                # LDA 结果
        ├── etm/                # ETM 结果
        ├── ctm/                # CTM 结果
        └── dtm/                # DTM 结果
```

### 9.4 Embedding 模型目录

```
/root/autodl-tmp/embedding_models/
├── qwen3_embedding_0.6B/       # Qwen 0.6B 模型
├── qwen3_embedding_4B/         # Qwen 4B 模型
└── qwen3_embedding_8B/         # Qwen 8B 模型
```

---

## 10. 参数参考表

### 10.1 prepare_data.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | string | 必填 | 数据集名称 |
| `--model` | string | 必填 | 模型类型: theta / baseline / dtm |
| `--model_size` | string | 0.6B | Qwen 模型大小: 0.6B / 4B / 8B |
| `--mode` | string | zero_shot | 训练模式: zero_shot / supervised / unsupervised |
| `--vocab_size` | int | 5000 | 词表大小 (1000-20000) |
| `--batch_size` | int | 32 | 批次大小 (8-128) |
| `--max_length` | int | 512 | Embedding 最大输入长度 (128-2048) |
| `--gpu` | int | 0 | GPU 设备 ID |
| `--clean` | flag | False | 先清洗数据 |
| `--raw-input` | string | None | 原始数据路径 |
| `--language` | string | english | 清洗语言: english / chinese |
| `--bow-only` | flag | False | 只生成 BOW |
| `--check-only` | flag | False | 只检查文件 |
| `--time_column` | string | year | 时间列名 (DTM 用) |

### 10.2 run_pipeline.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | string | 必填 | 数据集名称 |
| `--models` | string | 必填 | 模型列表 (逗号分隔): theta / lda / etm / ctm / dtm |
| `--model_size` | string | 0.6B | Qwen 模型大小: 0.6B / 4B / 8B |
| `--mode` | string | zero_shot | THETA 模式: zero_shot / supervised / unsupervised |
| `--num_topics` | int | 20 | 主题数量 (5-100) |
| `--epochs` | int | 100 | 训练轮数 (10-500) |
| `--batch_size` | int | 64 | 批次大小 (8-512) |
| `--hidden_dim` | int | 512 | 编码器隐藏层维度 (128-1024) |
| `--learning_rate` | float | 0.002 | 学习率 (0.00001-0.1) |
| `--kl_start` | float | 0.0 | KL 退火起始权重 (0-1) |
| `--kl_end` | float | 1.0 | KL 退火结束权重 (0-1) |
| `--kl_warmup` | int | 50 | KL 预热轮数 |
| `--patience` | int | 10 | 早停耐心值 (1-50) |
| `--no_early_stopping` | flag | False | 禁用早停 |
| `--gpu` | int | 0 | GPU 设备 ID |
| `--language` | string | en | 可视化语言: en / zh |
| `--skip-train` | flag | False | 跳过训练 |
| `--skip-eval` | flag | False | 跳过评估 |
| `--skip-viz` | flag | False | 跳过可视化 |
| `--check-only` | flag | False | 只检查文件 |
| `--prepare` | flag | False | 先预处理数据 |

### 10.3 visualization.run_visualization 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--result_dir` | string | 必填 | 结果目录路径 |
| `--dataset` | string | 必填 | 数据集名称 |
| `--mode` | string | zero_shot | THETA 模式 (THETA 模型用) |
| `--model_size` | string | 0.6B | Qwen 模型大小 (THETA 模型用) |
| `--baseline` | flag | False | 是否为 Baseline 模型 |
| `--model` | string | None | Baseline 模型名称: lda / etm / ctm / dtm |
| `--num_topics` | int | 20 | 主题数量 (Baseline 模型用) |
| `--language` | string | en | 可视化语言: en / zh |
| `--dpi` | int | 300 | 图片 DPI |

---

*文档版本: 2026-02-03*
*项目: THETA Topic Model*

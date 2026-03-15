# prepare_data.py

**[English](prepare-data.md)** | **[中文](prepare-data.zh.md)**

---

用于生成嵌入和词袋表示的数据预处理脚本。

---

## 基本用法

```bash
python prepare_data.py --dataset 数据集名称 --model 模型类型 [选项]
```

---

## 必需参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `--dataset` | 字符串 | 数据集名称（必须匹配 `/root/autodl-tmp/data/` 中的目录名） |
| `--model` | 字符串 | 模型类型：`theta`、`baseline` 或 `dtm` |

## 模型配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--model_size` | 字符串 | `0.6B` | 通义千问模型规模：`0.6B`、`4B` 或 `8B`（仅限THETA） |
| `--mode` | 字符串 | `zero_shot` | 训练模式：`zero_shot`、`supervised` 或 `unsupervised`（仅限THETA） |

## 数据处理

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|-----------|------|---------|-------|-------------|
| `--vocab_size` | 整数 | `5000` | 1000-20000 | 词袋表示的词汇表大小 |
| `--batch_size` | 整数 | `32` | 8-128 | 嵌入生成的批处理大小 |
| `--max_length` | 整数 | `512` | 128-2048 | 嵌入的最大序列长度 |

## GPU配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--gpu` | 整数 | `0` | GPU设备ID（0、1、2等） |

## 数据清洗

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--clean` | 标志 | False | 预处理前清洗数据 |
| `--raw-input` | 字符串 | None | 原始CSV文件路径（需要 `--clean`） |
| `--language` | 字符串 | `english` | 清洗语言：`english` 或 `chinese` |

## 实用选项

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--bow-only` | 标志 | False | 仅生成词袋，跳过嵌入 |
| `--check-only` | 标志 | False | 检查预处理文件是否存在 |
| `--time_column` | 字符串 | `year` | DTM的时间列名（仅限DTM） |

---

## 示例

**基本THETA预处理：**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000
```

**基线模型预处理：**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model baseline \
    --vocab_size 5000
```

**组合清洗和预处理：**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --vocab_size 5000 \
    --clean \
    --raw-input /path/to/raw.csv \
    --language english
```

**检查预处理文件：**
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --check-only
```

---

## 输出文件

预处理数据保存到：
```
/root/autodl-tmp/result/{model_size}/{dataset}/bow/
```

生成的文件：
- `qwen_embeddings_{mode}.npy`：文档嵌入
- `vocab.pkl`：词汇表字典
- `doc_indices.npy`：文档-词项索引
- `bow_matrix.npz`：稀疏词袋矩阵
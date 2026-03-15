# visualization.run_visualization

**[English](visualization.md)** | **[中文](visualization.zh.md)**

---

独立的可视化生成工具。

---

## 基本用法

```bash
python -m visualization.run_visualization --result_dir 目录路径 --dataset 数据集名称 [选项]
```

---

## 必需参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `--result_dir` | 字符串 | 结果目录路径 |
| `--dataset` | 字符串 | 数据集名称 |

## THETA模型参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--mode` | 字符串 | `zero_shot` | 训练模式（用于THETA模型） |
| `--model_size` | 字符串 | `0.6B` | 通义千问模型规模（用于THETA模型） |

## 基线模型参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--baseline` | 标志 | False | 指示为基线模型 |
| `--model` | 字符串 | None | 基线模型名称：`lda`、`etm`、`ctm` 或 `dtm` |
| `--num_topics` | 整数 | `20` | 主题数量（用于基线模型） |

## 输出配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--language` | 字符串 | `en` | 可视化语言：`en` 或 `zh` |
| `--dpi` | 整数 | `300` | 图像分辨率（每英寸点数） |

---

## 示例

**THETA模型可视化：**
```bash
python -m visualization.run_visualization \
    --result_dir /root/autodl-tmp/result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en \
    --dpi 300
```

**LDA模型可视化：**
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

**高分辨率可视化：**
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

---

## 输出文件

可视化结果保存到与模型结果相同的目录：
- `topic_words_bars.png`：主题词条形图
- `topic_similarity.png`：主题相似性热图
- `doc_topic_umap.png`：文档-主题UMAP投影
- `topic_wordclouds.png`：每个主题的词云
- `metrics.png`：评估指标比较
- `pyldavis.html`：交互式可视化

---

# dataclean.main

用于预处理原始文本的数据清洗模块。

## 基本用法

```bash
python -m dataclean.main --input 输入路径 --output 输出路径 --language 语言
```

## 参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `--input` | 字符串 | 输入的CSV文件路径或目录 |
| `--output` | 字符串 | 输出的CSV文件路径或目录 |
| `--language` | 字符串 | 语言：`english` 或 `chinese` |

## 示例

**清洗单个文件（英文）：**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language english
```

**清洗单个文件（中文）：**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language chinese
```

**清洗目录：**
```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw/ \
    --output /root/autodl-tmp/data/cleaned/ \
    --language english
```

## 清洗操作

**英文清洗：**
- 移除HTML标签和实体
- 移除URL和电子邮件地址
- 移除特殊字符（基本标点符号除外）
- 规范化空白
- 移除非ASCII字符（可选）
- 转换为小写（可选）

**中文清洗：**
- 移除HTML标签和实体
- 移除URL和电子邮件地址
- 规范化全角和半角字符
- 处理中文标点符号
- 移除非中文字符（可选）
- 保留词边界
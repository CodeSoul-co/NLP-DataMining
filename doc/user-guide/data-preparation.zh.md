# 数据准备

**[English](data-preparation.md)** | **[中文](data-preparation.zh.md)**

---

本指南涵盖数据格式要求和清洗流程。

---

## 数据格式要求

THETA接受具有特定列要求的CSV文件。预处理流程识别几个标准的文本内容列名。

**可接受的文本列名：**
- `text`
- `content`
- `cleaned_content`
- `clean_text`

**可选列：**
- `label` 或 `category` - 有监督模式必需
- `year`、`timestamp` 或 `date` - DTM（时间分析）必需

示例CSV结构：

```csv
text,label,year
"关于可再生能源和太阳能电池板的文档。",环境,2020
"讨论机器学习应用的文章。",技术,2021
"关于医疗改革的政策文件。",医疗,2022
```

---

## 数据清洗

原始文本通常包含降低主题质量的噪声。数据清洗模块处理英文和中文文本中的常见问题。

### 英文数据清洗

```bash
cd /root/autodl-tmp/ETM

python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language english
```

清洗过程移除：
- HTML标签和标记
- URL和电子邮件地址
- 特殊字符和符号
- 多余空白
- 不可打印字符

### 中文数据清洗

中文文本需要专门处理以实现正确的分词和清洗。

```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language chinese
```

中文的额外步骤：
- 移除繁体标点符号
- 处理全角和半角字符
- 保留中文词边界

### 批量清洗

处理目录中的多个文件：

```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw/ \
    --output /root/autodl-tmp/data/cleaned/ \
    --language english
```

输入目录中的所有CSV文件都将被处理并以相同的文件名保存到输出目录。
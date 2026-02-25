# DataClean

一个用于文本文件处理的工具，可以将各种格式的文本文件转换为CSV，使用NLP技术进行文本清洗，并将清洗后的数据整合为CSV文件，每个文件占一行。

## 功能特点

- 支持多种文件格式转换为CSV：
  - 文本文件 (.txt)
  - PDF文件 (.pdf)
  - CSV文件 (.csv)
  - JSON文件 (.json)
  - XML文件 (.xml)
  - HTML文件 (.html)
  - Word文档 (.docx)
  
- 支持中英文的NLP技术进行文本数据清洗：
  - 移除URL和HTML标签
  - 移除标点符号和特殊字符
  - 移除停用词（支持中文停用词）
  - 文本规范化
  - 中文分词处理
  
- 将清洗后的数据整合为CSV文件，每个文件占一行，便于后续分析

## 安装

1. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行界面

该项目提供了一个命令行界面，可以通过以下命令使用：

#### 1. 转换文本文件为CSV（一行一个文件）

```bash
python main.py convert <input_path> <output_csv> [--recursive] [--language {english,chinese}] [--clean/--no-clean] [--operations OPERATIONS...]
```

示例：
```bash
python main.py convert example_paper.pdf output/converted.csv --language chinese
```

#### 2. 清洗文本数据

```bash
python main.py clean <input_file> <output_file> [--language {english,chinese}] [--operations OPERATIONS...]
```

示例：
```bash
python main.py clean example_paper.pdf output/cleaned_text.txt --language chinese
```

#### 3. 批量处理文件

```bash
python main.py batch <input_dir> <output_dir> [--recursive] [--language {english,chinese}] [--operations OPERATIONS...]
```

示例：
```bash
python main.py batch ./data output/processed --recursive --language chinese
```

#### 4. 查看支持的文件格式

```bash
python main.py supported-formats
```

### 清洗操作选项

可用的清洗操作：
- `remove_urls`：移除URL
- `remove_html_tags`：移除HTML标签
- `remove_punctuation`：移除标点符号
- `remove_stopwords`：移除停用词
- `normalize_whitespace`：规范化空白字符
- `remove_numbers`：移除数字
- `remove_special_chars`：移除特殊字符

### 编程接口

也可以在Python代码中使用该项目的API：

```python
from src.converter import TextConverter
from src.cleaner import TextCleaner
from src.consolidator import DataConsolidator

# 初始化组件
converter = TextConverter()
cleaner = TextCleaner(language='chinese')
consolidator = DataConsolidator()

# 提取文本
text = converter.extract_text('example_paper.pdf')

# 清洗文本
cleaned_text = cleaner.clean_text(text, operations=['remove_urls', 'remove_stopwords'])

# 保存为CSV
csv_path = consolidator.create_oneline_csv(
    ['example_paper.pdf'],
    'output/result.csv',
    converter.extract_text,
    lambda text: cleaner.clean_text(text, operations=['remove_urls', 'remove_stopwords'])
)
```

## 项目结构

```
dataclean/
├── main.py                # 主入口点
├── requirements.txt       # 依赖列表
├── README.md              # 文档
└── src/                   # 源代码
    ├── __init__.py
    ├── converter.py       # 文件格式转换模块
    ├── cleaner.py         # 文本清洗模块
    ├── consolidator.py    # 数据整合模块
    └── processor.py       # 主处理模块
```

## 示例用法

### 1. 转换PDF文件为CSV并进行中文清洗

```bash
python main.py convert example_paper.pdf output/result.csv --language chinese
```

### 2. 批量处理目录中的所有文件

```bash
python main.py batch ./data output/processed --recursive --language chinese
```

### 3. 清洗单个文件

```bash
python main.py clean example_paper.pdf output/cleaned.txt --language chinese --operations remove_urls remove_stopwords
```

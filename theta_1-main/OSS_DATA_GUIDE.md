# OSS 数据和模型使用指南

## 概述

本项目的训练数据和预训练模型存储在阿里云 OSS 中，不直接上传到 Git 仓库。本文档说明如何从 OSS 获取和使用这些资源。

---

## 1. OSS Bucket 信息

| 配置项 | 值 |
|--------|-----|
| **Bucket 名称** | `theta-prod-20260123` |
| **区域** | 华东2（上海） |
| **Endpoint** | `oss-cn-shanghai.aliyuncs.com` |

---

## 2. OSS 目录结构

```
oss://theta-prod-20260123/
├── code/                          # 训练代码（DLC 挂载到 /mnt/code/）
│   └── ETM/                       # 主要训练代码
│
├── data/                          # 训练数据（DLC 挂载到 /mnt/data/）
│   ├── edu_data/                  # 教育数据集
│   │   └── edu_data_cleaned.csv
│   └── {job_id}/                  # 用户上传的数据
│       └── uploaded_file.csv
│
├── result/                        # 训练结果（DLC 挂载到 /mnt/result/）
│   ├── baseline/                  # 基线模型结果
│   │   └── {dataset}/
│   │       ├── data/              # 预处理数据
│   │       └── models/            # 模型结果
│   └── {model_size}/              # THETA 模型结果
│       └── {dataset}/
│
├── embedding_models/              # 嵌入模型（DLC 挂载到 /mnt/embedding_models/）
│   └── sbert/
│       └── all-MiniLM-L6-v2/      # SBERT 模型文件
│
└── sbert/                         # SBERT 模型备用路径（DLC 挂载到 /mnt/sbert/）
    └── all-MiniLM-L6-v2/
```

---

## 3. 本地开发：下载数据和模型

### 3.1 安装 ossutil

**Windows:**
```powershell
# 下载并解压 ossutil
Invoke-WebRequest -Uri "https://gosspublic.alicdn.com/ossutil/1.7.19/ossutil-v1.7.19-windows-amd64.zip" -OutFile "ossutil.zip"
Expand-Archive -Path "ossutil.zip" -DestinationPath "C:\Tools\ossutil"
```

**Linux/Mac:**
```bash
curl -o ossutil https://gosspublic.alicdn.com/ossutil/1.7.19/ossutil64
chmod +x ossutil
```

### 3.2 配置 ossutil

```bash
ossutil config
# 输入 Endpoint: oss-cn-shanghai.aliyuncs.com
# 输入 AccessKey ID
# 输入 AccessKey Secret
```

### 3.3 下载嵌入模型（本地开发必需）

```bash
# 下载 SBERT 模型到 embedding_models 目录
ossutil cp -r oss://theta-prod-20260123/embedding_models/sbert/ ./embedding_models/sbert/

# 或下载到 ETM/model/baselines/sbert 目录
ossutil cp -r oss://theta-prod-20260123/sbert/ ./ETM/model/baselines/sbert/sentence-transformers/
```

### 3.4 下载示例数据（可选）

```bash
# 下载 edu_data 数据集
ossutil cp -r oss://theta-prod-20260123/data/edu_data/ ./data/edu_data/

# 下载训练结果
ossutil cp -r oss://theta-prod-20260123/result/baseline/edu_data/ ./result/baseline/edu_data/
```

---

## 4. DLC 环境：自动挂载

在 DLC 训练任务中，OSS 数据会自动挂载到 `/mnt` 目录：

| OSS 路径 | DLC 挂载路径 | 说明 |
|----------|--------------|------|
| `oss://theta-prod-20260123/code/` | `/mnt/code/` | 训练代码 |
| `oss://theta-prod-20260123/data/` | `/mnt/data/` | 训练数据 |
| `oss://theta-prod-20260123/result/` | `/mnt/result/` | 训练结果 |
| `oss://theta-prod-20260123/embedding_models/` | `/mnt/embedding_models/` | 嵌入模型 |
| `oss://theta-prod-20260123/sbert/` | `/mnt/sbert/` | SBERT 模型 |

---

## 5. 环境变量配置

### 本地开发

```bash
# Linux/Mac
export ETM_BASE_DIR="."
export ETM_DATA_DIR="./data"
export ETM_RESULT_DIR="./result"
export ETM_CODE_DIR="./ETM"
export ETM_EMBEDDING_MODELS_DIR="./embedding_models"

# Windows PowerShell
$env:ETM_BASE_DIR = "."
$env:ETM_DATA_DIR = "./data"
$env:ETM_RESULT_DIR = "./result"
$env:ETM_CODE_DIR = "./ETM"
$env:ETM_EMBEDDING_MODELS_DIR = "./embedding_models"
```

### DLC 环境（自动设置）

```bash
export ETM_BASE_DIR="/mnt"
export ETM_DATA_DIR="/mnt/data"
export ETM_RESULT_DIR="/mnt/result"
export ETM_CODE_DIR="/mnt/code/ETM"
export ETM_EMBEDDING_MODELS_DIR="/mnt/embedding_models"
```

---

## 6. 代码中的路径处理

代码会自动检测运行环境并使用正确的路径：

```python
# ETM/config.py 中的路径配置
import os

# 自动检测环境
if os.path.exists('/mnt/code/ETM'):
    # DLC 环境
    BASE_DIR = '/mnt'
    DATA_DIR = '/mnt/data'
    RESULT_DIR = '/mnt/result'
else:
    # 本地环境
    BASE_DIR = os.environ.get('ETM_BASE_DIR', '.')
    DATA_DIR = os.environ.get('ETM_DATA_DIR', './data')
    RESULT_DIR = os.environ.get('ETM_RESULT_DIR', './result')
```

---

## 7. 常见问题

### Q: 本地运行时找不到 SBERT 模型？

A: 需要先从 OSS 下载模型：
```bash
ossutil cp -r oss://theta-prod-20260123/embedding_models/sbert/ ./embedding_models/sbert/
```

或者从 HuggingFace 下载：
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Q: 如何上传新的数据集到 OSS？

A: 使用 ossutil 上传：
```bash
ossutil cp ./my_data.csv oss://theta-prod-20260123/data/my_dataset/my_data.csv
```

### Q: 如何查看 OSS 中的文件？

A: 使用 ossutil 列出文件：
```bash
ossutil ls oss://theta-prod-20260123/data/
ossutil ls oss://theta-prod-20260123/result/baseline/edu_data/
```

---

## 8. 联系方式

如有问题，请联系后端开发团队。

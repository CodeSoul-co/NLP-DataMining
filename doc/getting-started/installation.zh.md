# 安装

本指南将帮助您在系统上安装THETA。

---

## 系统要求

THETA需要以下系统配置：

**操作系统：**
- Linux（推荐Ubuntu 18.04或更高版本）
- macOS 10.14或更高版本
- Windows 10/11（带WSL2）

**硬件要求：**

| 组件 | 最低配置 | 推荐配置 |
|-----------|---------|-------------|
| Python | 3.8+ | 3.9+ |
| 内存 | 8GB | 16GB+ |
| 显存 | 4GB（0.6B模型） | 12GB+（4B模型） |
| CUDA | 11.8+ | 12.1+ |
| 存储空间 | 20GB | 50GB+ |

**模型特定GPU要求：**

| 模型规模 | 参数量 | 嵌入维度 | 所需显存 | 适用场景 |
|-----------|-----------|---------------|---------------|----------|
| 0.6B | 6亿 | 1024 | ~4GB | 快速实验，资源有限 |
| 4B | 40亿 | 2560 | ~12GB | 性能和速度平衡 |
| 8B | 80亿 | 4096 | ~24GB | 最佳质量结果 |

---

## 安装步骤

### 步骤1：克隆仓库

```bash
git clone https://github.com/CodeSoul-co/THETA.git
cd THETA
```

### 步骤2：创建虚拟环境

使用conda（推荐）：

```bash
conda create -n theta python=3.9
conda activate theta
```

使用venv：

```bash
python -m venv theta-env
source theta-env/bin/activate  # 在Linux/macOS上
# theta-env\Scripts\activate   # 在Windows上
```

### 步骤3：安装依赖

```bash
pip install -r requirements.txt
```

安装包含以下关键包：
- PyTorch（带CUDA支持）
- Transformers
- Sentence-Transformers
- Gensim
- scikit-learn
- NumPy、Pandas
- Matplotlib、Seaborn
- UMAP-learn

### 步骤4：下载嵌入模型

下载Qwen3-Embedding模型：

```bash
# 0.6B模型（推荐首次用户使用）
python scripts/download_models.py --model 0.6B

# 4B模型
python scripts/download_models.py --model 4B

# 8B模型
python scripts/download_models.py --model 8B
```

模型将默认下载到 `/root/embedding_models/` 目录。

---

## 验证安装

检查PyTorch和CUDA是否正确安装：

```bash
python -c "import torch; print(f'PyTorch版本：{torch.__version__}')"
python -c "import torch; print(f'CUDA可用：{torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA版本：{torch.version.cuda}')"
```

预期输出：
```
PyTorch版本：2.0.1+cu118
CUDA可用：True
CUDA版本：11.8
```

检查THETA安装：

```bash
python -c "from src.model import etm; print('THETA安装成功')"
```

---

## 下一步

- [快速入门教程](quickstart.md) - 5分钟内训练您的第一个模型
- [数据准备指南](../user-guide/data-preparation.md) - 了解数据格式
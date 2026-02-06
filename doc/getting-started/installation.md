# Installation

This guide will help you install THETA on your system.

---

## System Requirements

THETA requires the following system specifications:

**Operating System:**
- Linux (Ubuntu 18.04 or later recommended)
- macOS 10.14 or later
- Windows 10/11 with WSL2

**Hardware Requirements:**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.9+ |
| RAM | 8GB | 16GB+ |
| GPU Memory | 4GB (0.6B model) | 12GB+ (4B model) |
| CUDA | 11.8+ | 12.1+ |
| Storage | 20GB | 50GB+ |

**Model-Specific GPU Requirements:**

| Model Size | Parameters | Embedding Dim | VRAM Required | Use Case |
|-----------|-----------|---------------|---------------|----------|
| 0.6B | 600M | 1024 | ~4GB | Quick experiments, limited resources |
| 4B | 4B | 2560 | ~12GB | Balanced performance and speed |
| 8B | 8B | 4096 | ~24GB | Best quality results |

---

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/CodeSoul-co/THETA.git
cd THETA
```

### Step 2: Create Virtual Environment

Using conda (recommended):

```bash
conda create -n theta python=3.9
conda activate theta
```

Using venv:

```bash
python -m venv theta-env
source theta-env/bin/activate  # On Linux/macOS
# theta-env\Scripts\activate   # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The installation includes the following key packages:
- PyTorch (with CUDA support)
- Transformers
- Sentence-Transformers
- Gensim
- scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn
- UMAP-learn

### Step 4: Download Embedding Models

Download the Qwen3-Embedding models:

```bash
# For 0.6B model (recommended for first-time users)
python scripts/download_models.py --model 0.6B

# For 4B model
python scripts/download_models.py --model 4B

# For 8B model
python scripts/download_models.py --model 8B
```

Models will be downloaded to `/root/embedding_models/` by default.

---

## Verify Installation

Check that PyTorch and CUDA are properly installed:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch version: 2.0.1+cu118
CUDA available: True
CUDA version: 11.8
```

Check THETA installation:

```bash
python -c "from src.model import etm; print('THETA installed successfully')"
```

---

## Next Steps

- [Quick Start Tutorial](quickstart.md) - Train your first model in 5 minutes
- [Data Preparation Guide](../user-guide/data-preparation.md) - Learn about data formats

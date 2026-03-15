# Troubleshooting

Common issues and solutions for THETA topic modeling.

---

## Installation Issues

### CUDA Not Available

**Problem:**
```
RuntimeError: CUDA is not available
torch.cuda.is_available() returns False
```

**Solutions:**

Check CUDA installation:
```bash
nvidia-smi
nvcc --version
```

Reinstall PyTorch with CUDA support:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Version Conflicts

Create fresh virtual environment:
```bash
conda create -n theta_clean python=3.9
conda activate theta_clean
pip install -r requirements.txt
```

### Model Download Failures

**Problem:**
```
OSError: Can't load model from 'Qwen/Qwen-Embedding-0.6B'
```

**Solution:** Download manually:
```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen-Embedding-0.6B
mv Qwen-Embedding-0.6B /root/embedding_models/qwen3_embedding_0.6B/
```

---

## Data Issues

### File Not Found

**Problem:**
```
FileNotFoundError: /root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv
```

**Solution:** Verify naming convention `{dataset_name}_cleaned.csv`:
```bash
mkdir -p /root/autodl-tmp/data/my_dataset
cp your_file.csv /root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv
```

### Missing Required Columns

**Problem:**
```
KeyError: 'text'
```

**Solution:** Rename column to standard name:
```python
import pandas as pd
df = pd.read_csv('data.csv')
df.rename(columns={'content': 'text'}, inplace=True)
df.to_csv('data_fixed.csv', index=False)
```

Accepted text column names: `text`, `content`, `cleaned_content`, `clean_text`

### Encoding Errors

```bash
iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
```

### Empty or Invalid Data

Check data statistics:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data.csv')
print(f'Rows: {len(df)}')
print(f'Empty text: {df[\"text\"].isna().sum()}')
print(f'Avg length: {df[\"text\"].str.len().mean():.1f}')
"
```

---

## Training Issues

### CUDA Out of Memory

**Solutions:**

Reduce batch size:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --batch_size 16 \
    --gpu 0
```

Memory requirements by configuration:

| Model Size | Batch Size | VRAM Required |
|-----------|-----------|---------------|
| 0.6B | 16 | ~6GB |
| 0.6B | 32 | ~8GB |
| 0.6B | 64 | ~12GB |
| 4B | 8 | ~10GB |
| 4B | 16 | ~14GB |
| 8B | 8 | ~18GB |
| 8B | 16 | ~28GB |

### Training Not Converging

**Solutions:**

Reduce learning rate:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --learning_rate 0.001 \
    --gpu 0
```

Adjust KL annealing:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 80 \
    --gpu 0
```

### Early Stopping Too Soon

Increase patience:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --patience 20 \
    --gpu 0
```

Or disable early stopping:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --epochs 200 \
    --no_early_stopping \
    --gpu 0
```

### NaN or Inf Values

Reduce learning rate significantly:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --learning_rate 0.0005 \
    --gpu 0
```

Check for data issues:
```bash
python -c "
import numpy as np
embeddings = np.load('result/0.6B/my_dataset/bow/qwen_embeddings_zeroshot.npy')
print(f'Contains NaN: {np.isnan(embeddings).any()}')
print(f'Contains Inf: {np.isinf(embeddings).any()}')
"
```

---

## Evaluation Issues

### Poor Metric Scores

**Solutions:**

- Train longer with `--epochs 200 --no_early_stopping`
- Adjust topic count: try 10, 15, 20, 25, 30
- Improve data quality: clean text more thoroughly, remove short documents
- Tune hyperparameters: `--hidden_dim 768 --learning_rate 0.001 --kl_warmup 80`

### Metric Computation Errors

Minimum requirements:
- Documents: 500+
- Average length: 20+ words
- Vocabulary: 1000+ words

---

## Visualization Issues

### Visualization Generation Fails

Install required fonts:
```bash
# Ubuntu/Debian
apt-get install fonts-liberation fonts-noto-cjk

# macOS
brew install font-liberation font-noto-cjk
```

Set matplotlib backend:
```bash
export MPLBACKEND=Agg
python run_pipeline.py --dataset my_dataset --models theta
```

### Chinese Characters Not Displaying

Install Chinese fonts:
```bash
apt-get install fonts-noto-cjk fonts-wqy-zenhei
```

Specify language parameter:
```bash
python run_pipeline.py \
    --dataset chinese_dataset \
    --models theta \
    --language zh \
    --gpu 0
```

### Low Resolution Images

Increase DPI:
```bash
python -m visualization.run_visualization \
    --result_dir result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --dpi 600 \
    --language en
```

DPI recommendations: Screen=150, Document=300, Publication=600, Poster=1200

---

## Performance Issues

### Slow Preprocessing

Increase batch size:
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --batch_size 64 \
    --gpu 0
```

Monitor GPU utilization: `nvidia-smi dmon`

### Memory Leaks

Clear cache periodically:
```python
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

---

## Specific Error Messages

| Error | Solution |
|-------|----------|
| "Dataset directory does not exist" | `mkdir -p /root/autodl-tmp/data/my_dataset` |
| "Preprocessed files not found" | Run `prepare_data.py` first |
| "Model checkpoint not found" | Run training first |
| "Invalid number of topics" | Use range 5-100 |
| "Supervised mode requires labels" | Add label column or use `--mode zero_shot` |
| "DTM requires time column" | Add `--time_column year` to preprocessing |

---

## Getting Help

### Report Issues

When reporting issues, include:
1. Complete error message
2. Command that produced error
3. System information (GPU, CUDA version)
4. Dataset characteristics (size, language)

System information:
```bash
python -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Community Resources

- GitHub Issues: [Report bugs](https://github.com/CodeSoul-co/THETA/issues)
- GitHub Discussions: [Ask questions](https://github.com/CodeSoul-co/THETA/discussions)
- Documentation: [https://theta.code-soul.com](https://theta.code-soul.com)
- Email: support@theta.code-soul.com

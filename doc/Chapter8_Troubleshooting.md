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

**Causes:**
1. CUDA not installed on system
2. PyTorch installed without CUDA support
3. CUDA version mismatch

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

Verify installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**

Install missing dependencies:
```bash
pip install -r requirements.txt
```

If specific package missing:
```bash
pip install transformers sentence-transformers gensim
```

### Version Conflicts

**Problem:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solution:**

Create fresh virtual environment:
```bash
conda create -n theta_clean python=3.9
conda activate theta_clean
pip install -r requirements.txt
```

Or use specific versions:
```bash
pip install torch==2.0.1 transformers==4.30.0
```

### Model Download Failures

**Problem:**
```
OSError: Can't load model from 'Qwen/Qwen-Embedding-0.6B'
```

**Causes:**
1. Network issues
2. Insufficient disk space
3. Permission problems

**Solutions:**

Download manually:
```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen-Embedding-0.6B
mv Qwen-Embedding-0.6B /root/embedding_models/qwen3_embedding_0.6B/
```

Check disk space:
```bash
df -h /root/embedding_models/
```

Fix permissions:
```bash
chmod -R 755 /root/embedding_models/
```

---

## Data Issues

### File Not Found

**Problem:**
```
FileNotFoundError: /root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv
```

**Causes:**
1. Incorrect filename
2. Wrong directory structure
3. Dataset name mismatch

**Solutions:**

Verify file location:
```bash
ls -la /root/autodl-tmp/data/my_dataset/
```

Check naming convention:
```bash
# Correct: {dataset_name}_cleaned.csv
# Example: news_cleaned.csv for dataset 'news'
```

Create correct structure:
```bash
mkdir -p /root/autodl-tmp/data/my_dataset
cp your_file.csv /root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv
```

### Missing Required Columns

**Problem:**
```
KeyError: 'text'
ValueError: CSV must contain a text column
```

**Causes:**
1. Text column named differently
2. Column name has spaces or special characters

**Solutions:**

Rename column to standard name:
```python
import pandas as pd

df = pd.read_csv('data.csv')
df.rename(columns={'content': 'text'}, inplace=True)
df.to_csv('data_fixed.csv', index=False)
```

Accepted text column names:
- `text`
- `content`
- `cleaned_content`
- `clean_text`

### Encoding Errors

**Problem:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions:**

Convert file encoding:
```bash
iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
```

Or specify encoding in Python:
```python
import pandas as pd

df = pd.read_csv('data.csv', encoding='latin-1')
df.to_csv('data_utf8.csv', index=False, encoding='utf-8')
```

### Empty or Invalid Data

**Problem:**
```
ValueError: Cannot process empty dataset
RuntimeError: All documents are too short
```

**Solutions:**

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

Filter invalid entries:
```python
import pandas as pd

df = pd.read_csv('data.csv')
df = df[df['text'].notna()]  # Remove NaN
df = df[df['text'].str.len() > 10]  # Remove very short
df.to_csv('data_cleaned.csv', index=False)
```

---

## Training Issues

### CUDA Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Causes:**
1. Batch size too large
2. Model size too large for GPU
3. Memory leak from previous run

**Solutions:**

Reduce batch size:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --batch_size 16 \
    --gpu 0
```

Use smaller model:
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --batch_size 16
```

Clear GPU cache:
```python
import torch
torch.cuda.empty_cache()
```

Monitor memory usage:
```bash
watch -n 1 nvidia-smi
```

Memory requirements by configuration:

| Model Size | Batch Size | VRAM Required |
|-----------|-----------|---------------|
| 0.6B | 16 | ~6GB |
| 0.6B | 32 | ~8GB |
| 0.6B | 64 | ~12GB |
| 4B | 8 | ~10GB |
| 4B | 16 | ~14GB |
| 4B | 32 | ~22GB |
| 8B | 8 | ~18GB |
| 8B | 16 | ~28GB |

### Training Not Converging

**Problem:**
Loss not decreasing or oscillating:
```
Epoch 10: Loss=245.32
Epoch 20: Loss=243.18
Epoch 30: Loss=244.67
Epoch 40: Loss=246.23
```

**Causes:**
1. Learning rate too high
2. Improper KL annealing
3. Poor initialization

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

Monitor training curves:
```bash
cat result/0.6B/my_dataset/zero_shot/checkpoints/training_history.json
```

### Early Stopping Too Soon

**Problem:**
Training stops at epoch 15 with high loss:
```
Early stopping triggered at epoch 15
Final loss: 189.34
```

**Causes:**
1. Patience too low
2. Learning rate too high causing oscillation
3. Poor validation split

**Solutions:**

Increase patience:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --patience 20 \
    --gpu 0
```

Disable early stopping:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --epochs 200 \
    --no_early_stopping \
    --gpu 0
```

Adjust learning rate:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --learning_rate 0.001 \
    --patience 15 \
    --gpu 0
```

### NaN or Inf Values

**Problem:**
```
RuntimeError: Loss is NaN
ValueError: Invalid value encountered in training
```

**Causes:**
1. Learning rate too high
2. Numerical instability
3. Data preprocessing issues

**Solutions:**

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

Use gradient clipping:
```python
# In training code
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Evaluation Issues

### Poor Metric Scores

**Problem:**
All evaluation metrics are low:
```json
{
  "TD": 0.42,
  "NPMI": 0.08,
  "C_V": 0.31,
  "PPL": 285.4
}
```

**Causes:**
1. Insufficient training
2. Too many or too few topics
3. Poor data quality
4. Suboptimal hyperparameters

**Solutions:**

Train longer:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --epochs 200 \
    --no_early_stopping \
    --gpu 0
```

Adjust topic count:
```bash
# Try different values
for K in 10 15 20 25 30; do
    python run_pipeline.py \
        --dataset my_dataset \
        --models theta \
        --num_topics $K \
        --gpu 0
done
```

Improve data quality:
- Clean text more thoroughly
- Remove very short documents
- Filter noise and spam

Tune hyperparameters:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --hidden_dim 768 \
    --learning_rate 0.001 \
    --kl_warmup 80 \
    --gpu 0
```

### Metric Computation Errors

**Problem:**
```
ValueError: Not enough data for coherence calculation
RuntimeError: NPMI computation failed
```

**Solutions:**

Check corpus size:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/my_dataset/my_dataset_cleaned.csv')
print(f'Documents: {len(df)}')
print(f'Avg words: {df[\"text\"].str.split().str.len().mean():.1f}')
"
```

Minimum requirements:
- Documents: 500+
- Average length: 20+ words
- Vocabulary: 1000+ words

Skip problematic metrics:
```bash
# Evaluate with different metric subsets
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --skip-train \
    --gpu 0
```

---

## Visualization Issues

### Visualization Generation Fails

**Problem:**
```
OSError: Cannot open font file
RuntimeError: Failed to generate visualizations
```

**Causes:**
1. Missing font files
2. Matplotlib configuration issues
3. Insufficient permissions

**Solutions:**

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

Skip visualization during training:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --skip-viz \
    --gpu 0

# Generate later
python -m visualization.run_visualization \
    --result_dir result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --language en
```

### Chinese Characters Not Displaying

**Problem:**
Chinese text shows as boxes or garbled characters in visualizations.

**Solutions:**

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

Configure matplotlib for Chinese:
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### Low Resolution Images

**Problem:**
Visualization images appear blurry or pixelated.

**Solutions:**

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

DPI recommendations:
- Screen viewing: 150
- Document embedding: 300
- Publication: 600
- Poster printing: 1200

---

## Performance Issues

### Slow Preprocessing

**Problem:**
Preprocessing takes hours for medium-sized dataset.

**Causes:**
1. Batch size too small
2. CPU bottleneck
3. Disk I/O limitations

**Solutions:**

Increase batch size:
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --batch_size 64 \
    --gpu 0
```

Monitor GPU utilization:
```bash
nvidia-smi dmon
```

If GPU underutilized, increase batch size further.

Use faster storage:
```bash
# Move data to SSD if available
mv /hdd/data /ssd/data
ln -s /ssd/data /root/autodl-tmp/data
```

### Slow Training

**Problem:**
Each epoch takes very long to complete.

**Solutions:**

Profile training:
```bash
python -m cProfile -o profile.stats run_pipeline.py ...
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

Optimize batch size:
```bash
# Find maximum stable batch size
for bs in 32 64 128 256; do
    python run_pipeline.py \
        --dataset my_dataset \
        --models theta \
        --batch_size $bs \
        --epochs 5 \
        --gpu 0 || break
done
```

Enable mixed precision:
```python
# In training code
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
```

### Memory Leaks

**Problem:**
Memory usage grows continuously during training.

**Solutions:**

Clear cache periodically:
```python
import gc
import torch

# After each epoch
gc.collect()
torch.cuda.empty_cache()
```

Monitor memory:
```bash
watch -n 1 'nvidia-smi && free -h'
```

Reduce data loader workers:
```bash
# In training script
DataLoader(dataset, batch_size=64, num_workers=2)  # Reduce from 4
```

---

## Specific Error Messages

### "Dataset directory does not exist"

**Error:**
```
FileNotFoundError: Dataset directory does not exist: /root/autodl-tmp/data/my_dataset
```

**Solution:**
```bash
mkdir -p /root/autodl-tmp/data/my_dataset
cp your_data.csv /root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv
```

### "Preprocessed files not found"

**Error:**
```
RuntimeError: Preprocessed files not found for dataset my_dataset
```

**Solution:**

Run preprocessing first:
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --mode zero_shot \
    --gpu 0
```

### "Model checkpoint not found"

**Error:**
```
FileNotFoundError: Model checkpoint not found
```

**Solution:**

Train model first or check path:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --gpu 0
```

Verify checkpoint exists:
```bash
ls result/0.6B/my_dataset/zero_shot/checkpoints/
```

### "Invalid number of topics"

**Error:**
```
ValueError: num_topics must be between 5 and 100
```

**Solution:**

Use valid topic range:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --num_topics 20 \
    --gpu 0
```

### "Supervised mode requires labels"

**Error:**
```
ValueError: Supervised mode requires label column
```

**Solutions:**

Add label column to CSV:
```python
import pandas as pd

df = pd.read_csv('data.csv')
# Add labels from separate file or manual annotation
df['label'] = labels
df.to_csv('data_with_labels.csv', index=False)
```

Or use zero_shot mode:
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --mode zero_shot \
    --gpu 0
```

### "DTM requires time column"

**Error:**
```
ValueError: DTM model requires time_column parameter
```

**Solution:**

Specify time column:
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model dtm \
    --time_column year \
    --vocab_size 5000
```

Ensure CSV has time column:
```csv
text,year
"Document text...",2020
```

---

## Getting Help

### Check Logs

Enable verbose logging:
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --gpu 0 2>&1 | tee training.log
```

### Reproduce Issues

Create minimal example:
```bash
# Small test dataset
head -100 large_dataset.csv > test_dataset.csv

# Quick test run
python run_pipeline.py \
    --dataset test \
    --models theta \
    --num_topics 5 \
    --epochs 10 \
    --gpu 0
```

### Report Issues

When reporting issues, include:
1. Complete error message
2. Command that produced error
3. System information (GPU, CUDA version)
4. Dataset characteristics (size, language)
5. Relevant configuration parameters

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

- GitHub Issues: Report bugs and feature requests
- GitHub Discussions: Ask questions and share ideas
- Documentation: https://theta.code-soul.com
- Email: support@theta.code-soul.com

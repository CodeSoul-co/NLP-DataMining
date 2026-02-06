# Distributed & Memory-Efficient Training

Guide for scaling THETA to larger datasets and constrained environments.

---

## Memory-Efficient Training

For limited VRAM:

**0.6B with reduced batch size:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --batch_size 32 \
    --gpu 0
```

**4B with minimal batch size:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 4B \
    --mode zero_shot \
    --num_topics 20 \
    --batch_size 16 \
    --gpu 0
```

**8B requiring high-end GPU:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 8B \
    --mode zero_shot \
    --num_topics 20 \
    --batch_size 8 \
    --gpu 0
```

Reduce batch size if out-of-memory errors occur.

---

## Memory Requirements

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

---

## Multi-GPU Processing

Train different configurations in parallel using separate GPUs:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset dataset1 --models theta --gpu 0 &

# Terminal 2  
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --dataset dataset2 --models theta --gpu 0 &

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python run_pipeline.py \
    --dataset dataset3 --models theta --gpu 0 &
```

Each process uses a different GPU.

---

## Environment Variables

### CUDA Configuration

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --dataset my_dataset --models theta

# Limit GPU memory fraction
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python run_pipeline.py ...

# Enable memory debugging
PYTORCH_NO_CUDA_MEMORY_CACHING=1 python run_pipeline.py ...
```

### Logging

```bash
# Disable progress bars
TQDM_DISABLE=1 python run_pipeline.py ...

# Reduce logging
export PYTHONWARNINGS="ignore"
python run_pipeline.py ...
```

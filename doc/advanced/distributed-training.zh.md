# 分布式与内存高效训练

**[English](distributed-training.md)** | **[中文](distributed-training.zh.md)**

---

将THETA扩展到更大数据集和受限环境的指南。

---

## 内存高效训练

适用于有限显存：

**0.6B模型配合减小批大小：**
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

**4B模型配合最小批大小：**
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

**8B模型需要高端GPU：**
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

如果出现内存不足错误，请减小批大小。

---

## 内存需求

| 模型规模 | 批大小 | 所需显存 |
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

## 多GPU处理

使用独立的GPU并行训练不同的配置：

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset dataset1 --models theta --gpu 0 &

# 终端2  
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --dataset dataset2 --models theta --gpu 0 &

# 终端3
CUDA_VISIBLE_DEVICES=2 python run_pipeline.py \
    --dataset dataset3 --models theta --gpu 0 &
```

每个进程使用不同的GPU。

---

## 环境变量

### CUDA配置

```bash
# 使用特定GPU
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --dataset my_dataset --models theta

# 限制GPU内存分配
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python run_pipeline.py ...

# 启用内存调试
PYTORCH_NO_CUDA_MEMORY_CACHING=1 python run_pipeline.py ...
```

### 日志记录

```bash
# 禁用进度条
TQDM_DISABLE=1 python run_pipeline.py ...

# 减少日志输出
export PYTHONWARNINGS="ignore"
python run_pipeline.py ...
```
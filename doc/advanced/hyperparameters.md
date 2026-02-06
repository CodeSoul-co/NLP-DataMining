# Hyperparameter Tuning

Systematic guide to optimizing THETA hyperparameters.

---

## Learning Rate Scheduling

**Conservative approach (unstable training):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --learning_rate 0.0005 \
    --epochs 150 \
    --gpu 0
```

**Standard approach:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --learning_rate 0.002 \
    --epochs 100 \
    --gpu 0
```

**Aggressive approach (slow convergence):**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --learning_rate 0.01 \
    --epochs 80 \
    --gpu 0
```

Monitor training loss curves to determine if adjustment is needed.

---

## Batch Size Optimization

| Batch Size | Advantages | Disadvantages |
|-----------|-----------|---------------|
| 32 | Lower memory, better exploration | Noisy updates, slower convergence |
| 64 | Balanced (default) | — |
| 128 | Stable updates, faster epochs | Higher memory, may overfit |

---

## KL Annealing Strategies

**No annealing (immediate full KL):**
`--kl_start 1.0 --kl_end 1.0 --kl_warmup 0`
Risk: Posterior collapse, poor topic quality

**Standard annealing (recommended):**
`--kl_start 0.0 --kl_end 1.0 --kl_warmup 50`

**Slow annealing (complex data):**
`--kl_start 0.0 --kl_end 1.0 --kl_warmup 80`

**Partial annealing (fine-tuning):**
`--kl_start 0.2 --kl_end 0.8 --kl_warmup 40`

---

## Hidden Dimension Tuning

| Hidden Dim | Use Case |
|-----------|----------|
| 256 | Small datasets or memory constrained |
| 512 | Default choice for most applications |
| 1024 | Large complex datasets when VRAM permits |

---

## Early Stopping Configuration

| Patience | Behavior |
|----------|----------|
| 5 | Stops quickly if validation loss plateaus |
| 10 | Default setting |
| 20 | Allows longer training before stopping |
| Disabled (`--no_early_stopping`) | Trains for all specified epochs |

---

## Vocabulary Size Selection

| Corpus Size | Vocabulary Size | Coverage |
|------------|----------------|----------|
| < 1K docs | 2000-3000 | ~85% |
| 1K-10K docs | 5000 | ~90% |
| 10K-100K docs | 8000-10000 | ~92% |
| > 100K docs | 10000-15000 | ~95% |

---

## Using Different Model Sizes

### Scaling Strategy

**Development workflow:**
1. Start with 0.6B model
2. Optimize hyperparameters
3. Scale to 4B for production
4. Use 8B for final results if needed

**Quick comparison:**
```bash
for size in 0.6B 4B 8B; do
    python run_pipeline.py \
        --dataset my_dataset \
        --models theta \
        --model_size $size \
        --mode zero_shot \
        --num_topics 20 \
        --gpu 0
done
```

### Quality vs Cost Analysis

**0.6B → 4B:**
- Topic diversity: +3-5%
- Coherence (NPMI): +10-15%
- Training time: +60-80%

**4B → 8B:**
- Topic diversity: +1-2%
- Coherence (NPMI): +5-8%
- Training time: +80-100%

Diminishing returns suggest 4B is often the best choice for production.

---

## Grid Search

Systematic hyperparameter exploration:

```bash
#!/bin/bash
topics=(15 20 25 30)
learning_rates=(0.001 0.002 0.005)
hidden_dims=(256 512 768)

for K in "${topics[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for hd in "${hidden_dims[@]}"; do
            echo "Training K=$K, lr=$lr, hd=$hd"
            
            python run_pipeline.py \
                --dataset my_dataset \
                --models theta \
                --model_size 0.6B \
                --mode zero_shot \
                --num_topics $K \
                --learning_rate $lr \
                --hidden_dim $hd \
                --epochs 100 \
                --batch_size 64 \
                --gpu 0

            mkdir -p results_grid/K${K}_lr${lr}_hd${hd}
            cp -r result/0.6B/my_dataset/zero_shot/* results_grid/K${K}_lr${lr}_hd${hd}/
        done
    done
done
```

---

## Batch Processing Multiple Datasets

```bash
#!/bin/bash
datasets=("news" "reviews" "papers" "social")

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    
    python prepare_data.py \
        --dataset $dataset \
        --model theta \
        --model_size 0.6B \
        --mode zero_shot \
        --vocab_size 5000 \
        --gpu 0
    
    python run_pipeline.py \
        --dataset $dataset \
        --models theta \
        --model_size 0.6B \
        --mode zero_shot \
        --num_topics 20 \
        --gpu 0
done
```

---

## Parallel Processing on Multiple GPUs

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

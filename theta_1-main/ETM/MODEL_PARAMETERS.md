# THETA 主题模型 - 完整参数参考

本文档列出每个模型的所有可调参数和完整运行命令。

---

## 1. LDA (Latent Dirichlet Allocation)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_topics` | int | 20 | 主题数量 |
| `--max_iter` | int | 100 | 最大迭代次数 |
| `--learning_method` | str | batch | 学习方法 (batch/online) |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py \
    --dataset edu_data \
    --models lda \
    --num_topics 20 \
    --language zh
```

---

## 2. HDP (Hierarchical Dirichlet Process)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--max_topics` | int | 150 | 最大主题数上限 (自动推断实际数量) |
| `--alpha` | float | 1.0 | 文档-主题先验 |
| `--gamma` | float | 1.0 | 主题-词先验 |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py \
    --dataset edu_data \
    --models hdp \
    --language zh
```

**注意**: HDP 自动推断主题数，不需要指定 `--num_topics`

---

## 3. BTM (Biterm Topic Model)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_topics` | int | 20 | 主题数量 |
| `--n_iter` | int | 100 | Gibbs采样迭代次数 |
| `--alpha` | float | 1.0 | 主题先验 |
| `--beta` | float | 0.01 | 词先验 |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py \
    --dataset edu_data \
    --models btm \
    --num_topics 20 \
    --language zh
```

**注意**: BTM 对长文档效率较低，适合短文本

---

## 4. STM (Structural Topic Model)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_topics` | int | 20 | 主题数量 |
| `--max_iter` | int | 100 | 最大迭代次数 |
| `--covariates` | array | None | 协变量矩阵 |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
python run_pipeline.py \
    --dataset edu_data \
    --models stm \
    --num_topics 20 \
    --language zh
```

---

## 5. NVDM (Neural Variational Document Model)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_topics` | int | 20 | 主题数量 |
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 64 | 批大小 |
| `--hidden_dim` | int | 256 | 隐藏层维度 |
| `--learning_rate` | float | 0.002 | 学习率 |
| `--gpu` | int | 0 | GPU编号 |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset edu_data \
    --models nvdm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 256 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language zh
```

---

## 6. GSM (Gaussian Softmax Model)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_topics` | int | 20 | 主题数量 |
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 64 | 批大小 |
| `--hidden_dim` | int | 256 | 隐藏层维度 |
| `--learning_rate` | float | 0.002 | 学习率 |
| `--gpu` | int | 0 | GPU编号 |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset edu_data \
    --models gsm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 256 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language zh
```

---

## 7. ProdLDA (Product of Experts LDA)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_topics` | int | 20 | 主题数量 |
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 64 | 批大小 |
| `--hidden_dim` | int | 256 | 隐藏层维度 |
| `--learning_rate` | float | 0.002 | 学习率 |
| `--gpu` | int | 0 | GPU编号 |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset edu_data \
    --models prodlda \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 256 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language zh
```

---

## 8. CTM (Contextualized Topic Model)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_topics` | int | 20 | 主题数量 |
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 64 | 批大小 |
| `--hidden_sizes` | tuple | (100,100) | 编码器隐藏层 |
| `--learning_rate` | float | 0.002 | 学习率 |
| `--inference_type` | str | zeroshot | 推断类型 (zeroshot/combined) |
| `--early_stopping_patience` | int | 10 | 早停耐心值 |
| `--gpu` | int | 0 | GPU编号 |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset edu_data \
    --models ctm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language zh
```

---

## 9. DTM (Dynamic Topic Model)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_topics` | int | 20 | 主题数量 |
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 64 | 批大小 |
| `--hidden_dim` | int | 256 | 隐藏层维度 |
| `--learning_rate` | float | 0.002 | 学习率 |
| `--gpu` | int | 0 | GPU编号 |

### 数据要求

CSV 文件必须包含时间列 (`year`, `timestamp`, `date`)

### 运行命令

```bash
cd /root/autodl-tmp/ETM
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset edu_data \
    --models dtm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 256 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language zh
```

---

## 10. ETM (Embedded Topic Model - 原始版本)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_topics` | int | 20 | 主题数量 |
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 64 | 批大小 |
| `--hidden_dim` | int | 256 | 隐藏层维度 |
| `--embedding_dim` | int | 300 | 词嵌入维度 |
| `--learning_rate` | float | 0.002 | 学习率 |
| `--gpu` | int | 0 | GPU编号 |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset edu_data \
    --models etm \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 256 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language zh
```

---

## 11. THETA (我们的方法)

### 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_size` | str | 0.6B | Qwen模型大小 (0.6B/4B/8B) |
| `--mode` | str | zero_shot | 训练模式 (zero_shot/supervised/unsupervised) |
| `--num_topics` | int | 20 | 主题数量 |
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 64 | 批大小 |
| `--hidden_dim` | int | 512 | 隐藏层维度 |
| `--learning_rate` | float | 0.002 | 学习率 |
| `--kl_start` | float | 0.0 | KL退火起始权重 |
| `--kl_end` | float | 1.0 | KL退火结束权重 |
| `--kl_warmup` | int | 50 | KL预热轮数 |
| `--patience` | int | 10 | 早停耐心值 |
| `--gpu` | int | 0 | GPU编号 |

### 运行命令

```bash
cd /root/autodl-tmp/ETM
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset edu_data \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 50 \
    --patience 10 \
    --gpu 0 \
    --language zh
```

---

## 通用参数

所有模型都支持以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | 必需 | 数据集名称 |
| `--models` | str | 必需 | 模型名称 (可逗号分隔多个) |
| `--language` | str | en | 可视化语言 (en/zh) |
| `--skip-train` | flag | false | 跳过训练 |
| `--skip-eval` | flag | false | 跳过评估 |
| `--skip-viz` | flag | false | 跳过可视化 |
| `--check-only` | flag | false | 只检查数据文件 |

---

## 使用 Bash 脚本

### 使用 05_train_baseline.sh

```bash
# 完整参数示例
bash scripts/05_train_baseline.sh \
    --dataset edu_data \
    --models lda \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --gpu 0 \
    --language zh
```

### 使用 04_train_theta.sh

```bash
# 完整参数示例
bash scripts/04_train_theta.sh \
    --dataset edu_data \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 50 \
    --patience 10 \
    --gpu 0 \
    --language zh
```

---

## 多模型同时训练

```bash
# 训练多个传统模型
bash scripts/05_train_baseline.sh \
    --dataset edu_data \
    --models lda,hdp,stm \
    --num_topics 20 \
    --language zh

# 训练多个神经网络模型
CUDA_VISIBLE_DEVICES=0 bash scripts/05_train_baseline.sh \
    --dataset edu_data \
    --models nvdm,gsm,prodlda \
    --num_topics 20 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 256 \
    --learning_rate 0.002 \
    --language zh
```

---

*文档更新日期: 2026-02-05*

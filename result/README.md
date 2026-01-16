# Result Directory Structure

All pipeline outputs are organized here by dataset and embedding mode.

## Directory Structure

```
result/
├── {dataset}/                     # e.g., socialTwitter, hatespeech
│   │
│   ├── bow/                       # BOW矩阵 (所有mode共用)
│   │   ├── bow_matrix.npz              # N×V 稀疏矩阵
│   │   ├── vocab.json                  # 词汇表
│   │   ├── vocab_embeddings.npy        # V×D 词汇embedding
│   │   └── bow_metadata.json           # BOW元数据
│   │
│   ├── zero_shot/                 # Zero-shot模式结果
│   │   ├── embeddings/
│   │   │   ├── doc_embeddings.npy      # N×D 文档embedding
│   │   │   ├── labels.npy              # 标签 (如有)
│   │   │   └── metadata.json           # 元数据
│   │   ├── model/                      # 训练好的模型
│   │   ├── evaluation/                 # 评估结果
│   │   └── visualization/              # 可视化
│   │
│   ├── supervised/                # Supervised模式结果
│   │   ├── embeddings/
│   │   │   ├── doc_embeddings.npy      # N×D 文档embedding (LoRA微调后)
│   │   │   ├── labels.npy              # 标签
│   │   │   └── metadata.json
│   │   ├── model/
│   │   │   ├── etm_model_{timestamp}.pt    # 模型权重
│   │   │   ├── theta_{timestamp}.npy       # N×K 文档-主题分布
│   │   │   ├── beta_{timestamp}.npy        # K×V 主题-词分布
│   │   │   ├── topic_words_{timestamp}.json # 主题词列表
│   │   │   ├── training_history_{timestamp}.json
│   │   │   └── config_{timestamp}.json
│   │   ├── evaluation/
│   │   │   └── metrics_{timestamp}.json    # 评估指标
│   │   └── visualization/
│   │       ├── topic_wordclouds.png
│   │       ├── topic_similarity.png
│   │       └── doc_topic_dist.png
│   │
│   └── unsupervised/              # Unsupervised模式结果
│       ├── embeddings/
│       │   └── doc_embeddings.npy      # N×D 文档embedding (SimCSE训练后)
│       ├── model/
│       ├── evaluation/
│       └── visualization/
```

## 重要说明

**BOW矩阵是共用的**: `bow/` 目录位于数据集根目录下，所有mode共用同一个BOW矩阵和词汇表。
这样设计是为了公平比较不同embedding模式的效果。

**Embedding来源**:
- `doc_embeddings.npy` 来自 `/root/autodl-tmp/embedding/outputs/{mode}/` 目录
- 复制时重命名为统一格式: `{dataset}_{mode}_embeddings.npy` → `doc_embeddings.npy`

## Key Matrices

| File | Shape | Description |
|------|-------|-------------|
| `doc_embeddings.npy` | N × D | Qwen document embeddings (D=1024) |
| `bow_matrix.npz` | N × V | Bag-of-Words sparse matrix |
| `vocab_embeddings.npy` | V × D | Vocabulary embeddings |
| `theta.npy` | N × K | Document-topic distribution |
| `beta.npy` | K × V | Topic-word distribution |
| `topic_embeddings.npy` | K × D | Topic embeddings |

Where:
- N = number of documents
- D = embedding dimension (1024)
- V = vocabulary size
- K = number of topics

## Usage

Results are automatically saved to this structure when running:

```bash
python ETM/main.py pipeline --dataset socialTwitter --mode zero_shot --num_topics 20
```

To load results:

```python
import numpy as np
from scipy import sparse

dataset = "socialTwitter"
mode = "zero_shot"
base_path = f"/root/autodl-tmp/result/{dataset}/{mode}"

# Load matrices
theta = np.load(f"{base_path}/model/theta.npy")
beta = np.load(f"{base_path}/model/beta.npy")
bow = sparse.load_npz(f"{base_path}/bow/bow_matrix.npz")
```

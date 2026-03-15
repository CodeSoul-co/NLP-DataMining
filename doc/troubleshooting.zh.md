# 故障排除

THETA主题建模的常见问题及解决方案。

---

## 安装问题

### CUDA不可用

**问题：**
```
RuntimeError: CUDA is not available
torch.cuda.is_available() returns False
```

**解决方案：**

检查CUDA安装：
```bash
nvidia-smi
nvcc --version
```

重新安装支持CUDA的PyTorch：
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 导入错误

**问题：**
```
ModuleNotFoundError: No module named 'transformers'
```

**解决方案：**
```bash
pip install -r requirements.txt
```

### 版本冲突

创建新的虚拟环境：
```bash
conda create -n theta_clean python=3.9
conda activate theta_clean
pip install -r requirements.txt
```

### 模型下载失败

**问题：**
```
OSError: Can't load model from 'Qwen/Qwen-Embedding-0.6B'
```

**解决方案：** 手动下载：
```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen-Embedding-0.6B
mv Qwen-Embedding-0.6B /root/embedding_models/qwen3_embedding_0.6B/
```

---

## 数据问题

### 文件未找到

**问题：**
```
FileNotFoundError: /root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv
```

**解决方案：** 验证命名规则 `{数据集名称}_cleaned.csv`：
```bash
mkdir -p /root/autodl-tmp/data/my_dataset
cp your_file.csv /root/autodl-tmp/data/my_dataset/my_dataset_cleaned.csv
```

### 缺少必需列

**问题：**
```
KeyError: 'text'
```

**解决方案：** 将列重命名为标准名称：
```python
import pandas as pd
df = pd.read_csv('data.csv')
df.rename(columns={'content': 'text'}, inplace=True)
df.to_csv('data_fixed.csv', index=False)
```

可接受的文本列名：`text`、`content`、`cleaned_content`、`clean_text`

### 编码错误

```bash
iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
```

### 空数据或无效数据

检查数据统计：
```bash
python -c "
import pandas as pd
df = pd.read_csv('data.csv')
print(f'行数：{len(df)}')
print(f'空文本：{df[\"text\"].isna().sum()}')
print(f'平均长度：{df[\"text\"].str.len().mean():.1f}')
"
```

---

## 训练问题

### CUDA内存不足

**解决方案：**

减小批大小：
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --batch_size 16 \
    --gpu 0
```

各配置的内存需求：

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

### 训练不收敛

**解决方案：**

降低学习率：
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --learning_rate 0.001 \
    --gpu 0
```

调整KL退火：
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 80 \
    --gpu 0
```

### 早停过早

增加耐心值：
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --patience 20 \
    --gpu 0
```

或禁用早停：
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --epochs 200 \
    --no_early_stopping \
    --gpu 0
```

### NaN或Inf值

大幅降低学习率：
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --learning_rate 0.0005 \
    --gpu 0
```

检查数据问题：
```bash
python -c "
import numpy as np
embeddings = np.load('result/0.6B/my_dataset/bow/qwen_embeddings_zeroshot.npy')
print(f'包含NaN：{np.isnan(embeddings).any()}')
print(f'包含Inf：{np.isinf(embeddings).any()}')
"
```

---

## 评估问题

### 指标得分低

**解决方案：**

- 使用 `--epochs 200 --no_early_stopping` 延长训练
- 调整主题数量：尝试 10、15、20、25、30
- 提高数据质量：更彻底地清洗文本，移除短文档
- 调整超参数：`--hidden_dim 768 --learning_rate 0.001 --kl_warmup 80`

### 指标计算错误

最低要求：
- 文档数量：500+
- 平均长度：20+词
- 词汇表：1000+词

---

## 可视化问题

### 可视化生成失败

安装所需字体：
```bash
# Ubuntu/Debian
apt-get install fonts-liberation fonts-noto-cjk

# macOS
brew install font-liberation font-noto-cjk
```

设置matplotlib后端：
```bash
export MPLBACKEND=Agg
python run_pipeline.py --dataset my_dataset --models theta
```

### 中文字符不显示

安装中文字体：
```bash
apt-get install fonts-noto-cjk fonts-wqy-zenhei
```

指定语言参数：
```bash
python run_pipeline.py \
    --dataset chinese_dataset \
    --models theta \
    --language zh \
    --gpu 0
```

### 图像分辨率低

增加DPI：
```bash
python -m visualization.run_visualization \
    --result_dir result/0.6B \
    --dataset my_dataset \
    --mode zero_shot \
    --model_size 0.6B \
    --dpi 600 \
    --language en
```

DPI建议：屏幕=150，文档=300，发表=600，海报=1200

---

## 性能问题

### 预处理慢

增加批大小：
```bash
python prepare_data.py \
    --dataset my_dataset \
    --model theta \
    --model_size 0.6B \
    --batch_size 64 \
    --gpu 0
```

监控GPU利用率：`nvidia-smi dmon`

### 内存泄漏

定期清除缓存：
```python
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

---

## 特定错误信息

| 错误 | 解决方案 |
|-------|----------|
| "Dataset directory does not exist" | `mkdir -p /root/autodl-tmp/data/my_dataset` |
| "Preprocessed files not found" | 先运行 `prepare_data.py` |
| "Model checkpoint not found" | 先运行训练 |
| "Invalid number of topics" | 使用5-100范围 |
| "Supervised mode requires labels" | 添加标签列或使用 `--mode zero_shot` |
| "DTM requires time column" | 在预处理中添加 `--time_column year` |

---

## 获取帮助

### 报告问题

报告问题时，请包含：
1. 完整的错误信息
2. 产生错误的命令
3. 系统信息（GPU、CUDA版本）
4. 数据集特征（规模、语言）

系统信息：
```bash
python -c "
import torch
import sys
print(f'Python：{sys.version}')
print(f'PyTorch：{torch.__version__}')
print(f'CUDA：{torch.version.cuda}')
print(f'GPU：{torch.cuda.get_device_name(0)}')
"
```

### 社区资源

- GitHub Issues：[报告错误](https://github.com/CodeSoul-co/THETA/issues)
- GitHub Discussions：[提问](https://github.com/CodeSoul-co/THETA/discussions)
- 文档：[https://theta.code-soul.com](https://theta.code-soul.com)
- 邮箱：support@theta.code-soul.com
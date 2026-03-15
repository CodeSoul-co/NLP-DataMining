# 评估

**[English](evaluation.md)** | **[中文](evaluation.zh.md)**

---

训练过程会自动使用多个指标进行评估。结果以JSON格式保存。

---

## 评估指标

**主题多样性（TD）**
- 范围：0-1
- 越高越好
- 衡量主题的独特性
- 计算方式：所有主题前N个词中唯一词的比例

**逆排名偏重重叠（iRBO）**
- 范围：0-1
- 越高越好
- 衡量主题的区分度
- 较低值表示主题冗余或重叠

**标准化点互信息（NPMI）**
- 范围：-1到1
- 越高越好
- 衡量主题词语义连贯性
- 基于外部语料库的点互信息

**C_V连贯性**
- 范围：0-1
- 越高越好
- 备选连贯性指标
- 基于滑动窗口共现

**UMass连贯性**
- 范围：负值
- 越接近0越好
- 经典连贯性指标
- 基于文档共现

**专有性（Exclusivity）**
- 范围：0-1
- 越高越好
- 衡量主题特异性
- 使用FREX分数计算

**困惑度（PPL）**
- 范围：正值
- 越低越好
- 衡量模型在保留数据上的拟合度
- 概率模型的标准评估

---

## 查看评估结果

结果保存在metrics目录中：

```bash
cat /root/autodl-tmp/result/0.6B/my_dataset/zero_shot/metrics/evaluation_results.json
```

示例输出：
```json
{
  "TD": 0.891,
  "iRBO": 0.762,
  "NPMI": 0.418,
  "C_V": 0.654,
  "UMass": -2.341,
  "Exclusivity": 0.823,
  "PPL": 145.23,
  "training_time": 1425.6,
  "num_topics": 20,
  "num_documents": 5000
}
```

---

## 单独运行评估

跳过训练，仅评估现有模型：

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --skip-train \
    --gpu 0 \
    --language en
```

这将从检查点加载训练好的模型并重新计算所有指标。

---

## 比较多个模型

评估所有基线模型：

```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --skip-train \
    --gpu 0 \
    --language en
```

每个模型的结果分别保存。使用指标文件构建比较表。
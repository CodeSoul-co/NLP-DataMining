# 项目概述

理解THETA的架构和工作流程将帮助您更有效地使用它。

---

## 架构概述

THETA由三个主要组件组成：

1. **嵌入模块**：使用Qwen3-Embedding生成上下文嵌入
2. **主题模型**：用于主题发现的神经变分推理
3. **评估与可视化**：全面的评估和展示

**数据流：**

```
原始文本 → 数据清洗 → 预处理 → 训练 → 评估 → 可视化
    ↓         ↓         ↓       ↓       ↓       ↓
  清洗CSV   嵌入+词袋  模型检查点 指标    图表
```

---

## 支持的模型

THETA支持多种主题建模方法：

### THETA模型（我们的方法）

**架构：**
- 基于Qwen3-Embedding的变分自编码器
- 用于主题分布推断的神经编码器
- 通过主题-词分布进行重建

**训练模式：**

| 模式 | 描述 | 适用场景 | 要求 |
|------|-------------|----------|--------------|
| zero_shot | 无监督学习 | 无可用标签 | 仅需文本列 |
| supervised | 标签引导学习 | 有可用标签 | 文本 + 标签列 |
| unsupervised | 无监督（忽略标签） | 与零样本比较 | 仅需文本列 |

**模型规模：**

三种规模共享相同架构，但嵌入质量不同：
- **0.6B**：最快，适合开发和测试
- **4B**：性能平衡，适用于生产环境
- **8B**：质量最佳，适用于研究和关键应用

### 基线模型

**LDA（潜在狄利克雷分配）**
- 经典概率主题模型
- 无神经组件
- 快速且可解释

**ETM（嵌入主题模型）**
- 使用Word2Vec嵌入
- 神经主题模型
- 优于LDA，快于THETA

**CTM（上下文主题模型）**
- 使用SBERT嵌入
- 上下文表示
- 质量和速度的良好平衡

**DTM（动态主题模型）**
- 时序主题建模
- 追踪主题随时间演化
- 需要时间戳信息

---

## 目录结构

THETA按以下结构组织文件：

### 项目目录

```
/root/autodl-tmp/ETM/
├── main.py                   # THETA训练脚本
├── run_pipeline.py           # 统一入口点
├── prepare_data.py           # 数据预处理
├── config.py                 # 配置
├── requirements.txt          # 依赖项
├── dataclean/               # 数据清洗模块
│   └── main.py
├── src/
│   ├── bow/                 # 词袋生成
│   ├── model/               # 模型定义
│   │   ├── etm.py          # THETA/ETM模型
│   │   ├── lda.py          # LDA模型
│   │   ├── ctm.py          # CTM模型
│   │   └── baseline_trainer.py
│   ├── evaluation/          # 评估指标
│   │   ├── topic_metrics.py
│   │   └── unified_evaluator.py
│   ├── visualization/       # 可视化
│   │   ├── run_visualization.py
│   │   ├── topic_visualizer.py
│   │   └── visualization_generator.py
│   └── utils/               # 工具函数
│       └── result_manager.py
└── scripts/
    └── download_models.py
```

### 数据目录

```
/root/autodl-tmp/data/
└── {数据集名称}/
    └── {数据集名称}_cleaned.csv
```

### 结果目录

```
/root/autodl-tmp/result/
├── 0.6B/                    # THETA 0.6B结果
│   └── {数据集名称}/
│       ├── bow/             # 所有模式共享
│       ├── zero_shot/       # 零样本结果
│       │   ├── checkpoints/
│       │   ├── metrics/
│       │   └── visualizations/
│       ├── supervised/      # 有监督结果
│       └── unsupervised/    # 无监督结果
├── 4B/                      # THETA 4B结果
├── 8B/                      # THETA 8B结果
└── baseline/                # 基线结果
    └── {数据集名称}/
        ├── bow/
        ├── lda/
        │   └── K20/        # 20个主题
        ├── etm/
        ├── ctm/
        └── dtm/
```

### 嵌入模型目录

```
/root/embedding_models/
├── qwen3_embedding_0.6B/
├── qwen3_embedding_4B/
└── qwen3_embedding_8B/
```

---

## 工作流程总结

典型的THETA工作流程包括四个阶段：

**阶段1：数据准备**
1. 收集原始文本数据
2. 清洗并格式化为CSV
3. 确保正确的列名

**阶段2：预处理**
1. 运行 `prepare_data.py` 生成嵌入
2. 创建词袋表示
3. 构建词汇表
4. 保存预处理文件

**阶段3：训练**
1. 运行 `run_pipeline.py` 训练模型
2. 模型带早停机制训练
3. 自动进行多指标评估
4. 自动生成可视化

**阶段4：分析**
1. 查看评估指标
2. 检查可视化
3. 分析发现的主题
4. 与基线模型比较

---

## 下一步

现在您已经了解了架构，可以：

- 探索 **[用户指南](../user-guide/data-preparation.md)** 获取每个组件的详细文档
- 尝试不同的**训练模式**（有监督、无监督）
- 实验不同的**模型规模**（4B、8B）
- 在高级使用部分学习**[超参数调优](../advanced/hyperparameters.md)**
- 将THETA与**[基线模型](../models/baselines.md)**（LDA、ETM、CTM）进行比较
- 使用专门流程处理**[中文文本数据](../advanced/chinese-data.md)**
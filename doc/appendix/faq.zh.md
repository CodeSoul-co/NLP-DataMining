# 附录

**[English](faq.md)** | **[中文](faq.zh.md)**

---

参考资料和补充信息。

---

## 完整参数参考

### prepare_data.py

| 参数 | 类型 | 默认值 | 范围/选项 | 必需 | 描述 |
|-----------|------|---------|--------------|----------|-------------|
| `--dataset` | 字符串 | - | - | 是 | 数据集名称 |
| `--model` | 字符串 | - | theta/baseline/dtm | 是 | 模型类型 |
| `--model_size` | 字符串 | 0.6B | 0.6B/4B/8B | 否 | 通义千问模型规模 |
| `--mode` | 字符串 | zero_shot | zero_shot/supervised/unsupervised | 否 | 训练模式 |
| `--vocab_size` | 整数 | 5000 | 1000-20000 | 否 | 词汇表大小 |
| `--batch_size` | 整数 | 32 | 8-128 | 否 | 批处理大小 |
| `--max_length` | 整数 | 512 | 128-2048 | 否 | 最大序列长度 |
| `--gpu` | 整数 | 0 | 0-7 | 否 | GPU设备ID |
| `--clean` | 标志 | False | - | 否 | 先清洗数据 |
| `--raw-input` | 字符串 | None | 文件路径 | 否 | 原始CSV路径 |
| `--language` | 字符串 | english | english/chinese | 否 | 清洗语言 |
| `--bow-only` | 标志 | False | - | 否 | 仅词袋 |
| `--check-only` | 标志 | False | - | 否 | 仅检查文件 |
| `--time_column` | 字符串 | year | 列名 | 否 | 时间列（DTM） |

### run_pipeline.py

| 参数 | 类型 | 默认值 | 范围/选项 | 必需 | 描述 |
|-----------|------|---------|--------------|----------|-------------|
| `--dataset` | 字符串 | - | - | 是 | 数据集名称 |
| `--models` | 字符串 | - | theta,lda,etm,ctm,dtm | 是 | 模型列表 |
| `--model_size` | 字符串 | 0.6B | 0.6B/4B/8B | 否 | 通义千问模型规模 |
| `--mode` | 字符串 | zero_shot | zero_shot/supervised/unsupervised | 否 | 训练模式 |
| `--num_topics` | 整数 | 20 | 5-100 | 否 | 主题数量 |
| `--epochs` | 整数 | 100 | 10-500 | 否 | 训练轮数 |
| `--batch_size` | 整数 | 64 | 8-512 | 否 | 批处理大小 |
| `--hidden_dim` | 整数 | 512 | 128-1024 | 否 | 隐藏层维度 |
| `--learning_rate` | 浮点数 | 0.002 | 0.00001-0.1 | 否 | 学习率 |
| `--kl_start` | 浮点数 | 0.0 | 0.0-1.0 | 否 | KL起始权重 |
| `--kl_end` | 浮点数 | 1.0 | 0.0-1.0 | 否 | KL结束权重 |
| `--kl_warmup` | 整数 | 50 | 0-200 | 否 | KL预热轮数 |
| `--patience` | 整数 | 10 | 1-50 | 否 | 早停耐心值 |
| `--no_early_stopping` | 标志 | False | - | 否 | 禁用早停 |
| `--gpu` | 整数 | 0 | 0-7 | 否 | GPU设备ID |
| `--language` | 字符串 | en | en/zh | 否 | 可视化语言 |
| `--skip-train` | 标志 | False | - | 否 | 跳过训练 |
| `--skip-eval` | 标志 | False | - | 否 | 跳过评估 |
| `--skip-viz` | 标志 | False | - | 否 | 跳过可视化 |

### visualization.run_visualization

| 参数 | 类型 | 默认值 | 范围/选项 | 必需 | 描述 |
|-----------|------|---------|--------------|----------|-------------|
| `--result_dir` | 字符串 | - | 目录 | 是 | 结果目录 |
| `--dataset` | 字符串 | - | - | 是 | 数据集名称 |
| `--mode` | 字符串 | zero_shot | zero_shot/supervised/unsupervised | 否 | THETA模式 |
| `--model_size` | 字符串 | 0.6B | 0.6B/4B/8B | 否 | 模型规模 |
| `--baseline` | 标志 | False | - | 否 | 基线标志 |
| `--model` | 字符串 | None | lda/etm/ctm/dtm | 否 | 基线模型 |
| `--num_topics` | 整数 | 20 | 5-100 | 否 | 主题数量 |
| `--language` | 字符串 | en | en/zh | 否 | 语言 |
| `--dpi` | 整数 | 300 | 72-1200 | 否 | 图像分辨率 |

---

## 目录结构

```
/root/autodl-tmp/
├── ETM/
│   ├── main.py
│   ├── run_pipeline.py
│   ├── prepare_data.py
│   └── src/
├── data/
│   └── {dataset}/
│       └── {dataset}_cleaned.csv
├── result/
│   ├── 0.6B/
│   ├── 4B/
│   ├── 8B/
│   └── baseline/
└── embedding_models/
```

---

## 硬件要求

| 配置 | CPU | 内存 | GPU | CUDA | 存储 |
|-------|-----|-----|-----|------|---------|
| 最低 | 4核 | 8GB | 4GB显存 | 11.8+ | 20GB |
| 推荐 | 8核 | 16GB | 12GB显存 | 12.1+ | 50GB SSD |
| 高性能 | 16+核 | 32GB+ | A100 40GB | 12.1+ | 200GB NVMe |

---

## 常见问题

**问：THETA有什么不同？**  
答：THETA使用通义千问嵌入和神经变分推理，相比LDA或ETM具有更好的语义理解能力。

**问：应该使用哪个模型规模？**  
答：原型开发用0.6B，生产环境用4B，追求最高质量用8B。

**问：最小数据集规模？**  
答：建议至少500篇文档，平均每篇50词以上。

**问：训练时间？**  
答：5K文档在V100上：0.6B约25分钟，4B约50分钟。

**问：需要GPU吗？**  
答：是的。预处理和训练都需要GPU。

---

## 引用

```bibtex
@article{theta2024,
  title={THETA：基于通义千问嵌入的先进主题建模},
  author={CodeSoul团队},
  year={2024},
  url={https://github.com/CodeSoul-co/THETA}
}
```

---

## 联系方式

- 网站：[https://theta.code-soul.com](https://theta.code-soul.com)
- GitHub：[https://github.com/CodeSoul-co/THETA](https://github.com/CodeSoul-co/THETA)
- 邮箱：support@theta.code-soul.com

---

**文档版本**：1.0.0  
**最后更新**：2026年2月6日
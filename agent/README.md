# Topic Model Agent System

基于LangChain的主题模型分析多Agent系统，集成ETM（Embedded Topic Model）实现端到端的主题建模分析流程。

## 项目结构

```
THETA-main/
├── agents/                     # Agent模块
│   ├── __init__.py
│   ├── data_cleaning_agent.py  # 数据验证与配置Agent
│   ├── bow_agent.py            # BOW生成Agent
│   ├── embedding_agent.py      # 嵌入生成Agent
│   ├── etm_agent.py            # ETM主题建模Agent
│   ├── visualization_agent.py  # 可视化生成Agent
│   ├── report_agent.py         # Word报告生成Agent
│   ├── text_qa_agent.py        # 文本问答Agent
│   ├── vision_qa_agent.py      # 可视化问答Agent
│   ├── orchestrator_agent.py   # 编排Agent（总控）
│   └── langchain_agent.py      # LangChain集成
├── app/
│   ├── agent_integration.py    # Agent集成层
│   └── api.py                  # FastAPI接口
├── scripts/
│   ├── run_engine_a.py         # Engine A: BOW生成
│   ├── run_engine_b.py         # Engine B: Embedding生成
│   ├── run_engine_c.py         # Engine C: ETM训练
│   └── run_visualization.py    # 可视化生成
├── tools/                      # Agent工具函数
├── data/                       # 输入数据目录
│   └── {job_id}/data.csv
├── ETM/outputs/                # ETM输出目录
│   ├── vocab/{job_id}_vocab.json
│   ├── bow/{job_id}_bow.npz
│   ├── theta/{job_id}_theta.npy
│   ├── beta/{job_id}_beta.npy
│   ├── alpha/{job_id}_alpha.npy
│   └── topic_words/{job_id}_topics.json
├── embedding/outputs/          # Embedding输出目录
│   └── zero_shot/
│       ├── {job_id}_embeddings.npy
│       └── {job_id}_vocab_emb.npy
├── visualization/outputs/      # 可视化输出目录
│   └── {job_id}/
│       ├── wordcloud_topic_{i}.png
│       ├── topic_distribution.png
│       ├── heatmap_doc_topic.png
│       ├── coherence_curve.png
│       └── topic_similarity.png
├── result/                     # 结果输出目录
│   └── {job_id}/
│       ├── config.yaml
│       ├── metrics.json
│       ├── analysis_result.json
│       ├── report.docx
│       ├── theta.csv
│       ├── beta.csv
│       └── log.txt
├── run_full_pipeline.py        # 完整流程入口
├── requirements.txt            # 依赖
└── .env                        # 环境变量
```

---

## Agent功能说明

### 1. DataCleaningAgent（数据验证Agent）
**职责**：验证输入数据格式，创建配置文件

| 项目 | 说明 |
|------|------|
| **输入** | `data/{job_id}/data.csv` |
| **输出** | `result/{job_id}/config.yaml`, `result/{job_id}/log.txt` |
| **验证** | CSV格式、非空数据、必要列存在 |

### 2. BowAgent（词袋生成Agent）
**职责**：调用Engine A生成词汇表和BOW矩阵

| 项目 | 说明 |
|------|------|
| **输入** | `data/{job_id}/data.csv` |
| **输出** | `ETM/outputs/vocab/{job_id}_vocab.json`, `ETM/outputs/bow/{job_id}_bow.npz` |
| **参数** | `max_vocab_size=10000`, `min_word_freq=2` |

### 3. EmbeddingAgent（嵌入生成Agent）
**职责**：调用Engine B生成文档和词汇嵌入

| 项目 | 说明 |
|------|------|
| **输入** | 词汇表JSON, BOW矩阵 |
| **输出** | `embedding/outputs/zero_shot/{job_id}_embeddings.npy` (N×1024), `embedding/outputs/zero_shot/{job_id}_vocab_emb.npy` (V×1024) |
| **模式** | Zero-shot（使用Qwen3-Embedding或简单模式） |

### 4. ETMAgent（主题建模Agent）
**职责**：调用Engine C运行ETM训练

| 项目 | 说明 |
|------|------|
| **输入** | BOW矩阵, 文档嵌入, 词汇嵌入 |
| **输出** | `theta.npy` (N×K), `beta.npy` (K×V), `alpha.npy` (K×1024), `topics.json` |
| **参数** | `num_topics=20`, `epochs=100` |

### 5. VisualizationAgent（可视化Agent）
**职责**：生成5类可视化图表

| 图表类型 | 文件名 | 说明 |
|----------|--------|------|
| 词云 | `wordcloud_topic_{i}.png` | 每个主题的关键词词云 |
| 主题分布 | `topic_distribution.png` | 各主题占比柱状图 |
| 热力图 | `heatmap_doc_topic.png` | 文档-主题关联热力图 |
| 一致性曲线 | `coherence_curve.png` | 主题数vs一致性得分 |
| 相似度矩阵 | `topic_similarity.png` | 主题间相似度热力图 |

### 6. ReportAgent（报告生成Agent）
**职责**：自动生成结构化Word分析报告

**报告结构**：
- 封面：项目信息、生成时间
- 第一章：数据概览
- 第二章：方法说明
- 第三章：主题识别结果（含词云）
- 第四章：主题分布分析
- 第五章：质量评估
- 附录：完整参数配置

### 7. TextQAAgent（文本问答Agent）
**职责**：基于分析结果回答用户问题

**核心能力**：
- 主题语义解读：根据关键词推断内容领域
- 内容结构分析：识别核心话题和边缘话题
- 主题关联分析：揭示主题间语义关系
- 业务洞察：提供可操作的业务建议

### 8. VisionQAAgent（可视化问答Agent）
**职责**：基于可视化图表回答用户问题

**分析能力**：
- 词云图语义解读
- 主题分布内容分析
- 热力图文档聚类识别
- 相似度矩阵主题关联分析

### 9. OrchestratorAgent（编排Agent）
**职责**：统一入口，管理依赖链和流程控制

**依赖链**：
```
DataCleaning → BOW → Embedding → ETM → Visualization → Report
```

---

## 交互式问答示例

### 文本问答示例

**用户问题**：这些主题分别讨论了什么内容？

**Agent回答**：
```
根据主题关键词分析，各主题讨论的内容领域如下：

**主题0 - 监管评估**
关键词：审核、评估、备案、标准、机构
内容解读：该主题聚焦于行业监管和评估体系的讨论，涉及备案审核流程、
标准制定以及监管机构的职责。

**主题1 - 资金投入**
关键词：普惠、补助、财政、扶持、投入
内容解读：该主题关注财政支持政策，特别是普惠性补助和财政扶持措施。

...
```

### 可视化问答示例

**用户问题**：从词云图来看，哪些主题最重要？

**Agent回答**：
```
从词云图分析，以下主题在文档集中占据核心地位：

1. **主题0（监管评估）**：词云中"审核"、"评估"等词汇权重最高，
   表明监管评估是文档集的核心议题。

2. **主题1（资金投入）**：词云显示"普惠"、"财政"等词汇突出，
   反映财政支持政策是重要讨论领域。

建议：关注这两个主题的交叉内容，可能揭示政策执行与资金配置的关联。
```

---

## 输出文件格式规范

### metrics.json
```json
{
  "job_id": "job_20250106_001",
  "num_documents": 397,
  "vocab_size": 10000,
  "num_topics": 19,
  "topic_quality": {
    "coherence_npmi": 0.356,
    "diversity": 0.84
  },
  "model_performance": {
    "perplexity": 108.5,
    "reconstruction_loss": 1284.5
  },
  "training_info": {
    "total_epochs": 100,
    "training_time_seconds": 342
  }
}
```

### analysis_result.json（前端数据源）
```json
{
  "job_id": "job_20250106_001",
  "status": "success",
  "completed_at": "2025-01-06 14:38:35",
  "duration_seconds": 520,
  "metrics": {
    "coherence_score": 0.356,
    "diversity_score": 0.84,
    "optimal_k": 19
  },
  "topics": [
    {
      "id": 0,
      "name": "监管评估",
      "keywords": ["审核", "评估", "备案", "标准", "机构"],
      "proportion": 0.217,
      "wordcloud_url": "/api/download/job_001/wordcloud_topic_0.png"
    }
  ],
  "charts": {
    "topic_distribution": "/api/download/job_001/topic_distribution.png",
    "heatmap": "/api/download/job_001/heatmap_doc_topic.png",
    "coherence_curve": "/api/download/job_001/coherence_curve.png",
    "topic_similarity": "/api/download/job_001/topic_similarity.png"
  },
  "downloads": {
    "report": "/api/download/job_001/report.docx",
    "theta_csv": "/api/download/job_001/theta.csv",
    "beta_csv": "/api/download/job_001/beta.csv"
  }
}
```

---

## 后端API接口规范

### 1. 上传文件
```
POST /api/data/upload
Request: FormData { file }
Response: {
  "file_id": "uuid",
  "job_id": "job_20250106_001",
  "filename": "data.csv",
  "rows": 397,
  "columns": ["ID", "内容", "时间", "地区"],
  "preview": [[...], [...]]
}
```

### 2. 提交分析任务
```
POST /api/analysis/start
Request: {
  "job_id": "job_20250106_001",
  "text_col": "内容",
  "time_col": "时间",
  "num_topics": 0
}
Response: {
  "job_id": "job_20250106_001",
  "status": "queued",
  "queue_position": 3
}
```

### 3. 查询任务状态
```
GET /api/analysis/status/{job_id}
Response: {
  "job_id": "job_20250106_001",
  "status": "running",
  "queue_position": 0,
  "progress": {
    "current_stage": 3,
    "total_stages": 5,
    "percentage": 60
  },
  "estimated_completion": "2025-01-06 14:45:00"
}
```

### 4. 获取分析结果
```
GET /api/results/{job_id}
Response: { /* analysis_result.json 内容 */ }
```

### 5. 下载文件
```
GET /api/download/{job_id}/report.docx
GET /api/download/{job_id}/theta.csv
GET /api/download/{job_id}/wordcloud_topic_0.png
```

### 6. 交互式问答
```
POST /api/qa/{job_id}
Request: {
  "question": "这些主题分别讨论了什么内容？"
}
Response: {
  "status": "success",
  "job_id": "job_20250106_001",
  "question": "这些主题分别讨论了什么内容？",
  "answer": "根据主题关键词分析..."
}
```

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
# .env 文件
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
QWEN_MODEL=qwen-flash
```

### 3. 运行完整分析
```bash
python run_full_pipeline.py your_data.csv --job_id my_analysis --interactive
```

### 4. 使用Python API
```python
from app.agent_integration import AgentIntegration

# 初始化
integration = AgentIntegration(base_dir=".")

# 运行完整分析
result = integration.run_full_analysis("my_job_id")

# 交互式问答
answer = integration.handle_query("my_job_id", "主题0讨论了什么内容？")
print(answer["answer"])
```

---

## 处理流程

### Engine A - 数据预处理
- **职责**：将CSV清洗并转换为BOW格式
- **输入**：`data/{job_id}/data.csv`
- **输出**：`ETM/outputs/vocab/{job_id}_vocab.json`, `ETM/outputs/bow/{job_id}_bow.npz`
- **参数**：`max_vocab_size=10000`, `min_word_freq=2`, `remove_stopwords=true`

### Engine B - 语义向量化
- **职责**：调用Qwen3模型生成稠密向量
- **模式**：Zero-shot Only
- **输入**：`data/{job_id}/data.csv`, 词汇表
- **输出**：`embedding/outputs/zero_shot/{job_id}_embeddings.npy` (N×1024), `{job_id}_vocab_emb.npy` (V×1024)

### Engine C - 主题建模
- **职责**：结合BOW和Embedding进行ETM训练
- **输入**：BOW矩阵, 文档嵌入, 词汇嵌入
- **输出**：`theta.npy` (N×K), `beta.npy` (K×V), `alpha.npy` (K×1024), `topics.json`
- **参数**：`num_topics=0`(自动推荐), `epochs=100`

### 主题数自动推荐算法
```python
if num_topics == 0:
    num_topics = max(2, min(50, int(np.sqrt(num_docs))))
```

---

## 技术栈

- **Agent框架**：LangChain
- **主题模型**：ETM (Embedded Topic Model)
- **嵌入模型**：Qwen3-Embedding-0.6B
- **LLM**：Qwen-Flash (通义千问)
- **可视化**：matplotlib, seaborn, wordcloud
- **报告生成**：python-docx
- **Web框架**：FastAPI

---

## 环境要求

- Python 3.10+
- CUDA 11.8+ (GPU加速)
- 显存：~10GB (Qwen Embedding)

---

## License

MIT License

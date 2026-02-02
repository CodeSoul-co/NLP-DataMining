# THETA API Documentation

## Overview

THETA is a LangGraph-based Agent System for ETM (Embedded Topic Model) pipeline management.

**Base URL**: `http://localhost:8000`

---

## LangGraph Agent Architecture

### Pipeline Flow (DAG)

```
┌─────────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐    ┌──────────────┐
│ preprocess  │───▶│ embedding │───▶│ training │───▶│ evaluation │───▶│visualization │───▶ END
│             │    │           │    │          │    │            │    │              │
│ BOW+Vocab   │    │ Load Emb  │    │ ETM Model│    │ Metrics    │    │ Plots+HTML   │
└─────────────┘    └───────────┘    └──────────┘    └────────────┘    └──────────────┘
       │                 │                │                │                 │
       ▼                 ▼                ▼                ▼                 ▼
    [error]          [error]          [error]          [error]           [error]
       │                 │                │                │                 │
       └─────────────────┴────────────────┴────────────────┴─────────────────┘
                                          │
                                          ▼
                                        END
```

### AgentState (Core State Object)

The state object flows through all nodes. Key fields:

```typescript
interface AgentState {
  // Task identification
  task_id: string;
  dataset: string;           // e.g., "socialTwitter"
  mode: string;              // "zero_shot" | "supervised" | "unsupervised"
  
  // Paths (auto-generated)
  data_path: string;
  result_dir: string;
  bow_dir: string;
  model_dir: string;
  evaluation_dir: string;
  visualization_dir: string;
  
  // Hyperparameters
  num_topics: number;        // default: 20
  vocab_size: number;        // default: 5000
  epochs: number;            // default: 50
  batch_size: number;        // default: 64
  learning_rate: number;     // default: 0.002
  hidden_dim: number;        // default: 512
  
  // Execution state
  current_step: string;      // "preprocess" | "embedding" | "training" | "evaluation" | "visualization"
  status: string;            // "pending" | "running" | "completed" | "failed" | "cancelled"
  error_message?: string;
  
  // Step completion flags
  preprocess_completed: boolean;
  embedding_completed: boolean;
  training_completed: boolean;
  evaluation_completed: boolean;
  visualization_completed: boolean;
  
  // Results
  metrics?: Record<string, number>;
  topic_words?: Record<string, string[]>;
  visualization_paths?: string[];
  
  // Timestamps
  created_at: string;
  updated_at: string;
  completed_at?: string;
  
  // Execution log
  logs: Array<{step: string, status: string, message: string, timestamp: string}>;
}
```

---

## REST API Endpoints

### Health & Status

#### GET /api/health
Check system health and GPU status.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 1,
  "gpu_id": 1,
  "etm_dir_exists": true,
  "data_dir_exists": true,
  "result_dir_exists": true
}
```

#### GET /api/project
Get project overview.

**Response:**
```json
{
  "name": "THETA",
  "version": "1.0.0",
  "datasets_count": 5,
  "results_count": 3,
  "active_tasks": 1,
  "gpu_available": true,
  "gpu_id": 1,
  "recent_results": [...]
}
```

---

### Datasets

#### GET /api/datasets
List all available datasets.

**Response:**
```json
[
  {
    "name": "socialTwitter",
    "path": "/root/autodl-tmp/data/socialTwitter",
    "size": 10000,
    "columns": ["cleaned_content", "label"],
    "has_labels": true,
    "language": "english"
  }
]
```

#### GET /api/datasets/{dataset_name}
Get detailed dataset info.

**Parameters:**
- `dataset_name` (path): Dataset name

**Response:** Same as above, single object.

#### GET /api/datasets/{dataset_name}/columns ⭐ NEW
Get column information with auto-detected column types. Used for column mapping UI.

**Response:**
```json
{
  "all_columns": ["id", "content", "publish_date", "province", "category"],
  "text_columns": [{"name": "content", "dtype": "object", "sample_values": ["..."], "unique_count": 1000}],
  "time_columns": [{"name": "publish_date", "dtype": "object", "sample_values": ["2023-01-15"]}],
  "dimension_columns": [{"name": "province", "dtype": "object", "sample_values": ["北京", "上海"]}],
  "label_columns": [],
  "other_columns": [...],
  "auto_detected": {
    "text_column": "content",
    "time_column": "publish_date",
    "dimension_column": "province"
  }
}
```

**Frontend Usage:** Display auto-detected results, let user confirm or modify column mappings.

---

### Results

#### GET /api/results
List all training results.

**Response:**
```json
[
  {
    "dataset": "socialTwitter",
    "mode": "zero_shot",
    "timestamp": "20240115_123456",
    "path": "/root/autodl-tmp/result/socialTwitter/zero_shot",
    "num_topics": 20,
    "epochs_trained": 50,
    "metrics": {
      "topic_coherence_avg": 0.58,
      "topic_diversity_td": 0.72,
      "topic_diversity_irbo": 0.85
    },
    "has_model": true,
    "has_theta": true,
    "has_beta": true,
    "has_topic_words": true,
    "has_visualizations": true
  }
]
```

#### GET /api/results/{dataset}/{mode}
Get specific result details.

#### GET /api/results/{dataset}/{mode}/metrics
Get detailed evaluation metrics.

**Response:**
```json
{
  "dataset": "socialTwitter",
  "mode": "zero_shot",
  "timestamp": "20240115_123456",
  "topic_coherence_avg": 0.58,
  "topic_coherence_per_topic": [0.55, 0.62, ...],
  "topic_diversity_td": 0.72,
  "topic_diversity_irbo": 0.85,
  "additional": {...}
}
```

#### GET /api/results/{dataset}/{mode}/topic-words
Get topic words.

**Query Parameters:**
- `top_k` (int, default=10): Number of words per topic

**Response:**
```json
{
  "0": ["word1", "word2", "word3", ...],
  "1": ["word1", "word2", "word3", ...],
  ...
}
```

#### GET /api/results/{dataset}/{mode}/visualizations
List visualization files.

**Response:**
```json
[
  {
    "name": "topic_words_socialTwitter_zero_shot.png",
    "path": "/root/autodl-tmp/result/.../visualization/...",
    "type": "image",
    "size": 123456
  },
  {
    "name": "pyldavis_socialTwitter_zero_shot.html",
    "path": "...",
    "type": "html",
    "size": 234567
  }
]
```

#### GET /api/results/{dataset}/{mode}/visualizations/{filename}
Download/view a visualization file.

#### GET /api/results/{dataset}/{mode}/topic-overview ⭐ NEW
Get structured topic overview data for frontend rendering (Tab 1).

**Query Parameters:**
- `top_k` (int, default=20): Number of topics to return

**Response:**
```json
{
  "topics": [
    {
      "id": 0,
      "name": "Topic 0",
      "keywords": [
        {"word": "政策", "weight": 0.08},
        {"word": "发展", "weight": 0.06}
      ],
      "proportion": 0.12,
      "document_count": 1500,
      "wordcloud_path": "topic_words_xxx_topic0.png"
    }
  ],
  "total_documents": 10000,
  "num_topics": 20
}
```

#### GET /api/results/{dataset}/{mode}/temporal-analysis ⭐ NEW
Get temporal/time-series analysis data (Tab 2/3).

**Query Parameters:**
- `freq` (string, default="Y"): Aggregation frequency - "Y" (year), "M" (month), "Q" (quarter)
- `top_k` (int, default=10): Number of top topics to include

**Success Response:**
```json
{
  "available": true,
  "document_volume": {
    "labels": ["2019", "2020", "2021", "2022", "2023"],
    "values": [1200, 1500, 1800, 2000, 2200]
  },
  "topic_evolution": {
    "labels": ["2019", "2020", "2021", "2022", "2023"],
    "series": [
      {"topic_id": 0, "name": "科技创新", "values": [0.12, 0.15, 0.18, 0.20, 0.22]},
      {"topic_id": 1, "name": "金融服务", "values": [0.10, 0.11, 0.12, 0.11, 0.10]}
    ]
  },
  "topic_trends": {
    "rising": [{"topic_id": 0, "name": "科技创新", "change": 0.10}],
    "falling": [{"topic_id": 5, "name": "传统产业", "change": -0.05}]
  }
}
```

**Unavailable Response (when data lacks timestamp column):**
```json
{
  "available": false,
  "message": "📊 时序分析功能",
  "reason": "当前数据集未包含时间列。",
  "suggestions": [
    "在原始数据中添加时间列（如 publish_date）",
    "重新上传数据并在"列映射"中选择时间列"
  ],
  "available_columns": ["id", "content", "label"],
  "actions": [
    {"label": "跳过", "action": "skip"},
    {"label": "去添加", "action": "add_column"}
  ]
}
```

**Frontend Handling:** Check `available` field. If `false`, display friendly message with suggestions.

#### GET /api/results/{dataset}/{mode}/dimension-analysis ⭐ NEW
Get dimension/spatial analysis data (Tab 4).

**Query Parameters:**
- `top_k_topics` (int, default=10): Number of topics
- `top_k_dimensions` (int, default=20): Number of dimension values

**Success Response:**
```json
{
  "available": true,
  "heatmap_data": {
    "x_labels": ["Topic 0", "Topic 1", "Topic 2"],
    "y_labels": ["北京", "上海", "广东", "浙江"],
    "values": [
      [0.15, 0.10, 0.08],
      [0.12, 0.18, 0.05],
      [0.10, 0.12, 0.15],
      [0.08, 0.09, 0.20]
    ]
  },
  "dimension_stats": {
    "total_dimensions": 31,
    "top_dimensions": [
      {"name": "北京", "count": 1500, "dominant_topic": 0},
      {"name": "上海", "count": 1200, "dominant_topic": 1}
    ]
  }
}
```

**Unavailable Response:** Same structure as temporal-analysis with `available: false`.

---

### Tasks

#### POST /api/tasks
Create and start a new training task.

**Request Body:**
```json
{
  "dataset": "socialTwitter",
  "mode": "zero_shot",
  "num_topics": 20,
  "vocab_size": 5000,
  "epochs": 50,
  "batch_size": 64,
  "learning_rate": 0.002,
  "hidden_dim": 512,
  "dev_mode": false
}
```

**Response:**
```json
{
  "task_id": "task_20240115_123456_abc12345",
  "status": "pending",
  "current_step": "preprocess",
  "progress": 0,
  "created_at": "2024-01-15T12:34:56",
  "updated_at": "2024-01-15T12:34:56"
}
```

#### GET /api/tasks
List all tasks.

#### GET /api/tasks/{task_id}
Get task status and results.

**Response:**
```json
{
  "task_id": "task_20240115_123456_abc12345",
  "status": "completed",
  "current_step": "visualization",
  "progress": 100,
  "metrics": {
    "topic_coherence_avg": 0.58,
    "topic_diversity_td": 0.72
  },
  "topic_words": {...},
  "visualization_paths": [...],
  "created_at": "2024-01-15T12:34:56",
  "updated_at": "2024-01-15T12:45:00",
  "completed_at": "2024-01-15T12:45:00"
}
```

#### DELETE /api/tasks/{task_id}
Cancel a running task.

---

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws');

// Or for specific task:
const ws = new WebSocket('ws://localhost:8000/api/ws/task/{task_id}');
```

### Subscribe to Task Updates

```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  task_id: 'task_20240115_123456_abc12345'
}));
```

### Receive Updates

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  // data structure:
  // {
  //   type: 'step_update' | 'task_complete' | 'error',
  //   task_id: string,
  //   step: string,
  //   status: string,
  //   message: string,
  //   progress?: number,
  //   metrics?: object,
  //   timestamp: string
  // }
};
```

### Message Types

| Type | Description |
|------|-------------|
| `step_update` | Progress update for current step |
| `task_complete` | Task finished successfully |
| `error` | Error occurred |
| `connected` | Connection established |

---

## File Structure

```
/root/autodl-tmp/
├── data/                          # Datasets
│   └── {dataset}/
│       └── {dataset}_text_only.csv
├── result/                        # Results
│   └── {dataset}/
│       └── {mode}/
│           ├── bow/
│           │   ├── bow_matrix.npz
│           │   ├── vocab.txt
│           │   └── vocab_embeddings.npy
│           ├── embeddings/
│           │   └── {dataset}_{mode}_embeddings.npy  # REQUIRED before training
│           ├── model/
│           │   ├── etm_model_{timestamp}.pt
│           │   ├── theta_{timestamp}.npy
│           │   ├── beta_{timestamp}.npy
│           │   ├── topic_words_{timestamp}.json
│           │   └── training_history_{timestamp}.json
│           ├── evaluation/
│           │   └── metrics_{timestamp}.json
│           └── visualization/
│               ├── topic_words_{prefix}.png
│               ├── topic_similarity_{prefix}.png
│               ├── doc_topics_{prefix}.png
│               ├── topic_proportions_{prefix}.png
│               └── pyldavis_{prefix}.html
└── langgraph_agent/               # This project
    └── backend/
        └── app/
```

---

## Prerequisites for Training

Before starting a training task, ensure:

1. **Dataset exists**: `/root/autodl-tmp/data/{dataset}/{dataset}_text_only.csv`
2. **Embeddings exist**: `/root/autodl-tmp/result/{dataset}/{mode}/embeddings/{dataset}_{mode}_embeddings.npy`

If embeddings don't exist, generate them first using the ETM project's embedding pipeline.

---

## Error Handling

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 500 | Internal server error |

Error response format:
```json
{
  "detail": "Error message here"
}
```

---

## Configuration

Key settings in `backend/app/core/config.py`:

| Setting | Value | Description |
|---------|-------|-------------|
| `GPU_ID` | 1 | GPU to use (DO NOT use GPU 0) |
| `DATA_DIR` | `/root/autodl-tmp/data` | Datasets location |
| `RESULT_DIR` | `/root/autodl-tmp/result` | Results output |
| `ETM_DIR` | `/root/autodl-tmp/ETM` | ETM source code |
| `QWEN_MODEL_PATH` | `/root/autodl-tmp/qwen3_embedding_0.6B` | Embedding model |

---

## Usage Example (Frontend Integration)

```javascript
// 1. Check available datasets
const datasets = await fetch('/api/datasets').then(r => r.json());

// 2. Start training
const task = await fetch('/api/tasks', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    dataset: 'socialTwitter',
    mode: 'zero_shot',
    num_topics: 20
  })
}).then(r => r.json());

// 3. Subscribe to updates via WebSocket
const ws = new WebSocket(`ws://localhost:8000/api/ws/task/${task.task_id}`);
ws.onmessage = (e) => {
  const update = JSON.parse(e.data);
  console.log(`[${update.step}] ${update.message} (${update.progress}%)`);
};

// 4. Poll for final results
const result = await fetch(`/api/tasks/${task.task_id}`).then(r => r.json());

// 5. Get topic words
const topics = await fetch(`/api/results/${dataset}/${mode}/topic-words`).then(r => r.json());

// 6. Get visualizations
const vizList = await fetch(`/api/results/${dataset}/${mode}/visualizations`).then(r => r.json());
```

# THETA API 接口规范文档

本文档定义了THETA主题模型系统的API接口规范，用于前后端对接。

## 基本信息

- 基础URL: `http://<server_ip>:5000/api`
- 认证方式: Bearer Token
- 响应格式: JSON
- 编码: UTF-8

## 状态码

| 状态码 | 描述 |
|--------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 404 | 资源不存在 |
| 409 | 资源冲突 |
| 500 | 服务器内部错误 |

## 错误响应格式

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述信息"
  }
}
```

## API端点

### 1. 文件上传

#### 请求

```
POST /data/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| file | File | 是 | CSV文件 |
| dataset | String | 否 | 数据集名称，默认为"default" |

#### 响应

```json
{
  "file_id": "uuid",
  "job_id": "job_20260128_001",
  "filename": "data.csv",
  "rows": 397,
  "columns": ["ID", "内容", "时间", "地区"],
  "preview": [
    ["1", "这是第一条内容", "2026-01-01", "北京"],
    ["2", "这是第二条内容", "2026-01-02", "上海"]
  ]
}
```

### 2. 提交分析任务

#### 请求

```
POST /analysis/start
Content-Type: application/json
Authorization: Bearer <token>
```

```json
{
  "job_id": "job_20260128_001",
  "text_col": "内容",
  "time_col": "时间",
  "num_topics": 0
}
```

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| job_id | String | 是 | 任务ID（从上传接口获取） |
| text_col | String | 是 | 文本列名 |
| time_col | String | 否 | 时间列名（用于时间序列分析） |
| num_topics | Integer | 否 | 主题数量，0表示自动确定 |

#### 响应

```json
{
  "job_id": "job_20260128_001",
  "status": "queued",
  "queue_position": 3,
  "estimated_wait_time": 900
}
```

### 3. 查询任务状态

#### 请求

```
GET /analysis/status/{job_id}
Authorization: Bearer <token>
```

#### 响应

```json
{
  "job_id": "job_20260128_001",
  "status": "running",
  "queue_position": 0,
  "progress": {
    "current_stage": 3,
    "total_stages": 5,
    "percentage": 60,
    "stage_name": "ETM模型训练"
  },
  "estimated_completion": "2026-01-28 14:45:00"
}
```

| 状态值 | 描述 |
|--------|------|
| queued | 排队中 |
| running | 运行中 |
| completed | 已完成 |
| failed | 失败 |

### 4. 获取分析结果

#### 请求

```
GET /results/{job_id}
Authorization: Bearer <token>
```

#### 响应

```json
{
  "job_id": "job_20260128_001",
  "status": "success",
  "completed_at": "2026-01-28 14:38:35",
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
      "wordcloud_url": "/api/download/job_20260128_001/wordcloud_topic_0.png"
    },
    {
      "id": 1,
      "name": "资金投入",
      "keywords": ["普惠", "补助", "财政", "扶持", "投入"],
      "proportion": 0.183,
      "wordcloud_url": "/api/download/job_20260128_001/wordcloud_topic_1.png"
    }
  ],
  
  "charts": {
    "topic_distribution": "/api/download/job_20260128_001/topic_distribution.png",
    "heatmap": "/api/download/job_20260128_001/heatmap_doc_topic.png",
    "coherence_curve": "/api/download/job_20260128_001/coherence_curve.png",
    "topic_similarity": "/api/download/job_20260128_001/topic_similarity.png"
  },
  
  "downloads": {
    "report": "/api/download/job_20260128_001/report.docx",
    "theta_csv": "/api/download/job_20260128_001/theta.csv",
    "beta_csv": "/api/download/job_20260128_001/beta.csv"
  }
}
```

### 5. 下载文件

#### 请求

```
GET /download/{job_id}/{filename}
Authorization: Bearer <token>
```

#### 响应

文件内容（根据文件类型设置相应的Content-Type）

### 6. 获取任务列表

#### 请求

```
GET /jobs
Authorization: Bearer <token>
```

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| status | String | 否 | 筛选状态 (queued, running, completed, failed) |
| page | Integer | 否 | 页码，默认1 |
| per_page | Integer | 否 | 每页数量，默认20 |

#### 响应

```json
{
  "total": 42,
  "page": 1,
  "per_page": 20,
  "jobs": [
    {
      "job_id": "job_20260128_001",
      "dataset": "hatespeech",
      "status": "completed",
      "submitted_at": "2026-01-28 12:30:45",
      "completed_at": "2026-01-28 14:38:35"
    },
    {
      "job_id": "job_20260127_005",
      "dataset": "news",
      "status": "running",
      "submitted_at": "2026-01-27 18:20:15",
      "completed_at": null
    }
  ]
}
```

## 数据结构

### Job（任务）

```json
{
  "job_id": "job_20260128_001",
  "dataset": "hatespeech",
  "user_id": "user_001",
  "status": "running",
  "priority": 0,
  "submitted_at": "2026-01-28 12:30:45",
  "started_at": "2026-01-28 13:15:20",
  "completed_at": null,
  "params": {
    "text_col": "内容",
    "time_col": "发布时间",
    "num_topics": 0
  },
  "gpu_id": 0
}
```

### Topic（主题）

```json
{
  "id": 0,
  "name": "监管评估",
  "keywords": ["审核", "评估", "备案", "标准", "机构"],
  "proportion": 0.217,
  "coherence": 0.42,
  "wordcloud_url": "/api/download/job_20260128_001/wordcloud_topic_0.png",
  "top_docs": [
    {
      "doc_id": 156,
      "score": 0.89,
      "text": "关于加强金融机构监管评估的通知..."
    }
  ]
}
```

## 实现注意事项

1. **文件路径约定**：
   - 用户上传的原始数据：`/root/autodl-tmp/data/job_{job_id}/data.csv`
   - 处理结果：`/root/autodl-tmp/result/job_{job_id}/`
   - 可视化图表：`/root/autodl-tmp/result/job_{job_id}/visualization/`

2. **任务状态管理**：
   - 任务状态文件：`/root/autodl-tmp/job_status/{job_id}.json`
   - 任务队列文件：`/root/autodl-tmp/job_queue.json`

3. **错误处理**：
   - 所有API端点应返回统一格式的错误响应
   - 任务处理过程中的错误应记录到日志文件

4. **安全性考虑**：
   - 实现适当的认证和授权机制
   - 限制上传文件大小和类型
   - 防止路径遍历攻击

## 版本控制

为了支持平滑迁移，API接口应实现版本控制：

- V1版本：`/api/v1/...`（原有格式）
- V2版本：`/api/v2/...`（新格式，支持job_id）

## 测试环境

测试服务器信息：
- URL: `http://test-server:5000/api`
- 测试账号: `test@example.com` / `test123`

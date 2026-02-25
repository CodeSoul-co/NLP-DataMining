# THETA 前端对接文档

## 概述

本文档说明前端如何与 THETA 主题模型训练系统对接，包括：
- 用户认证
- 数据上传
- 训练任务提交
- 结果下载

---

## 1. 测试账号

| 字段 | 值 |
|------|-----|
| 用户名 | `admin` |
| 密码 | `admin123` |

---

## 2. API 基础信息

### 2.1 API 地址

```
开发环境: http://localhost:8000
生产环境: https://api.theta.example.com (待部署)
```

### 2.2 认证方式

使用 Bearer Token 认证：

```
Authorization: Bearer {access_token}
```

---

## 3. API 端点

### 3.1 认证相关

#### 登录

```http
POST /api/auth/login
Content-Type: application/json

{
    "username": "admin",
    "password": "admin123"
}
```

**响应：**
```json
{
    "access_token": "xxxxxx",
    "token_type": "bearer",
    "expires_in": 86400,
    "user": {
        "username": "admin",
        "role": "admin",
        "created_at": "2026-02-18"
    }
}
```

#### 获取当前用户

```http
GET /api/auth/me
Authorization: Bearer {access_token}
```

#### 登出

```http
POST /api/auth/logout
Authorization: Bearer {access_token}
```

---

### 3.2 数据上传流程

#### 步骤 1：获取签名上传 URL

```http
POST /api/data/presigned-url
Content-Type: application/json
Authorization: Bearer {access_token}

{
    "filename": "my_data.csv",
    "content_type": "text/csv"
}
```

**响应：**
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "upload_url": "https://theta-prod-20260123.oss-cn-shanghai.aliyuncs.com/data/550e8400.../my_data.csv?Signature=...",
    "oss_path": "data/550e8400-e29b-41d4-a716-446655440000/my_data.csv",
    "expires_in": 3600
}
```

#### 步骤 2：前端直传 OSS

使用返回的 `upload_url` 直接上传文件到 OSS：

```javascript
// JavaScript 示例
const uploadFile = async (file, uploadUrl) => {
    const response = await fetch(uploadUrl, {
        method: 'PUT',
        body: file,
        headers: {
            'Content-Type': 'text/csv'
        }
    });
    return response.ok;
};
```

#### 步骤 3：通知上传完成并启动训练

```http
POST /api/data/upload-complete
Content-Type: application/json
Authorization: Bearer {access_token}

{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "dataset_name": "my_dataset",
    "num_topics": 20,
    "epochs": 100,
    "mode": "zero_shot",
    "model_size": "0.6B",
    "models": "theta,lda,etm"
}
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| job_id | string | 是 | 步骤1返回的 job_id |
| dataset_name | string | 否 | 数据集名称，默认使用文件名 |
| num_topics | int | 否 | 主题数量，默认 20 |
| epochs | int | 否 | 训练轮数，默认 100 |
| mode | string | 否 | 训练模式：zero_shot/supervised/unsupervised |
| model_size | string | 否 | THETA 模型大小：0.6B/4B/8B |
| models | string | 否 | 要训练的模型，逗号分隔 |

**支持的模型：**
- `theta` - THETA 主题模型（主模型）
- `lda` - 传统 LDA
- `hdp` - 层次狄利克雷过程
- `btm` - 双词主题模型
- `etm` - 嵌入主题模型
- `ctm` - 组合主题模型
- `nvdm` - 神经变分文档模型
- `gsm` - 高斯 Softmax 模型
- `prodlda` - 产品 LDA
- `bertopic` - BERTopic

**响应：**
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "training",
    "message": "DLC 训练任务已提交",
    "dlc_job_id": "dlc-xxxxxxxx"
}
```

---

### 3.3 任务状态查询

```http
GET /api/data/jobs/{job_id}/status
Authorization: Bearer {access_token}
```

**响应：**
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "training",
    "dlc_job_id": "dlc-xxxxxxxx",
    "dlc_status": "Running",
    "created_at": "2026-02-18T16:00:00",
    "completed_at": null,
    "error": null
}
```

**状态说明：**

| status | 说明 |
|--------|------|
| pending_upload | 等待上传 |
| submitting_dlc | 正在提交训练任务 |
| training | 训练中 |
| completed | 训练完成 |
| error | 训练失败 |

---

### 3.4 获取训练结果

```http
GET /api/data/jobs/{job_id}/results
Authorization: Bearer {access_token}
```

**响应：**
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "result_base": "result/550e8400-e29b-41d4-a716-446655440000",
    "files": {
        "metrics_k20.json": "https://oss.../metrics_k20.json?Signature=...",
        "topic_words_k20.json": "https://oss.../topic_words_k20.json?Signature=...",
        "visualization/topic_wordcloud.png": "https://oss.../topic_wordcloud.png?Signature=..."
    }
}
```

---

### 3.5 列出所有任务

```http
GET /api/data/jobs?limit=20
Authorization: Bearer {access_token}
```

---

## 4. 文件路径说明

### 4.1 OSS Bucket 结构

```
oss://theta-prod-20260123/
├── data/                          # 用户上传的数据
│   └── {job_id}/                  # 按 job_id 隔离
│       └── {filename}.csv         # 原始数据文件
│
├── result/                        # 训练结果
│   └── {job_id}/                  # 按 job_id 隔离
│       ├── theta/                 # THETA 模型结果
│       │   └── exp_xxx/
│       │       ├── metrics_k20.json
│       │       ├── theta_k20.npy
│       │       ├── beta_k20.npy
│       │       └── visualization/
│       ├── lda/                   # LDA 模型结果
│       ├── etm/                   # ETM 模型结果
│       └── ...
│
├── code/                          # 训练代码（只读）
│   └── ETM/
│
├── embedding_models/              # 预训练嵌入模型（只读）
│   └── sbert/
│
└── sbert/                         # SBERT 模型（只读）
```

### 4.2 数据文件格式

上传的 CSV 文件需要包含以下列：

| 列名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | 是 | 文档文本内容 |
| label | string/int | 否 | 文档标签（监督模式需要） |
| timestamp | datetime | 否 | 时间戳（DTM 模型需要） |

**示例：**
```csv
text,label
"这是第一篇文档的内容",教育
"这是第二篇文档的内容",科技
```

---

## 5. 完整前端调用示例

### JavaScript/TypeScript

```typescript
// api.ts
const API_BASE = 'http://localhost:8000';

interface LoginResponse {
    access_token: string;
    token_type: string;
    expires_in: number;
    user: {
        username: string;
        role: string;
    };
}

// 登录
async function login(username: string, password: string): Promise<LoginResponse> {
    const response = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    });
    return response.json();
}

// 上传数据并训练
async function uploadAndTrain(
    file: File,
    token: string,
    options: {
        numTopics?: number;
        epochs?: number;
        models?: string;
    } = {}
) {
    // 1. 获取签名 URL
    const presignedRes = await fetch(`${API_BASE}/api/data/presigned-url`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
            filename: file.name,
            content_type: 'text/csv'
        })
    });
    const { job_id, upload_url } = await presignedRes.json();

    // 2. 直传 OSS
    await fetch(upload_url, {
        method: 'PUT',
        body: file,
        headers: { 'Content-Type': 'text/csv' }
    });

    // 3. 通知完成并启动训练
    const completeRes = await fetch(`${API_BASE}/api/data/upload-complete`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
            job_id,
            num_topics: options.numTopics || 20,
            epochs: options.epochs || 100,
            models: options.models || 'theta,lda'
        })
    });

    return completeRes.json();
}

// 查询任务状态
async function getJobStatus(jobId: string, token: string) {
    const response = await fetch(`${API_BASE}/api/data/jobs/${jobId}/status`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    return response.json();
}

// 获取结果
async function getJobResults(jobId: string, token: string) {
    const response = await fetch(`${API_BASE}/api/data/jobs/${jobId}/results`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    return response.json();
}

// 使用示例
async function main() {
    // 登录
    const { access_token } = await login('admin', 'admin123');
    
    // 上传文件并训练
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = fileInput.files![0];
    
    const { job_id } = await uploadAndTrain(file, access_token, {
        numTopics: 20,
        models: 'theta,lda,etm'
    });
    
    // 轮询状态
    const checkStatus = async () => {
        const status = await getJobStatus(job_id, access_token);
        console.log('Status:', status);
        
        if (status.status === 'completed') {
            const results = await getJobResults(job_id, access_token);
            console.log('Results:', results);
        } else if (status.status === 'error') {
            console.error('Training failed:', status.error);
        } else {
            setTimeout(checkStatus, 10000); // 10秒后再查
        }
    };
    
    checkStatus();
}
```

---

## 6. 数据预处理流程

系统会自动执行以下预处理步骤：

1. **数据清洗** - 去除空行、重复行、特殊字符
2. **分词** - 中文使用 jieba，英文使用 NLTK
3. **停用词过滤** - 移除常见停用词
4. **词频过滤** - 移除低频词和高频词
5. **构建词汇表** - 默认 5000 词
6. **生成 BOW 矩阵** - 词袋表示
7. **生成嵌入** - SBERT 文档嵌入、Word2Vec 词嵌入

预处理结果保存在：
```
result/{job_id}/baseline/{dataset}/data/exp_xxx/
├── bow_matrix.npy        # BOW 矩阵
├── vocab.txt             # 词汇表
├── sbert_embeddings.npy  # SBERT 嵌入
├── word2vec.npy          # Word2Vec 嵌入
└── data_config.json      # 数据配置
```

---

## 7. 错误处理

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 未认证或 Token 无效 |
| 403 | 权限不足 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

### 错误响应格式

```json
{
    "detail": "错误描述信息"
}
```

---

## 8. 环境变量配置

后端需要配置以下环境变量：

```bash
# 阿里云凭证
ALIBABA_CLOUD_ACCESS_KEY_ID=your_access_key_id
ALIBABA_CLOUD_ACCESS_KEY_SECRET=your_access_key_secret

# OSS 配置
OSS_ENDPOINT=oss-cn-shanghai.aliyuncs.com
OSS_BUCKET=theta-prod-20260123

# DLC 配置
DLC_WORKSPACE_ID=464377
OSS_DATASET_ID=d-cvx2t6q7t8w3bnrvgl

# API 配置
API_HOST=0.0.0.0
API_PORT=8000
TOKEN_EXPIRE_HOURS=24
```

---

## 9. 快速启动

### 本地开发

```bash
cd api
pip install -r requirements.txt
python main.py
```

API 文档访问：http://localhost:8000/docs

### Docker 部署

```bash
docker build -t theta-api .
docker run -p 8000:8000 \
    -e ALIBABA_CLOUD_ACCESS_KEY_ID=xxx \
    -e ALIBABA_CLOUD_ACCESS_KEY_SECRET=xxx \
    theta-api
```

---

## 10. 联系方式

如有问题，请联系后端开发团队。

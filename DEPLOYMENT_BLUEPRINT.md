# THETA 部署蓝图

> 本文档基于阿里云香港部署方案

## 架构图

```mermaid
graph TD
    User((用户)) --> |HTTPS:443| Nginx[Nginx 反向代理]
    
    subgraph "阿里云香港 ECS/轻量服务器 (2核4G)"
        Nginx --> |/| Frontend[Next.js 容器 (3000)]
        Nginx --> |/api| Backend[FastAPI 容器 (8000)]
        Backend --> |读写| DB[(PostgreSQL 容器)]
        Backend --> |缓存| Redis[(Redis 容器)]
    end
    
    subgraph "阿里云云原生产品 (按量付费)"
        Backend --> |存取文件| OSS[对象存储 OSS]
        Backend --> |1. 提交训练指令| DLC[PAI-DLC (训练集群)]
        Backend --> |2. 调用推理API| EAS[PAI-EAS (推理服务)]
        DLC & EAS -- 拉取镜像 --> ACR[容器镜像服务 ACR]
        DLC & EAS -- 读写数据 --> OSS
    end
```

---

## 架构说明

### 一、Web 服务层（阿里云香港 ECS/轻量服务器 2核4G）

| 组件 | 端口 | 说明 |
|------|------|------|
| **Nginx** | 443 (HTTPS) | 反向代理，SSL 终止，路由分发 |
| **Next.js 前端** | 3000 | React/Next.js 容器，处理 `/` 路由 |
| **FastAPI 后端** | 8000 | Python 后端容器，处理 `/api` 路由 |
| **PostgreSQL** | 5432 | 持久化数据库（用户、任务、结果元数据） |
| **Redis** | 6379 | 缓存、会话、任务队列 |

### 二、阿里云云原生产品（按量付费）

| 产品 | 用途 |
|------|------|
| **OSS (对象存储)** | 存储上传文件、数据集、模型权重、训练结果 |
| **PAI-DLC (训练集群)** | GPU 训练任务：ETM/DETM 模型训练、LoRA 微调 |
| **PAI-EAS (推理服务)** | 模型推理 API：嵌入生成、主题推断 |
| **ACR (容器镜像服务)** | 存储训练/推理 Docker 镜像 |

---

## 数据流

### 1. 用户访问流程
```
用户 → HTTPS:443 → Nginx → / → Next.js (3000)
                         → /api → FastAPI (8000)
```

### 2. 训练任务流程
```
前端 → POST /api/tasks → FastAPI → 
    1. 任务元数据 → PostgreSQL
    2. 数据文件 → OSS
    3. 提交训练 → PAI-DLC
       → 拉取镜像 ← ACR
       → 读取数据 ← OSS
       → 写入结果 → OSS
    4. 更新状态 → PostgreSQL
```

### 3. 推理流程
```
前端 → POST /api/analyze → FastAPI → 
    1. 调用推理 → PAI-EAS
       → 拉取镜像 ← ACR
       → 加载模型 ← OSS
    2. 返回结果 → 前端
```

---

## 端口映射

| 服务 | 内部端口 | 外部访问 |
|------|----------|----------|
| Nginx | 80/443 | 公网 HTTPS |
| Next.js | 3000 | 通过 Nginx `/` |
| FastAPI | 8000 | 通过 Nginx `/api` |
| PostgreSQL | 5432 | 内部网络 |
| Redis | 6379 | 内部网络 |

---

## 环境变量配置

### 后端 (FastAPI)
```env
# 数据库
DATABASE_URL=postgresql://user:pass@db:5432/theta

# Redis
REDIS_URL=redis://redis:6379/0

# OSS
OSS_ACCESS_KEY_ID=xxx
OSS_ACCESS_KEY_SECRET=xxx
OSS_BUCKET_NAME=theta-data
OSS_ENDPOINT=oss-cn-hongkong.aliyuncs.com

# PAI-DLC
PAI_ACCESS_KEY_ID=xxx
PAI_ACCESS_KEY_SECRET=xxx
PAI_REGION=cn-hongkong
PAI_PROJECT_ID=xxx

# PAI-EAS
EAS_ENDPOINT=xxx.cn-hongkong.pai-eas.aliyuncs.com
EAS_TOKEN=xxx

# CORS
CORS_ORIGINS=https://your-domain.com
```

### 前端 (Next.js)
```env
NEXT_PUBLIC_API_URL=https://your-domain.com
```

---

## Docker Compose 参考

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - frontend
      - backend

  frontend:
    build: ./theta-frontend3
    expose:
      - "3000"
    environment:
      - NEXT_PUBLIC_API_URL=https://your-domain.com

  backend:
    build: ./langgraph_agent/backend
    expose:
      - "8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/theta
      - REDIS_URL=redis://redis:6379/0
      - OSS_ACCESS_KEY_ID=${OSS_ACCESS_KEY_ID}
      - OSS_ACCESS_KEY_SECRET=${OSS_ACCESS_KEY_SECRET}
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=theta
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

---

## 成本估算

| 资源 | 规格 | 预估月费 |
|------|------|----------|
| 香港轻量服务器 | 2核4G | ¥50-100 |
| OSS | 按量 | ¥10-50 |
| PAI-DLC | GPU 按量 | 按训练时长 |
| PAI-EAS | 按调用量 | 按推理次数 |
| ACR | 基础版免费 | ¥0 |

**总计**: 基础 ¥60-150/月 + GPU 训练按需

---

## 扩展说明

1. **PAI-DLC 训练**: 支持多种 GPU 规格，按训练时长计费，适合批量训练任务
2. **PAI-EAS 推理**: 支持弹性伸缩，可按 QPS 自动扩缩容
3. **OSS**: 支持生命周期管理，可自动归档冷数据降低成本
4. **ACR**: 支持镜像安全扫描，建议开启漏洞扫描

---

**最后更新**: 2025-01-23

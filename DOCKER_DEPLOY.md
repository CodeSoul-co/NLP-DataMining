# THETA - Docker 部署指南

本项目使用 **Docker Compose** 进行部署，包含三个服务：

1. **Frontend** - Next.js 前端应用
2. **Backend API** - LangGraph Agent FastAPI 后端
3. **DataClean API** - 数据清洗服务

## 📋 目录结构

```
THETA/
├── docker-compose.prod.yml    # Docker Compose 生产环境配置
├── theta-frontend3/
│   └── Dockerfile              # 前端 Dockerfile
├── langgraph_agent/backend/
│   └── Dockerfile.backend     # 后端 Dockerfile
└── ETM/dataclean/
    └── Dockerfile              # DataClean Dockerfile
```

## 🚀 快速部署

### 前置要求

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- 至少 4GB 可用内存
- 至少 10GB 可用磁盘空间

### 步骤 1: 克隆代码

```bash
git clone -b frontend-3 https://github.com/CodeSoul-co/THETA.git
cd THETA
```

### 步骤 2: 创建 Docker Compose 配置文件

创建 `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./theta-frontend3
      dockerfile: Dockerfile
    ports:
      - "3002:3000"  # 根据实际情况调整端口
    environment:
      - NEXT_PUBLIC_API_URL=http://your-domain.com:8000
      - NEXT_PUBLIC_DATACLEAN_API_URL=http://your-domain.com:8001
    restart: unless-stopped
    networks:
      - theta-network

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - CORS_ORIGINS=http://your-domain.com:3002,http://your-domain.com,https://your-domain.com
      - SIMULATION_MODE=true
      - HOST=0.0.0.0
      - PORT=8000
    volumes:
      - theta-data:/app/data
      - theta-result:/app/result
    restart: unless-stopped
    networks:
      - theta-network

  dataclean:
    build:
      context: ./ETM/dataclean
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - CORS_ORIGINS=http://your-domain.com:3002,http://your-domain.com,https://your-domain.com
      - PORT=8001
    restart: unless-stopped
    networks:
      - theta-network

networks:
  theta-network:
    driver: bridge

volumes:
  theta-data:
  theta-result:
```

**重要**: 将 `your-domain.com` 替换为你的实际域名或 IP 地址。

### 步骤 3: 创建 Dockerfile

#### 前端 Dockerfile (`theta-frontend3/Dockerfile`)

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
RUN npm install -g pnpm
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
ARG NEXT_PUBLIC_API_URL=http://your-domain.com:8000
ARG NEXT_PUBLIC_DATACLEAN_API_URL=http://your-domain.com:8001
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_DATACLEAN_API_URL=$NEXT_PUBLIC_DATACLEAN_API_URL
RUN pnpm build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV HOSTNAME=0.0.0.0
ENV PORT=3000
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

#### 后端 Dockerfile (`Dockerfile.backend`)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY langgraph_agent/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY langgraph_agent/backend/app ./app
COPY ETM ./ETM
ENV PYTHONPATH=/app:/app/ETM
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### DataClean Dockerfile (`ETM/dataclean/Dockerfile`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p temp_uploads temp_processed && \
    chmod 755 temp_uploads temp_processed

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 步骤 4: 构建和启动服务

```bash
# 构建所有服务
docker compose -f docker-compose.prod.yml build

# 启动所有服务
docker compose -f docker-compose.prod.yml up -d

# 查看服务状态
docker compose -f docker-compose.prod.yml ps

# 查看日志
docker compose -f docker-compose.prod.yml logs -f
```

### 步骤 5: 配置防火墙

```bash
# Ubuntu/Debian
ufw allow 3002/tcp
ufw allow 8000/tcp
ufw allow 8001/tcp

# CentOS/RHEL
firewall-cmd --permanent --add-port=3002/tcp
firewall-cmd --permanent --add-port=8000/tcp
firewall-cmd --permanent --add-port=8001/tcp
firewall-cmd --reload
```

## 🌐 访问地址

部署成功后，可以通过以下地址访问：

- **前端界面**: http://your-domain.com:3002
- **后端 API 文档**: http://your-domain.com:8000/docs
- **DataClean API**: http://your-domain.com:8001/health

## 📝 常用命令

### 查看服务状态

```bash
docker compose -f docker-compose.prod.yml ps
```

### 查看日志

```bash
# 查看所有服务日志
docker compose -f docker-compose.prod.yml logs -f

# 查看特定服务日志
docker compose -f docker-compose.prod.yml logs -f frontend
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f dataclean
```

### 重启服务

```bash
# 重启所有服务
docker compose -f docker-compose.prod.yml restart

# 重启特定服务
docker compose -f docker-compose.prod.yml restart frontend
```

### 停止服务

```bash
docker compose -f docker-compose.prod.yml down
```

### 更新代码并重新部署

```bash
# 拉取最新代码
git pull origin frontend-3

# 重新构建并启动
docker compose -f docker-compose.prod.yml up -d --build
```

### 清理资源

```bash
# 停止并删除容器
docker compose -f docker-compose.prod.yml down

# 删除所有相关资源（包括卷）
docker compose -f docker-compose.prod.yml down -v

# 清理未使用的镜像
docker image prune -a
```

## ⚙️ 环境变量配置

### 前端环境变量

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `NEXT_PUBLIC_API_URL` | 后端 API 地址 | `http://your-domain.com:8000` |
| `NEXT_PUBLIC_DATACLEAN_API_URL` | DataClean API 地址 | `http://your-domain.com:8001` |

### 后端环境变量

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `CORS_ORIGINS` | 允许的跨域来源（逗号分隔） | `http://your-domain.com:3002,https://your-domain.com` |
| `SIMULATION_MODE` | 模拟模式（无 GPU 时使用） | `true` |
| `HOST` | 监听地址 | `0.0.0.0` |
| `PORT` | 监听端口 | `8000` |

### DataClean 环境变量

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `CORS_ORIGINS` | 允许的跨域来源（逗号分隔） | `http://your-domain.com:3002,https://your-domain.com` |
| `PORT` | 监听端口 | `8001` |

## 🔧 故障排除

### 端口被占用

如果遇到端口被占用错误：

```bash
# 检查端口占用
lsof -i :3002
lsof -i :8000
lsof -i :8001

# 停止占用端口的进程
fuser -k 3002/tcp
fuser -k 8000/tcp
fuser -k 8001/tcp
```

### 构建失败

```bash
# 清理构建缓存
docker compose -f docker-compose.prod.yml build --no-cache

# 查看详细构建日志
docker compose -f docker-compose.prod.yml build --progress=plain
```

### 容器无法启动

```bash
# 查看容器日志
docker logs theta-frontend-1
docker logs theta-backend-1
docker logs theta-dataclean-1

# 检查容器状态
docker ps -a
```

### 内存不足

如果遇到内存不足问题：

```bash
# 检查系统资源
docker stats

# 限制容器资源使用（在 docker-compose.prod.yml 中添加）
services:
  frontend:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

## 📊 资源要求

### 最低配置

- CPU: 2 核
- 内存: 4GB
- 磁盘: 20GB

### 推荐配置

- CPU: 4 核
- 内存: 8GB
- 磁盘: 50GB

## 🔒 安全建议

1. **使用 HTTPS**: 配置 Nginx 反向代理并启用 SSL 证书
2. **限制访问**: 使用防火墙规则限制特定 IP 访问
3. **定期更新**: 保持 Docker 和镜像更新到最新版本
4. **备份数据**: 定期备份 `theta-data` 和 `theta-result` 卷

## 📚 相关文档

- [README.md](./README.md) - 项目概述
- [README_CN.md](./README_CN.md) - 中文文档
- [RAILWAY_DEPLOY.md](./RAILWAY_DEPLOY.md) - Railway 部署指南（如果存在）

## 💡 提示

- 首次构建可能需要 10-15 分钟，请耐心等待
- 建议在服务器上使用 `screen` 或 `tmux` 来保持会话
- 可以使用 `docker-compose` 替代 `docker compose`（旧版本 Docker）

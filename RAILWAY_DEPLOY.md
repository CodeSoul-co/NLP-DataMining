# THETA - Railway 部署指南

本项目是一个 monorepo，包含三个服务需要部署到 Railway：

1. **Frontend** - Next.js 前端应用 (`theta-frontend3/`)
2. **Backend API** - LangGraph Agent FastAPI 后端 (`langgraph_agent/backend/`)
3. **DataClean API** - 数据清洗服务 (`ETM/dataclean/`)

## 🚀 快速部署

### 方式一：通过 Railway Dashboard（推荐）

#### 步骤 1：创建 Railway 项目

1. 登录 [Railway](https://railway.app/)
2. 点击 "New Project" -> "Deploy from GitHub repo"
3. 选择 `CodeSoul-co/THETA` 仓库

#### 步骤 2：部署前端服务

1. 在项目中点击 "New Service" -> "GitHub Repo"
2. 选择相同仓库
3. 配置服务：
   - **Root Directory**: `theta-frontend3`
   - **Build Command**: `pnpm install && pnpm build`
   - **Start Command**: `pnpm start`
4. 添加环境变量：
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-api.railway.app
   NEXT_PUBLIC_DATACLEAN_API_URL=https://your-dataclean-api.railway.app
   PORT=3000
   ```
5. 生成域名（Settings -> Networking -> Generate Domain）

#### 步骤 3：部署后端 API 服务

1. 在项目中点击 "New Service" -> "GitHub Repo"
2. 选择相同仓库
3. 配置服务：
   - **Root Directory**: `langgraph_agent/backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. 添加环境变量：
   ```
   CORS_ORIGINS=https://your-frontend.railway.app
   SIMULATION_MODE=true
   LOG_LEVEL=INFO
   ```
5. 生成域名

#### 步骤 4：部署 DataClean API 服务

1. 在项目中点击 "New Service" -> "GitHub Repo"
2. 选择相同仓库
3. 配置服务：
   - **Root Directory**: `ETM/dataclean`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. 添加环境变量：
   ```
   CORS_ORIGINS=https://your-frontend.railway.app
   LOG_LEVEL=INFO
   ```
5. 生成域名

#### 步骤 5：更新前端环境变量

部署完后端服务后，获取它们的域名，然后更新前端服务的环境变量：
- `NEXT_PUBLIC_API_URL` = 后端 API 的域名
- `NEXT_PUBLIC_DATACLEAN_API_URL` = DataClean API 的域名

然后重新部署前端服务。

---

### 方式二：通过 Railway CLI

#### 安装 Railway CLI

```bash
# macOS
brew install railway

# 或使用 npm
npm install -g @railway/cli
```

#### 登录

```bash
railway login
```

#### 部署

```bash
# 进入项目目录
cd /path/to/THETA

# 创建新项目
railway init

# 部署前端
cd theta-frontend3
railway up

# 部署后端 API
cd ../langgraph_agent/backend
railway up

# 部署 DataClean API
cd ../../ETM/dataclean
railway up
```

---

## 📝 环境变量配置

### 前端 (theta-frontend3)

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `NEXT_PUBLIC_API_URL` | 后端 API 地址 | `https://theta-api.railway.app` |
| `NEXT_PUBLIC_DATACLEAN_API_URL` | DataClean API 地址 | `https://theta-dataclean.railway.app` |
| `PORT` | 服务端口（Railway 自动设置） | `3000` |

### 后端 API (langgraph_agent/backend)

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `CORS_ORIGINS` | 允许的跨域来源 | `https://theta.railway.app` |
| `SIMULATION_MODE` | 模拟模式（无 GPU 时使用） | `true` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `PORT` | 服务端口（Railway 自动设置） | `8000` |

### DataClean API (ETM/dataclean)

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `CORS_ORIGINS` | 允许的跨域来源 | `https://theta.railway.app` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `PORT` | 服务端口（Railway 自动设置） | `8001` |

---

## ⚠️ 注意事项

### 1. 模拟模式

由于 Railway 免费版不提供 GPU，建议开启 `SIMULATION_MODE=true`。这样后端会使用模拟数据而不是实际运行 ETM 模型训练。

### 2. 文件存储

Railway 服务是无状态的，重启后本地文件会丢失。如果需要持久化存储：
- 使用 Railway 的 Volume 功能
- 或接入外部存储服务（如 S3、Cloudflare R2）

### 3. 内存限制

Railway 免费版有 512MB 内存限制。如果遇到内存不足：
- 升级到付费计划
- 或优化代码减少内存使用

### 4. 构建超时

如果构建时间过长，可以：
- 使用 `.dockerignore` 排除不必要的文件
- 预先安装依赖并缓存

---

## 🔧 故障排除

### 构建失败

```bash
# 查看构建日志
railway logs --build
```

### 运行时错误

```bash
# 查看运行日志
railway logs
```

### CORS 错误

确保后端服务的 `CORS_ORIGINS` 环境变量包含前端的完整域名（包括 `https://`）。

### 连接超时

检查环境变量中的 API 地址是否正确，确保使用 HTTPS。

---

## 📊 监控

Railway 提供内置的监控功能：
- CPU 使用率
- 内存使用率
- 网络流量
- 请求日志

在 Railway Dashboard 的服务详情页可以查看这些指标。

---

## 💰 费用估算

Railway 提供：
- **免费版**: $5/月免费额度
- **Hobby**: $5/月起
- **Pro**: $20/月起

三个服务的基本运行费用约为 $10-20/月（取决于使用量）。

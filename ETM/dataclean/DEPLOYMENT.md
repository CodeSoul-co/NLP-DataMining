# DataClean API 部署指南

## 推荐部署平台

### 🚀 Railway（推荐 - 最简单）

**优点：**
- 配置最简单
- 自动检测 Python 项目
- 支持环境变量
- 免费额度充足

**部署步骤：**
1. 访问 https://railway.app
2. 使用 GitHub 登录
3. 点击 "New Project" → "Deploy from GitHub repo"
4. 选择 `THETA` 仓库
5. 设置 Root Directory: `ETM/dataclean`
6. Railway 会自动检测并部署

**环境变量：**
- 无需额外配置，Railway 会自动设置 `PORT`

### 🌐 Render（推荐 - 免费层）

**优点：**
- 提供免费层
- 配置简单
- 支持自动部署

**部署步骤：**
1. 访问 https://render.com
2. 使用 GitHub 登录
3. 点击 "New" → "Web Service"
4. 连接 GitHub 仓库 `THETA`
5. 配置：
   - **Name**: dataclean-api
   - **Root Directory**: ETM/dataclean
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt && pip install fastapi uvicorn[standard] python-multipart`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
6. 点击 "Create Web Service"

**环境变量：**
- Render 会自动设置 `PORT` 环境变量

### ✈️ Fly.io（Docker 部署）

**优点：**
- 使用 Docker，部署灵活
- 全球边缘网络
- 免费层可用

**部署步骤：**
```bash
# 安装 flyctl
curl -L https://fly.io/install.sh | sh

# 登录
fly auth login

# 在 ETM/dataclean 目录下初始化
cd ETM/dataclean
fly launch

# 部署
fly deploy
```

### ⚡ Vercel（不推荐，但可行）

**注意：** Vercel 主要面向 serverless functions，对于完整的 FastAPI 应用支持有限。

**部署步骤：**
1. 访问 https://vercel.com
2. 导入 GitHub 仓库
3. 设置 Root Directory: `ETM/dataclean`
4. Vercel 会自动检测 `vercel.json` 配置

**限制：**
- 函数执行时间限制（10秒免费层）
- 文件上传大小限制
- 不适合长时间运行的任务

## 环境变量配置

所有平台都需要设置以下环境变量（如果需要）：

```bash
# 端口（大多数平台自动设置）
PORT=8001

# CORS 允许的源（生产环境）
ALLOWED_ORIGINS=https://your-frontend-domain.netlify.app
```

## 更新前端 API URL

部署后端后，更新前端的 API URL：

**Netlify 环境变量：**
```
NEXT_PUBLIC_DATACLEAN_API_URL=https://your-backend-url.railway.app
```

或

```
NEXT_PUBLIC_DATACLEAN_API_URL=https://your-backend-url.onrender.com
```

## 测试部署

部署后测试 API：

```bash
# 健康检查
curl https://your-backend-url.railway.app/health

# 获取支持格式
curl https://your-backend-url.railway.app/api/formats
```

## 推荐方案

**最佳组合：**
- **前端**: Netlify（Next.js 支持好）
- **后端**: Railway 或 Render（Python/FastAPI 支持好）

这样前后端分离部署，各自使用最适合的平台。

# 启动后端服务指南

## 🚀 快速启动后端

### 方法一：使用 Docker Compose（推荐）

```bash
# 在项目根目录执行
cd /www/wwwroot/theta.code-soul.com

# 启动所有服务（前端+后端）
docker-compose up -d

# 或只启动后端
docker-compose up -d etm-agent-api dataclean-api

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f etm-agent-api
```

### 方法二：在 Docker 容器中启动

```bash
# 如果容器已存在但未运行
docker start etm-agent-api

# 查看日志
docker logs -f etm-agent-api

# 进入容器
docker exec -it etm-agent-api bash
```

### 方法三：手动启动（开发环境）

```bash
# 进入后端目录
cd langgraph_agent/backend

# 安装依赖（如果还没安装）
pip install -r requirements.txt

# 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 或使用 run.py
python run.py --host 0.0.0.0 --port 8000
```

---

## 📝 创建测试账号（后端未启动时）

如果后端未启动，可以使用独立脚本直接操作数据库创建账号：

### 在本地环境

```bash
# 进入后端目录
cd langgraph_agent/backend

# 安装依赖（如果还没安装）
pip install passlib[bcrypt]

# 运行独立脚本
python scripts/create_test_user_standalone.py
```

### 在 Docker 容器中

```bash
# 如果容器已存在但未运行，先启动
docker start etm-agent-api

# 进入容器
docker exec -it etm-agent-api bash

# 运行脚本
python scripts/create_test_user_standalone.py
```

### 直接操作数据库（SQLite）

如果无法运行 Python 脚本，可以直接操作数据库：

```bash
# 找到数据库文件
# 通常在: langgraph_agent/backend/data/users.db

# 使用 sqlite3 命令行工具
sqlite3 langgraph_agent/backend/data/users.db
```

在 SQLite 中执行：

```sql
-- 创建表（如果不存在）
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name TEXT,
    created_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);

-- 注意：密码需要先加密，这里只是示例
-- 实际应该使用 Python 脚本生成哈希密码
-- 或者先启动后端，通过 API 注册
```

---

## ✅ 验证后端是否启动

### 检查健康状态

```bash
# 使用 curl
curl http://localhost:8000/health

# 或
curl http://localhost:8000/api/health
```

### 检查 API 文档

访问：`http://localhost:8000/docs`

### 检查容器状态

```bash
docker ps | grep etm-agent-api
```

---

## 🔧 常见问题

### 1. 端口被占用

```bash
# 检查端口占用
sudo netstat -tlnp | grep 8000
# 或
sudo ss -tlnp | grep 8000

# 停止占用端口的进程
sudo kill -9 <PID>
```

### 2. 容器启动失败

```bash
# 查看详细日志
docker logs etm-agent-api

# 检查环境变量
docker exec etm-agent-api env | grep -E "QWEN_API_KEY|SECRET_KEY|DATABASE_URL"
```

### 3. 数据库文件权限问题

```bash
# 检查数据库文件权限
ls -la langgraph_agent/backend/data/users.db

# 修复权限
chmod 644 langgraph_agent/backend/data/users.db
chown $(whoami) langgraph_agent/backend/data/users.db
```

---

## 📚 相关文档

- 测试账号说明：`TEST_ACCOUNTS.md`
- 后端结构：`langgraph_agent/backend/BACKEND_STRUCTURE.md`
- Docker 部署：`DOCKER_DEPLOY.md`

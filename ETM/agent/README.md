# ETM Agent API 服务

ETM Agent API 是一个基于 FastAPI 的后端服务，提供主题感知的智能代理功能。

## 快速启动

### 1. 安装依赖

```bash
# 安装 FastAPI 和 Uvicorn
pip install fastapi uvicorn

# 安装其他依赖（如果需要）
pip install -r ../requirements.txt
```

### 2. 启动服务

```bash
# 使用启动脚本（推荐）
./start_api.sh

# 或直接运行
cd api
python3 app.py
```

### 3. 验证服务

服务启动后，访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## 环境变量

可以通过环境变量配置服务：

```bash
# 设置端口（默认 8000）
export PORT=8000

# ETM 模型路径
export ETM_MODEL_PATH=/path/to/etm_model.pt

# 词汇表路径
export VOCAB_PATH=/path/to/vocab.json
```

## 前端集成

前端需要设置环境变量：

```bash
# .env.local 或 .env
NEXT_PUBLIC_ETM_AGENT_API_URL=http://localhost:8000
NEXT_PUBLIC_ETM_AGENT_WS_URL=ws://localhost:8000
```

## 故障排查

### 服务无法启动

1. 检查 Python 版本：`python3 --version`（需要 Python 3.8+）
2. 检查依赖：`pip list | grep -E "(fastapi|uvicorn)"`
3. 检查端口占用：`lsof -i :8000`

### 前端连接失败

1. 确认服务正在运行：`curl http://localhost:8000/health`
2. 检查 CORS 配置（已在代码中配置为允许所有来源）
3. 检查环境变量 `NEXT_PUBLIC_ETM_AGENT_API_URL` 是否正确

## API 端点

- `GET /health` - 健康检查
- `GET /tasks` - 获取所有任务
- `POST /tasks` - 创建新任务
- `GET /tasks/{task_id}` - 获取任务详情
- `GET /results` - 获取所有结果
- `GET /results/{dataset}/{mode}/topic-words` - 获取主题词
- `GET /results/{dataset}/{mode}/metrics` - 获取指标
- `GET /results/{dataset}/{mode}/visualizations` - 获取可视化

更多详情请访问 http://localhost:8000/docs

#!/bin/bash

# THETA LangGraph Agent Backend Startup Script

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "THETA LangGraph Agent Backend"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Script Dir: $SCRIPT_DIR"

# 设置 Python 路径
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/ETM:$SCRIPT_DIR/backend"

# 检查依赖
echo ""
echo "Checking dependencies..."

# 检查必要的 Python 包
python3 -c "import fastapi" 2>/dev/null || {
    echo "Installing FastAPI dependencies..."
    pip install fastapi uvicorn[standard] websockets python-multipart pydantic pydantic-settings
}

python3 -c "import langgraph" 2>/dev/null || {
    echo "Installing LangGraph dependencies..."
    pip install langgraph langchain langchain-core
}

# 进入后端目录
cd "$SCRIPT_DIR/backend"

echo ""
echo "Starting THETA LangGraph Agent API..."
echo "API will be available at: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "WebSocket: ws://localhost:8000/api/ws"
echo ""

# 启动服务
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

#!/bin/bash
# ETM Agent API 启动脚本

# 设置端口（可通过环境变量覆盖）
PORT=${PORT:-8000}

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$SCRIPT_DIR"
API_DIR="$AGENT_DIR/api"
ETM_DIR="$AGENT_DIR/.."
PROJECT_ROOT="$ETM_DIR/.."

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python 3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import fastapi, uvicorn, numpy, faiss, websockets" 2>/dev/null || {
    echo "⚠️  缺少依赖，正在安装..."
    if [ -f "$ETM_DIR/requirements.txt" ]; then
        python3 -m pip install -q -r "$ETM_DIR/requirements.txt"
    else
        python3 -m pip install -q fastapi uvicorn numpy scipy torch pandas
    fi
    # 安装 faiss（用于向量相似性搜索）
    python3 -m pip install -q faiss-cpu 2>/dev/null || python3 -m pip install -q faiss 2>/dev/null || true
    # 安装 WebSocket 支持
    python3 -m pip install -q 'uvicorn[standard]' websockets 2>/dev/null || true
}

# 设置 Python 路径（项目根目录，这样可以从 ETM.agent 导入）
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 切换到 API 目录
cd "$API_DIR"

# 启动 API 服务
echo "🚀 启动 ETM Agent API 服务..."
echo "   端口: $PORT"
echo "   API 文档: http://localhost:$PORT/docs"
echo "   健康检查: http://localhost:$PORT/health"
echo "   按 Ctrl+C 停止服务"
echo ""

python3 app.py

#!/bin/bash

# THETA 项目启动脚本
# 启动所有服务：后端 API、DataClean API 和前端应用

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  THETA 项目启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到 Python 3${NC}"
    exit 1
fi

# 检查 Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到 Node.js${NC}"
    echo "请先安装 Node.js: https://nodejs.org/"
    exit 1
fi

# 检查 pnpm（推荐）或 npm
if command -v pnpm &> /dev/null; then
    PACKAGE_MANAGER="pnpm"
elif command -v npm &> /dev/null; then
    PACKAGE_MANAGER="npm"
else
    echo -e "${RED}❌ 错误: 未找到 pnpm 或 npm${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python: $(python3 --version)"
echo -e "${GREEN}✓${NC} Node.js: $(node --version)"
echo -e "${GREEN}✓${NC} 包管理器: $PACKAGE_MANAGER"
echo ""

# 函数：清理后台进程
cleanup() {
    echo ""
    echo -e "${YELLOW}正在停止所有服务...${NC}"
    kill $BACKEND_PID $DATACLEAN_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $DATACLEAN_PID $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}所有服务已停止${NC}"
    exit 0
}

# 捕获 Ctrl+C
trap cleanup SIGINT SIGTERM

# 设置 Python 路径
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/ETM:$PROJECT_ROOT/langgraph_agent/backend"

# 检查后端依赖
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo -e "${YELLOW}安装后端依赖...${NC}"
    pip install -q fastapi uvicorn[standard] websockets python-multipart pydantic pydantic-settings langgraph langchain langchain-core || true
fi

# ==========================================
# 1. 启动 LangGraph Agent API (端口 8000)
# ==========================================
echo -e "${YELLOW}[1/3] 启动 LangGraph Agent API...${NC}"

cd "$PROJECT_ROOT/langgraph_agent/backend"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/theta_backend.log 2>&1 &
BACKEND_PID=$!

sleep 2

if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} LangGraph Agent API 已启动"
    echo -e "   ${GREEN}→${NC} API: http://localhost:8000"
    echo -e "   ${GREEN}→${NC} 文档: http://localhost:8000/docs"
else
    echo -e "${RED}⚠️  LangGraph Agent API 可能未正常启动${NC}"
fi

echo ""

# ==========================================
# 2. 启动 DataClean API (端口 8001)
# ==========================================
echo -e "${YELLOW}[2/3] 启动 DataClean API...${NC}"

cd "$PROJECT_ROOT/ETM/dataclean"
PORT=8001 python3 api.py > /tmp/theta_dataclean.log 2>&1 &
DATACLEAN_PID=$!

sleep 2

if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} DataClean API 已启动"
    echo -e "   ${GREEN}→${NC} API: http://localhost:8001"
    echo -e "   ${GREEN}→${NC} 文档: http://localhost:8001/docs"
else
    echo -e "${RED}⚠️  DataClean API 可能未正常启动${NC}"
fi

echo ""

# ==========================================
# 3. 启动前端应用 (端口 3000)
# ==========================================
echo -e "${YELLOW}[3/3] 启动前端应用...${NC}"

cd "$PROJECT_ROOT/theta-frontend3"

# 检查 node_modules
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}安装前端依赖（这可能需要几分钟）...${NC}"
    $PACKAGE_MANAGER install
fi

# 检查环境变量
if [ ! -f ".env.local" ]; then
    echo -e "${YELLOW}创建 .env.local 文件...${NC}"
    cat > .env.local << EOF
NEXT_PUBLIC_ETM_AGENT_API_URL=http://localhost:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001
EOF
fi

# 启动前端（后台运行）
$PACKAGE_MANAGER run dev > /tmp/theta_frontend.log 2>&1 &
FRONTEND_PID=$!

# 等待前端启动
echo -e "${YELLOW}等待前端启动...${NC}"
sleep 5

# 检查前端是否启动成功
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} 前端应用已启动"
    echo -e "   ${GREEN}→${NC} 前端: http://localhost:3000"
else
    echo -e "${RED}⚠️  前端可能未正常启动${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  所有服务已启动！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}访问地址:${NC}"
echo -e "  ${GREEN}前端:${NC}            http://localhost:3000"
echo -e "  ${GREEN}LangGraph API:${NC}   http://localhost:8000"
echo -e "  ${GREEN}DataClean API:${NC}   http://localhost:8001"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止所有服务${NC}"
echo ""

# 显示日志
echo -e "${YELLOW}实时日志:${NC}"
echo ""

# 等待用户中断
tail -f /tmp/theta_backend.log /tmp/theta_dataclean.log /tmp/theta_frontend.log 2>/dev/null || wait

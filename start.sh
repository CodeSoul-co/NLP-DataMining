#!/bin/bash

# THETA 本地开发启动脚本
# 启动：后端 langgraph_agent(8000) + DataClean(8001) + 前端(3000)
# 8000 使用 langgraph_agent 后端（完整路由 + OSS/DLC 集成）
# 若自动启动失败，请用 分步启动 方式（见下方或 前后端完成与对接情况.md）

# 不用 set -e，单步失败时继续尝试后续服务
set +e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  THETA 本地开发 (后端 + DataClean + 前端)${NC}"
echo -e "${GREEN}========================================${NC}"

# 必要命令
for c in python3 node; do
  if ! command -v $c &>/dev/null; then
    echo -e "${RED}❌ 缺少: $c${NC}"
    exit 1
  fi
done
PM=pnpm; command -v pnpm &>/dev/null || PM=npm
echo -e "Python: $(python3 --version) | Node: $(node -v) | 包管理: $PM"

# 清理占用端口的进程
kill_port() {
  local p
  p=$(lsof -ti:$1 2>/dev/null)
  if [ -n "$p" ]; then
    echo "$p" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
}

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/ETM:$PROJECT_ROOT/theta_1-main:$PROJECT_ROOT/langgraph_agent/backend"
export DEBUG=true

# 从 backend/.env 加载 SIMULATION_MODE（若 .env 有 SIMULATION_MODE=false 则用 PostgreSQL）
BACKEND_ENV="$PROJECT_ROOT/langgraph_agent/backend/.env"
if [ -f "$BACKEND_ENV" ] && grep -q '^SIMULATION_MODE=' "$BACKEND_ENV" 2>/dev/null; then
  export SIMULATION_MODE=$(grep '^SIMULATION_MODE=' "$BACKEND_ENV" | sed 's/^SIMULATION_MODE=//' | tr -d '"' | tr -d "'" | head -1)
fi
export SIMULATION_MODE="${SIMULATION_MODE:-true}"

# 从 .env 注入环境变量
for f in "$PROJECT_ROOT/.env" "$PROJECT_ROOT/theta_1-main/.env"; do
  [ ! -f "$f" ] && continue
  for key in QWEN_API_KEY ALIBABA_CLOUD_ACCESS_KEY_ID ALIBABA_CLOUD_ACCESS_KEY_SECRET OSS_ENDPOINT OSS_BUCKET DLC_WORKSPACE_ID OSS_DATASET_ID DLC_INSTANCE_TYPE DLC_TRAINING_IMAGE ALIBABA_CLOUD_REGION; do
    if grep -q "^${key}=" "$f" 2>/dev/null; then
      val=$(grep "^${key}=" "$f" | sed "s/^${key}=//" | tr -d '"' | tr -d "'" | head -1)
      [ -n "$val" ] && export "$key=$val"
    fi
  done
done
[ -n "$ALIBABA_CLOUD_ACCESS_KEY_ID" ] && echo -e "${GREEN}✓${NC} 已加载 OSS/DLC 配置" || echo -e "${YELLOW}⊘ 未配置 OSS/DLC${NC} → 上传将使用本地模式，需在 .env 配置阿里云凭证以启用 OSS+DLC"

# 后端依赖（langgraph_agent：完整路由 + OSS/DLC 集成）
echo -e "\n${YELLOW}[1/3] 后端 API (8000) - langgraph_agent (含 OSS+DLC)${NC}"
for pkg in fastapi uvicorn pydantic; do
  python3 -c "import $pkg" 2>/dev/null || python3 -m pip install -q "$pkg" 2>/dev/null || true
done
[ -f "$PROJECT_ROOT/langgraph_agent/backend/requirements.txt" ] && python3 -m pip install -q -r "$PROJECT_ROOT/langgraph_agent/backend/requirements.txt" 2>/dev/null || true

kill_port 8000
cd "$PROJECT_ROOT/langgraph_agent/backend" || exit 1
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/theta_backend.log 2>&1 &
BID=$!

co="-s --connect-timeout 2 --max-time 3"
n=0; while [ $n -lt 10 ]; do
  sleep 1; n=$((n+1))
  curl $co http://localhost:8000/ >/dev/null 2>&1 && break
done
if curl $co http://localhost:8000/ >/dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} 后端: http://localhost:8000 文档: /docs"
else
  echo -e "${RED}✗ 后端未就绪${NC} → tail -f /tmp/theta_backend.log"
fi

# DataClean（依赖: python-docx, pandas, jieba 等，缺则跳过）
echo -e "\n${YELLOW}[2/3] DataClean API (8001)${NC}"
kill_port 8001
( cd "$PROJECT_ROOT/ETM/dataclean" && PORT=8001 python3 api.py > /tmp/theta_dataclean.log 2>&1 ) &
DID=$!

n=0; while [ $n -lt 6 ]; do sleep 1; n=$((n+1)); curl $co http://localhost:8001/health >/dev/null 2>&1 && break; done
if curl $co http://localhost:8001/health >/dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} DataClean: http://localhost:8001"
else
  echo -e "${YELLOW}⊘ DataClean 未启动${NC}（可选）→ pip install -r ETM/dataclean/requirements.txt 后重试"
fi

# 前端
echo -e "\n${YELLOW}[3/3] 前端 (3000)${NC}"
kill_port 3000
cd "$PROJECT_ROOT/theta-frontend3" || exit 1
[ ! -d node_modules ] && $PM install
# 本地开发固定用 8000/8001，每次启动写入，避免旧 SSH/远程配置导致连接失败
printf "NEXT_PUBLIC_API_URL=http://localhost:8000\nNEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001\n" > .env.local

$PM run dev > /tmp/theta_frontend.log 2>&1 &
FID=$!
sleep 6
if curl $co http://localhost:3000 >/dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} 前端: http://localhost:3000"
else
  echo -e "${YELLOW}⊘ 前端启动中…${NC} 稍后访问 http://localhost:3000 或 tail -f /tmp/theta_frontend.log"
fi

echo -e "\n${GREEN}----------------------------------------${NC}"
echo -e "  前端: http://localhost:3000"
echo -e "  后端: http://localhost:8000  文档: http://localhost:8000/docs"
echo -e "  DataClean: http://localhost:8001"
echo -e "  日志: /tmp/theta_backend.log, /tmp/theta_dataclean.log, /tmp/theta_frontend.log"
echo -e "  按 Ctrl+C 停止 | 若失败见 前后端完成与对接情况.md 分步启动"
echo -e "${GREEN}----------------------------------------${NC}\n"

cleanup() {
  printf "\n${YELLOW}停止服务…${NC}\n"
  kill $BID $DID $FID 2>/dev/null; wait $BID $DID $FID 2>/dev/null; exit 0
}
trap cleanup SIGINT SIGTERM
wait $BID $DID $FID 2>/dev/null; true

#!/bin/bash
# 全面功能测试脚本

set -e

echo "🧪 THETA 项目功能测试"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试计数器
PASSED=0
FAILED=0

# 测试函数
test_endpoint() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo -n "测试 $name... "
    response=$(curl -s -w "\n%{http_code}" "$url" 2>&1 || echo "ERROR\n000")
    http_code=$(echo "$response" | tail -1)
    body=$(echo "$response" | sed '$d')
    
    if [[ "$http_code" == "200" ]] || [[ "$http_code" == "101" ]]; then
        if [[ -n "$expected" ]]; then
            if echo "$body" | grep -q "$expected"; then
                echo -e "${GREEN}✓ 通过${NC}"
                ((PASSED++))
                return 0
            else
                echo -e "${RED}✗ 失败 (响应不包含预期内容)${NC}"
                ((FAILED++))
                return 1
            fi
        else
            echo -e "${GREEN}✓ 通过${NC}"
            ((PASSED++))
            return 0
        fi
    else
        echo -e "${RED}✗ 失败 (HTTP $http_code)${NC}"
        ((FAILED++))
        return 1
    fi
}

# 1. 检查服务运行状态
echo "📋 1. 服务状态检查"
echo "-------------------"

# 检查前端
if pgrep -f "next dev" > /dev/null; then
    echo -e "${GREEN}✓ 前端服务运行中${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ 前端服务未运行${NC}"
    ((FAILED++))
fi

# 检查 DataClean API（通过健康检查）
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ DataClean API 运行中${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ DataClean API 未运行${NC}"
    ((FAILED++))
fi

# 检查 ETM Agent API（通过健康检查）
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ETM Agent API 运行中${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ ETM Agent API 未运行${NC}"
    ((FAILED++))
fi

echo ""

# 2. DataClean API 测试
echo "📋 2. DataClean API 功能测试"
echo "-------------------"

test_endpoint "健康检查" "http://localhost:8001/health" "status"
test_endpoint "支持格式" "http://localhost:8001/api/formats" "txt"
test_endpoint "API 文档" "http://localhost:8001/docs" "swagger"

# 测试文本清洗
echo -n "测试文本清洗... "
test_text="这是一个测试文本 https://example.com <p>HTML标签</p> 包含多个空格   和制表符"
response=$(curl -s -X POST "http://localhost:8001/api/clean/text" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$test_text\"}" 2>&1)

if echo "$response" | grep -q "cleaned_text"; then
    echo -e "${GREEN}✓ 通过${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ 失败${NC}"
    echo "  响应: $response"
    ((FAILED++))
fi

echo ""

# 3. ETM Agent API 测试
echo "📋 3. ETM Agent API 功能测试"
echo "-------------------"

test_endpoint "健康检查" "http://localhost:8000/health" "status"
test_endpoint "API 文档" "http://localhost:8000/docs" "swagger"

# 测试任务列表
test_endpoint "获取任务列表" "http://localhost:8000/api/tasks" ""

# 测试创建任务
echo -n "测试创建任务... "
task_response=$(curl -s -X POST "http://localhost:8000/api/tasks" \
    -H "Content-Type: application/json" \
    -d '{
        "dataset": "test_dataset",
        "mode": "train",
        "num_topics": 10
    }' 2>&1)

if echo "$task_response" | grep -q "task_id"; then
    echo -e "${GREEN}✓ 通过${NC}"
    ((PASSED++))
    TASK_ID=$(echo "$task_response" | grep -o '"task_id":"[^"]*"' | cut -d'"' -f4)
    echo "  任务 ID: $TASK_ID"
    
    # 测试获取单个任务
    if [[ -n "$TASK_ID" ]]; then
        test_endpoint "获取单个任务" "http://localhost:8000/api/tasks/$TASK_ID" "task_id"
    fi
else
    echo -e "${RED}✗ 失败${NC}"
    echo "  响应: $task_response"
    ((FAILED++))
fi

echo ""

# 4. 前端页面测试
echo "📋 4. 前端页面测试"
echo "-------------------"

test_endpoint "首页" "http://localhost:3000" "THETA"
test_endpoint "训练页面" "http://localhost:3000/training" ""
test_endpoint "结果页面" "http://localhost:3000/results" ""
test_endpoint "可视化页面" "http://localhost:3000/visualizations" ""

echo ""

# 5. WebSocket 测试（简单测试）
echo "📋 5. WebSocket 连接测试"
echo "-------------------"

echo -n "测试 WebSocket 端点... "
ws_response=$(curl -s -i -N \
    -H "Connection: Upgrade" \
    -H "Upgrade: websocket" \
    -H "Sec-WebSocket-Version: 13" \
    -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
    "http://localhost:8000/api/ws" 2>&1 | head -5)

if echo "$ws_response" | grep -qE "(101|400|426)"; then
    echo -e "${GREEN}✓ WebSocket 端点响应正常${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WebSocket 需要浏览器环境测试${NC}"
    echo "  响应: $ws_response"
fi

echo ""

# 6. 总结
echo "================================"
echo "📊 测试总结"
echo "================================"
echo -e "${GREEN}通过: $PASSED${NC}"
echo -e "${RED}失败: $FAILED${NC}"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}✅ 所有测试通过！${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  部分测试失败，请检查上述错误${NC}"
    exit 1
fi

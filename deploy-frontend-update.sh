#!/bin/bash
# THETA 前端更新部署脚本（包含 Git 更新）

set -e

echo "🔄 THETA 前端更新部署脚本"
echo "================================"

# 检查是否在 Git 仓库中
if [ ! -d ".git" ]; then
    echo "❌ 错误: 当前目录不是 Git 仓库"
    echo "请先初始化 Git 仓库或克隆项目"
    exit 1
fi

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未找到 Docker"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ 错误: 未找到 Docker Compose"
    exit 1
fi

# 检查 Docker 服务是否运行
if ! docker info &> /dev/null; then
    echo "❌ 错误: Docker 服务未运行"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 步骤 1: 拉取最新代码
echo "📥 [1/4] 拉取最新代码..."

# 配置 Git 默认行为（避免分支分歧提示）
git config pull.rebase true 2>/dev/null || true

# 获取当前分支名
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")

# 拉取最新代码
git fetch origin
if git pull --rebase origin $CURRENT_BRANCH 2>/dev/null; then
    echo "✅ 代码更新完成（使用 rebase）"
elif git pull origin $CURRENT_BRANCH 2>/dev/null; then
    echo "✅ 代码更新完成（使用 merge）"
else
    echo "⚠️  Git pull 失败，继续使用当前代码"
    echo "   提示: 如果遇到分支分歧，请手动执行: git pull --rebase"
fi
echo ""

# 步骤 2: 检查 .env.frontend 文件
if [ ! -f ".env.frontend" ]; then
    echo "📝 [2/4] 创建 .env.frontend 文件..."
    if [ -f ".env.frontend.example" ]; then
        cp .env.frontend.example .env.frontend
        echo "✅ 已从 .env.frontend.example 创建 .env.frontend 文件"
        echo "⚠️  请编辑 .env.frontend 文件，设置正确的配置值"
        echo ""
        read -p "是否现在编辑 .env.frontend 文件? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ${EDITOR:-nano} .env.frontend
        fi
    else
        echo "⚠️  未找到 .env.frontend.example，请手动创建 .env.frontend 文件"
        exit 1
    fi
else
    echo "✅ [2/4] .env.frontend 文件已存在"
fi
echo ""

# 步骤 3: 检查端口占用
echo "🔍 [3/5] 检查端口占用..."
FRONTEND_PORT=$(grep FRONTEND_PORT .env.frontend 2>/dev/null | cut -d '=' -f2 | tr -d ' ' || echo "80")
if [ -z "$FRONTEND_PORT" ] || [ "$FRONTEND_PORT" = "" ]; then
    FRONTEND_PORT=80
fi

# 检查端口是否被占用
if command -v netstat &> /dev/null; then
    PORT_IN_USE=$(sudo netstat -tlnp 2>/dev/null | grep ":$FRONTEND_PORT " || true)
elif command -v ss &> /dev/null; then
    PORT_IN_USE=$(sudo ss -tlnp 2>/dev/null | grep ":$FRONTEND_PORT " || true)
else
    PORT_IN_USE=""
fi

if [ -n "$PORT_IN_USE" ]; then
    echo "⚠️  警告: 端口 $FRONTEND_PORT 已被占用"
    echo "   占用信息: $PORT_IN_USE"
    echo ""
    echo "💡 解决方案："
    echo "   1. 如果使用 Nginx 反向代理，将容器端口改为 3000，Nginx 监听 80"
    echo "   2. 或者停止占用端口的服务"
    echo "   3. 或者修改 .env.frontend 中的 FRONTEND_PORT 为其他端口（如 3000）"
    echo ""
    read -p "是否继续部署？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 部署已取消"
        exit 1
    fi
else
    echo "✅ 端口 $FRONTEND_PORT 可用"
fi
fi
echo ""

# 步骤 4: 停止现有容器
echo "🛑 [4/5] 停止现有容器..."
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down 2>/dev/null || true
echo "✅ 容器已停止"
echo ""

# 步骤 5: 重新构建并启动
echo "🔨 [5/5] 重新构建并启动服务..."
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend build --no-cache
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo ""
echo "📊 服务状态:"
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend ps

# 检查健康状态
echo ""
echo "🏥 健康检查:"
FRONTEND_PORT=$(grep FRONTEND_PORT .env.frontend 2>/dev/null | cut -d '=' -f2 || echo "80")
if [ -z "$FRONTEND_PORT" ] || [ "$FRONTEND_PORT" = "" ]; then
    FRONTEND_PORT=80
fi

echo -n "前端 (端口 $FRONTEND_PORT): "
if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
    echo "✅ 运行正常"
else
    echo "❌ 无法访问，请检查日志: docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs theta-frontend"
fi

echo ""
echo "✅ 前端更新部署完成！"
echo ""
echo "📋 常用命令:"
echo "  查看日志:     docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f"
echo "  停止服务:     docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down"
echo "  重启服务:     docker-compose -f docker-compose.frontend.yml --env-file .env.frontend restart"
echo ""
echo "🌐 访问地址:"
if [ "$FRONTEND_PORT" = "80" ]; then
    echo "  前端: http://localhost (标准 HTTP 端口 80)"
else
    echo "  前端: http://localhost:$FRONTEND_PORT"
fi

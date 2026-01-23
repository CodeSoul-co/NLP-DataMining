#!/bin/bash
# THETA 项目 Docker 一键部署脚本

set -e

echo "🚀 THETA 项目 Docker 部署脚本"
echo "================================"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未找到 Docker"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ 错误: 未找到 Docker Compose"
    echo "请先安装 Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# 检查 Docker 服务是否运行
if ! docker info &> /dev/null; then
    echo "❌ 错误: Docker 服务未运行"
    echo "请启动 Docker 服务: sudo systemctl start docker"
    exit 1
fi

echo "✅ Docker 环境检查通过"
echo ""

# 选择 docker compose 命令
if docker compose version &>/dev/null; then
    DCO="docker compose"
else
    DCO="docker-compose"
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "📝 创建 .env 文件..."
    if [ -f "docker.env.template" ]; then
        cp docker.env.template .env
        echo "✅ 已从 docker.env.template 创建 .env 文件"
        echo "⚠️  请编辑 .env，必填: QWEN_API_KEY, POSTGRES_PASSWORD, SECRET_KEY, DOMAIN"
        echo ""
        read -p "是否现在编辑 .env 文件? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ${EDITOR:-nano} .env
        fi
    elif [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ 已从 .env.example 创建 .env 文件"
        echo "⚠️  请编辑 .env，必填: QWEN_API_KEY, POSTGRES_PASSWORD, SECRET_KEY, DOMAIN"
        echo ""
        read -p "是否现在编辑 .env 文件? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ${EDITOR:-nano} .env
        fi
    else
        echo "⚠️  未找到 docker.env.template 或 .env.example，请手动创建 .env"
        exit 1
    fi
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p ETM/dataclean/temp_uploads ETM/dataclean/temp_processed
mkdir -p nginx/certs data result
chmod 755 ETM/dataclean/temp_uploads ETM/dataclean/temp_processed

# 停止现有容器（如果存在）
echo "🛑 停止现有容器..."
$DCO down 2>/dev/null || true

# 构建镜像
echo "🔨 构建 Docker 镜像（--no-cache）..."
$DCO build --no-cache

# 启动服务
echo "🚀 启动服务..."
$DCO up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 15

# 检查服务状态
echo ""
echo "📊 服务状态:"
$DCO ps

# 健康检查（通过 Nginx 80 端口）
echo ""
echo "🏥 健康检查（通过 http://localhost）:"
echo -n "Nginx /health: "
if curl -sf http://localhost/health >/dev/null; then
    echo "✅"
else
    echo "❌ 检查: $DCO logs nginx"
fi

echo -n "后端 /api/health: "
if curl -sf http://localhost/api/health >/dev/null; then
    echo "✅"
else
    echo "❌ 检查: $DCO logs backend"
fi

echo -n "前端 /: "
if curl -sf http://localhost/ >/dev/null; then
    echo "✅"
else
    echo "❌ 检查: $DCO logs frontend"
fi

echo ""
echo "✅ 部署完成！"
echo ""
echo "📋 常用: $DCO logs -f | $DCO down | $DCO ps"
echo "🔄 更新: git pull && $DCO build --no-cache && $DCO up -d"
echo "🌐 访问: http://服务器IP 或 http://域名（/api/, /dataclean/）"

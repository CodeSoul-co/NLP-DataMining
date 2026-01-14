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

# 检查并配置 Docker 镜像加速器（腾讯云优化）
if ! docker info 2>/dev/null | grep -q "mirror.ccs.tencentyun.com"; then
    echo "🔧 检测到未配置 Docker 镜像加速器..."
    read -p "是否自动配置腾讯云镜像加速器? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "setup-docker-mirror.sh" ]; then
            chmod +x setup-docker-mirror.sh
            sudo ./setup-docker-mirror.sh
        else
            echo "⚠️  未找到 setup-docker-mirror.sh，请手动配置"
            echo "参考: https://cloud.tencent.com/document/product/1141/50332"
        fi
    fi
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "📝 创建 .env 文件..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ 已从 .env.example 创建 .env 文件"
        echo "⚠️  请编辑 .env 文件，设置正确的配置值"
        echo ""
        read -p "是否现在编辑 .env 文件? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ${EDITOR:-nano} .env
        fi
    elif [ -f "docker.env.template" ]; then
        cp docker.env.template .env
        echo "✅ 已从 docker.env.template 创建 .env 文件"
        echo "⚠️  请编辑 .env 文件，设置正确的配置值"
        echo ""
        read -p "是否现在编辑 .env 文件? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ${EDITOR:-nano} .env
        fi
    else
        echo "⚠️  未找到 .env.example，请手动创建 .env 文件"
    fi
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p ETM/dataclean/temp_uploads ETM/dataclean/temp_processed
chmod 755 ETM/dataclean/temp_uploads ETM/dataclean/temp_processed

# 停止现有容器（如果存在）
echo "🛑 停止现有容器..."
docker-compose down 2>/dev/null || true

# 构建镜像
echo "🔨 构建 Docker 镜像..."
docker-compose build --no-cache

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo ""
echo "📊 服务状态:"
docker-compose ps

# 检查健康状态
echo ""
echo "🏥 健康检查:"
echo -n "后端 API: "
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ 运行正常"
else
    echo "❌ 无法访问，请检查日志: docker-compose logs dataclean-api"
fi

echo -n "前端: "
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ 运行正常"
else
    echo "❌ 无法访问，请检查日志: docker-compose logs theta-frontend"
fi

echo ""
echo "✅ 部署完成！"
echo ""
echo "📋 常用命令:"
echo "  查看日志:     docker-compose logs -f"
echo "  停止服务:     docker-compose down"
echo "  重启服务:     docker-compose restart"
echo "  查看状态:     docker-compose ps"
echo "  更新代码:     git pull && docker-compose up -d --build"
echo ""
echo "🌐 访问地址:"
echo "  前端: http://localhost:3000"
echo "  后端 API: http://localhost:8001"
echo "  API 健康检查: http://localhost:8001/health"

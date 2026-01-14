#!/bin/bash
# 配置 Docker 镜像加速器（腾讯云优化版）

set -e

echo "🔧 配置 Docker 镜像加速器（腾讯云）..."

# 检查是否有 daemon.json
if [ ! -f /etc/docker/daemon.json ]; then
    echo "📝 创建 /etc/docker/daemon.json..."
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://docker.m.daocloud.io",
    "https://hub-mirror.c.163.com"
  ],
  "dns": ["8.8.8.8", "114.114.114.114"],
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 5
}
EOF
else
    echo "📝 更新 /etc/docker/daemon.json..."
    # 备份原文件
    sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
    
    # 使用 jq 更新（如果安装了）
    if command -v jq &> /dev/null; then
        sudo jq '.registry-mirrors = ["https://mirror.ccs.tencentyun.com", "https://docker.m.daocloud.io", "https://hub-mirror.c.163.com"] | .dns = ["8.8.8.8", "114.114.114.114"]' /etc/docker/daemon.json > /tmp/daemon.json
        sudo mv /tmp/daemon.json /etc/docker/daemon.json
    else
        echo "⚠️  未安装 jq，请手动编辑 /etc/docker/daemon.json"
        echo "添加以下内容到 registry-mirrors:"
        echo '  "https://mirror.ccs.tencentyun.com"'
    fi
fi

# 重启 Docker
echo "🔄 重启 Docker 服务..."
sudo systemctl daemon-reload
sudo systemctl restart docker

# 等待 Docker 启动
sleep 3

# 验证配置
echo "✅ 验证配置..."
docker info | grep -i "registry mirror" || echo "⚠️  无法验证，但配置已应用"

echo ""
echo "✅ Docker 镜像加速器配置完成！"
echo ""
echo "测试拉取镜像:"
echo "  docker pull python:3.11-slim"
echo "  docker pull node:20-alpine"

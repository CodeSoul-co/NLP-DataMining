# 腾讯云服务器 Docker 部署指南

## 🚀 快速部署（腾讯云优化版）

### 步骤 1: 配置 Docker 镜像加速器（必须）

腾讯云服务器访问 Docker Hub 较慢，必须先配置镜像加速器：

```bash
# 运行自动配置脚本
chmod +x setup-docker-mirror.sh
sudo ./setup-docker-mirror.sh
```

或者手动配置：

```bash
# 创建/编辑 Docker 配置文件
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://docker.m.daocloud.io",
    "https://hub-mirror.c.163.com"
  ],
  "dns": ["8.8.8.8", "114.114.114.114"]
}
EOF

# 重启 Docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# 验证配置
docker info | grep -i "registry mirror"
```

### 步骤 2: 测试镜像拉取

```bash
# 测试拉取基础镜像
docker pull python:3.11-slim
docker pull node:20-alpine

# 如果成功，继续下一步
# 如果失败，检查网络或使用其他镜像源
```

### 步骤 3: 克隆项目

```bash
cd /opt
sudo git clone https://github.com/CodeSoul-co/THETA.git
cd THETA
sudo git checkout frontend-3
```

### 步骤 4: 一键部署

```bash
# 运行部署脚本
sudo chmod +x docker-deploy.sh
sudo ./docker-deploy.sh
```

## 🔧 腾讯云特定优化

### 1. Dockerfile 已优化

- **后端 Dockerfile**: 使用腾讯云 pip 镜像源
- **前端 Dockerfile**: 使用腾讯云 npm 镜像源
- **apt 源**: 自动使用腾讯云 Debian 镜像

### 2. 网络优化

如果仍然遇到网络问题，可以：

#### 方案 A: 使用腾讯云容器镜像服务

```bash
# 登录腾讯云容器镜像服务
docker login ccr.ccs.tencentyun.com

# 在 docker-compose.yml 中使用腾讯云镜像
# 需要先将镜像推送到腾讯云容器镜像服务
```

#### 方案 B: 使用代理（如果有）

```bash
# 配置 Docker 代理
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<EOF
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:8080"
Environment="HTTPS_PROXY=http://proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1"
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
```

#### 方案 C: 离线导入镜像

在能联网的机器上：

```bash
# 拉取镜像
docker pull python:3.11-slim
docker pull node:20-alpine

# 导出镜像
docker save python:3.11-slim > python-3.11-slim.tar
docker save node:20-alpine > node-20-alpine.tar
```

在服务器上：

```bash
# 导入镜像
docker load < python-3.11-slim.tar
docker load < node-20-alpine.tar
```

## 🛠️ 常见问题

### 问题 1: 镜像拉取超时

**解决方案：**
1. 确保已配置镜像加速器
2. 检查防火墙设置
3. 尝试使用其他镜像源

```bash
# 测试镜像源
curl -I https://mirror.ccs.tencentyun.com
```

### 问题 2: pip 安装慢

Dockerfile 已配置使用腾讯云 pip 镜像，如果仍然慢：

```bash
# 在 Dockerfile 中已配置，无需手动操作
# 如果构建时仍然慢，检查网络连接
```

### 问题 3: npm/pnpm 安装慢

Dockerfile 已配置使用腾讯云 npm 镜像，如果仍然慢：

```bash
# 在 Dockerfile 中已配置，无需手动操作
```

### 问题 4: 构建失败

```bash
# 查看详细日志
docker-compose build --progress=plain

# 清理缓存重新构建
docker-compose build --no-cache
```

## 📋 完整部署流程

```bash
# 1. 配置镜像加速器
sudo ./setup-docker-mirror.sh

# 2. 测试镜像拉取
docker pull python:3.11-slim
docker pull node:20-alpine

# 3. 部署项目
sudo ./docker-deploy.sh

# 4. 查看状态
docker-compose ps
docker-compose logs -f
```

## 🔒 安全建议

1. **配置防火墙**
```bash
# 腾讯云安全组配置
# 开放端口: 22 (SSH), 80 (HTTP), 443 (HTTPS), 3000 (前端), 8001 (后端)
```

2. **使用 HTTPS**
```bash
# 配置 Nginx + Let's Encrypt
# 参考 SERVER_DEPLOYMENT.md
```

3. **定期更新**
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 更新 Docker 镜像
docker-compose pull
docker-compose up -d
```

## 📞 技术支持

如果遇到问题：
1. 查看日志: `docker-compose logs -f`
2. 检查网络: `curl -I https://mirror.ccs.tencentyun.com`
3. 查看文档: `DOCKER_DEPLOY.md`

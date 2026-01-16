#!/usr/bin/env python3
"""
THETA 服务器部署脚本
自动部署到远程服务器
"""

import paramiko
import sys
import time

# 服务器配置
SERVER = "liguozheng.site"
USERNAME = "root"
PASSWORD = "P@ssw0rd130"
PORT = 22

def create_ssh_client():
    """创建 SSH 连接"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(SERVER, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)
        print(f"✅ 成功连接到服务器 {SERVER}")
        return client
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None

def run_command(client, command, show_output=True):
    """执行远程命令"""
    print(f"\n🔧 执行: {command}")
    stdin, stdout, stderr = client.exec_command(command, timeout=300)
    
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    exit_code = stdout.channel.recv_exit_status()
    
    if show_output and output:
        print(output)
    if error and exit_code != 0:
        print(f"⚠️ 错误: {error}")
    
    return exit_code, output, error

def main():
    print("=" * 60)
    print("THETA 服务器部署脚本")
    print("=" * 60)
    
    # 连接服务器
    client = create_ssh_client()
    if not client:
        sys.exit(1)
    
    try:
        # 1. 检查系统信息
        print("\n📋 步骤 1: 检查系统信息")
        run_command(client, "uname -a")
        run_command(client, "cat /etc/os-release | head -3")
        
        # 2. 检查/安装 Docker
        print("\n📋 步骤 2: 检查 Docker")
        code, out, _ = run_command(client, "docker --version 2>/dev/null || echo 'NOT_INSTALLED'")
        
        if "NOT_INSTALLED" in out:
            print("🔄 正在安装 Docker...")
            run_command(client, "curl -fsSL https://get.docker.com | sh")
            run_command(client, "systemctl start docker && systemctl enable docker")
        
        # 3. 检查/安装 Docker Compose
        print("\n📋 步骤 3: 检查 Docker Compose")
        code, out, _ = run_command(client, "docker compose version 2>/dev/null || docker-compose --version 2>/dev/null || echo 'NOT_INSTALLED'")
        
        if "NOT_INSTALLED" in out:
            print("🔄 正在安装 Docker Compose...")
            run_command(client, "apt-get update && apt-get install -y docker-compose-plugin")
        
        # 4. 检查/安装 Git
        print("\n📋 步骤 4: 检查 Git")
        run_command(client, "git --version || apt-get install -y git")
        
        # 5. 克隆或更新代码
        print("\n📋 步骤 5: 获取代码")
        run_command(client, "mkdir -p /opt/theta")
        
        code, out, _ = run_command(client, "[ -d /opt/theta/.git ] && echo 'EXISTS' || echo 'NOT_EXISTS'")
        
        if "NOT_EXISTS" in out:
            print("🔄 克隆代码仓库...")
            run_command(client, "cd /opt && rm -rf theta && git clone -b frontend-3 https://github.com/CodeSoul-co/THETA.git theta")
        else:
            print("🔄 更新代码...")
            run_command(client, "cd /opt/theta && git fetch origin && git checkout frontend-3 && git pull origin frontend-3")
        
        # 6. 创建 Docker Compose 配置
        print("\n📋 步骤 6: 创建 Docker Compose 配置")
        
        docker_compose_content = '''version: '3.8'

services:
  # 前端服务
  frontend:
    build:
      context: ./theta-frontend3
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://liguozheng.site:8000
      - NEXT_PUBLIC_DATACLEAN_API_URL=http://liguozheng.site:8001
    restart: unless-stopped
    networks:
      - theta-network

  # 后端 API 服务
  backend:
    build:
      context: .
      dockerfile: langgraph_agent/backend/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - CORS_ORIGINS=http://liguozheng.site:3000,http://liguozheng.site,https://liguozheng.site
      - SIMULATION_MODE=true
      - HOST=0.0.0.0
      - PORT=8000
    volumes:
      - ./data:/app/data
      - ./result:/app/result
    restart: unless-stopped
    networks:
      - theta-network

  # DataClean API 服务
  dataclean:
    build:
      context: ./ETM/dataclean
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - CORS_ORIGINS=http://liguozheng.site:3000,http://liguozheng.site,https://liguozheng.site
      - PORT=8001
    restart: unless-stopped
    networks:
      - theta-network

networks:
  theta-network:
    driver: bridge
'''
        
        # 写入 docker-compose.prod.yml
        run_command(client, f"cat > /opt/theta/docker-compose.prod.yml << 'EOFCOMPOSE'\n{docker_compose_content}\nEOFCOMPOSE")
        
        # 7. 创建前端 Dockerfile
        print("\n📋 步骤 7: 创建 Dockerfile")
        
        frontend_dockerfile = '''FROM node:20-alpine AS builder

WORKDIR /app

# 安装 pnpm
RUN npm install -g pnpm

# 复制 package 文件
COPY package.json pnpm-lock.yaml ./

# 安装依赖
RUN pnpm install --frozen-lockfile

# 复制源代码
COPY . .

# 设置环境变量
ENV NEXT_PUBLIC_API_URL=http://liguozheng.site:8000
ENV NEXT_PUBLIC_DATACLEAN_API_URL=http://liguozheng.site:8001

# 构建
RUN pnpm build

# 生产镜像
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# 复制构建产物
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["node", "server.js"]
'''
        
        run_command(client, f"cat > /opt/theta/theta-frontend3/Dockerfile << 'EOFDOCKER'\n{frontend_dockerfile}\nEOFDOCKER")
        
        # 8. 创建后端 Dockerfile
        backend_dockerfile = '''FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements
COPY langgraph_agent/backend/requirements.txt ./requirements.txt

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY langgraph_agent/backend/app ./app
COPY ETM ./ETM

# 设置 Python 路径
ENV PYTHONPATH=/app:/app/ETM

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        run_command(client, f"cat > /opt/theta/langgraph_agent/backend/Dockerfile << 'EOFDOCKER'\n{backend_dockerfile}\nEOFDOCKER")
        
        # 9. 检查 DataClean Dockerfile
        code, out, _ = run_command(client, "[ -f /opt/theta/ETM/dataclean/Dockerfile ] && echo 'EXISTS' || echo 'NOT_EXISTS'")
        
        if "NOT_EXISTS" in out:
            dataclean_dockerfile = '''FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements
COPY requirements.txt ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
'''
            run_command(client, f"cat > /opt/theta/ETM/dataclean/Dockerfile << 'EOFDOCKER'\n{dataclean_dockerfile}\nEOFDOCKER")
        
        # 10. 构建和启动服务
        print("\n📋 步骤 8: 构建和启动服务")
        print("⏳ 这可能需要几分钟...")
        
        run_command(client, "cd /opt/theta && docker compose -f docker-compose.prod.yml down 2>/dev/null || true")
        run_command(client, "cd /opt/theta && docker compose -f docker-compose.prod.yml build --no-cache")
        run_command(client, "cd /opt/theta && docker compose -f docker-compose.prod.yml up -d")
        
        # 11. 检查服务状态
        print("\n📋 步骤 9: 检查服务状态")
        time.sleep(10)  # 等待服务启动
        run_command(client, "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'")
        
        # 12. 配置防火墙
        print("\n📋 步骤 10: 配置防火墙")
        run_command(client, "ufw allow 3000/tcp 2>/dev/null || iptables -A INPUT -p tcp --dport 3000 -j ACCEPT 2>/dev/null || true")
        run_command(client, "ufw allow 8000/tcp 2>/dev/null || iptables -A INPUT -p tcp --dport 8000 -j ACCEPT 2>/dev/null || true")
        run_command(client, "ufw allow 8001/tcp 2>/dev/null || iptables -A INPUT -p tcp --dport 8001 -j ACCEPT 2>/dev/null || true")
        
        print("\n" + "=" * 60)
        print("✅ 部署完成!")
        print("=" * 60)
        print(f"""
🌐 访问地址:
   - 前端: http://liguozheng.site:3000
   - 后端 API: http://liguozheng.site:8000
   - DataClean API: http://liguozheng.site:8001
   - API 文档: http://liguozheng.site:8000/docs

📋 常用命令:
   - 查看日志: docker compose -f /opt/theta/docker-compose.prod.yml logs -f
   - 重启服务: docker compose -f /opt/theta/docker-compose.prod.yml restart
   - 停止服务: docker compose -f /opt/theta/docker-compose.prod.yml down
""")
        
    finally:
        client.close()

if __name__ == "__main__":
    main()

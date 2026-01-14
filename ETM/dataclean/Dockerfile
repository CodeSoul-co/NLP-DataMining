FROM python:3.11-slim

WORKDIR /app

# 配置 apt 使用国内镜像源（腾讯云优化）
RUN sed -i 's/deb.debian.org/mirrors.tencentyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.tencentyun.com/g' /etc/apt/sources.list 2>/dev/null || true

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 配置 pip 使用国内镜像源（腾讯云优化）
RUN pip config set global.index-url https://mirrors.tencent.com/pypi/simple && \
    pip config set global.trusted-host mirrors.tencent.com

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p temp_uploads temp_processed && \
    chmod 755 temp_uploads temp_processed

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]

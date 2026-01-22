# THETA Nginx 反向代理架构

## 架构概览

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   Docker Network                     │
                    │                  (theta-network)                     │
                    │                                                      │
    ┌───────┐       │  ┌─────────┐     ┌─────────────────┐                │
    │ 用户  │──80──▶│  │  Nginx  │────▶│  theta-frontend │ (:3000)        │
    │       │       │  │  (:80)  │     │   (Next.js)     │                │
    └───────┘       │  │         │     └─────────────────┘                │
                    │  │         │                                         │
                    │  │         │     ┌─────────────────┐                │
                    │  │  /api/* │────▶│  etm-agent-api  │ (:8000)        │
                    │  │         │     │    (FastAPI)    │                │
                    │  │         │     └─────────────────┘                │
                    │  │         │                                         │
                    │  │         │     ┌─────────────────┐                │
                    │  │ /data-  │────▶│  dataclean-api  │ (:8001)        │
                    │  │ clean/* │     │    (FastAPI)    │                │
                    │  └─────────┘     └─────────────────┘                │
                    │                                                      │
                    └─────────────────────────────────────────────────────┘
```

## 路由规则

| 路径 | 目标服务 | 说明 |
|------|----------|------|
| `/` | `theta-frontend:3000` | 前端 Next.js 应用 |
| `/api/*` | `etm-agent-api:8000` | ETM Agent 后端 API |
| `/api/ws` | `etm-agent-api:8000` | WebSocket 连接 |
| `/dataclean/*` | `dataclean-api:8001` | 数据清洗 API |

## 文件结构

```
THETA/
├── docker-compose.yml           # 完整部署（前端+后端+Nginx）
├── docker-compose.frontend.yml  # 仅前端部署（前端+Nginx）
└── nginx/
    ├── nginx.conf               # 完整部署的 Nginx 配置
    └── nginx.frontend.conf      # 仅前端部署的 Nginx 配置
```

## 部署命令

### 完整部署（前端 + 后端）

```bash
# 启动所有服务
docker-compose up -d --build

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 仅前端部署

```bash
# 启动前端服务
docker-compose -f docker-compose.frontend.yml up -d --build

# 或使用部署脚本
./deploy-frontend.sh
```

## 配置说明

### Nginx 配置特性

1. **WebSocket 支持**
   - `/api/ws` 路径配置了 WebSocket 长连接
   - 设置了 `Upgrade` 和 `Connection` 头
   - 超时时间设置为 7 天

2. **大文件上传**
   - `client_max_body_size 100M`
   - 支持上传大数据文件

3. **性能优化**
   - Gzip 压缩
   - 静态资源缓存
   - `sendfile` 和 `tcp_nopush` 优化

4. **健康检查**
   - `/nginx-health` 端点用于容器健康检查

### 环境变量

创建 `.env` 文件：

```bash
# API Keys
QWEN_API_KEY=your-qwen-api-key

# 安全配置
SECRET_KEY=your-secret-key

# CORS 配置
ALLOWED_ORIGINS=http://localhost,http://yourdomain.com

# 数据库
DATABASE_URL=sqlite:///./users.db
```

## 端口说明

| 服务 | 内部端口 | 外部端口 | 说明 |
|------|----------|----------|------|
| nginx | 80 | 80 | 唯一对外端口 |
| theta-frontend | 3000 | - | 内部访问 |
| etm-agent-api | 8000 | - | 内部访问 |
| dataclean-api | 8001 | - | 内部访问 |

## 故障排查

### 查看 Nginx 日志

```bash
docker logs theta-nginx
```

### 检查容器网络

```bash
docker network inspect theta_theta-network
```

### 测试 API 连接

```bash
# 测试前端
curl http://localhost/

# 测试后端 API
curl http://localhost/api/health

# 测试数据清洗 API
curl http://localhost/dataclean/health
```

### 常见问题

1. **端口 80 被占用**
   - 检查是否有其他服务占用：`sudo netstat -tlnp | grep :80`
   - 停止占用服务或修改 Nginx 端口

2. **容器无法通信**
   - 确保所有容器在同一网络：`docker network ls`
   - 检查服务名称是否正确

3. **WebSocket 连接失败**
   - 确认 Nginx 配置包含 WebSocket 头
   - 检查超时设置是否足够长

## 安全建议

1. **生产环境配置 SSL/TLS**
   ```nginx
   server {
       listen 443 ssl;
       ssl_certificate /etc/nginx/ssl/cert.pem;
       ssl_certificate_key /etc/nginx/ssl/key.pem;
       # ... 其他配置
   }
   ```

2. **限制访问来源**
   - 配置 CORS 允许的域名
   - 使用防火墙限制 IP

3. **定期更新**
   - 保持 Nginx 镜像最新
   - 定期检查安全漏洞

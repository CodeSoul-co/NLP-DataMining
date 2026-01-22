# 宝塔面板 Nginx 反向代理配置指南

## 概述

由于服务器上已运行宝塔面板（占用 80 端口），Docker 容器使用独立端口运行，通过宝塔面板的 Nginx 进行反向代理。

## 端口分配

| 服务 | Docker 容器端口 | 说明 |
|------|----------------|------|
| 前端 | 3000 | Next.js 应用 |
| ETM Agent API | 8000 | 主后端 API |
| DataClean API | 8001 | 数据清洗 API |

## 宝塔面板配置步骤

### 1. 登录宝塔面板

访问 `http://your-server-ip:8888` 登录宝塔面板。

### 2. 创建网站

1. 点击 **网站** → **添加站点**
2. 填写域名：`yourdomain.com`
3. 选择 **PHP 版本**：纯静态（或任意版本，不使用 PHP）
4. 点击 **提交**

### 3. 配置反向代理

1. 进入网站设置页面
2. 点击 **设置** → **反向代理**
3. 点击 **添加反向代理**
4. 配置如下：

#### 前端反向代理配置

```
代理名称: theta-frontend
目标URL: http://127.0.0.1:3000
发送域名: $host
```

**高级设置**（点击展开）：
```
# 添加以下配置
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

#### API 反向代理配置

```
代理名称: theta-api
目标URL: http://127.0.0.1:8000
发送域名: $host
```

**匹配路径**: `/api/`

**高级设置**：
```
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_connect_timeout 60s;
proxy_send_timeout 60s;
proxy_read_timeout 60s;
```

#### DataClean API 反向代理配置

```
代理名称: theta-dataclean
目标URL: http://127.0.0.1:8001
发送域名: $host
```

**匹配路径**: `/dataclean/`

### 4. 手动编辑 Nginx 配置（推荐）

如果通过界面配置不够灵活，可以直接编辑 Nginx 配置文件：

1. 在宝塔面板中，点击 **网站** → 找到你的网站 → **设置** → **配置文件**
2. 在 `server` 块中添加以下配置：

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    # 上传文件大小限制
    client_max_body_size 100M;
    
    # 前端（Next.js）
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        
        # WebSocket 支持
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # ETM Agent API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        
        # WebSocket 支持
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # 大文件上传支持
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    # WebSocket 专用路径
    location /api/ws {
        proxy_pass http://127.0.0.1:8000/api/ws;
        proxy_http_version 1.1;
        
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 长连接超时
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }
    
    # DataClean API
    location /dataclean/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_http_version 1.1;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 大文件上传支持
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    # Next.js 静态文件缓存
    location /_next/static {
        proxy_pass http://127.0.0.1:3000/_next/static;
        proxy_http_version 1.1;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 缓存静态资源
        proxy_cache_valid 200 60m;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }
}
```

3. 点击 **保存**，然后点击 **重载配置**

### 5. 配置 SSL（可选）

如果需要 HTTPS：

1. 点击 **SSL** → **Let's Encrypt**
2. 填写域名，点击 **申请**
3. 开启 **强制 HTTPS**

## 验证配置

### 测试前端

```bash
curl http://yourdomain.com
```

### 测试 API

```bash
curl http://yourdomain.com/api/health
```

### 测试 DataClean API

```bash
curl http://yourdomain.com/dataclean/health
```

## 常见问题

### 1. 502 Bad Gateway

- 检查 Docker 容器是否运行：`docker ps`
- 检查端口是否正确：`netstat -tlnp | grep -E '3000|8000|8001'`
- 检查防火墙是否开放端口

### 2. WebSocket 连接失败

- 确保 Nginx 配置包含 `Upgrade` 和 `Connection` 头
- 检查超时设置是否足够长

### 3. 大文件上传失败

- 确保 `client_max_body_size 100M;` 已配置
- 检查 `proxy_request_buffering off;` 是否设置

## 防火墙设置

如果使用宝塔面板防火墙，确保开放以下端口：

- **3000**: 前端（仅本地访问，不需要对外开放）
- **8000**: ETM Agent API（仅本地访问）
- **8001**: DataClean API（仅本地访问）
- **80/443**: HTTP/HTTPS（对外开放）

## 注意事项

1. **安全性**: 确保 Docker 容器端口（3000, 8000, 8001）仅监听 `127.0.0.1`，不要对外开放
2. **性能**: 使用宝塔面板的 Nginx 缓存功能可以提升性能
3. **日志**: 查看宝塔面板的网站日志可以排查问题

---

**配置完成后，访问 `http://yourdomain.com` 即可使用 THETA 系统！** 🚀

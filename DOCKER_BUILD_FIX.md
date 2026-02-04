# Docker 构建修复说明

## 问题

Docker 构建前端时失败：
```
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
...
"/app/.next/standalone": not found
```

## 原因

1. `next.config.mjs` 中 `output: 'standalone'` 被注释掉了
2. Dockerfile 期望 `.next/standalone` 目录，但构建时未生成

## 修复

### 1. 启用 standalone 输出

**文件**：`theta-frontend3/next.config.mjs`

```javascript
output: 'standalone',  // 已启用
```

### 2. Dockerfile 使用构建参数

**文件**：`theta-frontend3/Dockerfile`

- 添加 `ARG` 接收构建时环境变量
- 在构建阶段设置 `ENV`，确保 Next.js 构建时能读取到

### 3. docker-compose 传递构建参数

**文件**：`docker-compose.yml`

```yaml
frontend:
  build:
    args:
      NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL:-}  # 空字符串 = 相对路径
      NEXT_PUBLIC_DATACLEAN_API_URL: ${NEXT_PUBLIC_DATACLEAN_API_URL:-/dataclean}
```

### 4. API 客户端支持相对路径

**修复的文件**：
- `lib/api/etm-agent.ts`
- `lib/api/auth.ts`
- `lib/api/dataclean.ts`
- `hooks/use-etm-websocket.ts`
- `app/admin/monitor/page.tsx`

**逻辑**：
- 如果 `NEXT_PUBLIC_API_URL` 为空字符串，使用相对路径（浏览器自动使用当前域名）
- 如果未设置，使用默认值 `http://localhost:8000`
- WebSocket 需要完整 URL，空字符串时使用 `window.location.origin`

## 使用方式

### Docker Compose 部署（通过 nginx）

在 `.env` 中设置：
```bash
NEXT_PUBLIC_API_URL=              # 空字符串，使用相对路径
NEXT_PUBLIC_DATACLEAN_API_URL=/dataclean
```

前端请求会：
- `/api/health` → 浏览器发送到 `http://domain/api/health` → nginx → backend
- `/dataclean/health` → 浏览器发送到 `http://domain/dataclean/health` → nginx → dataclean

### 直接访问（开发或非 nginx 部署）

在 `.env` 中设置：
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001
```

## 重新构建

```bash
docker compose build --no-cache frontend
docker compose up -d
```

## 验证

1. 前端容器启动后，检查日志：
   ```bash
   docker logs theta-frontend
   ```

2. 访问前端，检查浏览器控制台网络请求：
   - 应该看到请求发送到 `/api/...`（相对路径）
   - 或 `http://domain/api/...`（如果设置了完整域名）

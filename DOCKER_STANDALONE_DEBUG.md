# Docker Standalone 构建调试指南

## 问题

Docker 构建时 `.next/standalone` 目录未生成，即使 `next.config.mjs` 中已设置 `output: 'standalone'`。

## 已实施的修复

### 1. 配置检查
- ✅ `next.config.mjs` 中已设置 `output: 'standalone'`
- ✅ Dockerfile 中添加了 `ENV NEXT_PRIVATE_STANDALONE=true`
- ✅ 添加了构建前配置验证步骤

### 2. 调试步骤
Dockerfile 现在包含详细的调试输出：
- 验证 `next.config.mjs` 配置
- 显示构建过程
- 检查 `.next/standalone` 目录是否存在
- 验证 `server.js` 文件是否存在

## 排查步骤

### 步骤 1: 检查构建日志

重新构建并查看详细输出：
```bash
docker compose build --no-cache frontend 2>&1 | tee build.log
```

查找以下信息：
- `=== Verifying next.config.mjs ===` - 确认配置正确
- `=== Starting build ===` - 构建开始
- `=== Build completed ===` - 构建完成
- `=== Checking for standalone ===` - standalone 目录检查结果

### 步骤 2: 检查构建错误

如果 standalone 目录未生成，检查：
1. **构建是否成功完成**：查找 `Build completed` 消息
2. **是否有错误**：检查构建日志中的 `ERROR` 或 `failed`
3. **Next.js 版本**：确认使用的是 Next.js 16.0.10

### 步骤 3: 本地测试

在本地测试 standalone 构建：
```bash
cd theta-frontend3
pnpm build
ls -la .next/standalone
```

如果本地也不生成，可能是：
- Next.js 配置问题
- 依赖问题
- 代码错误导致构建失败

### 步骤 4: 使用备用方案

如果 standalone 模式持续失败，可以使用备用 Dockerfile：
```bash
# 在 docker-compose.yml 中临时修改
frontend:
  build:
    dockerfile: Dockerfile.alternative
```

备用方案使用标准的 Next.js 输出，需要复制 `node_modules`，镜像会更大但更稳定。

## 可能的原因

1. **Next.js 16.0.10 的 bug**：某些情况下 standalone 模式可能不工作
2. **构建错误**：构建过程中有错误但被忽略（`ignoreBuildErrors: true`）
3. **环境变量问题**：某些环境变量导致 standalone 模式被禁用
4. **依赖问题**：某些依赖与 standalone 模式不兼容

## 解决方案

### 方案 A: 修复 standalone（推荐）

1. 检查构建日志，找出具体错误
2. 修复导致 standalone 未生成的错误
3. 重新构建

### 方案 B: 使用备用 Dockerfile

如果 standalone 模式无法修复：
```yaml
# docker-compose.yml
frontend:
  build:
    context: ./theta-frontend3
    dockerfile: Dockerfile.alternative  # 使用备用方案
```

### 方案 C: 升级 Next.js

如果确认是 Next.js 16.0.10 的 bug，考虑升级到更新版本：
```bash
cd theta-frontend3
pnpm add next@latest
```

## 验证

构建成功后，验证容器：
```bash
docker compose up -d frontend
docker exec theta-frontend ls -la /app/.next/standalone
docker logs theta-frontend
```

访问前端，确认服务正常运行。

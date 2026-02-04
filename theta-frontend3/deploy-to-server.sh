#!/bin/bash
# 前端部署到香港轻量服务器
# 服务器: 47.86.49.93 (code-soul.com)
# 目录: /www/wwwroot/theta.code-soul.com
# 使用: ./deploy-to-server.sh   （需先配置 SSH 免密或运行后输入 root 密码）

set -e
DEPLOY_HOST="root@47.86.49.93"
DEPLOY_PATH="/www/wwwroot/theta.code-soul.com"

echo "==> 1. 安装依赖..."
pnpm install --frozen-lockfile

echo "==> 2. 构建..."
pnpm build

echo "==> 3. 准备 standalone 输出..."
# standalone 需手动复制 static 和 public
cp -r .next/static .next/standalone/.next/static
cp -r public .next/standalone/public

echo "==> 4. 上传到服务器..."
rsync -avz --delete \
  .next/standalone/ \
  "${DEPLOY_HOST}:${DEPLOY_PATH}/"

echo "==> 5. 确保远程目录存在..."
ssh "${DEPLOY_HOST}" "mkdir -p ${DEPLOY_PATH}"

echo "==> 6. 远程重启（若已用 pm2 托管）..."
ssh "${DEPLOY_HOST}" "cd ${DEPLOY_PATH} && (pm2 delete theta-frontend 2>/dev/null; true); pm2 start server.js --name theta-frontend 2>/dev/null || echo '提示: 请 SSH 登录服务器后执行: cd ${DEPLOY_PATH} && PORT=3000 node server.js 或使用 pm2'"

echo "==> 部署完成. 请确认服务器已监听端口（如 3000）且 nginx 已反向代理至 theta.code-soul.com"

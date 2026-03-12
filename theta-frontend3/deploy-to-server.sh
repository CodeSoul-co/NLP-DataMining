#!/bin/bash
# THETA 前端部署到香港轻量服务器
# 服务器: 47.86.49.93 (code-soul.com)
# 配置: 2vCPU 4GiB / 50GiB ESSD / 200Mbps
# 使用: ./deploy-to-server.sh   （需先配置 SSH 免密或运行后输入 root 密码）

set -e
DEPLOY_HOST="root@47.86.49.93"
DEPLOY_PATH="/www/wwwroot/theta.code-soul.com"

echo "==> 1. 安装依赖..."
pnpm install --frozen-lockfile

echo "==> 2. 构建（读取 .env.production 若存在）..."
pnpm build

echo "==> 3. 准备 standalone 输出..."
cp -r .next/static .next/standalone/.next/static
cp -r public .next/standalone/public

echo "==> 4. 确保远程目录存在..."
ssh "${DEPLOY_HOST}" "mkdir -p ${DEPLOY_PATH}"

echo "==> 5. 上传到服务器..."
rsync -avz --delete \
  .next/standalone/ \
  "${DEPLOY_HOST}:${DEPLOY_PATH}/"

echo "==> 6. 远程重启 pm2..."
ssh "${DEPLOY_HOST}" "cd ${DEPLOY_PATH} && (pm2 delete theta-frontend 2>/dev/null; true) && pm2 start server.js --name theta-frontend && pm2 save" || echo "提示: 若 pm2 未安装，请 SSH 登录后执行: cd ${DEPLOY_PATH} && PORT=3000 node server.js"

echo "==> 部署完成. 访问 https://theta.code-soul.com"

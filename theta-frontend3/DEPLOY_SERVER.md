# 前端部署到香港轻量服务器

## 服务器信息
- **公网 IP**: 47.86.49.93
- **域名**: code-soul.com（前端子站建议 theta.code-soul.com）
- **目录**: /www/wwwroot/theta.code-soul.com
- **root 密码**: （请勿提交到仓库，本地保管）

## 一、本地已完成的步骤
- 已执行 `pnpm install` 和 `pnpm build`
- 已生成 standalone 输出（.next/standalone 并包含 static、public）

## 二、推送到服务器

### 方式 A：一键脚本（需本机可 SSH 到服务器）
```bash
cd theta-frontend3
./deploy-to-server.sh
# 按提示输入 root 密码（若未配置 SSH 免密）
```

### 方式 B：手动执行
```bash
cd theta-frontend3

# 1. 若未构建，先构建并准备 standalone
pnpm install --frozen-lockfile
pnpm build
cp -r .next/static .next/standalone/.next/static
cp -r public .next/standalone/public

# 2. 上传
rsync -avz --delete .next/standalone/ root@47.86.49.93:/www/wwwroot/theta.code-soul.com/

# 3. SSH 登录服务器后启动
ssh root@47.86.49.93
cd /www/wwwroot/theta.code-soul.com
# 使用 pm2（推荐）
pm2 delete theta-frontend 2>/dev/null; pm2 start server.js --name theta-frontend
# 或直接运行（调试用）
PORT=3000 node server.js
```

## 三、服务器上建议配置
1. **Node.js**: 需安装 Node 18+（如 `nvm install 18`）
2. **pm2**: `npm i -g pm2`，便于守护进程与开机自启
3. **Nginx 反向代理**（示例，域名 theta.code-soul.com 指到 47.86.49.93）:
   ```nginx
   server {
       listen 80;
       server_name theta.code-soul.com;
       location / {
           proxy_pass http://127.0.0.1:3000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```
4. **HTTPS**: 使用 certbot 或宝塔等为 theta.code-soul.com 申请证书并开启 443

## 四、后续更新
每次改完前端后，在本地执行：
```bash
cd theta-frontend3
pnpm build
cp -r .next/static .next/standalone/.next/static && cp -r public .next/standalone/public
rsync -avz --delete .next/standalone/ root@47.86.49.93:/www/wwwroot/theta.code-soul.com/
ssh root@47.86.49.93 "cd /www/wwwroot/theta.code-soul.com && pm2 restart theta-frontend"
```

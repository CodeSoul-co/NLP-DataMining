# THETA 前端部署 - 香港轻量服务器

## 服务器信息

| 项目 | 值 |
|------|-----|
| 公网 IP | 47.86.49.93 |
| 域名 | code-soul.com（已绑定） |
| 前端子域 | theta.code-soul.com（建议） |
| 配置 | 2vCPU 4GiB / 50GiB ESSD / 200Mbps |
| root 密码 | Codesoul120@（请勿提交到仓库） |

---

## 一、首次服务器初始化（SSH 登录后执行）

```bash
# 1. 安装 Node.js 18+（推荐 nvm）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20

# 2. 安装 pnpm 和 pm2
npm i -g pnpm pm2

# 3. 创建部署目录
mkdir -p /www/wwwroot/theta.code-soul.com

# 4. 安装 Nginx（若未安装）
# Ubuntu/Debian:
apt update && apt install -y nginx

# 5. 配置 Nginx 反向代理（静态资源由 nginx 直供，避免 Next.js standalone 404）
cat > /etc/nginx/sites-available/theta.code-soul.com << 'EOF'
server {
    listen 80;
    server_name theta.code-soul.com;
    root /www/wwwroot/theta.code-soul.com/public;

    # 案例库图片、头像等静态资源由 nginx 直供（解决部署后图片不加载）
    location /papers/ {
        alias /www/wwwroot/theta.code-soul.com/public/papers/;
        expires 7d;
    }
    location /avatars/ {
        alias /www/wwwroot/theta.code-soul.com/public/avatars/;
        expires 7d;
    }
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
EOF

ln -sf /etc/nginx/sites-available/theta.code-soul.com /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# 6. HTTPS（可选，推荐 certbot）
apt install -y certbot python3-certbot-nginx
certbot --nginx -d theta.code-soul.com
```

**若已部署过，仅需修复案例库图片不加载**：在服务器执行步骤 5 的 `cat > ...` 覆盖 nginx 配置，然后 `nginx -t && systemctl reload nginx`。

---

## 二、DNS 配置

在域名服务商添加 A 记录：

| 类型 | 主机记录 | 记录值 |
|------|----------|--------|
| A | theta | 47.86.49.93 |

或若主域为 code-soul.com，则 theta.code-soul.com 解析到 47.86.49.93。

---

## 三、环境变量（构建时生效）

在 `theta-frontend3` 目录创建 `.env.production`（勿提交到 Git）：

```env
# 后端 API 地址（替换为实际部署的后端地址）
NEXT_PUBLIC_API_URL=https://api.code-soul.com
NEXT_PUBLIC_AGENT_URL=https://api.code-soul.com
```

若后端与前端同机，可用：
```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_AGENT_URL=http://127.0.0.1:8000
```

---

## 四、本地推送部署

### 方式 A：一键脚本（推荐）

```bash
cd theta-frontend3

# 首次需配置 SSH 免密（可选）
ssh-copy-id root@47.86.49.93

# 执行部署
./deploy-to-server.sh
```

### 方式 B：手动步骤

```bash
cd theta-frontend3

# 1. 安装依赖并构建（会读取 .env.production）
pnpm install --frozen-lockfile
pnpm build

# 2. 准备 standalone 输出
cp -r .next/static .next/standalone/.next/static
cp -r public .next/standalone/public

# 3. 上传
rsync -avz --delete .next/standalone/ root@47.86.49.93:/www/wwwroot/theta.code-soul.com/

# 4. 远程重启
ssh root@47.86.49.93 "cd /www/wwwroot/theta.code-soul.com && pm2 delete theta-frontend 2>/dev/null; pm2 start server.js --name theta-frontend && pm2 save"
```

---

## 五、pm2 开机自启

```bash
ssh root@47.86.49.93
pm2 startup
pm2 save
```

---

## 六、常用命令

```bash
# 查看运行状态
ssh root@47.86.49.93 "pm2 status"

# 查看日志
ssh root@47.86.49.93 "pm2 logs theta-frontend"

# 重启
ssh root@47.86.49.93 "pm2 restart theta-frontend"
```

---

## 七、访问地址

部署成功后访问：**https://theta.code-soul.com**（或 http，若未配置 HTTPS）

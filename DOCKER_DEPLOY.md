# Docker 部署指南（全栈：Nginx + 前端 + 后端 + DataClean + PostgreSQL + Redis）

本指南适用于在服务器上使用 **根目录 `docker-compose.yml`** 进行一次性构建与部署。

## 前置要求

- Linux（Ubuntu 20.04+ / Debian 11+ / CentOS 7+），root 或 sudo
- Docker 与 Docker Compose（v2: `docker compose` 或 v1: `docker-compose`）
- 至少 2GB RAM，10GB 磁盘

## 一、安装 Docker 与 Docker Compose

### Ubuntu / Debian

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo systemctl enable --now docker
docker --version && docker-compose --version
```

### CentOS / RHEL

```bash
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo systemctl enable --now docker
docker --version && docker-compose --version
```

## 二、克隆项目

```bash
cd /opt
sudo git clone <你的仓库地址> THETA
cd THETA
git checkout frontend-3   # 如需要
```

## 三、配置环境变量（必做）

```bash
cp docker.env.template .env
nano .env   # 或 vi / vim
```

**必须修改的项：**

| 变量 | 说明 |
|------|------|
| `QWEN_API_KEY` | 千问 API Key，否则 AI 相关功能不可用 |
| `POSTGRES_PASSWORD` | 数据库密码，生产环境务必改成强密码 |
| `SECRET_KEY` | JWT 等认证密钥，请改为随机字符串 |
| `DOMAIN` | 对外域名，用于 CORS；仅本机可填 `localhost` |

**可选（按需）：**

- `NEXT_PUBLIC_API_URL`、`NEXT_PUBLIC_DATACLEAN_API_URL`：前端通过 Nginx 访问时，若使用域名，需在 **构建前端镜像前** 通过 build-arg 传入（当前 Dockerfile 未做则沿用编译时默认；仅暴露 80/443 时，用相对路径或同域即可）。
- `ALLOWED_ORIGINS`：若不用 `docker.env.template` 的 CORS，可在后端/DataClean 环境变量中单独配置。
- OSS / PAI / EAS：按需填写。

**注意：** `.env` 已在 `.gitignore`，切勿 `git add .env`。

## 四、目录与 Nginx 证书目录

```bash
mkdir -p ETM/dataclean/temp_uploads ETM/dataclean/temp_processed
mkdir -p nginx/certs data result
# nginx/certs：未配置 HTTPS 时可为空；启用 HTTPS 时放入 fullchain.pem、privkey.pem
# data、result：后端挂载目录，可留空由 compose 自动创建
```

当前 `nginx/nginx.conf` 默认只启用 HTTP（80），未启用 SSL；如需 HTTPS，需在 `nginx.conf` 中取消 443 与 `ssl_certificate` 等注释，并在 `nginx/certs` 中放置证书。

## 五、构建与启动（重新构建部署）

```bash
# 推荐：无缓存构建，保证用到最新代码与依赖
docker compose build --no-cache
# 或旧版： docker-compose build --no-cache

docker compose up -d
# 或： docker-compose up -d
```

如需先停掉旧容器再起：

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 六、一键脚本（可选）

```bash
chmod +x docker-deploy.sh
./docker-deploy.sh
```

脚本会：检查 Docker、若缺 `.env` 则从 `docker.env.template` 复制、创建目录、`build --no-cache`、`up -d`，并对 Nginx / 后端 / 前端做简单健康检查。

## 七、验证

- **Nginx：** `curl -s http://localhost/health` 应返回 `OK`
- **后端：** `curl -s http://localhost/api/health`
- **前端：** 浏览器访问 `http://服务器IP` 或 `http://域名`

通过 Nginx 的访问路径：

- 前端：`/`
- 后端 API：`/api/`
- DataClean：`/dataclean/`
- WebSocket：`/api/ws`

## 八、常用命令

```bash
docker compose ps
docker compose logs -f
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f dataclean

docker compose down
docker compose up -d
docker compose restart
```

## 九、更新代码后重新构建部署

```bash
git pull
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 十、故障排查

- **构建失败：** `docker compose build --no-cache`，查看 `docker compose logs build 出错的服务名`。
- **后端连不上 DB/Redis：** 等 `db`、`redis` 健康后再起 `backend`；看 `docker compose logs backend`。
- **前端 404/接口不对：** 确认 Nginx 将 `/api/`、`/dataclean/` 正确反代；若改过 `NEXT_PUBLIC_*`，需重新 `build` 前端镜像。
- **QWEN 报错：** 在 `.env` 中正确设置 `QWEN_API_KEY` 并重启：`docker compose up -d backend`。

## 十一、服务与端口（compose 内）

| 服务 | 容器名 | 内部端口 | 说明 |
|------|--------|----------|------|
| nginx | theta-nginx | 80, 443 | 反向代理，对外暴露 80/443 |
| frontend | theta-frontend | 3000 | Next.js |
| backend | theta-backend | 8000 | FastAPI（langgraph_agent/backend） |
| dataclean | theta-dataclean | 8001 | DataClean API |
| db | theta-db | 5432 | PostgreSQL |
| redis | theta-redis | 6379 | Redis |

---

部署完成后，可通过 **http://服务器IP** 或 **http://你的域名** 访问前端；API 为 **/api/**、**/dataclean/**。

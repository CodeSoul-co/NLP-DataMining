FROM node:20-alpine AS base

# 配置 npm 使用国内镜像源（腾讯云优化）
RUN npm config set registry https://mirrors.tencent.com/npm/

# 安装 pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# 配置 pnpm 使用国内镜像源
RUN pnpm config set registry https://mirrors.tencent.com/npm/

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY package.json pnpm-lock.yaml ./

# 安装依赖
RUN pnpm install --frozen-lockfile

# 复制源代码
COPY . .

# 构建参数（API URL）
ARG NEXT_PUBLIC_DATACLEAN_API_URL=http://dataclean-api:8001
ENV NEXT_PUBLIC_DATACLEAN_API_URL=$NEXT_PUBLIC_DATACLEAN_API_URL

# 构建应用
RUN pnpm build

# 生产镜像
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# 配置 npm 使用国内镜像源
RUN npm config set registry https://mirrors.tencent.com/npm/

# 安装 pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# 配置 pnpm 使用国内镜像源
RUN pnpm config set registry https://mirrors.tencent.com/npm/

# 复制必要文件
COPY --from=base /app/package.json ./
COPY --from=base /app/pnpm-lock.yaml ./
COPY --from=base /app/.next ./.next
COPY --from=base /app/public ./public
COPY --from=base /app/node_modules ./node_modules

# 创建非 root 用户
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs
USER nextjs

# 暴露端口
EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# 启动命令
CMD ["pnpm", "start"]

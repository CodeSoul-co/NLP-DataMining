FROM node:20-alpine AS base

# 安装 pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY package.json pnpm-lock.yaml ./

# 安装依赖
RUN pnpm install --frozen-lockfile

# 复制配置文件
COPY tsconfig.json next.config.mjs postcss.config.mjs components.json ./

# 复制源代码目录（确保 lib 目录被包含）
COPY app ./app
COPY components ./components
COPY lib ./lib
COPY public ./public
COPY styles ./styles
COPY hooks ./hooks

# 构建参数（API URL）
ARG NEXT_PUBLIC_DATACLEAN_API_URL=http://dataclean-api:8001
ENV NEXT_PUBLIC_DATACLEAN_API_URL=$NEXT_PUBLIC_DATACLEAN_API_URL

# 构建应用
RUN pnpm build

# 生产镜像
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# 安装 pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

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

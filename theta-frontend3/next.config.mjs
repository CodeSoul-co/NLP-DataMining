/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // 禁用严格模式以避免双重渲染导致的状态问题
  reactStrictMode: false,
  // Turbopack 配置 (Next.js 16+)
  turbopack: {
    // 配置 HMR WebSocket
    resolveAlias: {},
  },
  // 开发服务器配置
  devIndicators: {
    buildActivity: false,
  },
  // 优化字体加载
  optimizeFonts: true,
  // 环境变量配置 (Railway 部署)
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_DATACLEAN_API_URL: process.env.NEXT_PUBLIC_DATACLEAN_API_URL || 'http://localhost:8001',
  },
  // 输出配置 - standalone 模式优化 Docker/Railway 部署
  output: 'standalone',
}

export default nextConfig

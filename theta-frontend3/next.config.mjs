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
}

export default nextConfig

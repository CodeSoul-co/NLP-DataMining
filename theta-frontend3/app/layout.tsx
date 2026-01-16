import type React from "react"
import type { Metadata } from "next"
import { Analytics } from "@vercel/analytics/next"
import "./globals.css"

// 使用 CSS 变量定义字体，避免构建时访问 Google Fonts
const fontClassName = "font-sans"

export const metadata: Metadata = {
  title: "THETA 智能分析平台",
  description: "Enterprise-grade intelligent data analysis platform",
  generator: "v0.app",
  icons: {
    icon: [
      {
        url: "/icon-light-32x32.png",
        media: "(prefers-color-scheme: light)",
      },
      {
        url: "/icon-dark-32x32.png",
        media: "(prefers-color-scheme: dark)",
      },
      {
        url: "/icon.svg",
        type: "image/svg+xml",
      },
    ],
    apple: "/apple-icon.png",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN">
      <body className={`${fontClassName} antialiased`}>
        {children}
        <Analytics />
      </body>
    </html>
  )
}

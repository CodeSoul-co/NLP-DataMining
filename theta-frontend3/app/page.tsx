"use client"

import type React from "react"
import { useState, useEffect, useCallback, useRef } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import {
  Play,
  Sparkles,
  BrainCircuit,
  MessageSquare,
  Paperclip,
  Send,
  CheckCircle2,
  Eye,
  EyeOff,
  Loader2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  FileSpreadsheet,
  MessageCircle,
  BarChart3,
  FileDown,
  Shield,
  Globe,
  FileText,
  Minus,
  Plus,
  Infinity,
  Zap,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { LineChart, Line, XAxis, ResponsiveContainer } from "recharts"
import { useAuth } from "@/contexts/auth-context"
import { ETMAgentAPI } from "@/lib/api/etm-agent"
import type { ChatMessage } from "@/components/chat/ai-sidebar"
import { TypingMessage } from "@/components/typing-message"
import { useCyclingTypewriter } from "@/hooks/use-cycling-typewriter"
import { ParticlesBg } from "@/components/particles-bg"

function getTimestamp() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
}
function generateId() {
  return `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

const LANDING_GREETING = "您好！我已经准备好分析您的数据。请上传文件或直接提问。"

/** 首页标题下打字机循环展示的平台功能文案（每条不同，轮流出现） */
const HERO_TYPEWRITER_PHRASES = [
  "数据清洗与预处理",
  "主题模型 (ETM) 训练与评估",
  "智能对话与 AI 科研助手",
  "任务中心 · 异步训练与监控",
  "可视化与结果导出",
  "从上传到洞察的一站式分析",
  "多数据集管理与协作",
  "深度主题发现与词云展示",
]

/** 使用教程四步：图+文同步切换 */
const HOW_IT_WORKS_STEPS = [
  {
    title: "多源数据，一键清洗",
    titleEn: "Data Ingestion",
    text: "支持拖拽上传 Excel、CSV、PDF 及 JSONL 等多格式文件。系统将自动识别字段并完成智能清洗，让繁琐的数据预处理一步到位。",
    icon: FileSpreadsheet,
  },
  {
    title: "自然语言，对话分析",
    titleEn: "Interactive Analysis",
    text: "无需编程，对话即分析。只需用自然语言提问（如「分析近三个月负面情绪的主题」），AI 即可实时解析数据并生成可视化的深度洞察。",
    icon: MessageCircle,
  },
  {
    title: "图表交互，深挖归因",
    titleEn: "Drill-down Insight",
    text: "图表即入口，点击即可追溯原因。发现数据异常或波峰？直接点击图表上的关键点，AI 将自动定位原始文本，并解读数据波动背后的具体成因。",
    icon: BarChart3,
  },
  {
    title: "学术级报告，一键导出",
    titleEn: "Export & Reporting",
    text: "支持下载高清矢量图与完整分析文档。输出格式符合学术出版标准，无缝衔接您的论文撰写或行业研报制作。",
    icon: FileDown,
  },
]

/** 场景化分析实验室：上三下二 */
const SCENARIO_LAB_ROW1 = [
  {
    title: "心理健康与精细情感图谱",
    titleEn: "Mental Health & Micro-Emotion Mapping",
    tags: ["精神疾病类型", "负面情感检测"],
    icon: BrainCircuit,
  },
  {
    title: "金融合规与客诉风险洞察",
    titleEn: "Financial Compliance & Risk Insights",
    tags: ["FCPB 投诉分析", "风险洞察"],
    icon: Shield,
  },
  {
    title: "数字内容安全与净化",
    titleEn: "Content Moderation & Digital Safety",
    tags: ["仇恨言论识别", "垃圾账户过滤"],
    icon: MessageSquare,
  },
]
const SCENARIO_LAB_ROW2 = [
  {
    title: "跨语言与多文化语义分析",
    titleEn: "Cross-Lingual & Multicultural Analysis",
    tags: ["多语言混合处理", "德语专业文本"],
    icon: Globe,
  },
  {
    title: "长文本宏观语义理解",
    titleEn: "Long-Context Macro Understanding",
    tags: ["政治演讲全篇解析", "长帖语义聚合"],
    icon: FileText,
  },
]

/** 价格方案：三档（占位，可按需改价格） */
const PRICING_PLANS = [
  { name: "入门", desc: "个人课程作业", priceMonth: 0, priceYear: 0, features: ["小规模数据", "基础分析"], recommended: false },
  { name: "专业", desc: "科研与项目", priceMonth: 99, priceYear: 999, features: ["大规模数据", "完整功能", "优先支持"], recommended: true },
  { name: "企业", desc: "团队与定制", priceMonth: 299, priceYear: 2999, features: ["私有部署", "定制模型", "专属客服"], recommended: false },
]

/** FAQ 问答 */
const FAQ_ITEMS = [
  {
    q: "Theta 到底是什么？不需要编程也能用吗？",
    a: "Theta 是一个专为社会科学研究设计的 AI 分析平台。我们致力于降低科研门槛，您完全不需要编程基础。通过直观的对话界面，即可完成从数据清洗、主题建模到深度文本挖掘的全流程工作，让您专注于研究思路本身。",
  },
  {
    q: "我的数据安全吗？会被用于训练 AI 吗？",
    a: "这是我们最重视的原则。我们遵循严格的数据隐私协议（GDPR Compliant）。您上传的数据经过加密处理，仅供您当次分析使用，分析结束后数据支持一键销毁。我们承诺：绝不会将您的私有数据用于训练公共模型或分享给第三方。",
  },
  {
    q: "生成图表可以直接用于论文发表吗？",
    a: "完全可以。Theta 生成的所有可视化图表（矢量图）均符合主流学术期刊的出版标准。此外，我们还会提供详细的算法引用说明，方便您在论文的方法论部分准确撰写引用来源。",
  },
  {
    q: "支持多大的文件？支持哪些语言？",
    a: "我们专为大规模数据设计。数据量：支持 500MB 的 CSV/Excel 表格处理，轻松应对海量数据。语言支持：深度支持中文、英文及中英混合文本分析，同时兼容德语、法语等 20+ 种主流语言的语义理解。",
  },
]

export default function LandingPage() {
  const router = useRouter()
  const { user, loading: authLoading, login } = useAuth()
  const [showLoginModal, setShowLoginModal] = useState(false)
  const [loginUsername, setLoginUsername] = useState("")
  const [loginPassword, setLoginPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [rememberMe, setRememberMe] = useState(false)
  const [loginError, setLoginError] = useState("")
  const [isLoggingIn, setIsLoggingIn] = useState(false)
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [chatInputValue, setChatInputValue] = useState("")
  const [isAiLoading, setIsAiLoading] = useState(false)
  const chatCardRef = useRef<HTMLDivElement>(null)
  const [howItWorksStep, setHowItWorksStep] = useState(0)
  const [pricingMode, setPricingMode] = useState<"per-use" | "monthly" | "yearly">("per-use")
  const [faqOpenIndex, setFaqOpenIndex] = useState<number | null>(null)
  const [showCiteModal, setShowCiteModal] = useState(false)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)

  const howItWorksCurrent = HOW_IT_WORKS_STEPS[howItWorksStep]
  const HowItWorksStepIcon = howItWorksCurrent.icon

  const { displayedText: typewriterText } = useCyclingTypewriter({
    phrases: HERO_TYPEWRITER_PHRASES,
    typingSpeed: 90,
    deleteSpeed: 60,
    holdDuration: 1600,
    loop: true,
  })

  // Load remembered username
  useEffect(() => {
    const rememberedUsername = localStorage.getItem('remembered_username')
    if (rememberedUsername) {
      setLoginUsername(rememberedUsername)
      setRememberMe(true)
    }
  }, [])

  // 已登录用户停留在起始页，不自动跳转到 dashboard（仅通过点击「立即开始」「免费使用」或三个快捷卡片时跳转）
  // 若出现「过几秒自动跳到项目管理界面」，多半是误触了上述入口，此处不做任何自动跳转

  const handleStartAnalysis = () => {
    if (user) {
      // Already logged in, go to dashboard
      router.push("/dashboard")
    } else {
      // Not logged in, show login modal
      setShowLoginModal(true)
    }
  }

  const handleLogin = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    
    if (!loginUsername.trim() || !loginPassword.trim()) {
      setLoginError("请输入用户名和密码")
      return
    }

    setIsLoggingIn(true)
    setLoginError("")

    try {
      await login(loginUsername, loginPassword, rememberMe)
      
      // Save remembered username if selected
      if (rememberMe) {
        localStorage.setItem('remembered_username', loginUsername)
      } else {
        localStorage.removeItem('remembered_username')
      }
      
      setShowLoginModal(false)
      setLoginUsername("")
      setLoginPassword("")
      // Redirect to dashboard after successful login
      router.push("/dashboard")
    } catch (err: unknown) {
      let errorMessage = "登录失败，请检查用户名和密码"
      if (err instanceof Error) {
        errorMessage = err.message || errorMessage
        const isConnectionError =
          errorMessage.includes('连接') ||
          errorMessage.includes('超时') ||
          errorMessage.includes('fetch') ||
          errorMessage.includes('network') ||
          errorMessage.includes('Failed to fetch')
        if (isConnectionError) {
          errorMessage =
            '无法连接后端（未启动或地址错误）。请先启动后端：在项目根目录运行 ./start.sh，或按「前后端完成与对接情况.md」分步启动后端（端口 8000）。'
        } else if (errorMessage.includes('401') || errorMessage.includes('Unauthorized') || errorMessage.includes('用户名或密码错误')) {
          errorMessage = '用户名或密码错误'
        }
      }
      setLoginError(errorMessage)
    } finally {
      setIsLoggingIn(false)
    }
  }

  const handleGoToRegister = () => {
    setShowLoginModal(false)
    router.push("/register")
  }

  // 与 dashboard 一致的对话：真实 Qwen 接口 + 打字机效果
  const handleLandingChatSend = useCallback(async (content: string) => {
    if (!content.trim() || isAiLoading) return
    const userMessage: ChatMessage = {
      id: generateId(),
      role: "user",
      content: content.trim(),
      type: "text",
      timestamp: getTimestamp(),
    }
    setChatHistory((prev) => [...prev, userMessage])
    setChatInputValue("")
    setIsAiLoading(true)
    try {
      const response = await ETMAgentAPI.chat(content.trim(), {
        current_page: "landing",
        current_view_name: "首页",
        current_view: "landing",
        app_state: "idle",
        datasets_count: 0,
        datasets: [],
      })
      const text = response.message ?? (response as { response?: string }).response ?? ""
      const aiMessage: ChatMessage = {
        id: generateId(),
        role: "ai",
        content: text,
        type: "text",
        timestamp: getTimestamp(),
      }
      setChatHistory((prev) => [...prev, aiMessage])
    } catch {
      const aiMessage: ChatMessage = {
        id: generateId(),
        role: "ai",
        content: "无法连接服务，请稍后重试或登录后使用完整功能。",
        type: "text",
        timestamp: getTimestamp(),
      }
      setChatHistory((prev) => [...prev, aiMessage])
    } finally {
      setIsAiLoading(false)
    }
  }, [isAiLoading])

  const lastAiMessageId = chatHistory.length > 0
    ? [...chatHistory].reverse().find((m) => m.role === "ai")?.id
    : undefined

  const [authLoadTimeout, setAuthLoadTimeout] = useState(false)
  useEffect(() => {
    if (!authLoading) return
    const t = setTimeout(() => setAuthLoadTimeout(true), 4000)
    return () => clearTimeout(t)
  }, [authLoading])
  const showContent = !authLoading || authLoadTimeout

  if (typeof window !== "undefined" && !showContent) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-10 h-10 animate-spin text-blue-600" />
          <p className="text-slate-500 text-sm">加载中...</p>
        </div>
      </div>
    )
  }

  return (
    <div
      className="min-h-screen relative"
      onMouseMove={(e) => setMousePos({ x: e.clientX, y: e.clientY })}
      onMouseLeave={() => setMousePos(null)}
    >
      {/* 全页背景特效：渐变 + 点阵 + 柔光球 */}
      <div className="page-bg-effect" aria-hidden>
        <div className="page-bg-effect__gradient" />
        <div className="page-bg-effect__dots" />
        <div className="page-bg-effect__orb page-bg-effect__orb--1" />
        <div className="page-bg-effect__orb page-bg-effect__orb--2" />
        <div className="page-bg-effect__orb page-bg-effect__orb--3" />
        <div className="page-bg-effect__orb page-bg-effect__orb--4" />
        <div className="page-bg-effect__orb page-bg-effect__orb--5" />
      </div>
      {/* 粒子连线背景：点 + 线，鼠标靠近时连线并轻微吸附 */}
      <ParticlesBg zIndex={-1} opacity={0.4} color="59, 130, 246" count={80} />
      {/* 鼠标跟随光晕：独立层叠在内容之上，pointer-events: none 不挡点击 */}
      {mousePos && (
        <div
          className="page-bg-effect__mouse-glow"
          style={{ left: mousePos.x, top: mousePos.y }}
          aria-hidden
        />
      )}
      {/* Navigation Bar */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="sticky top-0 z-50 bg-white/98 backdrop-blur-md border-b border-slate-200/80 shadow-sm shadow-slate-900/5"
      >
        <div className="max-w-7xl mx-auto px-5 sm:px-6 h-14 sm:h-16 flex items-center justify-between">
          {/* Left: Logo - 使用上传的 THETA 图片（青绿+橙边） */}
          <div className="flex items-center">
            <img src="/theta-logo.png" alt="THETA" className="h-8 sm:h-9 w-auto object-contain" />
          </div>

          {/* Center: Navigation Links */}
          <nav className="hidden md:flex items-center gap-7">
            {[
              { label: "首页", href: "#" },
              { label: "关于THETA", href: "#core-features" },
              { label: "案例库", href: "#cases" },
              { label: "文档", href: "#docs" },
              { label: "团队成员", href: "/team" },
              { label: "帮助中心", href: "#faq" },
            ].map((link) => (
              <a
                key={link.label}
                href={link.href}
                className="nav-link text-[13px] sm:text-sm"
              >
                {link.label}
              </a>
            ))}
          </nav>

          {/* Right: Open Source Links + Auth Buttons */}
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/CodeSoul-co/THETA"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-link flex items-center gap-1.5 text-[13px] sm:text-sm"
              title="GitHub"
            >
              <img
                src="/github-mark.svg"
                alt="GitHub"
                width={18}
                height={18}
                className="h-4 w-4 sm:h-[18px] sm:w-[18px] object-contain shrink-0"
              />
              <span className="hidden sm:inline">GitHub</span>
            </a>
            <a
              href="https://huggingface.co/organizations/CodeSoulco"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-link flex items-center gap-1.5 text-[13px] sm:text-sm"
              title="Hugging Face - CodeSoul"
              aria-label="打开 Hugging Face CodeSoul 组织页"
            >
              <img
                src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
                alt="Hugging Face"
                className="h-4 w-4 sm:h-[18px] sm:w-[18px] object-contain"
              />
              <span className="hidden sm:inline">Hugging Face</span>
            </a>
            <span className="w-px h-5 bg-slate-200" aria-hidden />
            {user ? (
              <>
                <span className="text-[13px] sm:text-sm text-slate-600 font-medium">欢迎, {user.username}</span>
                <Button 
                  onClick={() => router.push("/dashboard")}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 sm:px-5 text-sm font-medium rounded-lg shadow-sm"
                >
                  进入工作台
                </Button>
              </>
            ) : (
              <>
                <Button 
                  variant="ghost" 
                  className="nav-link text-slate-700 hover:bg-slate-100/80 rounded-lg px-4"
                  onClick={() => setShowLoginModal(true)}
                >
                  登录
                </Button>
                <Button 
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 sm:px-5 text-sm font-medium rounded-lg shadow-sm hover:shadow-md transition-shadow"
                  onClick={handleStartAnalysis}
                >
                  免费使用
                </Button>
              </>
            )}
          </div>
        </div>
      </motion.header>

      {/* Hero Section - 填满视口，底部为跑马灯，一屏平铺 */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-50 via-blue-50/20 to-white pb-4 flex flex-col min-h-[calc(100vh-4rem)]">
        {/* Subtle data pattern background */}
        <div className="absolute inset-0 overflow-hidden">
          <svg className="absolute top-0 right-0 w-1/2 h-full opacity-[0.04]" viewBox="0 0 400 400">
            <defs>
              <pattern id="dots" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
                <circle cx="2" cy="2" r="1.5" fill="#3B82F6" />
              </pattern>
            </defs>
            <rect fill="url(#dots)" width="400" height="400" />
          </svg>
          <div className="absolute top-20 right-20 w-96 h-96 bg-blue-100/25 rounded-full blur-3xl" />
          <div className="absolute bottom-20 left-20 w-64 h-64 bg-indigo-100/15 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-7xl mx-auto px-5 sm:px-6 py-4 lg:py-6 flex-1 min-h-0 flex flex-col">
          <div className="grid grid-cols-1 lg:grid-cols-2 lg:grid-rows-[1fr] gap-6 lg:gap-8 items-stretch min-h-0 flex-1">
            {/* Left Column - 松散大气，主内容 + 跑马灯；min-w-0 防止内容撑开列宽 */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              className="flex flex-col justify-center min-w-0"
            >
              <h1 className="hero-title text-3xl sm:text-4xl md:text-5xl lg:text-[3rem] text-slate-900 mb-4 leading-[1.2] text-balance">
                零代码社科文本挖掘工作台
              </h1>
              <p className="hero-typewriter text-lg sm:text-xl md:text-2xl text-blue-600 mb-4 min-h-[2.25rem] sm:min-h-[2.5rem] flex items-center">
                <span>{typewriterText}</span>
                <span className="inline-block w-[3px] h-5 sm:h-6 bg-blue-500 ml-0.5 animate-pulse rounded-sm align-middle shrink-0" aria-hidden />
              </p>
              <p className="text-base sm:text-lg text-slate-600 mb-6 leading-[1.6] text-pretty max-w-xl">
                从数据清洗到深度洞察，AI 驱动的全流程科研助手。上传数据，对话式交互，即刻获取专业分析结果。
              </p>
              <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 items-start sm:items-center">
                <span className="hero-cta-marquee inline-block rounded-xl p-[2px]">
                  <Button
                    onClick={handleStartAnalysis}
                    size="lg"
                    className="bg-blue-600 hover:bg-blue-700 text-white px-6 sm:px-8 py-3.5 sm:py-4 text-sm font-semibold rounded-[10px] shadow-lg shadow-blue-600/25 hover:shadow-xl hover:shadow-blue-600/30 transition-all w-full sm:w-auto"
                  >
                    立即开始分析
                  </Button>
                </span>
                <Button
                  variant="outline"
                  size="lg"
                  className="border-slate-200 hover:border-blue-400 hover:text-blue-600 hover:bg-blue-50/50 px-6 sm:px-8 py-3.5 sm:py-4 text-sm font-medium rounded-xl bg-white/80"
                >
                  <Play className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
                  观看演示
                </Button>
              </div>
              <div className="flex items-center gap-8 sm:gap-10 mt-6 pt-5 border-t border-slate-200/80">
                {[
                  { value: "10K+", label: "研究者信赖" },
                  { value: "500+", label: "分析模型" },
                  { value: "99.2%", label: "准确率" },
                ].map((stat) => (
                  <div key={stat.label}>
                    <p className="stat-value text-lg sm:text-xl text-blue-600">{stat.value}</p>
                    <p className="stat-label text-xs sm:text-sm mt-0.5 text-slate-500">{stat.label}</p>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Right Column - AI 对话框：固定宽高，发消息前后窗口尺寸不变 */}
            <motion.div
              ref={chatCardRef}
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative flex justify-end items-center min-h-0"
            >
              <div className="w-[560px] h-[540px] shrink-0 flex flex-col bg-white rounded-2xl shadow-xl shadow-slate-200/60 border border-slate-200/90 overflow-hidden max-w-[calc(100vw-2rem)] max-h-[min(540px,65vh)]">
                  {/* Chat Header */}
                  <div className="flex items-center justify-between px-3 py-2 border-b border-slate-100 bg-slate-50/60 shrink-0">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl overflow-hidden shadow-sm flex items-center justify-center bg-slate-100/50">
                        <img src="/ai-avatar.png" alt="猫咪科学家" className="w-[180%] h-[180%] object-contain scale-75" />
                      </div>
                      <span className="font-semibold text-slate-800 tracking-tight">猫咪科学家</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                      <span className="text-xs text-slate-500 font-medium">在线</span>
                    </div>
                  </div>

                  {/* Chat Body - 填满剩余高度，内容可滚动 */}
                  <div className="p-3 space-y-2 bg-white flex-1 min-h-0 overflow-y-auto">
                    {chatHistory.length === 0 ? (
                      <>
                        {/* 使用样例：AI 欢迎语 */}
                        <div className="flex gap-3">
                          <div className="w-8 h-8 rounded-full overflow-hidden shrink-0 flex items-center justify-center bg-slate-100/50">
                            <img src="/ai-avatar.png" alt="" className="w-[140%] h-[140%] object-contain scale-75" aria-hidden />
                          </div>
                          <div className="bg-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%]">
                            <p className="text-sm text-slate-700 leading-relaxed">{LANDING_GREETING}</p>
                          </div>
                        </div>
                        {/* 使用样例：用户提问 */}
                        <div className="flex justify-end">
                          <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-[85%]">
                            <p className="text-sm leading-relaxed">
                              帮我分析一下近三个月用户评论的情绪趋势，重点关注负面反馈。
                            </p>
                          </div>
                        </div>
                        {/* 使用样例：AI 回复（含情绪趋势图） */}
                        <div className="flex gap-3">
                          <div className="w-8 h-8 rounded-full overflow-hidden shrink-0 flex items-center justify-center bg-slate-100/50">
                            <img src="/ai-avatar.png" alt="" className="w-[140%] h-[140%] object-contain scale-75" aria-hidden />
                          </div>
                          <div className="bg-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%] space-y-3">
                            <p className="text-xs text-slate-500 flex items-center gap-2">
                              <CheckCircle2 className="w-3 h-3 text-green-500" />
                              {"正在分析 '2024Q1_评论数据.csv'..."}
                            </p>
                            <p className="text-sm text-slate-700 leading-relaxed">
                              分析完成。数据显示，3月份负面情绪略有上升（环比+4.2%），主要集中在"物流配送"和"售后响应"两个主题上。
                            </p>
                            <div className="bg-white rounded-xl p-2 border border-slate-200">
                              <p className="text-xs font-medium text-slate-600 mb-1">情绪趋势分析</p>
                              <div className="h-20">
                                <ResponsiveContainer width="100%" height="100%">
                                  <LineChart data={[
                                    { month: "1月", negative: 12, positive: 65 },
                                    { month: "2月", negative: 14, positive: 63 },
                                    { month: "3月", negative: 18, positive: 62 },
                                  ]}>
                                    <XAxis
                                      dataKey="month"
                                      axisLine={false}
                                      tickLine={false}
                                      tick={{ fontSize: 10, fill: "#94a3b8" }}
                                    />
                                    <Line
                                      type="monotone"
                                      dataKey="positive"
                                      stroke="#22c55e"
                                      strokeWidth={2}
                                      dot={{ fill: "#22c55e", strokeWidth: 0, r: 3 }}
                                    />
                                    <Line
                                      type="monotone"
                                      dataKey="negative"
                                      stroke="#ef4444"
                                      strokeWidth={2}
                                      dot={{ fill: "#ef4444", strokeWidth: 0, r: 3 }}
                                    />
                                  </LineChart>
                                </ResponsiveContainer>
                              </div>
                              <div className="flex items-center justify-center gap-3 mt-1">
                                <span className="flex items-center gap-1.5 text-xs text-slate-500">
                                  <span className="w-2 h-2 rounded-full bg-green-500" />
                                  正面情绪
                                </span>
                                <span className="flex items-center gap-1.5 text-xs text-slate-500">
                                  <span className="w-2 h-2 rounded-full bg-red-500" />
                                  负面情绪
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </>
                    ) : (
                      chatHistory.map((msg) => {
                        const isUser = msg.role === "user"
                        const isLatestAi = msg.role === "ai" && msg.id === lastAiMessageId && msg.type === "text"
                        return (
                          <div
                            key={msg.id}
                            className={isUser ? "flex justify-end" : "flex gap-3"}
                          >
                            {!isUser && (
                              <div className="w-8 h-8 rounded-full overflow-hidden shrink-0 flex items-center justify-center bg-slate-100/50">
                                <img src="/ai-avatar.png" alt="" className="w-[140%] h-[140%] object-contain scale-75" aria-hidden />
                              </div>
                            )}
                            <div
                              className={
                                isUser
                                  ? "bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-[85%]"
                                  : "bg-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%]"
                              }
                            >
                              {isLatestAi ? (
                                <TypingMessage
                                  content={msg.content}
                                  isLatest={true}
                                  className="text-slate-700 text-sm"
                                  speed={12}
                                />
                              ) : (
                                <p className="text-sm leading-relaxed">{msg.content}</p>
                              )}
                            </div>
                          </div>
                        )
                      })
                    )}
                    {isAiLoading && (
                      <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-full overflow-hidden shrink-0 flex items-center justify-center bg-slate-100/50">
                          <img src="/ai-avatar.png" alt="" className="w-[140%] h-[140%] object-contain scale-75" aria-hidden />
                        </div>
                        <div className="bg-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%] flex items-center gap-2">
                          <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
                          <span className="text-sm text-slate-500">正在思考...</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Chat Input Footer - 真实输入与发送 */}
                  <div className="px-3 py-2 border-t border-slate-100 bg-slate-50/50 shrink-0">
                    <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-xl px-3 py-2 focus-within:ring-2 focus-within:ring-blue-200 focus-within:border-blue-300">
                      <Paperclip className="w-4 h-4 text-slate-400 shrink-0" />
                      <input
                        type="text"
                        value={chatInputValue}
                        onChange={(e) => setChatInputValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault()
                            handleLandingChatSend(chatInputValue)
                          }
                        }}
                        placeholder="输入您的分析指令..."
                        className="flex-1 min-w-0 bg-transparent text-sm text-slate-900 placeholder:text-slate-400 outline-none"
                        disabled={isAiLoading}
                      />
                      <button
                        type="button"
                        onClick={() => handleLandingChatSend(chatInputValue)}
                        disabled={!chatInputValue.trim() || isAiLoading}
                        className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
                      >
                        <Send className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>

              {/* Floating decoration */}
              <div className="absolute -bottom-4 -right-4 w-24 h-24 bg-blue-100/50 rounded-full blur-2xl -z-10" />
              <div className="absolute -top-4 -left-4 w-16 h-16 bg-indigo-100/50 rounded-full blur-xl -z-10" />
            </motion.div>
          </div>
        </div>

        {/* 跑马灯：页面底部，一打开即见 */}
        <div className="page-bg-effect__ghost-text page-bg-effect__ghost-text--hero mt-2 sm:mt-3 shrink-0">
          <div className="page-bg-effect__ghost-text-track">
            <svg className="page-bg-effect__ghost-text-svg" viewBox="0 0 1800 120" preserveAspectRatio="xMinYMid meet" aria-hidden>
              <text x="0" y="92" fill="none" stroke="rgba(148,163,184,0.5)" strokeWidth="2" fontSize="108" fontWeight="800">THETA · THETA · THETA · THETA · THETA · THETA · THETA ·</text>
            </svg>
            <svg className="page-bg-effect__ghost-text-svg" viewBox="0 0 1800 120" preserveAspectRatio="xMinYMid meet" aria-hidden>
              <text x="0" y="92" fill="none" stroke="rgba(148,163,184,0.5)" strokeWidth="2" fontSize="108" fontWeight="800">THETA · THETA · THETA · THETA · THETA · THETA · THETA ·</text>
            </svg>
          </div>
        </div>
      </section>

      {/* 二、核心功能引擎 - 三大横向模块（图文交替） */}
      <section id="core-features" className="max-w-7xl mx-auto px-5 sm:px-6 py-20 sm:py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <h2 className="section-heading text-2xl sm:text-3xl md:text-[1.75rem] mb-3">核心功能引擎：重构社科研究生产力</h2>
          <p className="text-slate-600 max-w-2xl mx-auto text-[15px] sm:text-base leading-relaxed">
            集成大语言模型与主题模型，打造遵循严谨学术范式的一站式研究工作流。
          </p>
        </motion.div>

        {/* 模块一：全栈式主题建模 - 左文案 | 右概念图 */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center mb-16 sm:mb-20"
        >
          <div>
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-4">全栈式主题建模</h3>
            <p className="text-slate-600 leading-[1.7] mb-4">
              不再受限于单一模型。Theta 集成了从经典统计学到最新深度学习的完整算法库，灵活适配短文本、长文档及动态演化数据，满足多元化的研究目标。
            </p>
            <p className="text-xs text-slate-500">LDA · ETM · CTM · DTM · <strong className="text-blue-600">BERTopic</strong> · <strong className="text-blue-600">LLM-Topic</strong></p>
          </div>
          <div className="rounded-2xl border border-slate-200/90 bg-gradient-to-br from-slate-50 to-blue-50/30 aspect-video overflow-hidden min-h-[200px]">
            <img src="/1.png" alt="全栈式主题建模" className="w-full h-full object-cover" />
          </div>
        </motion.div>

        {/* 模块二：云端即时数据处理 - 左概念图 | 右文案 */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center mb-16 sm:mb-20"
        >
          <div className="order-2 lg:order-1 rounded-2xl border border-slate-200/90 bg-gradient-to-br from-blue-50/50 to-slate-50 aspect-video overflow-hidden min-h-[200px]">
            <img src="/2.png" alt="云端即时数据处理" className="w-full h-full object-cover" />
          </div>
          <div className="order-1 lg:order-2">
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-4">云端即时数据处理</h3>
            <p className="text-slate-600 leading-[1.7] mb-4">
              告别繁琐的本地环境配置、Python 库依赖与算力瓶颈。
            </p>
            <ul className="space-y-2 text-slate-600">
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 零配置快速上手</li>
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 在线交互式处理</li>
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 安全云端算力</li>
            </ul>
          </div>
        </motion.div>

        {/* 模块三：交互式 AI 科学家 - 左文案 | 右交互演示图 */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center"
        >
          <div>
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-4">交互式 AI 科学家</h3>
            <p className="text-slate-600 leading-[1.7] mb-4">
              它不仅是一个图表生成器，更是一位深谙学术规范的虚拟合作者。猫咪科学家拒绝千篇一律的模板化描述，而是提供定制化的深度解读。
            </p>
            <ul className="space-y-2 text-slate-600">
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 非模板化深度解读</li>
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 图表级多轮追问</li>
              <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" /> 研究假设辅助验证</li>
            </ul>
          </div>
          <div className="rounded-2xl border border-slate-200/90 bg-gradient-to-br from-slate-50 to-indigo-50/20 aspect-video overflow-hidden min-h-[200px]">
            <img src="/3.png" alt="交互式 AI 科学家" className="w-full h-full object-cover" />
          </div>
        </motion.div>
      </section>

      {/* 三、使用教程 How it Works - 左右布局 + 图文轮播 */}
      <section id="how-it-works" className="max-w-7xl mx-auto px-5 sm:px-6 py-20 sm:py-24 bg-slate-50/50">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <h2 className="section-heading text-2xl sm:text-3xl md:text-[1.75rem] mb-3">怎么用？</h2>
          <p className="text-slate-600 max-w-xl mx-auto text-[15px] sm:text-base leading-relaxed">
            零代码基础也能用，四步完成从数据到洞察
          </p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-center"
        >
          {/* 左侧：当前步骤文案 */}
          <div className="order-2 lg:order-1">
            <p className="text-xs sm:text-sm text-blue-600 font-semibold tracking-wide mb-2">
              步骤 {howItWorksStep + 1} / 4 · {HOW_IT_WORKS_STEPS[howItWorksStep].titleEn}
            </p>
            <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-4 tracking-tight">
              {HOW_IT_WORKS_STEPS[howItWorksStep].title}
            </h3>
            <p className="text-slate-600 leading-[1.7]">
              {HOW_IT_WORKS_STEPS[howItWorksStep].text}
            </p>
          </div>
          {/* 右侧：GIF 占位 + 左右翻页 */}
          <div className="order-1 lg:order-2 flex flex-col items-center">
            <div className="relative w-full max-w-lg aspect-video rounded-2xl overflow-hidden border border-slate-200/90 bg-white shadow-lg">
              {/* 占位：后续替换为真实 GIF */}
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-blue-50 to-slate-100">
                <HowItWorksStepIcon className="w-20 h-20 sm:w-24 sm:h-24 text-blue-400/70" />
              </div>
              <button
                type="button"
                onClick={() => setHowItWorksStep((s) => (s === 0 ? 3 : s - 1))}
                className="absolute left-3 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-white/90 shadow-md flex items-center justify-center text-slate-600 hover:bg-white hover:text-blue-600 transition-colors"
                aria-label="上一步"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              <button
                type="button"
                onClick={() => setHowItWorksStep((s) => (s === 3 ? 0 : s + 1))}
                className="absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-white/90 shadow-md flex items-center justify-center text-slate-600 hover:bg-white hover:text-blue-600 transition-colors"
                aria-label="下一步"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
            <p className="text-xs text-slate-400 mt-3">左右切换查看各步骤（动图占位，可替换为真实 GIF）</p>
          </div>
        </motion.div>
      </section>

      {/* 四、场景化分析实验室 Scenario Lab - 上三下二 */}
      <section id="scenario-lab" className="max-w-7xl mx-auto px-5 sm:px-6 py-20 sm:py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <h2 className="section-heading text-2xl sm:text-3xl md:text-[1.75rem] mb-3">场景化分析实验室</h2>
          <p className="text-slate-600 max-w-2xl mx-auto text-[15px] sm:text-base leading-relaxed">
            选择您的研究数据类型，即刻微调您的领域嵌入模型，发现更深入的领域洞察。
          </p>
        </motion.div>
        <div className="space-y-6">
          {/* 第一行：3 个卡片 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {SCENARIO_LAB_ROW1.map((card, index) => (
              <Card
                key={card.title}
                className="group relative border border-slate-200/90 bg-white hover:border-blue-200 hover:shadow-lg hover:shadow-blue-100/30 hover:-translate-y-1 transition-all p-6 rounded-2xl flex flex-col"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold text-slate-900 mb-1 tracking-tight">{card.title}</h3>
                    <p className="text-xs text-blue-600 font-medium mb-3">{card.titleEn}</p>
                    <div className="flex flex-wrap gap-1.5 mb-4">
                      {card.tags.map((tag) => (
                        <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="opacity-0 group-hover:opacity-100 transition-opacity border-blue-200 text-blue-600 hover:bg-blue-50"
                      onClick={handleStartAnalysis}
                    >
                      立即分析
                    </Button>
                  </div>
                  <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center shrink-0">
                    <card.icon className="w-6 h-6" />
                  </div>
                </div>
              </Card>
            ))}
          </motion.div>
          {/* 第二行：2 个宽卡片 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            {SCENARIO_LAB_ROW2.map((card) => (
              <Card
                key={card.title}
                className="group relative border border-slate-200/90 bg-white hover:border-blue-200 hover:shadow-lg hover:shadow-blue-100/30 hover:-translate-y-1 transition-all p-6 rounded-2xl flex flex-col"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold text-slate-900 mb-1 tracking-tight">{card.title}</h3>
                    <p className="text-xs text-blue-600 font-medium mb-3">{card.titleEn}</p>
                    <div className="flex flex-wrap gap-1.5 mb-4">
                      {card.tags.map((tag) => (
                        <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="opacity-0 group-hover:opacity-100 transition-opacity border-blue-200 text-blue-600 hover:bg-blue-50"
                      onClick={handleStartAnalysis}
                    >
                      Load Template
                    </Button>
                  </div>
                  <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center shrink-0">
                    <card.icon className="w-6 h-6" />
                  </div>
                </div>
              </Card>
            ))}
          </motion.div>
        </div>
      </section>

      {/* 五、价格方案 Pricing */}
      <section id="pricing" className="max-w-7xl mx-auto px-5 sm:px-6 py-20 sm:py-24 bg-slate-50/50">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="section-heading text-2xl sm:text-3xl md:text-[1.75rem] mb-3">匹配您研究需求的灵活方案</h2>
          <p className="text-slate-600 max-w-xl mx-auto text-[15px] sm:text-base leading-relaxed mb-8">
            无论是个人课程作业，还是大规模科研项目，我们都有适合您的算力支持。
          </p>
          <div className="inline-flex items-center gap-3 p-1.5 rounded-full bg-slate-200/60">
            <button
              type="button"
              onClick={() => setPricingMode("per-use")}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${pricingMode === "per-use" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"}`}
            >
              按次付费
            </button>
            <button
              type="button"
              onClick={() => setPricingMode("monthly")}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${pricingMode === "monthly" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"}`}
            >
              按月付费
            </button>
            <button
              type="button"
              onClick={() => setPricingMode("yearly")}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${pricingMode === "yearly" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"}`}
            >
              按年付费 <span className="text-green-600 text-xs">省 17%</span>
            </button>
          </div>
        </motion.div>
        {pricingMode === "per-use" ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto"
          >
            {[
              { size: "500M", price: 29, desc: "适合小规模数据", recommended: false },
              { size: "20G", price: 299, desc: "适合大规模项目", recommended: true },
              { size: "100G", price: 999, desc: "上限封顶，超值选择", recommended: false },
            ].map((plan) => (
              <Card
                key={plan.size}
                className={`relative border rounded-2xl p-6 flex flex-col ${plan.recommended ? "border-blue-300 shadow-lg shadow-blue-100/40 scale-105 md:scale-105" : "border-slate-200/90 bg-white"}`}
              >
                {plan.recommended && (
                  <span className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-0.5 rounded-full bg-blue-600 text-white text-xs font-semibold">
                    热门推荐
                  </span>
                )}
                <h3 className="text-lg font-bold text-slate-900 mb-1">{plan.size}</h3>
                <p className="text-sm text-slate-500 mb-4">{plan.desc}</p>
                <div className="mb-6">
                  <span className="text-3xl font-bold text-slate-900">
                    ¥{plan.price}
                  </span>
                  <span className="text-slate-500 text-sm ml-1">/ 次</span>
                </div>
                <ul className="space-y-2 mb-6 flex-1">
                  <li className="text-sm text-slate-600 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                    按上传数据大小计费
                  </li>
                  <li className="text-sm text-slate-600 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                    {plan.size === "100G" ? "100G 为上限，超出不再收费" : `单次处理 ${plan.size} 数据`}
                  </li>
                  <li className="text-sm text-slate-600 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                    永久有效，无时间限制
                  </li>
                </ul>
                <Button
                  className={plan.recommended ? "bg-blue-600 hover:bg-blue-700" : "border-slate-200"}
                  variant={plan.recommended ? "default" : "outline"}
                  onClick={handleStartAnalysis}
                >
                  立即开始
                </Button>
              </Card>
            ))}
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto"
          >
            {PRICING_PLANS.map((plan, index) => (
              <Card
                key={plan.name}
                className={`relative border rounded-2xl p-6 flex flex-col ${plan.recommended ? "border-blue-300 shadow-lg shadow-blue-100/40 scale-105 md:scale-105" : "border-slate-200/90 bg-white"}`}
              >
                {plan.recommended && (
                  <span className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-0.5 rounded-full bg-blue-600 text-white text-xs font-semibold">
                    热门推荐
                  </span>
                )}
                <h3 className="text-lg font-bold text-slate-900 mb-1">{plan.name}</h3>
                <p className="text-sm text-slate-500 mb-4">{plan.desc}</p>
                <div className="mb-6">
                  <span className="text-3xl font-bold text-slate-900">
                    ¥{pricingMode === "yearly" ? plan.priceYear : plan.priceMonth}
                  </span>
                  <span className="text-slate-500 text-sm ml-1">/ {pricingMode === "yearly" ? "年" : "月"}</span>
                </div>
                <ul className="space-y-2 mb-6 flex-1">
                  {plan.features.map((f) => (
                    <li key={f} className="text-sm text-slate-600 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />
                      {f}
                    </li>
                  ))}
                </ul>
                <Button
                  className={plan.recommended ? "bg-blue-600 hover:bg-blue-700" : "border-slate-200"}
                  variant={plan.recommended ? "default" : "outline"}
                  onClick={handleStartAnalysis}
                >
                  立即开始
                </Button>
              </Card>
            ))}
          </motion.div>
        )}
      </section>

      {/* 六、常见问题 FAQ */}
      <section id="faq" className="max-w-3xl mx-auto px-5 sm:px-6 py-20 sm:py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="section-heading text-2xl sm:text-3xl md:text-[1.75rem] mb-3">关于 Theta 的一切</h2>
          <p className="text-slate-600 text-[15px] sm:text-base leading-relaxed">
            我们整理了用户最关心的问题，助您无忧开启科研。
          </p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="space-y-3"
        >
          {FAQ_ITEMS.map((item, index) => (
            <div
              key={index}
              className="border border-slate-200 rounded-xl bg-white overflow-hidden transition-[border-color,box-shadow] hover:border-blue-200 hover:shadow-md"
            >
              <button
                type="button"
                onClick={() => setFaqOpenIndex(faqOpenIndex === index ? null : index)}
                className="w-full flex items-center justify-between gap-4 px-5 py-4 text-left"
              >
                <span className="font-semibold text-slate-900">{item.q}</span>
                <span className="text-slate-400 shrink-0">
                  {faqOpenIndex === index ? <Minus className="w-5 h-5" /> : <Plus className="w-5 h-5" />}
                </span>
              </button>
              <div
                className={`grid transition-[grid-template-rows] duration-200 ease-in-out ${faqOpenIndex === index ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}
              >
                <div className="overflow-hidden">
                  <p className="px-5 pb-4 pt-0 text-slate-600 text-sm leading-relaxed border-t border-slate-100">
                    {item.a}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </motion.div>
        <p className="text-center text-slate-500 text-sm mt-8">
          还有其他问题？直接联系我们的产品团队。
        </p>
        <div className="flex flex-wrap justify-center items-center gap-4 mt-6">
          <Button variant="outline" size="sm" className="h-9 px-4 shrink-0" asChild>
            <a href="mailto:duanzhenke@code-soul.com">留言给支持团队</a>
          </Button>
          <span className="text-slate-400 text-sm shrink-0">或</span>
          <a href="mailto:duanzhenke@code-soul.com" className="text-blue-600 text-sm font-medium hover:underline shrink-0">
            duanzhenke@code-soul.com
          </a>
        </div>
      </section>

      {/* 七、页脚 Footer - 五列 */}
      <footer className="border-t border-slate-200/80 bg-slate-50/80">
        <div className="max-w-7xl mx-auto px-5 sm:px-6 py-12 sm:py-16">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-8 lg:gap-6">
            {/* 第一列：品牌与愿景 */}
            <div className="col-span-2 md:col-span-1">
              <div className="flex items-center mb-3">
                <a href="https://github.com/CodeSoul-co" target="_blank" rel="noopener noreferrer" className="inline-flex items-center">
                  <img src="/codesoul-logo.png" alt="CodeSoul" className="h-10 sm:h-12 w-auto object-contain hover:opacity-80 transition-opacity" />
                </a>
              </div>
              <p className="text-xs text-slate-500 mb-2">Theta：洞见，先于思考</p>
              <a href="mailto:duanzhenke@code-soul.com" className="text-xs text-blue-600 hover:underline">
                duanzhenke@code-soul.com
              </a>
            </div>
            {/* 第二列：产品 */}
            <div>
              <h4 className="text-sm font-semibold text-slate-900 mb-3">产品</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><a href="#changelog" className="hover:text-blue-600">功能更新日志</a></li>
                <li><a href="#api-docs" className="hover:text-blue-600">API 文档</a></li>
                <li><a href="#cases" className="hover:text-blue-600">案例库</a></li>
              </ul>
            </div>
            {/* 第三列：支持与社区 */}
            <div>
              <h4 className="text-sm font-semibold text-slate-900 mb-3">支持与社区</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><a href="#faq" className="hover:text-blue-600">帮助中心</a></li>
                <li><a href="#community" className="hover:text-blue-600">学术交流群</a></li>
                <li>
                  <button type="button" onClick={() => setShowCiteModal(true)} className="hover:text-blue-600 text-left">
                    如何引用 Theta？
                  </button>
                </li>
                <li><a href="#feedback" className="hover:text-blue-600">反馈建议</a></li>
              </ul>
            </div>
            {/* 第四列：社交媒体 */}
            <div>
              <h4 className="text-sm font-semibold text-slate-900 mb-3">社交媒体</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><a href="https://github.com/CodeSoul-co/THETA" target="_blank" rel="noopener noreferrer" className="hover:text-blue-600">GitHub</a></li>
                <li><a href="#" className="hover:text-blue-600">X</a></li>
                <li><a href="#" className="hover:text-blue-600">Facebook</a></li>
              </ul>
            </div>
            {/* 第五列：关于 */}
            <div>
              <h4 className="text-sm font-semibold text-slate-900 mb-3">关于</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><a href="#about" className="hover:text-blue-600">关于我们</a></li>
                <li><a href="#hiring" className="hover:text-blue-600">加入我们</a></li>
                <li><Link href="/security" className="hover:text-blue-600">安全白皮书</Link></li>
                <li><a href="#terms" className="hover:text-blue-600">用户协议</a></li>
                <li><a href="#privacy" className="hover:text-blue-600">隐私政策</a></li>
                <li className="text-slate-400 text-xs">ICP 备案号 / 公安联网备案</li>
              </ul>
            </div>
          </div>
          <p className="text-center text-slate-400 text-xs mt-10 pt-6 border-t border-slate-200/80">
            &copy; 2024 THETA · 零代码社科文本挖掘工作台
          </p>
        </div>
      </footer>

      {/* 如何引用 Theta - BibTeX 弹窗 */}
      <Dialog open={showCiteModal} onOpenChange={setShowCiteModal}>
        <DialogContent className="sm:max-w-lg bg-white">
          <DialogHeader>
            <DialogTitle>如何引用 Theta？</DialogTitle>
            <DialogDescription>在论文或报告中引用 THETA 时，可使用以下 BibTeX。</DialogDescription>
          </DialogHeader>
          <pre className="p-4 rounded-lg bg-slate-100 text-sm text-slate-800 overflow-x-auto font-mono whitespace-pre-wrap">
{`@software{theta2024,
  title = {THETA: 零代码社科文本挖掘工作台},
  author = {CodeSoul},
  year = {2024},
  url = {https://github.com/CodeSoul-co/THETA}
}`}
          </pre>
          <Button variant="outline" size="sm" onClick={() => { navigator.clipboard.writeText(`@software{theta2024,\n  title = {THETA: 零代码社科文本挖掘工作台},\n  author = {CodeSoul},\n  year = {2024},\n  url = {https://github.com/CodeSoul-co/THETA}\n}`); }}>
            复制 BibTeX
          </Button>
        </DialogContent>
      </Dialog>

      {/* Login Modal */}
      <Dialog open={showLoginModal} onOpenChange={setShowLoginModal}>
        <DialogContent className="sm:max-w-md bg-white">
          <DialogHeader className="space-y-3 pb-2">
            <div className="flex items-center justify-center mb-2">
              <img src="/theta-logo.png" alt="THETA" className="h-14 w-auto object-contain" />
            </div>
            <DialogTitle className="text-2xl font-bold text-center bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              登录 THETA
            </DialogTitle>
            <DialogDescription className="text-center text-slate-500 text-sm">
              登录您的账户以继续
            </DialogDescription>
          </DialogHeader>
          
          <form onSubmit={handleLogin} className="space-y-5 py-4">
            {loginError && (
              <Alert variant="destructive" className="animate-in slide-in-from-top-2">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{loginError}</AlertDescription>
              </Alert>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="login-username" className="text-slate-700">
                用户名或邮箱
              </Label>
              <Input
                id="login-username"
                type="text"
                placeholder="请输入用户名或邮箱"
                value={loginUsername}
                onChange={(e) => setLoginUsername(e.target.value)}
                disabled={isLoggingIn}
                className="h-11 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors"
                autoComplete="username"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="login-password" className="text-slate-700">
                密码
              </Label>
              <div className="relative">
                <Input
                  id="login-password"
                  type={showPassword ? "text" : "password"}
                  placeholder="请输入密码"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  disabled={isLoggingIn}
                  className="h-11 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors pr-10"
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                  tabIndex={-1}
                >
                  {showPassword ? (
                    <EyeOff className="w-5 h-5" />
                  ) : (
                    <Eye className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="remember-me"
                  checked={rememberMe}
                  onCheckedChange={(checked) => setRememberMe(checked === true)}
                  disabled={isLoggingIn}
                />
                <Label
                  htmlFor="remember-me"
                  className="text-sm font-normal cursor-pointer text-slate-600"
                >
                  记住我
                </Label>
              </div>
              <a href="#" className="text-sm text-blue-600 hover:text-blue-700 hover:underline">
                忘记密码？
              </a>
            </div>

            <Button 
              type="submit"
              className="w-full h-11 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md hover:shadow-lg transition-all"
              disabled={isLoggingIn}
            >
              {isLoggingIn ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  登录中...
                </>
              ) : (
                "登录"
              )}
            </Button>

            <div className="relative my-2">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-200"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-white px-2 text-slate-400">或</span>
              </div>
            </div>

            <div className="text-center text-sm text-slate-600">
              还没有账户？{" "}
              <button 
                type="button"
                onClick={handleGoToRegister}
                className="text-blue-600 hover:text-blue-700 font-medium hover:underline"
              >
                立即注册
              </button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  )
}

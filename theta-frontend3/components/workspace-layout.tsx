"use client"

import type React from "react"
import { useState, useCallback } from "react"
import { motion } from "framer-motion"
import { useRouter, usePathname } from "next/navigation"
import {
  Database,
  FileCog,
  GraduationCap,
  FileCheck,
  PieChart,
  PanelLeftClose,
  PanelLeft,
  User,
  Settings,
  LogOut,
  BrainCircuit,
  Paperclip,
  Send,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

// 工作流步骤配置
const workflowSteps = [
  { id: "data", label: "数据管理", icon: Database, description: "上传和管理数据集" },
  { id: "processing", label: "数据清洗", icon: FileCog, description: "清洗和预处理数据" },
  { id: "training", label: "模型训练", icon: GraduationCap, description: "训练主题模型" },
  { id: "results", label: "分析结果", icon: FileCheck, description: "查看分析结果" },
  { id: "visualizations", label: "可视化", icon: PieChart, description: "数据可视化展示" },
]

interface Message {
  id: string
  role: "user" | "assistant" | "system"
  content: string
}

interface WorkspaceLayoutProps {
  children: React.ReactNode
  title?: string
  description?: string
  currentStep?: string
}

function NavItem({
  icon: Icon,
  label,
  active = false,
  onClick,
  collapsed = false,
}: {
  icon: React.ElementType
  label: string
  active?: boolean
  onClick?: () => void
  collapsed?: boolean
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center ${collapsed ? "justify-center" : "gap-3"} w-full px-4 py-2.5 rounded-lg transition-all ${
        active ? "bg-blue-50 text-blue-600 font-medium" : "text-slate-700 hover:bg-slate-100 hover:text-slate-900"
      }`}
      title={collapsed ? label : undefined}
    >
      <Icon className="w-5 h-5" />
      {!collapsed && <span className="text-sm">{label}</span>}
    </button>
  )
}

function UserDropdown() {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="p-2 hover:bg-slate-100 rounded-full transition-colors">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
            <User className="w-5 h-5 text-white" />
          </div>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48 bg-white">
        <DropdownMenuItem className="cursor-pointer">
          <User className="w-4 h-4 mr-2" />
          个人资料
        </DropdownMenuItem>
        <DropdownMenuItem className="cursor-pointer">
          <Settings className="w-4 h-4 mr-2" />
          设置
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem className="cursor-pointer text-red-600">
          <LogOut className="w-4 h-4 mr-2" />
          退出登录
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function ChatInterface({
  messages,
  inputValue,
  onInputChange,
  onSend,
}: {
  messages: Message[]
  inputValue: string
  onInputChange: (value: string) => void
  onSend: () => void
}) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }

  return (
    <>
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-3">
              <div className="w-16 h-16 rounded-2xl bg-blue-100 flex items-center justify-center mx-auto">
                <BrainCircuit className="w-8 h-8 text-blue-600" />
              </div>
              <p className="text-slate-500 text-sm">开始对话，让 AI 助手帮您分析数据</p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  message.role === "user"
                    ? "bg-blue-600 text-white"
                    : message.role === "system"
                    ? "bg-amber-50 text-amber-900 border border-amber-200"
                    : "bg-slate-100 text-slate-900 border border-slate-200"
                }`}
              >
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="border-t border-slate-200 p-4 bg-white">
        <div className="flex items-end gap-2">
          <button className="p-2 hover:bg-slate-100 rounded-lg transition-colors flex-shrink-0">
            <Paperclip className="w-5 h-5 text-slate-500" />
          </button>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入消息..."
            className="flex-1 px-4 py-2 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          />
          <button
            onClick={onSend}
            disabled={!inputValue.trim()}
            className="p-2 bg-blue-600 hover:bg-blue-700 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
          >
            <Send className="w-5 h-5 text-white" />
          </button>
        </div>
      </div>
    </>
  )
}

export function WorkspaceLayout({ children, title, description, currentStep }: WorkspaceLayoutProps) {
  const router = useRouter()
  const pathname = usePathname()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [chatHistory, setChatHistory] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState("")

  // 根据路径确定当前步骤
  const activeStep = currentStep || (() => {
    if (pathname === "/training") return "training"
    if (pathname === "/results") return "results"
    if (pathname === "/visualizations") return "visualizations"
    return "data"
  })()

  const handleNavigation = useCallback((step: typeof workflowSteps[0]) => {
    // 导航到主页面并切换视图
    // 使用 router.push 因为可能从独立页面跳转
    router.push(`/?view=${step.id}`)
  }, [router])

  const handleSendMessage = useCallback(() => {
    if (inputValue.trim()) {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: inputValue,
      }
      setChatHistory(prev => [...prev, userMessage])

      // 模拟 AI 回复
      setTimeout(() => {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "我可以帮您分析数据。请告诉我您的具体需求。",
        }
        setChatHistory(prev => [...prev, aiMessage])
      }, 1000)

      setInputValue("")
    }
  }, [inputValue])

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      {/* 左侧导航栏 */}
      <motion.aside
        initial={{ x: -280, opacity: 0 }}
        animate={{
          x: 0,
          opacity: 1,
          width: sidebarCollapsed ? 80 : 256,
        }}
        transition={{ delay: 0.1, type: "spring", damping: 25, stiffness: 200 }}
        className="bg-white border-r border-slate-200 flex flex-col flex-shrink-0"
      >
        <div className="p-6 border-b border-slate-200 flex items-center justify-between">
          {!sidebarCollapsed && (
            <div>
              <h1 className="text-3xl font-bold text-blue-600">THETA</h1>
              <p className="text-xs text-slate-500 mt-1">智能分析平台</p>
            </div>
          )}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
          >
            {sidebarCollapsed ? (
              <PanelLeft className="w-5 h-5 text-slate-600" />
            ) : (
              <PanelLeftClose className="w-5 h-5 text-slate-600" />
            )}
          </button>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          {!sidebarCollapsed && (
            <p className="px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">工作流程</p>
          )}
          {workflowSteps.map((step, index) => (
            <NavItem
              key={step.id}
              icon={step.icon}
              label={sidebarCollapsed ? "" : `${index + 1}. ${step.label}`}
              active={activeStep === step.id}
              onClick={() => handleNavigation(step)}
              collapsed={sidebarCollapsed}
            />
          ))}

          {/* 工作流进度指示器 */}
          {!sidebarCollapsed && (
            <div className="mt-6 pt-4 border-t border-slate-200">
              <p className="px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">当前进度</p>
              <div className="px-4">
                <div className="flex items-center gap-1">
                  {workflowSteps.map((step, index) => {
                    const stepIndex = workflowSteps.findIndex(s => s.id === activeStep)
                    const isCompleted = index < stepIndex
                    const isCurrent = index === stepIndex
                    return (
                      <div key={step.id} className="flex items-center flex-1">
                        <div
                          className={`w-3 h-3 rounded-full transition-colors ${
                            isCompleted ? "bg-green-500" : isCurrent ? "bg-blue-500" : "bg-slate-200"
                          }`}
                        />
                        {index < workflowSteps.length - 1 && (
                          <div
                            className={`flex-1 h-0.5 transition-colors ${isCompleted ? "bg-green-500" : "bg-slate-200"}`}
                          />
                        )}
                      </div>
                    )
                  })}
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  {workflowSteps.find(s => s.id === activeStep)?.description || ""}
                </p>
              </div>
            </div>
          )}
        </nav>
      </motion.aside>

      {/* 中间主内容区 */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex-1 bg-slate-50 overflow-auto"
        >
          {children}
        </motion.div>
      </div>

      {/* 右侧 AI 助手面板 */}
      <motion.aside
        initial={{ x: 100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 0.2, type: "spring", damping: 25, stiffness: 200 }}
        className="w-96 border-l border-slate-200 bg-white flex flex-col flex-shrink-0"
      >
        <div className="border-b border-slate-200 p-4">
          <h3 className="font-semibold text-slate-900 flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-blue-600" />
            AI 助手
          </h3>
        </div>
        <ChatInterface
          messages={chatHistory}
          inputValue={inputValue}
          onInputChange={setInputValue}
          onSend={handleSendMessage}
        />
      </motion.aside>

      {/* 右侧用户菜单栏 */}
      <motion.aside
        initial={{ x: 100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 0.2, type: "spring", damping: 25, stiffness: 200 }}
        className="w-16 bg-white border-l border-slate-200 flex flex-col items-center py-4 justify-between flex-shrink-0"
      >
        <UserDropdown />
        <button className="p-2 hover:bg-slate-100 rounded-lg transition-colors">
          <Settings className="w-5 h-5 text-slate-600" />
        </button>
      </motion.aside>
    </div>
  )
}

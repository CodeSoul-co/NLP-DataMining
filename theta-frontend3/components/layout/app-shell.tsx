"use client"

import React, { useState, useRef, useCallback, useEffect } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Settings, X, LogOut, User, Activity, LayoutDashboard, FileText, BarChart3, FolderOpen, PanelRightClose, PanelRightOpen, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { AiSidebar, type ChatMessage } from "@/components/chat/ai-sidebar"
import { useAuth } from "@/contexts/auth-context"
import { ReactNode } from "react"

export type Tab = {
  id: string
  title: string
  closable: boolean
}

interface AppShellProps {
  tabs: Tab[]
  activeTabId: string
  onTabChange: (tabId: string) => void
  onTabClose: (tabId: string) => void
  children: ReactNode
  // AI Sidebar Props
  chatHistory: ChatMessage[]
  onSendMessage: (content: string) => void
  onDataUploaded?: (file: File) => void
  onFocusChart?: (chartId: string) => void
  onClearChat?: () => void
}

export function AppShell({
  tabs,
  activeTabId,
  onTabChange,
  onTabClose,
  children,
  chatHistory,
  onSendMessage,
  onDataUploaded,
  onFocusChart,
  onClearChat,
}: AppShellProps) {
  const router = useRouter()
  const { user, logout } = useAuth()
  const [showAiSidebar, setShowAiSidebar] = useState(true)
  const [sidebarWidth, setSidebarWidth] = useState(380)
  const [isResizing, setIsResizing] = useState(false)
  const sidebarMin = 280
  const sidebarMax = 640
  const startXRef = useRef(0)
  const startWidthRef = useRef(380)

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
    startXRef.current = e.clientX
    startWidthRef.current = sidebarWidth
  }, [sidebarWidth])

  useEffect(() => {
    if (!isResizing) return
    const onMove = (e: MouseEvent) => {
      const delta = startXRef.current - e.clientX
      setSidebarWidth((w) => Math.min(sidebarMax, Math.max(sidebarMin, startWidthRef.current + delta)))
    }
    const onUp = () => setIsResizing(false)
    window.addEventListener("mousemove", onMove)
    window.addEventListener("mouseup", onUp)
    return () => {
      window.removeEventListener("mousemove", onMove)
      window.removeEventListener("mouseup", onUp)
    }
  }, [isResizing, sidebarMin, sidebarMax])

  const handleLogout = () => {
    logout()
    router.push("/")
  }

  // Get user initials for avatar
  const getUserInitials = () => {
    if (!user?.username) return "U"
    return user.username.charAt(0).toUpperCase()
  }

  return (
    <div className="h-screen w-full min-w-0 flex flex-col bg-gradient-to-br from-slate-50 via-slate-50 to-blue-50/30 font-sans antialiased overflow-hidden">
      {/* Top Navigation Bar */}
      <header className="h-14 flex-shrink-0 bg-white/90 backdrop-blur-md border-b border-slate-200/60 flex items-center justify-between px-4 sm:px-6 shadow-[0_1px_3px_rgba(0,0,0,0.05)] min-w-0">
        {/* Left: Logo */}
        <div className="flex items-center gap-2 sm:gap-4 min-w-0 flex-1 overflow-hidden">
          <img src="/theta-logo.png" alt="THETA" className="h-9 sm:h-10 w-auto flex-shrink-0" />
          <div className="h-5 w-px bg-gradient-to-b from-transparent via-slate-200 to-transparent hidden sm:block" />
          <span className="text-xs font-medium text-slate-400 hidden sm:block truncate tracking-wide">智能分析平台</span>
        </div>

        {/* Center: Quick Navigation */}
        <div className="hidden md:flex items-center gap-1 flex-shrink-0">
          <Link href="/training">
            <Button variant="ghost" size="sm" className="text-slate-500 hover:text-slate-700 hover:bg-slate-100/80 rounded-lg h-8 px-3 text-xs font-medium">
              <LayoutDashboard className="h-3.5 w-3.5 mr-1.5" />
              训练
            </Button>
          </Link>
          <Link href="/results">
            <Button variant="ghost" size="sm" className="text-slate-500 hover:text-slate-700 hover:bg-slate-100/80 rounded-lg h-8 px-3 text-xs font-medium">
              <FileText className="h-3.5 w-3.5 mr-1.5" />
              结果
            </Button>
          </Link>
          <Link href="/visualizations">
            <Button variant="ghost" size="sm" className="text-slate-500 hover:text-slate-700 hover:bg-slate-100/80 rounded-lg h-8 px-3 text-xs font-medium">
              <BarChart3 className="h-3.5 w-3.5 mr-1.5" />
              可视化
            </Button>
          </Link>
        </div>

        {/* Right: Settings + User */}
        <div className="flex items-center gap-1.5 sm:gap-2.5 flex-shrink-0">
          <Link href="/admin/monitor">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 sm:h-9 sm:w-9 text-slate-400 hover:text-slate-700 hover:bg-slate-100/80 rounded-xl bg-transparent transition-all duration-200"
              title="系统监控"
            >
              <Activity className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
            </Button>
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 sm:h-9 sm:w-9 text-slate-400 hover:text-slate-700 hover:bg-slate-100/80 rounded-xl bg-transparent transition-all duration-200"
            title="设置"
          >
            <Settings className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
          </Button>
          
          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Avatar className="h-8 w-8 sm:h-9 sm:w-9 cursor-pointer ring-2 ring-white shadow-md hover:shadow-lg hover:scale-105 transition-all duration-200">
                <AvatarFallback className="bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600 text-white text-xs sm:text-sm font-semibold">
                  {getUserInitials()}
                </AvatarFallback>
              </Avatar>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <div className="px-3 py-2 border-b">
                <p className="text-sm font-medium text-slate-900">{user?.username || "用户"}</p>
                <p className="text-xs text-slate-500">{user?.email || ""}</p>
              </div>
              <DropdownMenuItem onClick={() => router.push("/profile")}>
                <User className="h-4 w-4 mr-2" />
                个人资料
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => router.push("/admin/monitor")}>
                <Activity className="h-4 w-4 mr-2" />
                系统监控
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleLogout} className="text-red-600 focus:text-red-600">
                <LogOut className="h-4 w-4 mr-2" />
                退出登录
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Main Content Area - min-w-0 so flex children can shrink */}
      <div className="flex-1 flex overflow-hidden min-h-0 min-w-0">
        {/* Center Workspace - Project Hub */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0 min-h-0">
          {/* Tab Bar */}
          <div className="h-11 bg-white/80 border-b border-slate-200 flex items-center px-2 sm:px-4 gap-1 overflow-x-auto">
            {tabs.map((tab) => {
              const isHub = tab.id === "hub"
              const isActive = activeTabId === tab.id
              return (
                <div
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  className={`group relative flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 h-8 rounded-lg cursor-pointer transition-all duration-200 whitespace-nowrap ${
                    isActive
                      ? isHub
                        ? "bg-blue-50 text-blue-700 ring-1 ring-blue-200/80"
                        : "bg-slate-100 text-slate-800 ring-1 ring-slate-200"
                      : "text-slate-500 hover:bg-slate-50 hover:text-slate-700"
                  }`}
                >
                  {isHub && <FolderOpen className="h-3.5 w-3.5 shrink-0" />}
                  <span className="text-xs sm:text-sm font-medium">{tab.title}</span>
                  {tab.closable && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onTabClose(tab.id)
                      }}
                      className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-slate-200 transition-all"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  )}
                </div>
              )
            })}
          </div>

          {/* Content Viewport */}
          <div className="flex-1 overflow-auto">
            {children}
          </div>
        </div>

        {/* 右侧：拖拽条（展开时）或展开按钮（收起时） */}
        {showAiSidebar ? (
          <>
            <div
              role="separator"
              aria-label="调整猫咪科学家宽度"
              onMouseDown={handleResizeStart}
              className={`flex-shrink-0 w-1.5 flex flex-col items-center justify-center bg-slate-200/80 hover:bg-blue-200/80 cursor-col-resize transition-colors select-none ${isResizing ? "bg-blue-300" : ""}`}
              style={{ minWidth: 6 }}
            >
              <div className="w-0.5 h-8 rounded-full bg-slate-400 pointer-events-none" />
            </div>
            <div
              className="flex flex-col shrink-0 overflow-hidden border-l border-slate-200/60 bg-white"
              style={{ width: sidebarWidth, minWidth: sidebarWidth, maxWidth: sidebarWidth }}
            >
              <AiSidebar
                chatHistory={chatHistory}
                onSendMessage={onSendMessage}
                onDataUploaded={onDataUploaded}
                onFocusChart={onFocusChart}
                onClearChat={onClearChat}
                onCollapse={() => setShowAiSidebar(false)}
              />
            </div>
          </>
        ) : (
          <button
            type="button"
            onClick={() => setShowAiSidebar(true)}
            className="flex-shrink-0 w-10 flex flex-col items-center justify-center gap-1.5 py-4 bg-slate-100 hover:bg-blue-50 border-l border-slate-200 text-slate-500 hover:text-blue-600 transition-colors"
            title="展开猫咪科学家"
          >
            <Sparkles className="h-5 w-5" />
            <span className="text-[10px] font-medium">AI</span>
          </button>
        )}
      </div>
    </div>
  )
}

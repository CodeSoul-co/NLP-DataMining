"use client"

import React from "react"

import { useState, useRef, useCallback } from "react"
import {
  Sparkles,
  Paperclip,
  Send,
  Clock,
  Activity,
  Zap,
  Upload,
  FileText,
  Check,
  BarChart3,
  ExternalLink,
  Trash2,
  Mic,
  PanelRightClose,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { TypingMessage } from "@/components/typing-message"

// Message Types
export type MessageType = "text" | "chart_widget" | "table_summary" | "file_upload"

export interface ChatMessage {
  id: string
  role: "user" | "ai"
  content: string
  type: MessageType
  timestamp: string
  data?: {
    chartData?: { label: string; value: number }[]
    chartId?: string
    file?: { name: string; size: string; parsed: boolean }
    tableSummary?: { rows: number; columns: number; preview: string[] }
  }
  followUpQuestions?: string[]
}

interface AiSidebarProps {
  chatHistory: ChatMessage[]
  onSendMessage: (content: string) => void
  onDataUploaded?: (file: File) => void
  onFocusChart?: (chartId: string) => void
  onClearChat?: () => void
  onCollapse?: () => void
}

// Mini Chart Component
function MiniChart({ data, chartId, onFocusChart }: {
  data: { label: string; value: number }[]
  chartId: string
  onFocusChart?: (chartId: string) => void
}) {
  const maxValue = Math.max(...data.map(d => d.value))
  
  return (
    <div className="mt-3 p-3 bg-slate-50 rounded-xl border border-slate-100">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-slate-500">数据可视化</span>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs text-blue-600 hover:text-blue-700 hover:bg-blue-50 gap-1"
          onClick={() => {
            console.log("[v0] Focus chart:", chartId)
            onFocusChart?.(chartId)
          }}
        >
          <ExternalLink className="h-3 w-3" />
          在工作区查看
        </Button>
      </div>
      <div className="flex items-end gap-1.5 h-20">
        {data.map((item, index) => (
          <div key={index} className="flex-1 flex flex-col items-center gap-1">
            <div
              className="w-full bg-gradient-to-t from-blue-500 to-blue-400 rounded-t-sm transition-all hover:from-blue-600 hover:to-blue-500"
              style={{ height: `${(item.value / maxValue) * 100}%` }}
            />
            <span className="text-[10px] text-slate-400 truncate w-full text-center">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// File Card Component
function FileCard({ file }: { file: { name: string; size: string; parsed: boolean } }) {
  return (
    <div className="mt-2 flex items-center gap-3 p-3 bg-slate-50 rounded-xl border border-slate-100">
      <div className="h-10 w-10 rounded-lg bg-blue-50 flex items-center justify-center">
        <FileText className="h-5 w-5 text-blue-600" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-800 truncate">{file.name}</p>
        <p className="text-xs text-slate-400">{file.size}</p>
      </div>
      {file.parsed && (
        <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center">
          <Check className="h-3.5 w-3.5 text-green-600" />
        </div>
      )}
    </div>
  )
}

// Table Summary Component
function TableSummary({ data }: { data: { rows: number; columns: number; preview: string[] } }) {
  return (
    <div className="mt-3 p-3 bg-slate-50 rounded-xl border border-slate-100">
      <div className="flex items-center gap-4 mb-2">
        <span className="text-xs text-slate-500">
          <span className="font-semibold text-slate-700">{data.rows.toLocaleString()}</span> 行
        </span>
        <span className="text-xs text-slate-500">
          <span className="font-semibold text-slate-700">{data.columns}</span> 列
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {data.preview.map((col, i) => (
          <Badge key={i} variant="secondary" className="text-xs bg-white border-slate-200">
            {col}
          </Badge>
        ))}
      </div>
    </div>
  )
}

// Message Bubble Component - AI 文本消息支持打字机效果
function MessageBubble({ 
  message, 
  isLatestAiMessage,
  onFocusChart,
  onFollowUpClick 
}: { 
  message: ChatMessage
  isLatestAiMessage?: boolean
  onFocusChart?: (chartId: string) => void
  onFollowUpClick?: (question: string) => void
}) {
  const isUser = message.role === "user"
  const showTyping = !isUser && message.type === "text" && isLatestAiMessage && message.content.length > 0
  
  return (
    <div className={`flex gap-2.5 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {!isUser && (
        <Avatar className="h-9 w-9 shrink-0 ring-2 ring-white shadow-sm overflow-hidden flex items-center justify-center bg-slate-100/50">
          <AvatarImage src="/ai-avatar.png" alt="猫咪科学家" className="scale-75 object-contain w-[140%] h-[140%]" />
          <AvatarFallback className="bg-gradient-to-br from-blue-500 to-blue-600 text-white text-xs">
            <Sparkles className="h-4 w-4" />
          </AvatarFallback>
        </Avatar>
      )}
      
      <div className={`max-w-[85%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`px-4 py-2.5 text-sm leading-relaxed ${
            isUser
              ? "bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-2xl rounded-tr-md shadow-md"
              : "bg-white border border-slate-100 text-slate-700 rounded-2xl rounded-tl-md shadow-sm"
          }`}
        >
          {showTyping ? (
            <TypingMessage
              content={message.content}
              isLatest={true}
              className="text-slate-700"
              speed={12}
            />
          ) : (
            <p>{message.content}</p>
          )}
          
          {/* Chart Widget */}
          {message.type === "chart_widget" && message.data?.chartData && (
            <MiniChart 
              data={message.data.chartData} 
              chartId={message.data.chartId || "chart-1"}
              onFocusChart={onFocusChart}
            />
          )}
          
          {/* File Upload Card */}
          {message.type === "file_upload" && message.data?.file && (
            <FileCard file={message.data.file} />
          )}
          
          {/* Table Summary */}
          {message.type === "table_summary" && message.data?.tableSummary && (
            <TableSummary data={message.data.tableSummary} />
          )}
        </div>
        
        {/* Follow-up Questions */}
        {!isUser && message.followUpQuestions && message.followUpQuestions.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2">
            {message.followUpQuestions.map((question, i) => (
              <Badge
                key={i}
                variant="outline"
                className="text-xs cursor-pointer bg-white hover:bg-blue-50 hover:border-blue-200 hover:text-blue-700 transition-all"
                onClick={() => onFollowUpClick?.(question)}
              >
                {question}
              </Badge>
            ))}
          </div>
        )}
        
        <p className={`text-[10px] text-slate-400 mt-1.5 ${isUser ? "text-right" : "text-left"}`}>
          {message.timestamp}
        </p>
      </div>
    </div>
  )
}

export function AiSidebar({
  chatHistory,
  onSendMessage,
  onDataUploaded,
  onFocusChart,
  onClearChat,
  onCollapse,
}: AiSidebarProps) {
  const [inputValue, setInputValue] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  // Handle file upload
  const handleFileUpload = useCallback((file: File) => {
    console.log("[v0] File uploaded:", file.name, file.size)
    
    // Trigger callback to main workspace
    onDataUploaded?.(file)
    
    // Send a message about the upload
    onSendMessage(`已上传文件: ${file.name}`)
  }, [onDataUploaded, onSendMessage])

  // Drag and drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }, [handleFileUpload])

  // Handle paperclip click
  const handleAttachClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileUpload(files[0])
    }
    // Reset input
    e.target.value = ""
  }, [handleFileUpload])

  // Handle send message
  const handleSend = useCallback(() => {
    if (inputValue.trim()) {
      onSendMessage(inputValue.trim())
      setInputValue("")
    }
  }, [inputValue, onSendMessage])

  // Handle follow-up click
  const handleFollowUpClick = useCallback((question: string) => {
    onSendMessage(question)
  }, [onSendMessage])

  // Handle key press
  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend])

  return (
    <aside 
      className="w-full min-w-0 h-full flex flex-col bg-gradient-to-b from-white to-slate-50/30 border-l border-slate-200/60 shadow-lg shadow-slate-200/30 relative overflow-hidden"
      style={{ maxWidth: "100%" }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag Overlay */}
      {isDragOver && (
        <div className="absolute inset-0 z-50 bg-gradient-to-br from-blue-500/15 to-indigo-500/15 backdrop-blur-md border-2 border-dashed border-blue-400 rounded-lg flex flex-col items-center justify-center">
          <div className="h-20 w-20 rounded-2xl bg-gradient-to-br from-blue-100 to-indigo-100 flex items-center justify-center mb-4 shadow-xl shadow-blue-200/50">
            <Upload className="h-10 w-10 text-blue-600" />
          </div>
          <p className="text-lg font-bold text-blue-700">拖放文件以用 AI 分析</p>
          <p className="text-sm text-blue-500/80 mt-1.5 font-medium">支持 CSV, Excel, JSON 格式</p>
        </div>
      )}

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept=".csv,.xlsx,.xls,.json"
        onChange={handleFileInputChange}
      />

      {/* Header - constrained to sidebar width */}
      <div className="h-14 flex-shrink-0 border-b border-slate-100/80 px-3 sm:px-4 flex items-center justify-between bg-white/80 backdrop-blur-sm min-w-0 w-full">
        <div className="flex items-center gap-2.5 sm:gap-3 min-w-0 overflow-hidden">
          <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl overflow-hidden shadow-lg shrink-0 flex items-center justify-center bg-slate-100/50">
          <img src="/ai-avatar.png" alt="猫咪科学家" className="w-[140%] h-[140%] object-contain scale-75" />
        </div>
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-bold text-slate-800 text-sm sm:text-base tracking-tight">猫咪科学家</span>
            <div className="hidden sm:flex items-center gap-1.5 px-2 py-0.5 bg-emerald-50 rounded-full">
              <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[10px] text-emerald-600 font-semibold">在线</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-8 w-8 text-slate-400 hover:text-red-500 hover:bg-red-50 bg-transparent rounded-xl transition-all duration-200"
            onClick={onClearChat}
            title="清除对话"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8 text-slate-400 hover:text-slate-600 hover:bg-slate-100 bg-transparent rounded-xl transition-all duration-200">
            <Activity className="h-3.5 w-3.5" />
          </Button>
          {onCollapse && (
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-8 w-8 text-slate-400 hover:text-slate-600 hover:bg-slate-100 bg-transparent rounded-xl transition-all duration-200"
              onClick={onCollapse}
              title="收起边栏"
            >
              <PanelRightClose className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>

      {/* Tabs - constrained */}
      <div className="flex-shrink-0 flex items-center gap-4 px-4 py-2.5 border-b border-slate-100/60 bg-gradient-to-r from-slate-50/50 to-white min-w-0 w-full">
        <button className="flex items-center gap-1.5 text-sm text-slate-700 hover:text-slate-900 transition-colors font-semibold">
          <Clock className="h-3.5 w-3.5" />
          <span>历史</span>
          <span className="ml-0.5 px-2 py-0.5 bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-[10px] rounded-full font-bold shadow-sm">
            {chatHistory.length}
          </span>
        </button>
        <button className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-600 transition-colors font-medium">
          <Activity className="h-3.5 w-3.5" />
          <span>状态</span>
        </button>
      </div>

      {/* Chat Body - flex-1 min-h-0 so it shrinks */}
      <ScrollArea className="flex-1 min-h-0 min-w-0 overflow-hidden" ref={scrollAreaRef}>
        {chatHistory.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-72 px-8">
            <div className="relative">
              <div className="h-28 w-28 rounded-3xl overflow-hidden mb-6 shadow-2xl shadow-slate-200/50 flex items-center justify-center bg-slate-100/50">
              <img src="/ai-avatar.png" alt="猫咪科学家" className="w-[140%] h-[140%] object-contain scale-75" />
            </div>
              <div className="absolute -top-1 -right-1 h-6 w-6 rounded-full bg-gradient-to-br from-emerald-400 to-emerald-500 flex items-center justify-center shadow-lg animate-bounce">
                <span className="text-white text-xs">+</span>
              </div>
            </div>
            <p className="text-sm text-slate-700 text-center leading-relaxed font-semibold">
              开始对话，让猫咪科学家帮您分析数据
            </p>
            <p className="text-xs text-slate-400 text-center mt-2 font-medium">
              您也可以直接拖放文件到此处
            </p>
          </div>
        ) : (
          <div className="p-4 space-y-4">
            {(() => {
              const lastAiMessageId = chatHistory.length > 0
                ? [...chatHistory].reverse().find((m) => m.role === "ai")?.id
                : undefined
              return chatHistory.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  isLatestAiMessage={message.role === "ai" && message.id === lastAiMessageId}
                  onFocusChart={onFocusChart}
                  onFollowUpClick={handleFollowUpClick}
                />
              ))
            })()}
          </div>
        )}
      </ScrollArea>

      {/* Smart Suggestions - compact */}
      {chatHistory.length === 0 && (
        <div className="flex-shrink-0 px-3 py-2.5 border-t border-slate-100/60 bg-gradient-to-b from-white via-white to-slate-50/50 min-w-0 w-full overflow-hidden">
          <div className="flex items-center gap-1.5 mb-2">
            <div className="h-5 w-5 rounded-md bg-gradient-to-br from-amber-100 to-orange-100 flex items-center justify-center">
              <Zap className="h-3 w-3 text-amber-600" />
            </div>
            <span className="text-[10px] font-bold text-slate-500 tracking-wide uppercase">智能建议</span>
          </div>
          <div className="space-y-1.5">
            <button 
              className="w-full flex items-center justify-between p-2.5 bg-white hover:bg-blue-50/80 border border-slate-200/60 hover:border-blue-200 rounded-lg transition-all duration-200 group shadow-sm"
              onClick={() => onSendMessage("如何开始数据分析？")}
            >
              <div className="flex flex-col items-start gap-0 min-w-0">
                <span className="text-xs font-semibold text-slate-800 group-hover:text-blue-700 transition-colors">开始分析</span>
                <span className="text-[10px] text-slate-400 group-hover:text-blue-500/70 transition-colors truncate max-w-full">了解如何上传和分析数据</span>
              </div>
              <div className="h-6 w-6 shrink-0 rounded-md bg-slate-100 group-hover:bg-blue-100 flex items-center justify-center transition-colors">
                <BarChart3 className="h-3.5 w-3.5 text-slate-400 group-hover:text-blue-600 transition-colors" />
              </div>
            </button>
            <button 
              className="w-full flex items-center justify-between p-2.5 bg-white hover:bg-blue-50/80 border border-slate-200/60 hover:border-blue-200 rounded-lg transition-all duration-200 group shadow-sm"
              onClick={() => onSendMessage("查看帮助文档")}
            >
              <div className="flex flex-col items-start gap-0 min-w-0">
                <span className="text-xs font-semibold text-slate-800 group-hover:text-blue-700 transition-colors">查看帮助</span>
                <span className="text-[10px] text-slate-400 group-hover:text-blue-500/70 transition-colors truncate max-w-full">了解如何使用 THETA 系统</span>
              </div>
              <div className="h-6 w-6 shrink-0 rounded-md bg-slate-100 group-hover:bg-blue-100 flex items-center justify-center transition-colors">
                <Zap className="h-3.5 w-3.5 text-slate-400 group-hover:text-blue-600 transition-colors" />
              </div>
            </button>
          </div>
        </div>
      )}

      {/* Footer Input - strict width so it never overflows at any zoom/size */}
      <div className="flex-shrink-0 p-3 sm:p-4 border-t border-slate-100/60 bg-white/50 w-full min-w-0 max-w-full overflow-hidden box-border">
        <div className="w-full max-w-full min-w-0 bg-white border border-slate-200/60 rounded-2xl transition-all duration-300 focus-within:ring-2 focus-within:ring-blue-200 focus-within:border-blue-300 overflow-hidden box-border">
          <div className="w-full min-w-0 max-w-full overflow-hidden box-border px-2 sm:px-3 pt-2 sm:pt-3">
            <Textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="输入消息或拖放文件..."
              className="w-full max-w-full min-h-[44px] sm:min-h-[52px] max-h-[100px] resize-none border-0 bg-transparent p-2 sm:p-3 text-sm focus-visible:ring-0 focus-visible:ring-offset-0 shadow-none placeholder:text-slate-400 min-w-0 box-border"
            />
          </div>
          <div className="flex items-center justify-between gap-1 sm:gap-2 px-2 sm:px-3 py-2 border-t border-slate-100/60 min-w-0 w-full max-w-full box-border">
            <div className="flex items-center gap-0.5 shrink-0 min-w-0">
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8 shrink-0 text-slate-400 hover:text-blue-600 hover:bg-blue-50 bg-transparent rounded-xl"
                onClick={handleAttachClick}
                title="上传文件"
              >
                <Paperclip className="h-4 w-4" />
              </Button>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8 shrink-0 text-slate-400 hover:text-blue-600 hover:bg-blue-50 bg-transparent rounded-xl"
                title="语音输入"
              >
                <Mic className="h-4 w-4" />
              </Button>
            </div>
            <Button 
              size="sm"
              className="h-8 px-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 shadow-md disabled:opacity-50 disabled:cursor-not-allowed text-xs font-semibold shrink-0"
              onClick={handleSend}
              disabled={!inputValue.trim()}
            >
              <Send className="h-3.5 w-3.5 mr-1" />
              发送
            </Button>
          </div>
        </div>
      </div>
    </aside>
  )
}

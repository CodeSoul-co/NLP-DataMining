"use client"

import type React from "react"
import { Suspense } from "react"

import { useState, useCallback, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { useRouter, useSearchParams } from "next/navigation"
import { useAuth } from "@/contexts/auth-context"
import Link from "next/link"
import {
  Upload,
  Menu,
  User,
  Database,
  FileCog,
  BrainCircuit,
  PieChart,
  FileText,
  Folder,
  Plus,
  Paperclip,
  Send,
  LogOut,
  Settings,
  PanelLeftClose,
  PanelLeft,
  PanelRightClose,
  PanelRight,
  ArrowLeft,
  Trash2,
  X,
  GraduationCap,
  FileCheck,
  MessageSquare,
  Activity,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
// 图表组件暂时不使用
// import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { DataProcessingView } from "@/components/data-processing"
import { InteractiveChart } from "@/components/interactive-chart"
import { Progress } from "@/components/ui/progress"
import { MarkdownRenderer } from "@/components/markdown-renderer"
import { TypingMessage } from "@/components/typing-message"
import { DataCleanAPI } from "@/lib/api/dataclean"
import { Download, CheckCircle2, XCircle, Loader2, Clock, Zap, BarChart3, ExternalLink, Image, TrendingUp, RefreshCw, AlertCircle, Search, Edit2, Filter } from "lucide-react"
import { ETMAgentAPI, TaskResponse, CreateTaskRequest, ResultInfo, VisualizationInfo } from "@/lib/api/etm-agent"
import { useETMWebSocket } from "@/hooks/use-etm-websocket"
import { ProtectedRoute } from "@/components/protected-route"

type ViewType = "data" | "processing" | "embedding" | "tasks" | "results" | "visualizations"

type AppState = "idle" | "chatting" | "workspace"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
}

type ProcessingJob = {
  id: string
  taskId?: string  // DataClean API 返回的任务ID
  name: string
  sourceDataset: string
  sourceDatasetId: string
  fileCount: number
  status: "pending" | "processing" | "completed" | "failed"
  progress: number
  date: string
  resultFile?: string  // 处理结果文件名
  error?: string
}

// 数据集中的文件类型
type DatasetFile = {
  id: string
  name: string
  size: string
  type: string
  uploadDate: string
}

// 数据集类型（包含文件列表）
type Dataset = {
  id: string
  name: string
  files: DatasetFile[]
  totalSize: string
  date: string
}

// 生成默认数据集名称
const generateDefaultDatasetName = (existingDatasets: Dataset[]): string => {
  const baseName = "未命名数据集"
  let counter = 1
  let newName = baseName
  
  while (existingDatasets.some(d => d.name === newName)) {
    counter++
    newName = `${baseName} ${counter}`
  }
  
  return newName
}

// 计算文件总大小
const calculateTotalSize = (files: DatasetFile[]): string => {
  let totalBytes = 0
  files.forEach(file => {
    const match = file.size.match(/^([\d.]+)\s*(KB|MB|GB)$/i)
    if (match) {
      const value = parseFloat(match[1])
      const unit = match[2].toUpperCase()
      if (unit === 'KB') totalBytes += value * 1024
      else if (unit === 'MB') totalBytes += value * 1024 * 1024
      else if (unit === 'GB') totalBytes += value * 1024 * 1024 * 1024
    }
  })
  
  if (totalBytes >= 1024 * 1024 * 1024) {
    return `${(totalBytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  } else if (totalBytes >= 1024 * 1024) {
    return `${(totalBytes / (1024 * 1024)).toFixed(2)} MB`
  } else if (totalBytes >= 1024) {
    return `${(totalBytes / 1024).toFixed(2)} KB`
  }
  return `${totalBytes} B`
}

// 图表数据已移动到独立页面

// 主页面内容组件
function HomeContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  
  // ============================================
  // 所有 useState 必须在任何条件返回之前声明
  // ============================================
  const [appState, setAppState] = useState<AppState | null>(null) // null 表示正在初始化
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [chatSidebarCollapsed, setChatSidebarCollapsed] = useState(false) // 右侧 AI 助手收纳状态
  const [currentView, setCurrentView] = useState<ViewType>("data")
  const [isInitialized, setIsInitialized] = useState(false)
  const [showNameModal, setShowNameModal] = useState(false)
  const [showSourceModal, setShowSourceModal] = useState(false)
  const [datasetName, setDatasetName] = useState("")
  const [selectedSource, setSelectedSource] = useState("")
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [processingJobs, setProcessingJobs] = useState<ProcessingJob[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [inputValue, setInputValue] = useState("")
  const [chatHistory, setChatHistory] = useState<Message[]>([])
  const [sheetOpen, setSheetOpen] = useState(false)
  const [suggestions, setSuggestions] = useState<Array<{ text: string; action: string; description: string; data?: Record<string, unknown> }>>([])
  const [loadingSuggestions, setLoadingSuggestions] = useState(false)
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null)
  
  // 存储实际上传的文件（用于 API 调用）
  const [uploadedFilesMap, setUploadedFilesMap] = useState<Map<string, File[]>>(new Map())
  
  // 文件上传状态
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadError, setUploadError] = useState<string | null>(null)
  
  // 上传已清洗数据弹窗状态
  const [showUploadCleanedModal, setShowUploadCleanedModal] = useState(false)
  const [cleanedFilesToUpload, setCleanedFilesToUpload] = useState<File[]>([])
  const [cleanedDatasetName, setCleanedDatasetName] = useState("")
  
  // 删除数据集相关状态
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [datasetToDelete, setDatasetToDelete] = useState<string | null>(null)
  
  // 重命名数据集相关状态
  const [showRenameModal, setShowRenameModal] = useState(false)
  const [datasetToRename, setDatasetToRename] = useState<string | null>(null)
  const [newDatasetName, setNewDatasetName] = useState("")
  
  // 聊天消息自动滚动引用
  const chatMessagesEndRef = useRef<HTMLDivElement>(null)
  
  // 对话历史持久化
  const CHAT_HISTORY_KEY = 'theta_chat_history'
  const CHAT_SESSIONS_KEY = 'theta_chat_sessions'
  const CURRENT_SESSION_KEY = 'theta_current_session'
  
  // 历史对话会话状态
  const [chatSessions, setChatSessions] = useState<Array<{
    id: string
    title: string
    messages: Message[]
    createdAt: string
    updatedAt: string
  }>>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [currentContextDisplay, setCurrentContextDisplay] = useState<string>('')
  const [showStatusPanel, setShowStatusPanel] = useState(false)
  
  // 从 localStorage 加载对话历史和会话
  useEffect(() => {
    try {
      // 加载会话列表
      const savedSessions = localStorage.getItem(CHAT_SESSIONS_KEY)
      if (savedSessions) {
        const parsed = JSON.parse(savedSessions)
        // 只保留最近 20 个会话
        const recentSessions = Array.isArray(parsed) ? parsed.slice(-20) : []
        setChatSessions(recentSessions)
      }
      
      // 加载当前会话 ID
      const savedCurrentSession = localStorage.getItem(CURRENT_SESSION_KEY)
      if (savedCurrentSession) {
        setCurrentSessionId(savedCurrentSession)
      }
      
      // 加载当前对话历史
      const savedHistory = localStorage.getItem(CHAT_HISTORY_KEY)
      if (savedHistory) {
        const parsed = JSON.parse(savedHistory)
        // 只保留最近 50 条消息
        const recentHistory = Array.isArray(parsed) ? parsed.slice(-50) : []
        setChatHistory(recentHistory)
      }
    } catch (error) {
      console.error('Failed to load chat history:', error)
    }
  }, [])
  
  // 保存对话历史到 localStorage
  useEffect(() => {
    if (chatHistory.length > 0) {
      try {
        localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatHistory))
      } catch (error) {
        console.error('Failed to save chat history:', error)
      }
    }
  }, [chatHistory])
  
  // 保存会话列表到 localStorage
  useEffect(() => {
    try {
      localStorage.setItem(CHAT_SESSIONS_KEY, JSON.stringify(chatSessions))
    } catch (error) {
      console.error('Failed to save chat sessions:', error)
    }
  }, [chatSessions])
  
  // 清空当前对话
  const handleClearChat = useCallback(() => {
    if (chatHistory.length === 0) return
    
    // 保存当前对话到历史会话
    const now = new Date().toISOString()
    const firstUserMsg = chatHistory.find(m => m.role === 'user')
    const title = firstUserMsg 
      ? firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? '...' : '')
      : '未命名对话'
    
    const newSession = {
      id: Date.now().toString(),
      title,
      messages: chatHistory,
      createdAt: now,
      updatedAt: now,
    }
    
    setChatSessions(prev => [...prev.slice(-19), newSession]) // 最多保留 20 个
    setChatHistory([])
    setCurrentSessionId(null)
    localStorage.removeItem(CHAT_HISTORY_KEY)
    localStorage.removeItem(CURRENT_SESSION_KEY)
  }, [chatHistory])
  
  // 加载历史会话
  const handleLoadSession = useCallback((sessionId: string) => {
    const session = chatSessions.find(s => s.id === sessionId)
    if (session) {
      // 如果当前有对话，先保存
      if (chatHistory.length > 0 && currentSessionId !== sessionId) {
        const now = new Date().toISOString()
        const firstUserMsg = chatHistory.find(m => m.role === 'user')
        const title = firstUserMsg 
          ? firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? '...' : '')
          : '未命名对话'
        
        if (currentSessionId) {
          // 更新现有会话
          setChatSessions(prev => prev.map(s => 
            s.id === currentSessionId 
              ? { ...s, messages: chatHistory, updatedAt: now }
              : s
          ))
        } else {
          // 创建新会话
          const newSession = {
            id: Date.now().toString(),
            title,
            messages: chatHistory,
            createdAt: now,
            updatedAt: now,
          }
          setChatSessions(prev => [...prev.slice(-19), newSession])
        }
      }
      
      // 加载选中的会话
      setChatHistory(session.messages)
      setCurrentSessionId(sessionId)
      localStorage.setItem(CURRENT_SESSION_KEY, sessionId)
    }
  }, [chatSessions, chatHistory, currentSessionId])
  
  // 删除历史会话
  const handleDeleteSession = useCallback((sessionId: string) => {
    setChatSessions(prev => prev.filter(s => s.id !== sessionId))
    if (currentSessionId === sessionId) {
      setChatHistory([])
      setCurrentSessionId(null)
      localStorage.removeItem(CHAT_HISTORY_KEY)
      localStorage.removeItem(CURRENT_SESSION_KEY)
    }
  }, [currentSessionId])
  
  // 清空对话但不保存到历史
  const handleClearWithoutSave = useCallback(() => {
    setChatHistory([])
    setCurrentSessionId(null)
    localStorage.removeItem(CHAT_HISTORY_KEY)
    localStorage.removeItem(CURRENT_SESSION_KEY)
  }, [])

  // 自动滚动聊天消息到底部
  useEffect(() => {
    if (chatMessagesEndRef.current) {
      chatMessagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [chatHistory])

  // 获取默认建议（当后端不可用时使用）
  const getDefaultSuggestions = useCallback((view: ViewType) => {
    const defaultSuggestions: Record<ViewType, Array<{ text: string; action: string; description: string; data?: Record<string, unknown> }>> = {
      data: [
        { text: "上传数据集", action: "navigate", description: "上传 CSV、JSON 或文本文件", data: { view: "data" } },
        { text: "查看现有数据", action: "navigate", description: "浏览已上传的数据集", data: { view: "data" } },
      ],
      processing: [
        { text: "开始数据清洗", action: "start_cleaning", description: "清洗和预处理数据" },
        { text: "上传已清洗数据", action: "navigate", description: "直接上传已处理的数据", data: { view: "processing" } },
      ],
      embedding: [
        { text: "生成向量化", action: "navigate", description: "生成 BOW 矩阵和词嵌入", data: { view: "embedding" } },
      ],
      tasks: [
        { text: "创建训练任务", action: "navigate", description: "配置并提交 ETM 训练任务", data: { view: "tasks" } },
        { text: "查看任务列表", action: "navigate", description: "查看所有任务的状态和进度", data: { view: "tasks" } },
      ],
      results: [
        { text: "查看分析结果", action: "navigate", description: "查看主题词和评估指标", data: { view: "results" } },
        { text: "导出结果", action: "navigate", description: "导出 CSV 或 JSON 格式", data: { view: "results" } },
      ],
      visualizations: [
        { text: "查看可视化", action: "navigate", description: "查看主题分布图和词云", data: { view: "visualizations" } },
      ],
    }
    return defaultSuggestions[view] || []
  }, [])

  // 收集当前界面上下文信息（用于 AI 助手感知界面状态）
  const getCurrentContext = useCallback(() => {
    const viewNames: Record<ViewType, string> = {
      data: "数据管理",
      processing: "数据清洗",
      embedding: "向量化",
      tasks: "任务中心",
      results: "分析结果",
      visualizations: "可视化图表",
    }
    
    return {
      current_view: currentView,
      current_view_name: viewNames[currentView],
      app_state: appState,
      datasets_count: datasets.length,
      datasets: datasets.map(d => ({
        name: d.name,
        fileCount: d.files.length,
        totalSize: d.totalSize,
      })),
      processing_jobs_count: processingJobs.length,
      selected_dataset: selectedDatasetId ? datasets.find(d => d.id === selectedDatasetId)?.name : null,
    }
  }, [currentView, appState, datasets, processingJobs.length, selectedDatasetId])

  // 获取当前界面状态显示（用于 AI 助手状态按钮）
  const handleGetStatus = useCallback(() => {
    const context = getCurrentContext()
    const viewNames: Record<ViewType, string> = {
      data: "数据管理",
      processing: "数据清洗",
      embedding: "向量化",
      tasks: "任务中心",
      results: "分析结果",
      visualizations: "可视化图表",
    }
    
    let statusText = `当前页面: ${viewNames[currentView]}\n`
    statusText += `数据集数量: ${datasets.length}\n`
    
    if (datasets.length > 0) {
      statusText += `数据集列表:\n`
      datasets.forEach((d, i) => {
        statusText += `  ${i + 1}. ${d.name} (${d.files.length} 文件, ${d.totalSize})\n`
      })
    }
    
    if (selectedDatasetId) {
      const selected = datasets.find(d => d.id === selectedDatasetId)
      if (selected) {
        statusText += `已选择: ${selected.name}\n`
      }
    }
    
    if (processingJobs.length > 0) {
      statusText += `处理任务: ${processingJobs.length} 个\n`
    }
    
    setCurrentContextDisplay(statusText)
  }, [getCurrentContext, currentView, datasets, selectedDatasetId, processingJobs.length])

  // 当页面状态改变且状态面板打开时，自动更新状态显示
  useEffect(() => {
    if (showStatusPanel) {
      handleGetStatus()
    }
  }, [currentView, datasets, selectedDatasetId, processingJobs.length, showStatusPanel, handleGetStatus])

  // 加载智能建议（当后端不可用时使用默认建议）
  useEffect(() => {
    const loadSuggestions = async () => {
      if (appState === "workspace") {
        setLoadingSuggestions(true)
        try {
          const context = getCurrentContext()
          const response = await ETMAgentAPI.getSuggestions(context)
          setSuggestions(response.suggestions || [])
        } catch {
          // 后端不可用时使用默认建议
          setSuggestions(getDefaultSuggestions(currentView))
        } finally {
          setLoadingSuggestions(false)
        }
      }
    }
    
    loadSuggestions()
  }, [currentView, appState, datasets.length, processingJobs.length, getDefaultSuggestions, getCurrentContext])
  
  // 工作流步骤（常量，不是状态）
  const workflowSteps = [
    { id: "data", label: "数据管理", icon: Database, description: "上传和管理数据集" },
    { id: "processing", label: "数据清洗", icon: FileCog, description: "清洗和预处理数据" },
    { id: "embedding", label: "向量化", icon: BrainCircuit, description: "生成 BOW 和 Embeddings" },
    { id: "tasks", label: "任务中心", icon: GraduationCap, description: "创建和管理训练任务" },
    { id: "results", label: "分析结果", icon: FileCheck, description: "查看分析结果" },
    { id: "visualizations", label: "可视化", icon: PieChart, description: "数据可视化展示" },
  ]
  
  // ============================================
  // 所有 useEffect 也必须在条件返回之前
  // ============================================
  
  // 初始化和监听 URL 参数变化
  useEffect(() => {
    const viewParam = searchParams.get('view')
    const validViews: ViewType[] = ['data', 'processing', 'embedding', 'tasks', 'results', 'visualizations']
    if (viewParam && validViews.includes(viewParam as ViewType)) {
      // 只在视图真正变化时更新
      setCurrentView(prev => prev !== viewParam ? viewParam as ViewType : prev)
      // 只在 appState 不是 workspace 时更新
      setAppState(prev => prev !== "workspace" ? "workspace" : prev)
    } else if (!isInitialized) {
      // 首次加载且没有 view 参数时，显示初始页面
      setAppState("idle")
    }
    setIsInitialized(true)
  }, [searchParams, isInitialized])

  // 页面加载时从后端获取数据集列表
  useEffect(() => {
    const loadDatasets = async () => {
      // 只在已初始化且进入工作空间时加载数据集
      if (!isInitialized || appState !== "workspace") {
        return
      }

      try {
        // 获取后端已有的数据集列表
        const backendDatasets = await ETMAgentAPI.getDatasets()
        
        // 将后端返回的数据集格式转换为前端 Dataset 格式
        const convertedDatasets: Dataset[] = await Promise.all(
          backendDatasets.map(async (ds, index) => {
            // 尝试获取数据集的详细信息
            let files: DatasetFile[] = []
            let datasetSize = ds.size || 0  // 文档数量
            
            try {
              const detail = await ETMAgentAPI.getDataset(ds.name)
              
              // 如果后端返回了详细信息，使用它
              datasetSize = detail.size || datasetSize
              
              // 创建一个表示数据集的虚拟文件条目（因为后端不返回文件列表）
              // 可以根据文档数量估算文件大小
              const estimatedSize = datasetSize > 0 ? `${datasetSize.toLocaleString()} 行` : '未知'
              files = [{
                id: `f-${ds.name}-0`,
                name: `${ds.name} (数据集)`,
                size: estimatedSize,
                type: 'CSV',
                uploadDate: new Date().toISOString().split("T")[0],
              }]
            } catch (error) {
              // 如果获取详情失败，创建一个基本的文件条目
              console.warn(`Failed to get details for dataset ${ds.name}:`, error)
              files = [{
                id: `f-${ds.name}-0`,
                name: `${ds.name} (数据集)`,
                size: datasetSize > 0 ? `${datasetSize.toLocaleString()} 行` : '未知',
                type: 'DATASET',
                uploadDate: new Date().toISOString().split("T")[0],
              }]
            }
            
            // 计算总大小（使用文件数量或文档数量）
            const totalSize = files.length > 0 
              ? files.map(f => f.size).join(' + ') 
              : datasetSize > 0 ? `${datasetSize.toLocaleString()} 行` : '未知'
            
            return {
              id: `ds-backend-${ds.name}`,
              name: ds.name,
              files,
              totalSize,
              date: new Date().toISOString().split("T")[0],
            }
          })
        )

        // 更新数据集列表（合并后端和前端数据集，去重）
        setDatasets(prev => {
          const existingNames = new Set(convertedDatasets.map(d => d.name))
          const frontendOnly = prev.filter(d => !existingNames.has(d.name))
          // 后端数据集优先（如果有重名，使用后端数据）
          return [...convertedDatasets, ...frontendOnly]
        })
      } catch (error) {
        console.error('Failed to load datasets from backend:', error)
        // 不阻止页面渲染，如果后端不可用，使用前端已有数据
      }
    }

    loadDatasets()
  }, [isInitialized, appState])
  
  // ============================================
  // 所有 useCallback 必须在条件返回之前声明
  // ============================================
  
  // 向已有数据集添加文件
  const handleAddFilesToDataset = useCallback((datasetId: string, files: File[]) => {
    const newFiles: DatasetFile[] = files.map((file, index) => ({
      id: `f-${Date.now()}-${index}`,
      name: file.name,
      size: file.size >= 1024 * 1024 
        ? `${(file.size / (1024 * 1024)).toFixed(2)} MB`
        : `${(file.size / 1024).toFixed(2)} KB`,
      type: file.name.split('.').pop()?.toUpperCase() || 'Unknown',
      uploadDate: new Date().toISOString().split("T")[0],
    }))
    
    setDatasets(prev => prev.map(dataset => {
      if (dataset.id === datasetId) {
        const updatedFiles = [...dataset.files, ...newFiles]
        return {
          ...dataset,
          files: updatedFiles,
          totalSize: calculateTotalSize(updatedFiles),
        }
      }
      return dataset
    }))
    
    // 存储实际文件用于后续 API 调用
    setUploadedFilesMap(prev => {
      const newMap = new Map(prev)
      const existing = newMap.get(datasetId) || []
      newMap.set(datasetId, [...existing, ...files])
      return newMap
    })
  }, [])
  
  // 从数据集中删除文件
  const handleRemoveFileFromDataset = useCallback((datasetId: string, fileId: string) => {
    setDatasets(prev => prev.map(dataset => {
      if (dataset.id === datasetId) {
        const updatedFiles = dataset.files.filter(f => f.id !== fileId)
        return {
          ...dataset,
          files: updatedFiles,
          totalSize: calculateTotalSize(updatedFiles),
        }
      }
      return dataset
    }))
  }, [])
  
  // 删除数据集
  const handleDeleteDataset = useCallback((datasetId: string) => {
    setDatasetToDelete(datasetId)
    setShowDeleteConfirm(true)
  }, [])
  
  // 确认删除数据集
  const confirmDeleteDataset = useCallback(() => {
    if (datasetToDelete) {
      // 删除数据集
      setDatasets(prev => prev.filter(d => d.id !== datasetToDelete))
      
      // 删除关联的文件映射
      setUploadedFilesMap(prev => {
        const newMap = new Map(prev)
        newMap.delete(datasetToDelete)
        return newMap
      })
      
      // 如果当前查看的是被删除的数据集，返回列表视图
      if (selectedDatasetId === datasetToDelete) {
        setSelectedDatasetId(null)
      }
      
      // 清理状态
      setDatasetToDelete(null)
      setShowDeleteConfirm(false)
    }
  }, [datasetToDelete, selectedDatasetId])
  
  // 重命名数据集
  const handleRenameDataset = useCallback((datasetId: string) => {
    const dataset = datasets.find(d => d.id === datasetId)
    if (dataset) {
      setDatasetToRename(datasetId)
      setNewDatasetName(dataset.name)
      setShowRenameModal(true)
    }
  }, [datasets])
  
  // 确认重命名数据集
  const confirmRenameDataset = useCallback(() => {
    if (datasetToRename && newDatasetName.trim()) {
      setDatasets(prev => prev.map(d => 
        d.id === datasetToRename 
          ? { ...d, name: newDatasetName.trim() }
          : d
      ))
      
      // 清理状态
      setDatasetToRename(null)
      setNewDatasetName("")
      setShowRenameModal(false)
    }
  }, [datasetToRename, newDatasetName])
  
  // ============================================
  // 在初始化完成前显示加载状态
  // ============================================
  if (appState === null) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-blue-600 mb-2">THETA</h1>
          <p className="text-slate-400">加载中...</p>
        </div>
      </div>
    )
  }

  // ============================================
  // 以下是普通函数（非 hooks），可以在条件返回之后
  // ============================================
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  // 初始页面拖入文件：先收集文件，确认后再命名
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFiles = Array.from(e.dataTransfer.files)
    if (droppedFiles.length > 0) {
      setPendingFiles(droppedFiles)
      // 生成默认名称
      setDatasetName(generateDefaultDatasetName(datasets))
      setShowNameModal(true)
    }
  }

  // 支持的文件格式（仅用于上传已清洗数据）
  const CLEANED_FILE_EXTENSIONS = ['.csv', '.txt', '.json']
  const CLEANED_MIME_TYPES = ['text/csv', 'text/plain', 'application/json', 'text/json']
  
  // 验证文件类型（用于已清洗数据）
  const isValidCleanedFileType = (file: File): boolean => {
    const fileName = file.name.toLowerCase()
    const extension = '.' + fileName.split('.').pop()
    
    // 检查文件扩展名
    if (CLEANED_FILE_EXTENSIONS.includes(extension)) {
      return true
    }
    
    // 检查 MIME 类型（有些浏览器可能不提供）
    if (file.type && CLEANED_MIME_TYPES.includes(file.type.toLowerCase())) {
      return true
    }
    
    return false
  }
  
  // 点击上传按钮：打开文件选择器（数据管理界面 - 允许任何格式的原始数据）
  const handleFileUpload = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    // 不设置 accept，允许选择任何格式的原始数据文件
    input.onchange = (e) => {
      const files = Array.from((e.target as HTMLInputElement).files || [])
      if (files.length > 0) {
        // 数据管理界面允许任何格式，直接设置文件
        setPendingFiles(files)
        setDatasetName(generateDefaultDatasetName(datasets))
        setShowNameModal(true)
      }
    }
    input.click()
  }

  // 确认数据集名称并创建（真正的上传到后端）
  const handleNameConfirm = async () => {
    // 使用用户输入的名称，如果为空则使用默认名称
    let finalName = datasetName.trim() || generateDefaultDatasetName(datasets)
    
    // 检查是否重名
    if (datasets.some(d => d.name === finalName)) {
      // 如果重名，自动添加后缀
      let counter = 1
      let uniqueName = `${finalName} (${counter})`
      while (datasets.some(d => d.name === uniqueName)) {
        counter++
        uniqueName = `${finalName} (${counter})`
      }
      finalName = uniqueName
    }
    
    if (pendingFiles.length === 0) {
      return
    }
    
    // 数据管理界面允许上传任何格式的原始数据（后端会接受所有格式）
    // 不需要在前端验证格式，后端会处理文件保存
    
    // 开始上传到后端
    setIsUploading(true)
    setUploadProgress(0)
    setUploadError(null)
    
    try {
      // 调用真实的上传 API
      const result = await ETMAgentAPI.uploadDataset(
        pendingFiles,
        finalName,
        (progress) => {
          setUploadProgress(progress)
        }
      )
      
      // 上传成功，刷新数据集列表
      try {
        const updatedDatasets = await ETMAgentAPI.getDatasets()
        // 将后端返回的数据集格式转换为前端 Dataset 格式
        const formatSize = (bytes: number): string => {
          if (bytes >= 1024 * 1024 * 1024) {
            return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
          } else if (bytes >= 1024 * 1024) {
            return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
          } else if (bytes >= 1024) {
            return `${(bytes / 1024).toFixed(2)} KB`
          }
          return `${bytes} B`
        }
        
        const convertedDatasets: Dataset[] = await Promise.all(
          updatedDatasets.map(async (ds, index) => {
            // 尝试获取数据集的详细信息以获取文件列表
            let files: DatasetFile[] = []
            try {
              const detail = await ETMAgentAPI.getDataset(ds.name)
              if (detail.files && Array.isArray(detail.files)) {
                files = detail.files.map((f, fIndex) => ({
                  id: `f-${index}-${fIndex}`,
                  name: f.name || f,
                  size: formatSize(f.size || 0),
                  type: (f.name || f).split('.').pop()?.toUpperCase() || 'UNKNOWN',
                  uploadDate: new Date().toISOString().split("T")[0],
                }))
              }
            } catch {
              // 如果获取详情失败，使用空文件列表
            }
            
            return {
              id: `ds-${Date.now()}-${index}`,
              name: ds.name,
              files,
              totalSize: formatSize(ds.size || 0),
              date: new Date().toISOString().split("T")[0],
            }
          })
        )
        setDatasets(convertedDatasets)
        
        // 选择新上传的数据集
        const uploadedDataset = convertedDatasets.find(d => d.name === result.dataset_name)
        if (uploadedDataset) {
          setSelectedDatasetId(uploadedDataset.id)
        }
      } catch (error) {
        console.error('Failed to refresh datasets:', error)
      }
      
      // 添加成功消息到聊天
      const successMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: `✅ 数据集 **${result.dataset_name}** 上传成功！\n\n- 文件数量: ${result.file_count}\n- 总大小: ${(result.total_size / 1024).toFixed(2)} KB\n- 文件: ${result.files.join(', ')}\n\n您可以在向量化步骤中选择此数据集进行处理。`,
      }
      setChatHistory(prev => [...prev, successMessage])
      
      // 关闭弹窗并重置状态
      setShowNameModal(false)
      setPendingFiles([])
      setDatasetName("")
      setAppState("workspace")
      setCurrentView("data")
    } catch (error) {
      console.error('Upload failed:', error)
      const errorMsg = error instanceof Error ? error.message : '上传失败'
      setUploadError(errorMsg)
      
      // 添加错误消息到聊天
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: `❌ 上传失败: ${errorMsg}\n\n请检查后端服务是否运行，然后重试。`,
      }
      setChatHistory(prev => [...prev, errorMessage])
    } finally {
      setIsUploading(false)
      setUploadProgress(0)
    }
  }

  const handleSourceConfirm = async () => {
    if (selectedSource) {
      const sourceDataset = datasets.find((d) => d.id === selectedSource)
      if (sourceDataset) {
        const jobId = `processed-${Date.now()}`
        const newJob: ProcessingJob = {
          id: jobId,
          name: `${sourceDataset.name}_cleaned`,
          sourceDataset: sourceDataset.name,
          sourceDatasetId: sourceDataset.id,
          fileCount: sourceDataset.files.length,
          status: "pending",
          progress: 0,
          date: new Date().toISOString().split("T")[0],
        }
        setProcessingJobs(prev => [...prev, newJob])
        setShowSourceModal(false)
        setCurrentView("processing")
        setSelectedSource("")

        // 获取该数据集的实际文件
        const files = uploadedFilesMap.get(sourceDataset.id)
        
        if (!files || files.length === 0) {
          // 没有实际文件，更新任务状态为失败
          setProcessingJobs(prev => prev.map(job => 
            job.id === jobId ? { ...job, status: "failed", progress: 0 } : job
          ))
          alert('请先上传文件后再进行处理')
        } else {
          // 有实际文件，调用 DataClean API
          try {
            setProcessingJobs(prev => prev.map(job => 
              job.id === jobId ? { ...job, status: "processing", progress: 10 } : job
            ))
            
            // 调用批量处理 API
            const response = await DataCleanAPI.processBatchFiles(
              files,
              'chinese',
              true,
              ['remove_urls', 'remove_html_tags', 'normalize_whitespace']
            )
            
            // 更新任务状态
            setProcessingJobs(prev => prev.map(job => 
              job.id === jobId ? { 
                ...job, 
                taskId: response.task_id,
                status: "completed", 
                progress: 100,
                resultFile: `${sourceDataset.name}_cleaned.csv`
              } : job
            ))
          } catch (error: any) {
            setProcessingJobs(prev => prev.map(job => 
              job.id === jobId ? { 
                ...job, 
                status: "failed", 
                progress: 100,
                error: error.message || "处理失败"
              } : job
            ))
          }
        }
      }
    }
  }

  // 下载处理结果
  const handleDownloadResult = async (job: ProcessingJob) => {
    if (job.taskId) {
      // 有真实的 taskId，调用 API 下载
      try {
        await DataCleanAPI.downloadResultFile(job.taskId, job.resultFile || 'result.csv')
      } catch (error) {
        console.error('下载失败:', error)
        // 如果 API 下载失败，提示用户
        alert('下载失败，请检查后端服务是否运行')
      }
    } else {
      // 没有后端 taskId，无法下载
      alert('此任务没有可下载的文件')
    }
  }

  // 直接处理数据集（从数据集详情页调用）
  const startProcessingDataset = async (dataset: Dataset) => {
    const jobId = `processed-${Date.now()}`
    const newJob: ProcessingJob = {
      id: jobId,
      name: `${dataset.name}_cleaned`,
      sourceDataset: dataset.name,
      sourceDatasetId: dataset.id,
      fileCount: dataset.files.length,
      status: "pending",
      progress: 0,
      date: new Date().toISOString().split("T")[0],
    }
    setProcessingJobs(prev => [...prev, newJob])
    setCurrentView("processing")

    // 获取该数据集的实际文件
    const files = uploadedFilesMap.get(dataset.id)
    
    if (!files || files.length === 0) {
      // 没有实际文件，更新任务状态为失败
      setProcessingJobs(prev => prev.map(job => 
        job.id === jobId ? { ...job, status: "failed", progress: 0 } : job
      ))
      alert('请先上传文件后再进行处理')
    } else {
      // 有实际文件，调用 DataClean API
      try {
        setProcessingJobs(prev => prev.map(job => 
          job.id === jobId ? { ...job, status: "processing", progress: 10 } : job
        ))
        
        // 调用批量处理 API
        const response = await DataCleanAPI.processBatchFiles(
          files,
          'chinese',
          true,
          ['remove_urls', 'remove_html_tags', 'normalize_whitespace']
        )
        
        // 更新任务状态
        setProcessingJobs(prev => prev.map(job => 
          job.id === jobId ? { 
            ...job, 
            taskId: response.task_id,
            status: "completed", 
            progress: 100,
            resultFile: `${dataset.name}_cleaned.csv`
          } : job
        ))
      } catch (error: any) {
        setProcessingJobs(prev => prev.map(job => 
          job.id === jobId ? { 
            ...job, 
            status: "failed", 
            progress: 100,
            error: error.message || "处理失败"
          } : job
        ))
      }
    }
  }

  // 删除处理任务
  const handleDeleteJob = async (jobId: string) => {
    const job = processingJobs.find(j => j.id === jobId)
    if (job?.taskId) {
      // 如果有真实 taskId，也删除服务器上的任务
      try {
        await DataCleanAPI.deleteTask(job.taskId)
      } catch (error) {
        console.error('删除服务器任务失败:', error)
      }
    }
    setProcessingJobs(prev => prev.filter(j => j.id !== jobId))
  }

  // 上传已清洗的数据文件 - 打开文件选择器（限制为 CSV/TXT/JSON）
  const handleUploadCleanedData = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = CLEANED_FILE_EXTENSIONS.join(',')
    input.multiple = true
    input.onchange = (e) => {
      const files = Array.from((e.target as HTMLInputElement).files || [])
      if (files.length === 0) return
      
      // 验证文件类型（已清洗数据必须是 CSV/TXT/JSON）
      const validFiles = files.filter(isValidCleanedFileType)
      const invalidFiles = files.filter(f => !isValidCleanedFileType(f))
      
      if (invalidFiles.length > 0) {
        const invalidNames = invalidFiles.map(f => f.name).join(', ')
        alert(`以下文件格式不支持: ${invalidNames}\n\n已清洗数据必须为以下格式: ${CLEANED_FILE_EXTENSIONS.join(', ')}`)
      }
      
      if (validFiles.length > 0) {
        // 生成默认数据集名称（基于第一个文件名）
        const firstFileName = validFiles[0].name.replace(/\.[^/.]+$/, '')
        setCleanedFilesToUpload(validFiles)
        setCleanedDatasetName(firstFileName)
        setShowUploadCleanedModal(true)
      }
    }
    input.click()
  }

  // 确认上传已清洗的数据
  const handleConfirmUploadCleaned = async () => {
    if (cleanedFilesToUpload.length === 0 || !cleanedDatasetName.trim()) return
    
    // 再次验证文件类型（双重检查）
    const invalidFiles = cleanedFilesToUpload.filter(f => !isValidCleanedFileType(f))
    if (invalidFiles.length > 0) {
      const invalidNames = invalidFiles.map(f => f.name).join(', ')
      setUploadError(`以下文件格式不支持: ${invalidNames}\n\n已清洗数据必须为以下格式: ${CLEANED_FILE_EXTENSIONS.join(', ')}`)
      return
    }
    
    const datasetName = cleanedDatasetName.trim()
    const files = cleanedFilesToUpload
    
    // 关闭弹窗
    setShowUploadCleanedModal(false)
    setCleanedFilesToUpload([])
    setCleanedDatasetName("")
    
    // 创建上传中的任务
    const jobId = `uploading-${Date.now()}`
    const newJob: ProcessingJob = {
      id: jobId,
      name: datasetName,
      sourceDataset: '直接上传',
      sourceDatasetId: '',
      fileCount: files.length,
      status: "processing",
      progress: 0,
      date: new Date().toISOString().split("T")[0],
      resultFile: files.map(f => f.name).join(', '),
    }
    setProcessingJobs(prev => [...prev, newJob])
    setIsUploading(true)
    setUploadProgress(0)
    setUploadError(null)
    
    try {
      // 调用上传 API
      const result = await ETMAgentAPI.uploadDataset(
        files,
        datasetName,
        (progress) => {
          setUploadProgress(progress)
          // 更新任务进度
          setProcessingJobs(prev => prev.map(j => 
            j.id === jobId ? { ...j, progress } : j
          ))
        }
      )
        
      // 上传成功，更新任务状态
      setProcessingJobs(prev => prev.map(j => 
        j.id === jobId ? { 
          ...j, 
          status: "completed" as const, 
          progress: 100,
          name: result.dataset_name,
          resultFile: result.files.join(', ')
        } : j
      ))
      
      // 添加成功消息到聊天
      const successMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: `✅ 数据集 **${result.dataset_name}** 上传成功！\n\n- 文件数量: ${result.file_count}\n- 总大小: ${(result.total_size / 1024).toFixed(2)} KB\n- 文件: ${result.files.join(', ')}\n\n您可以在向量化步骤中选择此数据集进行处理。`,
      }
      setChatHistory(prev => [...prev, successMessage])
      
    } catch (error) {
      console.error('Upload failed:', error)
      const errorMsg = error instanceof Error ? error.message : '上传失败'
      setUploadError(errorMsg)
      
      // 更新任务状态为失败
      setProcessingJobs(prev => prev.map(j => 
        j.id === jobId ? { ...j, status: "error" as const, progress: 0 } : j
      ))
      
      // 添加错误消息到聊天
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: `❌ 上传失败: ${errorMsg}\n\n请检查后端服务是否运行，然后重试。`,
      }
      setChatHistory(prev => [...prev, errorMessage])
    } finally {
      setIsUploading(false)
    }
  }

  const handleSendMessage = async () => {
    if (inputValue.trim()) {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: inputValue,
      }
      setChatHistory([...chatHistory, userMessage])
      const messageContent = inputValue
      setInputValue("")

      if (appState === "idle") {
        setAppState("chatting")
      }

      // 收集当前界面上下文
      const context = getCurrentContext()

      // 调用千问 API，传递上下文
      try {
        const response = await ETMAgentAPI.chat(messageContent, context)
        
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: response.message || response.response || "抱歉，我无法理解您的请求。",
        }
        setChatHistory((prev) => [...prev, aiMessage])

        // 如果有操作需要执行
        if (response.action === 'start_task' && response.data) {
          // 自动创建训练任务
          try {
            const task = await ETMAgentAPI.createTask(response.data as CreateTaskRequest)
            const taskMessage: Message = {
              id: (Date.now() + 2).toString(),
              role: "assistant",
              content: `✅ 任务已创建！\n任务 ID: ${task.task_id}\n状态: ${task.status}\n\n您可以在"模型训练"页面查看任务进度。`,
            }
            setChatHistory((prev) => [...prev, taskMessage])
          } catch (taskError) {
            console.error('创建任务失败:', taskError)
          }
        } else if (response.action === 'show_datasets') {
          // 切换到数据视图
          setCurrentView("data")
          setAppState("workspace")
        } else if (response.action === 'show_results') {
          // 切换到结果视图
          setCurrentView("results")
          setAppState("workspace")
        }
      } catch (error) {
        console.error('Chat API error:', error)
        
        let errorContent = "抱歉，AI 助手暂时无法响应。"
        if (error instanceof Error) {
          if (error.message.includes('Not Found') || error.message.includes('404')) {
            errorContent = "AI 助手服务暂时不可用。请检查后端服务是否正常运行，或稍后再试。"
          } else if (error.message.includes('fetch') || error.message.includes('network')) {
            errorContent = "无法连接到服务器。请检查后端是否在本地运行（如 ./start.sh）及网络连接。"
          } else {
            errorContent = `错误: ${error.message}`
          }
        }
        
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: errorContent + "\n\n您可以直接使用左侧菜单导航到相应功能。",
        }
        setChatHistory((prev) => [...prev, errorMessage])
      } finally {
        // 确保加载状态被清除
        setInputValue("")
      }
    }
  }

  const handleNavClick = (view: ViewType) => {
    setCurrentView(view)
    setAppState("workspace")
    setSheetOpen(false)
  }

  const handleNavToPage = (path: string) => {
    router.push(path)
    setSheetOpen(false)
  }

  const handleNewProcessingTask = () => {
    setShowSourceModal(true)
  }

  // 不再需要此变量，所有分析和可视化视图已移至独立页面
  // const isCenterChatView = currentView === "analysis" || currentView === "visualization"

  return (
    <div className="min-h-screen bg-white">
      <AnimatePresence mode="wait">
        {appState !== "workspace" ? (
          <motion.div
            key="conversational"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="h-screen flex flex-col"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <motion.header
              initial={{ opacity: 0 }}
              animate={{ opacity: appState === "chatting" ? 1 : 0 }}
              className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-md border-b border-slate-200"
              style={{ pointerEvents: appState === "chatting" ? "auto" : "none" }}
            >
              <div className="flex items-center justify-between px-6 h-16">
                <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
                  <SheetTrigger asChild>
                    <button className="p-2 hover:bg-slate-100 rounded-lg transition-colors">
                      <Menu className="w-5 h-5 text-slate-700" />
                    </button>
                  </SheetTrigger>
                  <SheetContent side="left" className="bg-white">
                    <SheetHeader>
                      <SheetTitle className="text-blue-600 text-2xl font-bold">THETA</SheetTitle>
                    </SheetHeader>
                    <nav className="mt-8 space-y-1">
                      <p className="px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">工作流程</p>
                      {workflowSteps.map((step, index) => (
                        <NavItem 
                          key={step.id}
                          icon={step.icon} 
                          label={`${index + 1}. ${step.label}`} 
                          onClick={() => {
                            if (step.id === "results" || step.id === "visualizations") {
                              handleNavToPage(`/${step.id}`)
                            } else {
                              handleNavClick(step.id as ViewType)
                            }
                          }} 
                        />
                      ))}
                    </nav>
                  </SheetContent>
                </Sheet>

                {appState === "chatting" && (
                  <motion.h1
                    layoutId="app-logo"
                    className="text-xl font-bold text-blue-600 tracking-tight absolute left-1/2 -translate-x-1/2"
                  >
                    THETA
                  </motion.h1>
                )}

                <div className="flex items-center">
                  <UserDropdown vertical={false} />
                </div>
              </div>
            </motion.header>

            <div className="flex-1 flex flex-col items-center justify-center px-8">
              <div className="max-w-2xl w-full flex flex-col items-center gap-16">
                {appState === "idle" && (
                  <motion.div layoutId="app-logo" className="text-center">
                    <h1 className="text-7xl font-bold text-blue-600 tracking-tight mb-2">THETA</h1>
                    <p className="text-slate-500 text-lg">智能分析平台</p>
                  </motion.div>
                )}

                {appState === "chatting" && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="w-full max-h-[50vh] overflow-y-auto space-y-4 px-4 mt-20"
                  >
                    {chatHistory.map((message, index) => (
                      <motion.div
                        key={message.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[80%] rounded-2xl px-5 py-3 ${
                            message.role === "user"
                              ? "bg-blue-600 text-white"
                              : "bg-slate-100 text-slate-900 border border-slate-200"
                          }`}
                        >
                          {message.role === "user" ? (
                            <p className="text-sm leading-relaxed">{message.content}</p>
                          ) : (
                            <TypingMessage 
                              content={message.content} 
                              isLatest={index === chatHistory.length - 1}
                              className="text-sm" 
                            />
                          )}
                        </div>
                      </motion.div>
                    ))}
                    <div ref={chatMessagesEndRef} />
                  </motion.div>
                )}

                <div className="w-full">
                  <ChatInput
                    value={inputValue}
                    onChange={setInputValue}
                    onSend={handleSendMessage}
                    onFileUpload={handleFileUpload}
                    isDragging={isDragging}
                    isLanding
                  />
                </div>

                {appState === "idle" && (
                  <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="flex flex-wrap justify-center gap-3"
                  >
                    {["清洗电商数据", "销售趋势预测", "客户行为分析", "智能报告生成"].map((chip) => (
                      <button
                        key={chip}
                        onClick={() => {
                          setInputValue(chip)
                        }}
                        className="px-5 py-2 bg-white border border-slate-200 rounded-full text-sm text-slate-700 hover:bg-slate-50 hover:border-blue-500 hover:text-blue-600 transition-all shadow-sm"
                      >
                        {chip}
                      </button>
                    ))}
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
        ) : (
          <div
            key="workspace"
            className="flex h-screen overflow-hidden"
          >
            <motion.aside
              animate={{
                width: sidebarCollapsed ? 80 : 256,
              }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="bg-white border-r border-slate-200 flex flex-col flex-shrink-0"
            >
              <div className="p-6 border-b border-slate-200 flex items-center justify-between">
                {!sidebarCollapsed && (
                  <motion.div layoutId="app-logo">
                    <h1 className="text-3xl font-bold text-blue-600">THETA</h1>
                    <p className="text-xs text-slate-500 mt-1">智能分析平台</p>
                  </motion.div>
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
                    active={currentView === step.id}
                    onClick={() => {
                      // 所有视图都在同一页面切换，不跳转路由
                      setCurrentView(step.id as ViewType)
                      // 同步更新 URL（不触发页面刷新）
                      window.history.replaceState(null, '', `/?view=${step.id}`)
                    }}
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
                          const stepIndex = workflowSteps.findIndex(s => s.id === currentView)
                          const isCompleted = index < stepIndex
                          const isCurrent = index === stepIndex
                          return (
                            <div key={step.id} className="flex items-center flex-1">
                              <div 
                                className={`w-3 h-3 rounded-full transition-colors ${
                                  isCompleted ? 'bg-green-500' : 
                                  isCurrent ? 'bg-blue-500' : 
                                  'bg-slate-200'
                                }`}
                              />
                              {index < workflowSteps.length - 1 && (
                                <div 
                                  className={`flex-1 h-0.5 transition-colors ${
                                    isCompleted ? 'bg-green-500' : 'bg-slate-200'
                                  }`}
                                />
                              )}
                            </div>
                          )
                        })}
                      </div>
                      <p className="text-xs text-slate-500 mt-2">
                        {workflowSteps.find(s => s.id === currentView)?.description || ""}
                      </p>
                    </div>
                  </div>
                )}
              </nav>
            </motion.aside>

            <div className="flex flex-1 overflow-hidden">
              <div className="flex-1 bg-slate-50 overflow-auto">
                <AnimatePresence mode="wait">
                  {currentView === "data" && (
                    <DataView 
                      key="data" 
                      datasets={datasets} 
                      onUpload={handleFileUpload}
                      selectedDatasetId={selectedDatasetId}
                      onSelectDataset={setSelectedDatasetId}
                      onAddFiles={handleAddFilesToDataset}
                      onRemoveFile={handleRemoveFileFromDataset}
                      onStartProcessing={(datasetId) => {
                        // 直接开始处理选定的数据集
                        setSelectedSource(datasetId)
                        // 使用 setTimeout 确保状态更新后再调用
                        setTimeout(() => {
                          const dataset = datasets.find(d => d.id === datasetId)
                          if (dataset) {
                            startProcessingDataset(dataset)
                          }
                        }, 0)
                      }}
                      onDeleteDataset={handleDeleteDataset}
                      onRenameDataset={handleRenameDataset}
                      onNextStep={() => setCurrentView("processing")}
                    />
                  )}
                  {currentView === "processing" && (
                    <ProcessingView 
                      key="processing" 
                      jobs={processingJobs} 
                      onNewTask={handleNewProcessingTask}
                      onDownload={handleDownloadResult}
                      onDelete={handleDeleteJob}
                      onUploadCleanedData={handleUploadCleanedData}
                      onNextStep={() => setCurrentView("embedding")}
                      onPrevStep={() => setCurrentView("data")}
                      isUploading={isUploading}
                      uploadProgress={uploadProgress}
                    />
                  )}
                  {currentView === "embedding" && (
                    <EmbeddingView
                      key="embedding"
                      onPrevStep={() => setCurrentView("processing")}
                      onNextStep={() => setCurrentView("tasks")}
                      cleanedDatasets={processingJobs
                        .filter(job => job.status === 'completed')
                        .map(job => ({
                          name: job.name,
                          path: job.resultFile ? `local://${job.resultFile}` : undefined,
                          size: job.fileCount,
                        }))}
                    />
                  )}
                  {currentView === "tasks" && (
                    <TasksView
                      key="tasks"
                      onPrevStep={() => setCurrentView("embedding")}
                      onNextStep={() => setCurrentView("results")}
                      datasets={datasets}
                    />
                  )}
                  {currentView === "results" && (
                    <ResultsView
                      key="results"
                      onPrevStep={() => setCurrentView("tasks")}
                      onNextStep={() => setCurrentView("visualizations")}
                    />
                  )}
                  {currentView === "visualizations" && (
                    <VisualizationsView
                      key="visualizations"
                      onPrevStep={() => setCurrentView("results")}
                    />
                  )}
                </AnimatePresence>
              </div>

              {/* AI 助手面板 - 仅在展开时显示 */}
              {!chatSidebarCollapsed && (
                <aside className="w-full sm:w-80 lg:w-96 border-l border-slate-200 bg-white flex flex-col flex-shrink-0 fixed sm:relative inset-0 sm:inset-auto z-50 sm:z-auto">
                  <div className="border-b border-slate-200 p-3 sm:p-4 flex items-center justify-between">
                    <h3 className="font-semibold text-slate-900 flex items-center gap-2 text-sm sm:text-base">
                      <BrainCircuit className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600" />
                      AI 助手
                    </h3>
                    <button
                      onClick={() => setChatSidebarCollapsed(true)}
                      className="p-1.5 sm:p-2 hover:bg-slate-100 rounded-lg transition-colors"
                      title="收起 AI 助手"
                    >
                      <PanelRightClose className="w-4 h-4 sm:w-5 sm:h-5 text-slate-600" />
                    </button>
                  </div>
                  <ChatInterface
                    messages={chatHistory}
                    inputValue={inputValue}
                    onInputChange={setInputValue}
                    onSend={handleSendMessage}
                    onFileUpload={handleFileUpload}
                    suggestions={suggestions}
                    loadingSuggestions={loadingSuggestions}
                    onClearChat={handleClearChat}
                    onClearWithoutSave={handleClearWithoutSave}
                    onGetStatus={handleGetStatus}
                    currentContext={currentContextDisplay}
                    chatSessions={chatSessions}
                    onLoadSession={handleLoadSession}
                    onDeleteSession={handleDeleteSession}
                    currentSessionId={currentSessionId || undefined}
                    showStatusPanel={showStatusPanel}
                    onToggleStatusPanel={() => setShowStatusPanel(!showStatusPanel)}
                    onSuggestionClick={async (suggestion) => {
                      if (suggestion.action === 'navigate' && suggestion.data) {
                        const view = suggestion.data.view || "data"
                        if (view === "results" || view === "visualizations") {
                          router.push(`/${view}`)
                        } else {
                          setCurrentView(view as ViewType)
                          setAppState("workspace")
                          router.push(`/?view=${view}`)
                        }
                      } else if (suggestion.action === 'start_training' || suggestion.action === 'start_cleaning') {
                        setInputValue(suggestion.text)
                        // Use a callback to ensure state is updated before sending
                        setTimeout(() => {
                          const sendButton = document.querySelector('button[onclick*="handleSendMessage"]') as HTMLButtonElement
                          if (sendButton && !sendButton.disabled) {
                            handleSendMessage()
                          }
                        }, 50)
                      } else {
                        setInputValue(suggestion.text)
                      }
                    }}
                  />
                </aside>
              )}

              {/* 右侧工具栏 */}
              <aside className="hidden sm:flex w-12 lg:w-16 bg-white border-l border-slate-200 flex-col items-center py-3 lg:py-4 flex-shrink-0">
                <div className="flex flex-col items-center gap-2 w-full px-2">
                  <UserDropdown vertical={true} />
                  
                  {/* AI 助手收起时的图标 */}
                  {chatSidebarCollapsed && (
                    <button
                      onClick={() => setChatSidebarCollapsed(false)}
                      className="p-2 hover:bg-blue-50 rounded-lg transition-colors group relative"
                      title="展开 AI 助手"
                    >
                      <MessageSquare className="w-5 h-5 text-slate-400 group-hover:text-blue-600" />
                      {chatHistory.length > 0 && (
                        <div className="absolute -top-1 -right-1 w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                      )}
                    </button>
                  )}
                </div>
              </aside>
            </div>
          </div>
        )}
      </AnimatePresence>

      <Dialog open={showNameModal} onOpenChange={(open) => {
        if (!open) {
          setPendingFiles([])
          setDatasetName("")
        }
        setShowNameModal(open)
      }}>
        <DialogContent className="sm:max-w-lg bg-white">
          <DialogHeader className="pr-8">
            <DialogTitle className="text-slate-900">创建数据集</DialogTitle>
            <DialogDescription className="text-slate-500">
              已选择 {pendingFiles.length} 个文件，请为数据集命名
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {/* 显示待上传的文件列表，支持添加和删除 */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-slate-700">已选文件 ({pendingFiles.length})</Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const input = document.createElement('input')
                    input.type = 'file'
                    input.multiple = true
                    input.onchange = (e) => {
                      const newFiles = Array.from((e.target as HTMLInputElement).files || [])
                      if (newFiles.length > 0) {
                        setPendingFiles(prev => [...prev, ...newFiles])
                      }
                    }
                    input.click()
                  }}
                  className="text-blue-600 hover:text-blue-700 hover:bg-blue-50 gap-1"
                >
                  <Plus className="w-4 h-4" />
                  添加更多
                </Button>
              </div>
              {pendingFiles.length > 0 ? (
                <div className="max-h-40 overflow-y-auto border border-slate-200 rounded-lg p-2 space-y-1">
                  {pendingFiles.map((file, index) => (
                    <div key={index} className="flex items-center text-sm py-1.5 px-3 bg-slate-50 rounded group hover:bg-slate-100">
                      <span className="text-slate-700 truncate flex-1 min-w-0">{file.name}</span>
                      <div className="flex items-center gap-3 flex-shrink-0 ml-3">
                        <span className="text-slate-400 text-xs whitespace-nowrap">
                          {file.size >= 1024 * 1024 
                            ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
                            : `${(file.size / 1024).toFixed(1)} KB`}
                        </span>
                        <button
                          onClick={() => setPendingFiles(prev => prev.filter((_, i) => i !== index))}
                          className="text-slate-400 hover:text-red-500 transition-colors"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div 
                  className="border-2 border-dashed border-slate-200 rounded-lg p-6 text-center cursor-pointer hover:border-blue-300 hover:bg-blue-50/50 transition-colors"
                  onClick={() => {
                    const input = document.createElement('input')
                    input.type = 'file'
                    input.multiple = true
                    input.onchange = (e) => {
                      const newFiles = Array.from((e.target as HTMLInputElement).files || [])
                      if (newFiles.length > 0) {
                        setPendingFiles(newFiles)
                      }
                    }
                    input.click()
                  }}
                >
                  <Upload className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                  <p className="text-sm text-slate-500">点击选择文件</p>
                </div>
              )}
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="dataset-name" className="text-slate-700">
                数据集名称
              </Label>
              <Input
                id="dataset-name"
                placeholder="留空将使用默认名称"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="border-slate-300"
                disabled={isUploading}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isUploading) {
                    handleNameConfirm()
                  }
                }}
              />
              <p className="text-xs text-slate-400">
                留空将自动命名为 "{generateDefaultDatasetName(datasets)}"
              </p>
            </div>
            
            {/* 上传进度显示 */}
            {isUploading && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">上传中...</span>
                  <span className="text-blue-600 font-medium">{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="h-2" />
              </div>
            )}
            
            {/* 错误信息显示 */}
            {uploadError && (
              <div className="rounded-lg bg-red-50 border border-red-200 p-3">
                <p className="text-sm text-red-700">{uploadError}</p>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => {
                if (!isUploading) {
                  setShowNameModal(false)
                  setPendingFiles([])
                  setDatasetName("")
                  setUploadError(null)
                }
              }} 
              disabled={isUploading}
              className="border-slate-300"
            >
              取消
            </Button>
            <Button 
              onClick={handleNameConfirm} 
              disabled={pendingFiles.length === 0 || isUploading}
              className="bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  上传中 {uploadProgress}%
                </>
              ) : (
                '创建数据集'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 上传已清洗数据弹窗 */}
      <Dialog open={showUploadCleanedModal} onOpenChange={(open) => {
        if (!open) {
          setCleanedFilesToUpload([])
          setCleanedDatasetName("")
        }
        setShowUploadCleanedModal(open)
      }}>
        <DialogContent className="sm:max-w-md bg-white">
          <DialogHeader className="pb-4">
            <DialogTitle className="text-lg font-semibold text-slate-900 flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center">
                <Upload className="w-4 h-4 text-blue-600" />
              </div>
              上传已清洗数据
            </DialogTitle>
            <DialogDescription className="text-sm text-slate-500 mt-1">
              将已清洗的数据文件上传到服务器，用于后续的向量化处理
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            {/* 文件列表 */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium text-slate-700">
                  已选文件 ({cleanedFilesToUpload.length})
                </Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const input = document.createElement('input')
                    input.type = 'file'
                    input.accept = '.csv,.txt,.json'
                    input.multiple = true
                    input.onchange = (e) => {
                      const newFiles = Array.from((e.target as HTMLInputElement).files || [])
                      if (newFiles.length > 0) {
                        setCleanedFilesToUpload(prev => [...prev, ...newFiles])
                      }
                    }
                    input.click()
                  }}
                  className="text-blue-600 hover:text-blue-700 hover:bg-blue-50 gap-1 h-8 text-xs"
                >
                  <Plus className="w-3.5 h-3.5" />
                  添加更多
                </Button>
              </div>
              
              {cleanedFilesToUpload.length > 0 ? (
                <div className="max-h-32 overflow-y-auto border border-slate-200 rounded-lg divide-y divide-slate-100">
                  {cleanedFilesToUpload.map((file, index) => (
                    <div 
                      key={index} 
                      className="flex items-center text-sm py-2 px-3 hover:bg-slate-50 group"
                    >
                      <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center flex-shrink-0 mr-3">
                        <FileText className="w-4 h-4 text-slate-500" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-slate-700 truncate text-sm">{file.name}</p>
                        <p className="text-slate-400 text-xs">
                          {file.size >= 1024 * 1024 
                            ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
                            : `${(file.size / 1024).toFixed(1)} KB`}
                        </p>
                      </div>
                      <button
                        onClick={() => setCleanedFilesToUpload(prev => prev.filter((_, i) => i !== index))}
                        className="ml-2 p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors opacity-0 group-hover:opacity-100"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div 
                  className="border-2 border-dashed border-slate-200 rounded-lg p-6 text-center cursor-pointer hover:border-blue-300 hover:bg-blue-50/50 transition-colors"
                  onClick={() => {
                    const input = document.createElement('input')
                    input.type = 'file'
                    input.accept = '.csv,.txt,.json'
                    input.multiple = true
                    input.onchange = (e) => {
                      const newFiles = Array.from((e.target as HTMLInputElement).files || [])
                      if (newFiles.length > 0) {
                        setCleanedFilesToUpload(newFiles)
                      }
                    }
                    input.click()
                  }}
                >
                  <Upload className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                  <p className="text-sm text-slate-500">点击选择文件</p>
                  <p className="text-xs text-slate-400 mt-1">支持 CSV, TXT, JSON 格式</p>
                </div>
              )}
            </div>
            
            {/* 数据集名称输入 */}
            <div className="space-y-2">
              <Label htmlFor="cleaned-dataset-name" className="text-sm font-medium text-slate-700">
                数据集名称 <span className="text-red-500">*</span>
              </Label>
              <Input
                id="cleaned-dataset-name"
                placeholder="请输入数据集名称"
                value={cleanedDatasetName}
                onChange={(e) => setCleanedDatasetName(e.target.value)}
                className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && cleanedDatasetName.trim() && cleanedFilesToUpload.length > 0) {
                    handleConfirmUploadCleaned()
                  }
                }}
              />
              <p className="text-xs text-slate-400">
                数据集名称将用于后续的向量化和训练步骤
              </p>
            </div>
          </div>
          
          <DialogFooter className="mt-4 gap-2 sm:gap-0">
            <Button 
              variant="outline" 
              onClick={() => {
                setShowUploadCleanedModal(false)
                setCleanedFilesToUpload([])
                setCleanedDatasetName("")
              }} 
              className="border-slate-300 hover:bg-slate-50"
            >
              取消
            </Button>
            <Button 
              onClick={handleConfirmUploadCleaned}
              disabled={cleanedFilesToUpload.length === 0 || !cleanedDatasetName.trim()}
              className="bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 disabled:cursor-not-allowed gap-2"
            >
              <Upload className="w-4 h-4" />
              开始上传
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showSourceModal} onOpenChange={setShowSourceModal}>
        <DialogContent className="sm:max-w-lg bg-white">
          <DialogHeader>
            <DialogTitle className="text-slate-900">选择数据源</DialogTitle>
            <DialogDescription className="text-slate-500">
              请选择一个完整的数据集文件夹作为处理源
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <RadioGroup value={selectedSource} onValueChange={setSelectedSource}>
              <div className="space-y-2">
                {datasets.map((dataset) => (
                  <div
                    key={dataset.id}
                    className="flex items-center space-x-3 p-3 border border-slate-200 rounded-lg hover:bg-slate-50 cursor-pointer"
                  >
                    <RadioGroupItem value={dataset.id} id={dataset.id} />
                    <Label htmlFor={dataset.id} className="flex-1 cursor-pointer min-w-0">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center flex-shrink-0">
                          <Folder className="w-5 h-5 text-blue-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-slate-900 truncate" title={dataset.name}>{dataset.name}</p>
                          <p className="text-xs text-slate-500">
                            {dataset.files.length} 文件 · {dataset.totalSize}
                          </p>
                        </div>
                      </div>
                    </Label>
                  </div>
                ))}
              </div>
            </RadioGroup>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowSourceModal(false)} className="border-slate-300">
              取消
            </Button>
            <Button
              onClick={handleSourceConfirm}
              disabled={!selectedSource}
              className="bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50"
            >
              确认选择
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 删除数据集确认对话框 */}
      <Dialog open={showDeleteConfirm} onOpenChange={(open) => {
        if (!open) {
          setDatasetToDelete(null)
        }
        setShowDeleteConfirm(open)
      }}>
        <DialogContent className="sm:max-w-md bg-white">
          <DialogHeader>
            <DialogTitle className="text-slate-900">确认删除数据集</DialogTitle>
            <DialogDescription className="text-slate-500">
              此操作将永久删除该数据集及其所有文件，且无法恢复。您确定要继续吗？
            </DialogDescription>
          </DialogHeader>
          {datasetToDelete && (
            <div className="py-4">
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-sm text-red-800 font-medium">
                  数据集: {datasets.find(d => d.id === datasetToDelete)?.name || '未知'}
                </p>
                <p className="text-xs text-red-600 mt-1">
                  包含 {datasets.find(d => d.id === datasetToDelete)?.files.length || 0} 个文件
                </p>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => {
                setShowDeleteConfirm(false)
                setDatasetToDelete(null)
              }} 
              className="border-slate-300"
            >
              取消
            </Button>
            <Button 
              onClick={confirmDeleteDataset} 
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              确认删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 重命名数据集对话框 */}
      <Dialog open={showRenameModal} onOpenChange={(open) => {
        if (!open) {
          setDatasetToRename(null)
          setNewDatasetName("")
        }
        setShowRenameModal(open)
      }}>
        <DialogContent className="sm:max-w-md bg-white">
          <DialogHeader>
            <DialogTitle className="text-slate-900 flex items-center gap-2">
              <Edit2 className="w-5 h-5 text-blue-600" />
              重命名数据集
            </DialogTitle>
            <DialogDescription className="text-slate-500">
              请输入新的数据集名称
            </DialogDescription>
          </DialogHeader>
          {datasetToRename && (
            <div className="space-y-4 py-4">
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                <p className="text-xs text-slate-500 mb-1">当前名称</p>
                <p className="text-sm text-slate-700 font-medium">
                  {datasets.find(d => d.id === datasetToRename)?.name || '未知'}
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-dataset-name" className="text-slate-700">
                  新名称 <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="new-dataset-name"
                  placeholder="请输入数据集名称"
                  value={newDatasetName}
                  onChange={(e) => setNewDatasetName(e.target.value)}
                  className="border-slate-300 focus:border-blue-500 focus:ring-blue-500"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && newDatasetName.trim()) {
                      confirmRenameDataset()
                    }
                  }}
                  autoFocus
                />
                <p className="text-xs text-slate-400">
                  数据集名称不能为空
                </p>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => {
                setShowRenameModal(false)
                setDatasetToRename(null)
                setNewDatasetName("")
              }} 
              className="border-slate-300"
            >
              取消
            </Button>
            <Button 
              onClick={confirmRenameDataset}
              disabled={!newDatasetName.trim()}
              className="bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
            >
              确认重命名
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function ChatInput({
  value,
  onChange,
  onSend,
  onFileUpload,
  isDragging,
  isLanding = false,
}: {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onFileUpload: () => void
  isDragging: boolean
  isLanding?: boolean
}) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }

  if (isDragging && isLanding) {
    return (
      <Card className="border-2 border-dashed border-blue-500 bg-blue-50 p-8 sm:p-12 lg:p-16 rounded-2xl shadow-lg">
        <div className="flex flex-col items-center gap-4 sm:gap-6 text-center">
          <div className="w-12 h-12 sm:w-14 sm:h-14 lg:w-16 lg:h-16 rounded-xl sm:rounded-2xl bg-blue-600 flex items-center justify-center shadow-lg">
            <Upload className="w-6 h-6 sm:w-7 sm:h-7 lg:w-8 lg:h-8 text-white" />
          </div>
          <div className="space-y-2">
            <h2 className="text-2xl font-semibold text-blue-900">释放以开始上传</h2>
            <p className="text-blue-700">支持 CSV, Excel, JSON 等格式</p>
          </div>
        </div>
      </Card>
    )
  }

  return (
    <div className="relative">
      <div className="relative flex items-center bg-white border-2 border-slate-200 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 overflow-hidden">
        <button
          onClick={onFileUpload}
          className="absolute left-4 p-2 hover:bg-slate-100 rounded-lg transition-colors z-10"
        >
          <Paperclip className="w-5 h-5 text-slate-500" />
        </button>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入分析目标，或直接拖入数据文件..."
          className="flex-1 px-16 py-5 text-slate-900 placeholder:text-slate-400 focus:outline-none bg-transparent"
        />
        <button
          onClick={onSend}
          className="absolute right-2 p-3 bg-blue-600 hover:bg-blue-700 rounded-xl transition-colors shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={!value.trim()}
        >
          <Send className="w-5 h-5 text-white" />
        </button>
      </div>
    </div>
  )
}

interface Suggestion {
  text: string
  description?: string
  action?: string
  data?: any
}

// 历史对话会话类型
type ChatSession = {
  id: string
  title: string
  messages: Message[]
  createdAt: string
  updatedAt: string
}

function ChatInterface({
  messages,
  inputValue,
  onInputChange,
  onSend,
  onFileUpload,
  suggestions = [],
  loadingSuggestions = false,
  onSuggestionClick,
  onClearChat,
  onClearWithoutSave,
  onGetStatus,
  currentContext,
  chatSessions = [],
  onLoadSession,
  onDeleteSession,
  currentSessionId,
  showStatusPanel: showStatusPanelProp,
  onToggleStatusPanel,
}: {
  messages: Message[]
  inputValue: string
  onInputChange: (value: string) => void
  onSend: () => void
  onFileUpload: () => void
  suggestions?: Suggestion[]
  loadingSuggestions?: boolean
  onSuggestionClick?: (suggestion: Suggestion) => void
  onClearChat?: () => void
  onClearWithoutSave?: () => void
  onGetStatus?: () => void
  currentContext?: string
  chatSessions?: ChatSession[]
  onLoadSession?: (sessionId: string) => void
  onDeleteSession?: (sessionId: string) => void
  currentSessionId?: string
  showStatusPanel?: boolean
  onToggleStatusPanel?: () => void
}) {
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [showHistory, setShowHistory] = useState(false)

  // 滚动到消息区域底部
  const scrollToMessages = useCallback(() => {
    // 使用 requestAnimationFrame 确保 DOM 已更新
    requestAnimationFrame(() => {
      if (messagesContainerRef.current) {
        // 如果有消息，滚动到底部
        if (messages.length > 0 && messagesEndRef.current) {
          messagesEndRef.current.scrollIntoView({ 
            behavior: 'smooth',
            block: 'end'
          })
        } else {
          // 如果没有消息，滚动到消息容器顶部
          messagesContainerRef.current.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start',
            inline: 'nearest'
          })
        }
      }
    })
  }, [messages.length])

  // 当消息更新时，滚动到底部
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  // 处理输入框获得焦点
  const handleInputFocus = () => {
    scrollToMessages()
  }

  // 处理输入变化
  const handleInputChange = (value: string) => {
    onInputChange(value)
    // 如果消息区域不在视图中，滚动到消息区域
    requestAnimationFrame(() => {
      if (messagesContainerRef.current) {
        const rect = messagesContainerRef.current.getBoundingClientRect()
        const container = messagesContainerRef.current
        const containerRect = container.getBoundingClientRect()
        const viewportHeight = window.innerHeight
        
        // 检查消息容器是否在视口中可见
        const isFullyVisible = containerRect.top >= 0 && containerRect.bottom <= viewportHeight
        // 检查消息容器是否部分可见但在视口上方
        const isAboveViewport = containerRect.bottom < 0
        
        if (!isFullyVisible || isAboveViewport) {
          scrollToMessages()
        }
      }
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }

  return (
    <>
      {/* 工具栏 */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-100 bg-slate-50/50">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowHistory(!showHistory)}
            className={`p-1.5 rounded-lg transition-colors text-xs flex items-center gap-1 ${
              showHistory ? 'bg-blue-100 text-blue-600' : 'hover:bg-slate-100 text-slate-500'
            }`}
            title="历史对话"
          >
            <Clock className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">历史</span>
            {chatSessions.length > 0 && (
              <span className="ml-0.5 px-1 py-0.5 bg-slate-200 text-slate-600 text-[10px] rounded-full min-w-[16px] text-center">
                {chatSessions.length}
              </span>
            )}
          </button>
          <button
            onClick={() => {
              onToggleStatusPanel?.()
              if (!showStatusPanelProp) {
                onGetStatus?.()
              }
            }}
            className={`p-1.5 rounded-lg transition-colors text-xs flex items-center gap-1 ${
              showStatusPanelProp ? 'bg-green-100 text-green-600' : 'hover:bg-slate-100 text-slate-500'
            }`}
            title="当前界面状态"
          >
            <AlertCircle className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">状态</span>
          </button>
        </div>
        <div className="flex items-center gap-1">
          {messages.length > 0 && (
            <>
              <button
                onClick={onClearChat}
                className="p-1.5 rounded-lg hover:bg-blue-50 text-slate-500 hover:text-blue-600 transition-colors text-xs flex items-center gap-1"
                title="保存并新建对话"
              >
                <Plus className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">新建</span>
              </button>
              <button
                onClick={() => {
                  if (confirm('确定要清空当前对话吗？（不会保存到历史）')) {
                    onClearWithoutSave?.()
                  }
                }}
                className="p-1.5 rounded-lg hover:bg-red-50 text-slate-500 hover:text-red-600 transition-colors text-xs flex items-center gap-1"
                title="清空对话（不保存）"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* 当前状态面板 */}
      {showStatusPanelProp && currentContext && (
        <div className="mx-4 mt-2 p-3 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg animate-in slide-in-from-top-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1">
              <p className="text-xs font-semibold text-green-700 mb-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />
                当前界面状态
              </p>
              <p className="text-xs text-green-600 whitespace-pre-wrap leading-relaxed">{currentContext}</p>
            </div>
            <button
              onClick={() => onToggleStatusPanel?.()}
              className="p-1 hover:bg-green-100 rounded transition-colors"
            >
              <X className="w-3 h-3 text-green-600" />
            </button>
          </div>
        </div>
      )}

      {/* 历史对话面板 */}
      {showHistory && (
        <div className="mx-4 mt-2 p-3 bg-slate-50 border border-slate-200 rounded-lg animate-in slide-in-from-top-2 max-h-48 overflow-y-auto">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-semibold text-slate-600 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              历史对话 ({chatSessions.length})
            </p>
            <button
              onClick={() => setShowHistory(false)}
              className="p-1 hover:bg-slate-200 rounded transition-colors"
            >
              <X className="w-3 h-3 text-slate-500" />
            </button>
          </div>
          {chatSessions.length === 0 ? (
            <p className="text-xs text-slate-400 text-center py-3">暂无历史对话</p>
          ) : (
            <div className="space-y-1.5">
              {chatSessions.map((session) => (
                <div
                  key={session.id}
                  className={`group flex items-center justify-between p-2 rounded-lg cursor-pointer transition-all ${
                    currentSessionId === session.id
                      ? 'bg-blue-100 border border-blue-200'
                      : 'hover:bg-white border border-transparent hover:border-slate-200'
                  }`}
                  onClick={() => onLoadSession?.(session.id)}
                >
                  <div className="flex-1 min-w-0">
                    <p className={`text-xs font-medium truncate ${
                      currentSessionId === session.id ? 'text-blue-700' : 'text-slate-700'
                    }`}>
                      {session.title || '未命名对话'}
                    </p>
                    <p className="text-[10px] text-slate-400">
                      {new Date(session.updatedAt).toLocaleString('zh-CN', { 
                        month: 'short', 
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                      {' · '}{session.messages.length} 条消息
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteSession?.(session.id)
                    }}
                    className="p-1 opacity-0 group-hover:opacity-100 hover:bg-red-100 rounded transition-all"
                    title="删除此对话"
                  >
                    <Trash2 className="w-3 h-3 text-red-500" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-6 space-y-4"
      >
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
          <>
            {messages.map((message, index) => (
              <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                    message.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-slate-100 text-slate-900 border border-slate-200"
                  }`}
                >
                  {message.role === "user" ? (
                    <p className="text-sm leading-relaxed">{message.content}</p>
                  ) : (
                    <TypingMessage 
                      content={message.content} 
                      isLatest={index === messages.length - 1}
                      className="text-sm" 
                    />
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* 智能建议 */}
      {suggestions.length > 0 && messages.length === 0 && (
        <div className="border-t border-slate-200 p-4 bg-slate-50">
          <p className="text-xs font-semibold text-slate-500 mb-2">💡 智能建议</p>
          <div className="space-y-2">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => onSuggestionClick?.(suggestion)}
                className="w-full text-left p-3 bg-white border border-slate-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all group"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <p className="text-sm font-medium text-slate-900 group-hover:text-blue-600">
                      {suggestion.text}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">{suggestion.description}</p>
                  </div>
                  <Zap className="w-4 h-4 text-slate-400 group-hover:text-blue-600 flex-shrink-0 mt-0.5" />
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="border-t border-slate-200 p-4 bg-white">
        <div className="flex items-end gap-2">
          <button onClick={onFileUpload} className="p-2 hover:bg-slate-100 rounded-lg transition-colors flex-shrink-0">
            <Paperclip className="w-5 h-5 text-slate-500" />
          </button>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => handleInputChange(e.target.value)}
            onFocus={handleInputFocus}
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

function DataView({ 
  datasets, 
  onUpload,
  selectedDatasetId,
  onSelectDataset,
  onAddFiles,
  onRemoveFile,
  onStartProcessing,
  onDeleteDataset,
  onRenameDataset,
  onNextStep,
}: { 
  datasets: Dataset[]
  onUpload: () => void
  selectedDatasetId: string | null
  onSelectDataset: (id: string | null) => void
  onAddFiles: (datasetId: string, files: File[]) => void
  onRemoveFile: (datasetId: string, fileId: string) => void
  onStartProcessing: (datasetId: string) => void
  onDeleteDataset: (datasetId: string) => void
  onRenameDataset: (datasetId: string) => void
  onNextStep?: () => void
}) {
  const [isDraggingInDetail, setIsDraggingInDetail] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [sortBy, setSortBy] = useState<"name" | "date" | "size">("date")
  
  // 筛选和排序数据集
  const filteredAndSortedDatasets = datasets
    .filter(dataset => {
      if (!searchQuery.trim()) return true
      const query = searchQuery.toLowerCase()
      return dataset.name.toLowerCase().includes(query) || 
             dataset.id.toLowerCase().includes(query)
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "name":
          return a.name.localeCompare(b.name)
        case "date":
          return new Date(b.date).getTime() - new Date(a.date).getTime()
        case "size":
          // 简单的大小比较（基于文件数量）
          return b.files.length - a.files.length
        default:
          return 0
      }
    })
  
  const selectedDataset = selectedDatasetId 
    ? datasets.find(d => d.id === selectedDatasetId) 
    : null

  // 在数据集详情页添加文件
  const handleAddFilesClick = () => {
    if (!selectedDatasetId) return
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.onchange = (e) => {
      const files = Array.from((e.target as HTMLInputElement).files || [])
      if (files.length > 0) {
        onAddFiles(selectedDatasetId, files)
      }
    }
    input.click()
  }

  // 拖拽添加文件到数据集
  const handleDetailDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDraggingInDetail(true)
  }

  const handleDetailDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDraggingInDetail(false)
  }

  const handleDetailDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDraggingInDetail(false)
    if (!selectedDatasetId) return
    const droppedFiles = Array.from(e.dataTransfer.files)
    if (droppedFiles.length > 0) {
      onAddFiles(selectedDatasetId, droppedFiles)
    }
  }

  // 数据集详情视图
  if (selectedDataset) {
    return (
      <motion.div 
        className="p-4 sm:p-6 lg:p-8"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        onDragOver={handleDetailDragOver}
        onDragLeave={handleDetailDragLeave}
        onDrop={handleDetailDrop}
      >
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              onClick={() => onSelectDataset(null)}
              variant="ghost"
              size="icon"
              className="hover:bg-slate-100"
            >
              <ArrowLeft className="w-5 h-5 text-slate-600" />
            </Button>
            <div>
              <h2 className="text-2xl font-semibold text-slate-900 mb-1">{selectedDataset.name}</h2>
              <p className="text-slate-500 text-sm">
                {selectedDataset.files.length} 个文件 · {selectedDataset.totalSize} · 创建于 {selectedDataset.date}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button onClick={handleAddFilesClick} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
              <Plus className="w-4 h-4" />
              添加文件
            </Button>
            <Button
              onClick={() => onRenameDataset(selectedDataset.id)}
              variant="outline"
              className="border-slate-300 hover:bg-slate-50 gap-2"
            >
              <Edit2 className="w-4 h-4" />
              重命名
            </Button>
            <Button
              onClick={() => onDeleteDataset(selectedDataset.id)}
              variant="outline"
              className="border-red-200 text-red-600 hover:bg-red-50 hover:border-red-300 gap-2"
            >
              <Trash2 className="w-4 h-4" />
              删除数据集
            </Button>
          </div>
        </div>

        {/* 拖拽提示 */}
        {isDraggingInDetail && (
          <div className="fixed inset-0 bg-blue-500/10 z-40 flex items-center justify-center pointer-events-none">
            <div className="bg-white border-2 border-dashed border-blue-500 rounded-2xl p-6 sm:p-8 shadow-xl">
              <div className="flex flex-col items-center gap-3 sm:gap-4">
                <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-xl sm:rounded-2xl bg-blue-600 flex items-center justify-center">
                  <Upload className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                </div>
                <p className="text-blue-600 font-medium">释放以添加文件到此数据集</p>
              </div>
            </div>
          </div>
        )}

        {selectedDataset.files.length === 0 ? (
          <Card className="border-2 border-dashed border-slate-200 bg-white p-8 sm:p-12 lg:p-16 rounded-xl text-center">
            <div className="flex flex-col items-center gap-3 sm:gap-4">
              <div className="w-12 h-12 sm:w-14 sm:h-14 lg:w-16 lg:h-16 rounded-xl sm:rounded-2xl bg-slate-100 flex items-center justify-center">
                <Upload className="w-6 h-6 sm:w-7 sm:h-7 lg:w-8 lg:h-8 text-slate-400" />
              </div>
              <div>
                <p className="text-slate-500 mb-2">此数据集暂无文件</p>
                <p className="text-sm text-slate-400">点击上方按钮或拖拽文件到此处添加</p>
              </div>
            </div>
          </Card>
        ) : (
          <div className="space-y-3">
            {selectedDataset.files.map((file, index) => (
              <motion.div
                key={file.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Card className="border border-slate-200 bg-white hover:shadow-md transition-all p-4 rounded-xl">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center flex-shrink-0">
                      <FileText className="w-6 h-6 text-blue-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-slate-900 truncate">{file.name}</h4>
                      <p className="text-sm text-slate-500">
                        {file.type} · {file.size} · 上传于 {file.uploadDate}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => onRemoveFile(selectedDataset.id, file.id)}
                      className="text-slate-400 hover:text-red-500 hover:bg-red-50"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        )}

        {/* 数据处理入口 */}
        {selectedDataset.files.length > 0 && (
          <div className="mt-8">
            <Card className="border border-slate-200 bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-blue-600 flex items-center justify-center">
                    <FileCog className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h4 className="font-medium text-slate-900">处理此数据集</h4>
                    <p className="text-sm text-slate-500">对文件进行文本清洗和格式转换</p>
                  </div>
                </div>
                <Button 
                  onClick={() => onStartProcessing(selectedDataset.id)}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  开始处理
                </Button>
              </div>
            </Card>
          </div>
        )}
      </motion.div>
    )
  }

  // 数据集列表视图
  return (
    <motion.div 
      className="p-4 sm:p-6 lg:p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="mb-4 sm:mb-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-4">
          <div>
            <h2 className="text-xl sm:text-2xl font-semibold text-slate-900 mb-1 sm:mb-2">我的数据</h2>
            <p className="text-sm sm:text-base text-slate-600">管理和查看您的数据集</p>
          </div>
          <Button onClick={onUpload} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
            <Upload className="w-4 h-4" />
            <span className="hidden sm:inline">上传数据集</span>
            <span className="sm:hidden">上传</span>
          </Button>
        </div>
        
        {/* 搜索和筛选栏 */}
        {datasets.length > 0 && (
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
              <Input
                type="text"
                placeholder="搜索数据集名称或ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 border-slate-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-500" />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as "name" | "date" | "size")}
                className="px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
              >
                <option value="date">按日期排序</option>
                <option value="name">按名称排序</option>
                <option value="size">按文件数排序</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {datasets.length === 0 ? (
        <Card className="border-2 border-dashed border-slate-200 bg-white p-8 sm:p-12 lg:p-16 rounded-xl text-center">
          <div className="flex flex-col items-center gap-3 sm:gap-4">
            <div className="w-12 h-12 sm:w-14 sm:h-14 lg:w-16 lg:h-16 rounded-xl sm:rounded-2xl bg-slate-100 flex items-center justify-center">
              <Database className="w-6 h-6 sm:w-7 sm:h-7 lg:w-8 lg:h-8 text-slate-400" />
            </div>
            <div>
              <p className="text-slate-500 mb-2">暂无数据集</p>
              <p className="text-sm text-slate-400">点击上方按钮或在首页拖拽文件创建数据集</p>
            </div>
          </div>
        </Card>
      ) : filteredAndSortedDatasets.length === 0 ? (
        <Card className="border-2 border-dashed border-slate-200 bg-white p-8 sm:p-12 rounded-xl text-center">
          <div className="flex flex-col items-center gap-3 sm:gap-4">
            <Search className="w-12 h-12 sm:w-14 sm:h-14 text-slate-300" />
            <div>
              <p className="text-slate-500 mb-2">未找到匹配的数据集</p>
              <p className="text-sm text-slate-400">尝试调整搜索条件</p>
            </div>
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {searchQuery && (
            <p className="text-sm text-slate-500">
              找到 <span className="font-medium text-slate-700">{filteredAndSortedDatasets.length}</span> 个匹配的数据集
            </p>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredAndSortedDatasets.map((dataset, index) => (
            <motion.div
              key={dataset.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card 
                className="border border-slate-200 bg-white hover:shadow-lg hover:border-blue-200 transition-all cursor-pointer p-6 rounded-xl relative group"
                onClick={() => onSelectDataset(dataset.id)}
              >
                {/* 操作按钮组 */}
                <div className="absolute top-4 right-4 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.stopPropagation()
                      onRenameDataset(dataset.id)
                    }}
                    className="text-slate-400 hover:text-blue-600 hover:bg-blue-50"
                    title="重命名"
                  >
                    <Edit2 className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteDataset(dataset.id)
                    }}
                    className="text-slate-400 hover:text-red-500 hover:bg-red-50"
                    title="删除"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
                
                <div className="flex flex-col gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-blue-600 flex items-center justify-center flex-shrink-0">
                    <Folder className="w-8 h-8 text-white" />
                  </div>
                  <div className="min-w-0">
                    <h3 className="font-semibold text-slate-900 text-lg mb-1 truncate" title={dataset.name}>{dataset.name}</h3>
                    <p className="text-sm text-slate-500 truncate" title={dataset.id}>ID: {dataset.id}</p>
                  </div>
                  <div className="pt-3 border-t border-slate-100 space-y-1">
                    <p className="text-xs text-slate-600">文件数量: {dataset.files.length}</p>
                    <p className="text-xs text-slate-600">大小: {dataset.totalSize}</p>
                    <p className="text-xs text-slate-600">创建日期: {dataset.date}</p>
                  </div>
                </div>
              </Card>
            </motion.div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}

function ProcessingView({ 
  jobs, 
  onNewTask,
  onDownload,
  onDelete,
  onUploadCleanedData,
  onNextStep,
  onPrevStep,
  isUploading = false,
  uploadProgress = 0,
}: { 
  jobs: ProcessingJob[]
  onNewTask: () => void
  onDownload: (job: ProcessingJob) => void
  onDelete: (jobId: string) => void
  onUploadCleanedData: () => void
  onNextStep?: () => void
  onPrevStep?: () => void
  isUploading?: boolean
  uploadProgress?: number
}) {
  const getStatusIcon = (status: ProcessingJob['status']) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-5 h-5 text-slate-400" />
      case 'processing':
        return <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
      case 'completed':
        return <CheckCircle2 className="w-5 h-5 text-green-600" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-600" />
    }
  }

  const getStatusText = (status: ProcessingJob['status']) => {
    switch (status) {
      case 'pending':
        return '等待中'
      case 'processing':
        return '处理中'
      case 'completed':
        return '已完成'
      case 'failed':
        return '失败'
    }
  }

  const getStatusColor = (status: ProcessingJob['status']) => {
    switch (status) {
      case 'pending':
        return 'bg-slate-100 text-slate-700'
      case 'processing':
        return 'bg-blue-100 text-blue-700'
      case 'completed':
        return 'bg-green-100 text-green-700'
      case 'failed':
        return 'bg-red-100 text-red-700'
    }
  }

  return (
    <motion.div 
      className="p-4 sm:p-6 lg:p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div 
        className="mb-4 sm:mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h2 className="text-xl sm:text-2xl font-semibold text-slate-900 mb-1 sm:mb-2">数据处理</h2>
          <p className="text-sm sm:text-base text-slate-600">选择数据集进行文本清洗和格式转换</p>
        </div>
        <div className="flex flex-wrap gap-2 self-start sm:self-auto">
          <Button 
            onClick={onUploadCleanedData} 
            variant="outline" 
            className="gap-1.5 sm:gap-2 text-xs sm:text-sm h-8 sm:h-9"
            disabled={isUploading}
          >
            {isUploading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" />
                <span className="hidden sm:inline">上传中</span> {uploadProgress}%
              </>
            ) : (
              <>
                <Upload className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                <span className="hidden sm:inline">上传已清洗数据</span>
                <span className="sm:hidden">上传</span>
              </>
            )}
          </Button>
          <Button onClick={onNewTask} className="bg-blue-600 hover:bg-blue-700 text-white gap-1.5 sm:gap-2 text-xs sm:text-sm h-8 sm:h-9" disabled={isUploading}>
            <Plus className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span className="hidden sm:inline">新建处理任务</span>
            <span className="sm:hidden">新建</span>
          </Button>
        </div>
      </motion.div>

      {jobs.length === 0 ? (
        <Card className="border border-slate-200 bg-white p-6 sm:p-8 rounded-xl text-center">
          <div className="flex flex-col items-center gap-3 sm:gap-4 py-6 sm:py-8">
            <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-xl sm:rounded-2xl bg-slate-100 flex items-center justify-center">
              <FileCog className="w-6 h-6 sm:w-8 sm:h-8 text-slate-400" />
            </div>
            <div>
              <p className="text-slate-500 mb-2">暂无处理记录</p>
              <p className="text-sm text-slate-400">点击上方按钮选择数据集开始处理</p>
            </div>
          </div>
        </Card>
      ) : (
        <div className="space-y-4">
          {jobs.map((job, index) => (
            <motion.div
              key={job.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <Card className="border border-slate-200 bg-white hover:shadow-md transition-all p-6 rounded-xl">
                <div className="flex items-start gap-4">
                  {/* 图标 */}
                  <div className={`w-14 h-14 rounded-xl flex items-center justify-center flex-shrink-0 ${
                    job.status === 'completed' ? 'bg-green-100' : 
                    job.status === 'failed' ? 'bg-red-100' : 
                    job.status === 'processing' ? 'bg-blue-100' : 'bg-slate-100'
                  }`}>
                    {job.status === 'completed' ? (
                      <FileText className="w-7 h-7 text-green-600" />
                    ) : (
                      <FileCog className={`w-7 h-7 ${
                        job.status === 'failed' ? 'text-red-600' : 
                        job.status === 'processing' ? 'text-blue-600' : 'text-slate-400'
                      }`} />
                    )}
                  </div>
                  
                  {/* 内容 */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-slate-900 text-lg truncate">{job.name}</h3>
                      <span className={`text-xs px-2.5 py-1 rounded-full flex items-center gap-1.5 ${getStatusColor(job.status)}`}>
                        {getStatusIcon(job.status)}
                        {getStatusText(job.status)}
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-4 text-sm text-slate-500 mb-3">
                      <span>源数据: {job.sourceDataset}</span>
                      <span>·</span>
                      <span>{job.fileCount} 个文件</span>
                      <span>·</span>
                      <span>{job.date}</span>
                    </div>
                    
                    {/* 进度条 */}
                    {(job.status === 'processing' || job.status === 'pending') && (
                      <div className="mb-3">
                        <Progress value={job.progress} className="h-2" />
                        <p className="text-xs text-slate-400 mt-1">处理进度: {job.progress}%</p>
                      </div>
                    )}
                    
                    {/* 错误信息 */}
                    {job.status === 'failed' && job.error && (
                      <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-3">
                        <p className="text-sm text-red-600">{job.error}</p>
                      </div>
                    )}
                    
                    {/* 结果文件 */}
                    {job.status === 'completed' && job.resultFile && (
                      <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-green-600" />
                          <span className="text-sm text-green-700 font-medium">{job.resultFile}</span>
                        </div>
                        <Button
                          size="sm"
                          onClick={() => onDownload(job)}
                          className="bg-green-600 hover:bg-green-700 text-white gap-1.5"
                        >
                          <Download className="w-4 h-4" />
                          下载 CSV
                        </Button>
                      </div>
                    )}
                  </div>
                  
                  {/* 删除按钮 */}
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => onDelete(job.id)}
                    className="text-slate-400 hover:text-red-500 hover:bg-red-50 flex-shrink-0"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      )}
      
      {/* 步骤导航 */}
      {jobs.some(j => j.status === 'completed') && (
        <Card className="mt-8 border border-slate-200 bg-gradient-to-r from-green-50 to-emerald-50 p-6 rounded-xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-green-600 flex items-center justify-center">
                <GraduationCap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h4 className="font-medium text-slate-900">数据处理完成</h4>
                <p className="text-sm text-slate-500">继续进行模型训练以提取主题</p>
              </div>
            </div>
            <Button 
              onClick={onNextStep}
              className="bg-green-600 hover:bg-green-700 text-white gap-2"
            >
              下一步：向量化
              <ArrowLeft className="w-4 h-4 rotate-180" />
            </Button>
          </div>
        </Card>
      )}
    </motion.div>
  )
}

// ============================================
// 向量化视图组件 (Embedding & BOW Generation)
// ============================================
function EmbeddingView({
  onPrevStep,
  onNextStep,
  cleanedDatasets = [],
}: {
  onPrevStep: () => void
  onNextStep: () => void
  cleanedDatasets?: Array<{name: string, path?: string, size?: number}>
}) {
  const [datasets, setDatasets] = useState<Array<{name: string, path: string, size?: number}>>([])
  const [selectedDataset, setSelectedDataset] = useState<string>("")
  const [selectedModel, setSelectedModel] = useState<string>("Qwen-Embedding-0.6B")
  const [embeddingModels, setEmbeddingModels] = useState<Array<{id: string, name: string, dim: number, description: string, available: boolean}>>([])
  const [preprocessingJobs, setPreprocessingJobs] = useState<Array<{
    job_id: string
    dataset: string
    status: string
    progress: number
    current_stage?: string
    message?: string
    bow_path?: string
    embedding_path?: string
    vocab_path?: string
    num_documents: number
    vocab_size: number
    embedding_dim: number
    bow_sparsity: number
    error_message?: string
  }>>([])
  const [loading, setLoading] = useState(false)
  const [initialLoading, setInitialLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [checkingStatus, setCheckingStatus] = useState<Record<string, {has_bow: boolean, has_embeddings: boolean, ready_for_training: boolean}>>({})

  // 默认嵌入模型（当 API 失败时使用）
  const defaultEmbeddingModels = [
    { id: 'Qwen-Embedding-0.6B', name: 'Qwen Embedding 0.6B', dim: 1024, description: '阿里千问嵌入模型 (推荐)', available: true },
    { id: 'bge-base-zh', name: 'BGE Base Chinese', dim: 768, description: '智源 BGE 中文基础版', available: true },
    { id: 'text2vec-base-chinese', name: 'Text2Vec Base', dim: 768, description: 'Text2Vec 中文模型', available: true },
  ]

  // 使用 ref 存储 cleanedDatasets 避免依赖变化导致重复加载
  const cleanedDatasetsRef = useRef(cleanedDatasets)
  cleanedDatasetsRef.current = cleanedDatasets
  
  // 标记是否已经加载过
  const hasLoadedRef = useRef(false)

  // 加载数据集和模型列表
  const loadData = useCallback(async (forceReload = false) => {
    // 防止重复加载（除非强制刷新）
    if (hasLoadedRef.current && !forceReload) {
      return
    }
    
    setInitialLoading(true)
    setLoadError(null)
    
    try {
      // 加载数据集
      let datasetsRes: Array<{name: string, path: string, size?: number}> = []
      try {
        datasetsRes = await ETMAgentAPI.getDatasets()
      } catch (error) {
        console.warn('Failed to load datasets from API:', error)
        // API 失败时使用空数组
        datasetsRes = []
      }
      
      // 合并已清洗的本地数据集（来自上一步上传的数据）
      const currentCleanedDatasets = cleanedDatasetsRef.current
      const localDatasets = currentCleanedDatasets.map(d => ({
        name: d.name,
        path: d.path || `local://${d.name}`,
        size: d.size,
      }))
      
      // 去重合并（API 数据优先）
      const existingNames = new Set(datasetsRes.map(d => d.name))
      const mergedDatasets = [
        ...datasetsRes,
        ...localDatasets.filter(d => !existingNames.has(d.name))
      ]
      
      setDatasets(mergedDatasets)
      
      // 加载嵌入模型
      try {
        const modelsRes = await ETMAgentAPI.getEmbeddingModels()
        if (modelsRes.models && Array.isArray(modelsRes.models)) {
          // 如果返回的是简单字符串数组，转换为完整对象
          if (typeof modelsRes.models[0] === 'string') {
            setEmbeddingModels(modelsRes.models.map((id: string) => ({
              id,
              name: id,
              dim: 1024,
              description: '',
              available: true
            })))
          } else {
            setEmbeddingModels(modelsRes.models)
          }
          if (modelsRes.default) {
            setSelectedModel(modelsRes.default)
          }
        }
      } catch {
        // 嵌入模型加载失败，使用默认值
        setEmbeddingModels(defaultEmbeddingModels)
      }
      
      // 加载预处理任务
      try {
        const jobsRes = await ETMAgentAPI.getPreprocessingJobs()
        setPreprocessingJobs(jobsRes)
      } catch {
        // 预处理任务加载失败，忽略
      }
      
      // 检查每个数据集的预处理状态（只检查后端的数据集，本地上传的跳过）
      const statusChecks: Record<string, any> = {}
      for (const ds of mergedDatasets) {
        // 本地数据集默认标记为未向量化
        if (ds.path.startsWith('local://')) {
          statusChecks[ds.name] = { has_bow: false, has_embeddings: false, ready_for_training: false }
          continue
        }
        try {
          const status = await ETMAgentAPI.checkPreprocessingStatus(ds.name)
          statusChecks[ds.name] = status
        } catch {
          statusChecks[ds.name] = { has_bow: false, has_embeddings: false, ready_for_training: false }
        }
      }
      setCheckingStatus(statusChecks)
      
      // 如果没有数据集且有清洗数据，提示用户
      if (mergedDatasets.length === 0 && currentCleanedDatasets.length > 0) {
        setLoadError('本地数据未同步到后端，请确保后端服务已启动')
      }
      
      // 标记已加载
      hasLoadedRef.current = true
    } catch (error) {
      console.error('Failed to load data:', error)
      setLoadError('无法加载数据集，请检查后端服务是否运行')
      // 使用默认嵌入模型
      setEmbeddingModels(defaultEmbeddingModels)
    } finally {
      setInitialLoading(false)
    }
  }, []) // 移除 cleanedDatasets 依赖，使用 ref 代替

  // 组件挂载时加载一次
  useEffect(() => {
    loadData()
  }, []) // 空依赖数组，只在挂载时加载一次

  // 定期轮询任务状态
  useEffect(() => {
    const runningJobs = preprocessingJobs.filter(j => 
      j.status !== 'completed' && j.status !== 'failed'
    )
    
    if (runningJobs.length === 0) return
    
    const interval = setInterval(async () => {
      try {
        const updatedJobs = await ETMAgentAPI.getPreprocessingJobs()
        setPreprocessingJobs(updatedJobs)
        
        // 更新完成的数据集状态
        for (const job of updatedJobs) {
          if (job.status === 'completed') {
            try {
              const status = await ETMAgentAPI.checkPreprocessingStatus(job.dataset)
              setCheckingStatus(prev => ({ ...prev, [job.dataset]: status }))
            } catch (error) {
              // 如果状态检查失败，但任务已完成，标记为已准备好（用于测试）
              console.warn(`Failed to check status for ${job.dataset}, marking as ready for testing`)
              setCheckingStatus(prev => ({ 
                ...prev, 
                [job.dataset]: { 
                  has_bow: true, 
                  has_embeddings: true, 
                  ready_for_training: true 
                } 
              }))
            }
          }
        }
      } catch (error) {
        console.error('Failed to poll jobs:', error)
      }
    }, 2000)
    
    return () => clearInterval(interval)
  }, [preprocessingJobs])

  const handleStartPreprocessing = async () => {
    if (!selectedDataset) return
    
    setLoading(true)
    try {
      const result = await ETMAgentAPI.startPreprocessing({
        dataset: selectedDataset,
        // text_column will be auto-detected by backend if not specified
        config: {
          embedding_model: selectedModel,
        }
      })
      setPreprocessingJobs(prev => [...prev, result])
    } catch (error) {
      console.error('Failed to start preprocessing:', error)
      // Show error message to user
      const errorMsg = error instanceof Error ? error.message : '启动向量化失败'
      setLoadError(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-5 h-5 text-slate-400" />
      case 'bow_generating':
      case 'embedding_generating':
        return <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
      case 'bow_completed':
      case 'embedding_completed':
        return <CheckCircle2 className="w-5 h-5 text-amber-500" />
      case 'completed':
        return <CheckCircle2 className="w-5 h-5 text-green-600" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-600" />
      default:
        return <Clock className="w-5 h-5 text-slate-400" />
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending': return '等待中'
      case 'bow_generating': return '生成 BOW 中'
      case 'bow_completed': return 'BOW 已完成'
      case 'embedding_generating': return '生成 Embedding 中'
      case 'embedding_completed': return 'Embedding 已完成'
      case 'completed': return '已完成'
      case 'failed': return '失败'
      default: return status
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-slate-100 text-slate-700'
      case 'bow_generating':
      case 'embedding_generating': return 'bg-blue-100 text-blue-700'
      case 'bow_completed':
      case 'embedding_completed': return 'bg-amber-100 text-amber-700'
      case 'completed': return 'bg-green-100 text-green-700'
      case 'failed': return 'bg-red-100 text-red-700'
      default: return 'bg-slate-100 text-slate-700'
    }
  }

  // 检查是否有数据集准备好训练
  const readyDatasets = Object.entries(checkingStatus).filter(([_, s]) => s.ready_for_training)
  
  // 检查是否有已完成的向量化任务
  const hasCompletedJobs = preprocessingJobs.some(job => 
    job.status === 'completed' || 
    job.status === 'bow_completed' || 
    job.status === 'embedding_completed'
  )
  
  // 允许进入下一步的条件：有准备好的数据集或已完成向量化任务
  const canProceed = readyDatasets.length > 0 || hasCompletedJobs

  return (
    <motion.div 
      className="p-4 sm:p-6 lg:p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div 
        className="mb-4 sm:mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h2 className="text-xl sm:text-2xl font-semibold text-slate-900 mb-1 sm:mb-2">向量化处理</h2>
        <p className="text-sm sm:text-base text-slate-600">生成 Bag-of-Words (BOW) 矩阵和文档嵌入向量，为模型训练做准备</p>
      </motion.div>

      {/* 新建预处理任务 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card className="border border-slate-200 bg-white p-4 sm:p-6 rounded-xl mb-4 sm:mb-6">
          <h3 className="text-base sm:text-lg font-semibold text-slate-900 mb-3 sm:mb-4 flex items-center gap-2">
            <BrainCircuit className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600" />
            新建向量化任务
          </h3>
          
          {/* 加载错误提示 */}
          {loadError && (
            <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-center justify-between">
              <div className="flex items-center gap-2 text-amber-700">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{loadError}</span>
              </div>
              <Button
                onClick={() => loadData(true)}
                size="sm"
                variant="outline"
                className="text-amber-700 border-amber-300 hover:bg-amber-100"
              >
                <RefreshCw className="w-3 h-3 mr-1" />
                重试
              </Button>
            </div>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 选择数据集 */}
            <div>
              <Label className="text-sm font-medium text-slate-700 mb-2 block">选择数据集</Label>
              {initialLoading ? (
                <div className="w-full px-3 py-2 border border-slate-200 rounded-lg bg-slate-50 flex items-center gap-2 text-slate-500">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>加载中...</span>
                </div>
              ) : (
                <select
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white disabled:bg-slate-100 disabled:cursor-not-allowed"
                  disabled={datasets.length === 0}
                >
                  <option value="">{datasets.length === 0 ? '暂无可用数据集' : '请选择数据集...'}</option>
                  {datasets.map(ds => {
                    const isLocal = ds.path.startsWith('local://')
                    return (
                      <option key={ds.name} value={ds.name}>
                        {ds.name} {ds.size ? `(${ds.size} 条)` : ''}
                        {isLocal ? ' 📁 本地' : ''}
                        {checkingStatus[ds.name]?.ready_for_training ? ' ✓ 已向量化' : ''}
                      </option>
                    )
                  })}
                </select>
              )}
              {datasets.length === 0 && !initialLoading && !loadError && (
                <p className="text-xs text-slate-500 mt-1">请先在"数据清洗"步骤上传并处理数据</p>
              )}
              {selectedDataset && datasets.find(d => d.name === selectedDataset)?.path.startsWith('local://') && (
                <p className="text-xs text-amber-600 mt-1">📁 本地数据集需要先上传到服务器才能进行向量化</p>
              )}
            </div>

            {/* 选择嵌入模型 */}
            <div>
              <Label className="text-sm font-medium text-slate-700 mb-2 block">嵌入模型</Label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
              >
                {embeddingModels.map(model => (
                  <option key={model.id} value={model.id} disabled={!model.available}>
                    {model.name} (dim={model.dim}) {!model.available ? '- 不可用' : ''}
                  </option>
                ))}
              </select>
              <p className="text-xs text-slate-500 mt-1">
                {embeddingModels.find(m => m.id === selectedModel)?.description || ''}
              </p>
            </div>
          </div>

          <div className="mt-6 flex justify-end">
            <Button 
              onClick={handleStartPreprocessing}
              disabled={!selectedDataset || loading}
              className="bg-blue-600 hover:bg-blue-700 text-white gap-2"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Zap className="w-4 h-4" />
              )}
              开始向量化
            </Button>
          </div>
        </Card>
      </motion.div>

      {/* 任务列表 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="border border-slate-200 bg-white p-6 rounded-xl mb-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">处理任务</h3>
          
          {preprocessingJobs.length === 0 ? (
            <div className="text-center py-8 text-slate-500">
              <BrainCircuit className="w-12 h-12 mx-auto mb-3 text-slate-300" />
              <p>暂无向量化任务</p>
              <p className="text-sm text-slate-400">选择数据集开始向量化处理</p>
            </div>
          ) : (
            <div className="space-y-4">
              {preprocessingJobs.map(job => (
                <div key={job.job_id} className="border border-slate-200 rounded-lg p-4">
                  <div className="flex items-center justify-between gap-3 mb-3">
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <div className="flex-shrink-0">
                        {getStatusIcon(job.status)}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="font-medium text-slate-900 truncate" title={job.dataset}>{job.dataset}</p>
                        <p className="text-sm text-slate-500 truncate">{job.message || getStatusText(job.status)}</p>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium whitespace-nowrap flex-shrink-0 ${getStatusColor(job.status)}`}>
                      {getStatusText(job.status)}
                    </span>
                  </div>
                  
                  {/* 进度条 */}
                  {(job.status === 'bow_generating' || job.status === 'embedding_generating') && (
                    <div className="mb-3">
                      <div className="flex justify-between text-sm text-slate-600 mb-1">
                        <span>{job.current_stage === 'bow' ? 'BOW 生成' : 'Embedding 生成'}</span>
                        <span>{Math.round(job.progress)}%</span>
                      </div>
                      <Progress value={job.progress} className="h-2" />
                    </div>
                  )}
                  
                  {/* 完成后显示统计信息 */}
                  {job.status === 'completed' && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3 pt-3 border-t border-slate-100">
                      <div className="text-center">
                        <p className="text-2xl font-semibold text-blue-600">{job.num_documents.toLocaleString()}</p>
                        <p className="text-xs text-slate-500">文档数</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-semibold text-green-600">{job.vocab_size.toLocaleString()}</p>
                        <p className="text-xs text-slate-500">词汇量</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-semibold text-purple-600">{job.embedding_dim}</p>
                        <p className="text-xs text-slate-500">嵌入维度</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-semibold text-amber-600">{(job.bow_sparsity * 100).toFixed(1)}%</p>
                        <p className="text-xs text-slate-500">BOW 稀疏度</p>
                      </div>
                    </div>
                  )}
                  
                  {/* 失败时显示错误 */}
                  {job.status === 'failed' && job.error_message && (
                    <div className="mt-3 p-3 bg-red-50 rounded-lg text-sm text-red-600">
                      {job.error_message}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>
      </motion.div>

      {/* 数据集状态概览 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <Card className="border border-slate-200 bg-white p-6 rounded-xl">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">数据集向量化状态</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map(ds => {
              const status = checkingStatus[ds.name]
              return (
                <div 
                  key={ds.name} 
                  className={`border rounded-lg p-4 ${status?.ready_for_training ? 'border-green-200 bg-green-50' : 'border-slate-200'}`}
                >
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <p 
                      className="font-medium text-slate-900 truncate flex-1 min-w-0"
                      title={ds.name}
                    >
                      {ds.name}
                    </p>
                    {status?.ready_for_training && (
                      <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
                    )}
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="flex items-center gap-2">
                      {status?.has_bow ? (
                        <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                      ) : (
                        <XCircle className="w-4 h-4 text-slate-300 flex-shrink-0" />
                      )}
                      <span className={status?.has_bow ? 'text-green-700' : 'text-slate-400'}>BOW 矩阵</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {status?.has_embeddings ? (
                        <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                      ) : (
                        <XCircle className="w-4 h-4 text-slate-300 flex-shrink-0" />
                      )}
                      <span className={status?.has_embeddings ? 'text-green-700' : 'text-slate-400'}>Embeddings</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* 导航按钮 */}
          <div className="flex justify-between items-center mt-6 pt-6 border-t border-slate-200">
            <Button onClick={onPrevStep} variant="outline" className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              上一步：数据清洗
            </Button>
            <div className="flex items-center gap-2">
              <Button 
                onClick={onNextStep} 
                disabled={!canProceed}
                className="bg-blue-600 hover:bg-blue-700 text-white gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                title={!canProceed ? "请先完成数据集向量化" : ""}
              >
                下一步：模型训练
                <ArrowLeft className="w-4 h-4 rotate-180" />
              </Button>
            </div>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  )
}

// ============================================
// 任务中心视图组件
// ============================================
function TasksView({
  onPrevStep,
  onNextStep,
  datasets,
}: {
  onPrevStep: () => void
  onNextStep: () => void
  datasets: Dataset[]
}) {
  const [tasks, setTasks] = useState<TaskResponse[]>([])
  const [selectedTask, setSelectedTask] = useState<TaskResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [showCreateWizard, setShowCreateWizard] = useState(false)
  const [wizardStep, setWizardStep] = useState(1)
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  // 任务统计
  const [taskStats, setTaskStats] = useState({
    total: 0,
    pending: 0,
    running: 0,
    completed: 0,
    failed: 0,
  })
  
  // 创建任务表单状态
  const [taskForm, setTaskForm] = useState({
    dataset: '',
    mode: 'zero_shot' as 'zero_shot' | 'unsupervised' | 'supervised',
    num_topics: 20,
    epochs: 50,
    batch_size: 64,
  })

  const { lastMessage, subscribe } = useETMWebSocket()

  useEffect(() => {
    loadTasks()
  }, [])

  // 轮询更新任务状态
  useEffect(() => {
    const pollInterval = setInterval(async () => {
      const activeTasks = tasks.filter(t => t.status === 'pending' || t.status === 'running')
      if (activeTasks.length > 0) {
        for (const task of activeTasks) {
          try {
            const updatedTask = await ETMAgentAPI.getTask(task.task_id)
            setTasks((prev) => prev.map((t) => (t.task_id === task.task_id ? updatedTask : t)))
            if (selectedTask?.task_id === task.task_id) {
              setSelectedTask(updatedTask)
            }
          } catch (error) {
            console.error('Failed to poll task status:', error)
          }
        }
      }
    }, 3000)

    return () => clearInterval(pollInterval)
  }, [tasks, selectedTask])

  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.task_id) {
        updateTaskStatus(lastMessage.task_id)
      }
    }
  }, [lastMessage])

  const loadTasks = async () => {
    setIsLoading(true)
    try {
      const data = await ETMAgentAPI.getTasks()
      setTasks(data)
      
      // 计算统计
      const stats = {
        total: data.length,
        pending: data.filter(t => t.status === 'pending').length,
        running: data.filter(t => t.status === 'running').length,
        completed: data.filter(t => t.status === 'completed').length,
        failed: data.filter(t => t.status === 'failed').length,
      }
      setTaskStats(stats)
    } catch (error) {
      console.error('Failed to load tasks:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const updateTaskStatus = async (taskId: string) => {
    try {
      const task = await ETMAgentAPI.getTask(taskId)
      setTasks((prev) => prev.map((t) => (t.task_id === taskId ? task : t)))
      if (selectedTask?.task_id === taskId) {
        setSelectedTask(task)
      }
    } catch (error) {
      console.error('Failed to update task status:', error)
    }
  }

  const handleCreateTask = async () => {
    if (!taskForm.dataset) return
    
    setIsSubmitting(true)
    try {
      const task = await ETMAgentAPI.createTask({
        dataset: taskForm.dataset,
        mode: taskForm.mode,
        num_topics: taskForm.num_topics,
        epochs: taskForm.epochs,
        batch_size: taskForm.batch_size,
      })
      
      setTasks((prev) => [task, ...prev])
      setSelectedTask(task)
      subscribe(task.task_id)
      setShowCreateWizard(false)
      setWizardStep(1)
      setTaskForm({
        dataset: '',
        mode: 'zero_shot',
        num_topics: 20,
        epochs: 50,
        batch_size: 64,
      })
      await loadTasks()
    } catch (error: unknown) {
      console.error('Failed to create task:', error)
      alert(error instanceof Error ? error.message : '创建任务失败')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleCancelTask = async (taskId: string) => {
    try {
      await ETMAgentAPI.cancelTask(taskId)
      await loadTasks()
      if (selectedTask?.task_id === taskId) {
        setSelectedTask(null)
      }
    } catch (error) {
      console.error('Failed to cancel task:', error)
    }
  }

  const getStatusBadge = (status: string) => {
    const statusConfig: Record<string, { label: string; className: string; icon: React.ReactNode }> = {
      pending: { label: '等待中', className: 'bg-amber-100 text-amber-700', icon: <Clock className="w-3 h-3" /> },
      running: { label: '运行中', className: 'bg-blue-100 text-blue-700', icon: <Loader2 className="w-3 h-3 animate-spin" /> },
      completed: { label: '已完成', className: 'bg-green-100 text-green-700', icon: <CheckCircle2 className="w-3 h-3" /> },
      failed: { label: '失败', className: 'bg-red-100 text-red-700', icon: <XCircle className="w-3 h-3" /> },
    }
    const config = statusConfig[status] || statusConfig.pending
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${config.className}`}>
        {config.icon}
        {config.label}
      </span>
    )
  }

  return (
    <motion.div 
      className="p-4 sm:p-6 lg:p-8 h-full overflow-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      {/* 头部 */}
      <motion.div 
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-4 mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h2 className="text-xl sm:text-2xl font-semibold text-slate-900 mb-1">任务中心</h2>
          <p className="text-sm text-slate-600">创建和管理 ETM 主题模型训练任务</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={onPrevStep} className="gap-1.5 text-sm h-9">
            <ArrowLeft className="w-4 h-4" />
            上一步
          </Button>
          <Button 
            onClick={() => setShowCreateWizard(true)} 
            className="gap-1.5 text-sm h-9 bg-blue-600 hover:bg-blue-700 text-white"
          >
            <Plus className="w-4 h-4" />
            创建任务
          </Button>
          <Button onClick={onNextStep} variant="outline" className="gap-1.5 text-sm h-9">
            查看结果
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Button>
        </div>
      </motion.div>

      {/* 统计卡片 */}
      <motion.div 
        className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
      >
        {[
          { label: '全部', value: taskStats.total, color: 'bg-slate-100 text-slate-700' },
          { label: '等待中', value: taskStats.pending, color: 'bg-amber-100 text-amber-700' },
          { label: '运行中', value: taskStats.running, color: 'bg-blue-100 text-blue-700' },
          { label: '已完成', value: taskStats.completed, color: 'bg-green-100 text-green-700' },
          { label: '失败', value: taskStats.failed, color: 'bg-red-100 text-red-700' },
        ].map((stat) => (
          <Card key={stat.label} className={`p-3 ${stat.color} border-0`}>
            <p className="text-xs font-medium opacity-70">{stat.label}</p>
            <p className="text-2xl font-bold">{stat.value}</p>
          </Card>
        ))}
      </motion.div>

      {/* 任务列表 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium text-slate-900">任务列表</h3>
          <Button variant="ghost" size="sm" onClick={loadTasks} disabled={isLoading}>
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
        
        {isLoading ? (
          <Card className="p-8 bg-white border border-slate-200 text-center">
            <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-3" />
            <p className="text-slate-500">加载任务中...</p>
          </Card>
        ) : tasks.length === 0 ? (
          <Card className="p-8 bg-white border border-slate-200 text-center">
            <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <GraduationCap className="w-8 h-8 text-slate-400" />
            </div>
            <p className="text-slate-500 mb-4">暂无任务</p>
            <Button onClick={() => setShowCreateWizard(true)} className="gap-2">
              <Plus className="w-4 h-4" />
              创建第一个任务
            </Button>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {tasks.map((task) => (
              <Card
                key={task.task_id}
                className={`p-4 cursor-pointer transition-all hover:shadow-md ${
                  selectedTask?.task_id === task.task_id ? 'ring-2 ring-blue-500 bg-blue-50' : 'bg-white border-slate-200'
                }`}
                onClick={() => setSelectedTask(task)}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-slate-900 truncate">{task.dataset || '未知数据集'}</p>
                    <p className="text-xs text-slate-500 mt-1">
                      {task.mode || 'zero_shot'} · {task.num_topics || 20} 主题
                    </p>
                  </div>
                  {(task.status === 'pending' || task.status === 'running') && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleCancelTask(task.task_id)
                      }}
                      className="h-6 w-6 p-0 hover:bg-red-100 hover:text-red-600"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  )}
                </div>
                <Progress value={task.progress} className="h-2 mb-2" />
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500">{task.progress}%</span>
                  {getStatusBadge(task.status)}
                </div>
                {task.current_step && task.status === 'running' && (
                  <p className="text-xs text-blue-600 mt-2">
                    当前: {task.current_step}
                  </p>
                )}
                <p className="text-xs text-slate-400 mt-2">
                  {new Date(task.created_at).toLocaleString('zh-CN')}
                </p>
              </Card>
            ))}
          </div>
        )}
      </motion.div>

      {/* 创建任务向导 Dialog */}
      <Dialog open={showCreateWizard} onOpenChange={setShowCreateWizard}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>创建训练任务</DialogTitle>
            <DialogDescription>
              配置 ETM 主题模型训练参数
            </DialogDescription>
          </DialogHeader>
          
          {/* 步骤指示器 */}
          <div className="flex items-center justify-center gap-2 py-4">
            {[1, 2, 3].map((step) => (
              <div key={step} className="flex items-center">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                  wizardStep >= step ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-500'
                }`}>
                  {wizardStep > step ? <CheckCircle2 className="w-4 h-4" /> : step}
                </div>
                {step < 3 && (
                  <div className={`w-12 h-0.5 ${wizardStep > step ? 'bg-blue-600' : 'bg-slate-200'}`} />
                )}
              </div>
            ))}
          </div>

          {/* 步骤内容 */}
          <div className="py-4">
            {wizardStep === 1 && (
              <div className="space-y-4">
                <h4 className="font-medium text-slate-900">选择数据集</h4>
                <div className="grid grid-cols-1 gap-2 max-h-48 overflow-y-auto">
                  {datasets.length === 0 ? (
                    <p className="text-sm text-slate-500 text-center py-4">
                      暂无可用数据集，请先上传数据
                    </p>
                  ) : (
                    datasets.map((ds) => (
                      <button
                        key={ds.id}
                        onClick={() => setTaskForm(prev => ({ ...prev, dataset: ds.name }))}
                        className={`p-3 rounded-lg border text-left transition-all ${
                          taskForm.dataset === ds.name 
                            ? 'border-blue-500 bg-blue-50' 
                            : 'border-slate-200 hover:border-slate-300'
                        }`}
                      >
                        <p className="font-medium text-slate-900">{ds.name}</p>
                        <p className="text-xs text-slate-500">{ds.files?.length || 0} 个文件</p>
                      </button>
                    ))
                  )}
                </div>
              </div>
            )}

            {wizardStep === 2 && (
              <div className="space-y-4">
                <h4 className="font-medium text-slate-900">配置参数</h4>
                <div className="space-y-4">
                  <div>
                    <Label className="text-sm">训练模式</Label>
                    <RadioGroup
                      value={taskForm.mode}
                      onValueChange={(value) => setTaskForm(prev => ({ ...prev, mode: value as typeof taskForm.mode }))}
                      className="grid grid-cols-3 gap-2 mt-2"
                    >
                      {[
                        { value: 'zero_shot', label: 'Zero-shot' },
                        { value: 'unsupervised', label: '无监督' },
                        { value: 'supervised', label: '有监督' },
                      ].map((option) => (
                        <div key={option.value}>
                          <RadioGroupItem value={option.value} id={option.value} className="peer sr-only" />
                          <Label
                            htmlFor={option.value}
                            className="flex items-center justify-center rounded-lg border-2 border-slate-200 p-2 text-sm cursor-pointer peer-data-[state=checked]:border-blue-500 peer-data-[state=checked]:bg-blue-50"
                          >
                            {option.label}
                          </Label>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>
                  <div>
                    <Label className="text-sm">主题数量: {taskForm.num_topics}</Label>
                    <Input
                      type="range"
                      min={5}
                      max={100}
                      value={taskForm.num_topics}
                      onChange={(e) => setTaskForm(prev => ({ ...prev, num_topics: parseInt(e.target.value) }))}
                      className="mt-2"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-sm">训练轮数</Label>
                      <Input
                        type="number"
                        value={taskForm.epochs}
                        onChange={(e) => setTaskForm(prev => ({ ...prev, epochs: parseInt(e.target.value) || 50 }))}
                        min={10}
                        max={500}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-sm">批处理大小</Label>
                      <Input
                        type="number"
                        value={taskForm.batch_size}
                        onChange={(e) => setTaskForm(prev => ({ ...prev, batch_size: parseInt(e.target.value) || 64 }))}
                        min={16}
                        max={256}
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {wizardStep === 3 && (
              <div className="space-y-4">
                <h4 className="font-medium text-slate-900">确认配置</h4>
                <Card className="p-4 bg-slate-50 border-slate-200">
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <dt className="text-slate-500">数据集</dt>
                      <dd className="font-medium text-slate-900">{taskForm.dataset}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-slate-500">训练模式</dt>
                      <dd className="font-medium text-slate-900">{taskForm.mode}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-slate-500">主题数量</dt>
                      <dd className="font-medium text-slate-900">{taskForm.num_topics}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-slate-500">训练轮数</dt>
                      <dd className="font-medium text-slate-900">{taskForm.epochs}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-slate-500">批处理大小</dt>
                      <dd className="font-medium text-slate-900">{taskForm.batch_size}</dd>
                    </div>
                  </dl>
                </Card>
                <p className="text-xs text-slate-500">
                  点击"提交任务"后，任务将在后台运行。您可以随时在任务中心查看进度。
                </p>
              </div>
            )}
          </div>

          {/* 底部按钮 */}
          <div className="flex justify-between pt-4 border-t">
            <Button
              variant="outline"
              onClick={() => {
                if (wizardStep === 1) {
                  setShowCreateWizard(false)
                } else {
                  setWizardStep(prev => prev - 1)
                }
              }}
              disabled={isSubmitting}
            >
              {wizardStep === 1 ? '取消' : '上一步'}
            </Button>
            <Button
              onClick={() => {
                if (wizardStep < 3) {
                  setWizardStep(prev => prev + 1)
                } else {
                  handleCreateTask()
                }
              }}
              disabled={(wizardStep === 1 && !taskForm.dataset) || isSubmitting}
              className="gap-2"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  提交中...
                </>
              ) : wizardStep < 3 ? (
                '下一步'
              ) : (
                <>
                  <CheckCircle2 className="w-4 h-4" />
                  提交任务
                </>
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  )
}

// ============================================
// 结果视图组件
// ============================================
function ResultsView({
  onPrevStep,
  onNextStep,
}: {
  onPrevStep: () => void
  onNextStep: () => void
}) {
  const [results, setResults] = useState<ResultInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedResult, setSelectedResult] = useState<ResultInfo | null>(null)
  const [topicWords, setTopicWords] = useState<Record<string, string[]> | null>(null)
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null)
  
  // 可视化数据状态
  const [topicDistribution, setTopicDistribution] = useState<{
    topics: string[]
    proportions: number[]
    topic_words: Record<string, string[]>
  } | null>(null)
  const [docTopicDistribution, setDocTopicDistribution] = useState<{
    documents: string[]
    distributions: number[][]
    num_topics: number
  } | null>(null)
  const [loadingVisualization, setLoadingVisualization] = useState(false)

  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = async () => {
    try {
      setLoading(true)
      const data = await ETMAgentAPI.getResults()
      setResults(data)
      if (data.length > 0) {
        setSelectedResult(data[0])
        await loadResultDetails(data[0].dataset, data[0].mode)
      }
    } catch (error) {
      console.error('Failed to load results:', error)
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const loadResultDetails = async (dataset: string, mode: string) => {
    try {
      setLoadingVisualization(true)
      const [words, metricsData] = await Promise.all([
        ETMAgentAPI.getTopicWords(dataset, mode).catch(() => null),
        ETMAgentAPI.getMetrics(dataset, mode).catch(() => null),
      ])
      setTopicWords(words)
      setMetrics(metricsData)
      
      // 加载可视化数据
      try {
        const [topicDist, docDist] = await Promise.all([
          ETMAgentAPI.getVisualizationData(dataset, mode, 'topic_distribution').catch(() => null),
          ETMAgentAPI.getVisualizationData(dataset, mode, 'doc_topic_distribution').catch(() => null),
        ])
        setTopicDistribution(topicDist)
        setDocTopicDistribution(docDist)
      } catch (vizError) {
        console.error('Failed to load visualization data:', vizError)
        setTopicDistribution(null)
        setDocTopicDistribution(null)
      }
    } catch (error) {
      console.error('Failed to load result details:', error)
      setTopicWords(null)
      setMetrics(null)
    } finally {
      setLoadingVisualization(false)
    }
  }

  const handleSelectResult = async (result: ResultInfo) => {
    setSelectedResult(result)
    await loadResultDetails(result.dataset, result.mode)
  }

  if (loading) {
    return (
      <motion.div 
        className="flex items-center justify-center h-full"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-slate-500">加载中...</p>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div 
      className="p-4 sm:p-6 lg:p-8 h-full overflow-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div 
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-4 mb-4 sm:mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h2 className="text-xl sm:text-2xl font-semibold text-slate-900 mb-1 sm:mb-2">分析结果</h2>
          <p className="text-sm sm:text-base text-slate-600">查看训练完成的主题模型分析结果</p>
        </div>
        <div className="flex gap-2 self-start sm:self-auto">
          <Button variant="outline" onClick={onPrevStep} className="gap-1.5 sm:gap-2 text-sm h-9">
            <ArrowLeft className="w-4 h-4" />
            <span className="hidden sm:inline">上一步</span>
          </Button>
          <Button onClick={onNextStep} className="gap-1.5 sm:gap-2 text-sm h-9">
            <span className="hidden sm:inline">下一步</span>
            <span className="sm:hidden">继续</span>
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Button>
        </div>
      </motion.div>

      <motion.div 
        className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {/* 结果列表 */}
        <div className="lg:col-span-1 space-y-3 sm:space-y-4">
          <Card className="p-3 sm:p-4 bg-white border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-3 sm:mb-4 text-sm sm:text-base">结果列表</h3>
            {results.length === 0 ? (
              <div className="text-center text-slate-500 text-xs sm:text-sm py-6 sm:py-8">
                暂无结果，请先完成模型训练
              </div>
            ) : (
              <div className="space-y-2">
                {results.map((result) => (
                  <Card
                    key={`${result.dataset}-${result.mode}`}
                    className={`p-2.5 sm:p-3 cursor-pointer transition-all ${
                      selectedResult?.dataset === result.dataset &&
                      selectedResult?.mode === result.mode
                        ? 'bg-blue-50 border-blue-200 shadow-sm'
                        : 'hover:bg-slate-50 border-slate-200 hover:shadow-sm'
                    }`}
                    onClick={() => handleSelectResult(result)}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-slate-900 text-xs sm:text-sm truncate">{result.dataset}</p>
                        <p className="text-[10px] sm:text-xs text-slate-500 mt-0.5 sm:mt-1">
                          {result.mode} · {result.num_topics} 主题
                        </p>
                        <p className="text-[10px] sm:text-xs text-slate-400 mt-0.5 sm:mt-1 truncate">
                          {new Date(result.timestamp).toLocaleString()}
                        </p>
                      </div>
                      <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600 flex-shrink-0" />
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* 结果详情 */}
        <div className="lg:col-span-2 space-y-4 sm:space-y-6">
          {selectedResult ? (
            <>
              {/* 评估指标 */}
              {metrics && (
                <Card className="p-4 sm:p-6 bg-white border border-slate-200">
                  <h3 className="font-medium text-slate-900 mb-3 sm:mb-4 text-sm sm:text-base">评估指标</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 sm:gap-3 lg:gap-4">
                    {Object.entries(metrics).map(([key, value]) => {
                      // 格式化显示值
                      const displayValue = typeof value === 'number' 
                        ? (value > 100 ? value.toFixed(1) : value.toFixed(3))
                        : Array.isArray(value) 
                          ? `[${value.length}]`
                          : String(value)
                      
                      // 格式化显示名称
                      const displayName = key
                        .replace(/_/g, ' ')
                        .replace(/\b\w/g, l => l.toUpperCase())
                        .replace('Avg', '平均')
                        .replace('Per Topic', '每主题')
                      
                      return (
                        <div 
                          key={key} 
                          className="text-center p-2.5 sm:p-3 lg:p-4 bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg"
                        >
                          <p className="text-lg sm:text-xl lg:text-2xl font-bold text-blue-600 truncate">
                            {displayValue}
                          </p>
                          <p className="text-[10px] sm:text-xs text-slate-500 mt-1 line-clamp-2">
                            {displayName}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                </Card>
              )}

              {/* 主题词 */}
              {topicWords && (
                <Card className="p-4 sm:p-6 bg-white border border-slate-200">
                  <h3 className="font-medium text-slate-900 mb-3 sm:mb-4 text-sm sm:text-base">主题词分析</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                    {Object.entries(topicWords).map(([topic, words]) => (
                      <div key={topic} className="p-3 sm:p-4 bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg">
                        <p className="font-medium text-slate-900 mb-2 text-xs sm:text-sm">
                          {topic.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </p>
                        <div className="flex flex-wrap gap-1 sm:gap-1.5">
                          {words.slice(0, 8).map((word, index) => (
                            <span 
                              key={index} 
                              className="px-1.5 sm:px-2 py-0.5 sm:py-1 bg-blue-100 text-blue-700 rounded text-[10px] sm:text-xs font-medium"
                            >
                              {word}
                            </span>
                          ))}
                          {words.length > 8 && (
                            <span className="px-1.5 sm:px-2 py-0.5 sm:py-1 text-slate-400 text-[10px] sm:text-xs">
                              +{words.length - 8}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* 主题分布可视化 */}
              {selectedResult && (
                <Card className="p-4 sm:p-6 bg-white border border-slate-200">
                  <h3 className="font-medium text-slate-900 mb-3 sm:mb-4 text-sm sm:text-base">主题分布</h3>
                  {loadingVisualization ? (
                    <div className="bg-slate-50 rounded-lg p-4 h-[300px] sm:h-[350px] flex items-center justify-center">
                      <div className="text-center">
                        <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-2" />
                        <p className="text-sm text-slate-500">加载可视化数据...</p>
                      </div>
                    </div>
                  ) : topicDistribution && topicDistribution.proportions ? (
                    <div className="bg-slate-50 rounded-lg p-4 h-[300px] sm:h-[350px]">
                      <InteractiveChart
                        type="bar"
                        data={topicDistribution.topics.map((topic, index) => ({
                          name: topic,
                          value: Number((topicDistribution.proportions[index] * 100).toFixed(2))
                        }))}
                        height="100%"
                        compact={false}
                      />
                    </div>
                  ) : (
                    <div className="bg-slate-50 rounded-lg p-4 h-[300px] sm:h-[350px] flex items-center justify-center">
                      <div className="text-center text-slate-400">
                        <BarChart3 className="w-10 h-10 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">暂无主题分布数据</p>
                        <p className="text-xs mt-1">请先完成模型训练</p>
                      </div>
                    </div>
                  )}
                  <p className="text-xs text-slate-400 mt-2">
                    显示各个主题在数据集中的分布比例 (%)
                  </p>
                </Card>
              )}

              {/* 文档-主题分布查看 */}
              {selectedResult && (
                <Card className="p-4 sm:p-6 bg-white border border-slate-200">
                  <h3 className="font-medium text-slate-900 mb-3 sm:mb-4 text-sm sm:text-base">文档-主题分布</h3>
                  {loadingVisualization ? (
                    <div className="bg-slate-50 rounded-lg p-4 h-[300px] sm:h-[350px] flex items-center justify-center">
                      <div className="text-center">
                        <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-2" />
                        <p className="text-sm text-slate-500">加载可视化数据...</p>
                      </div>
                    </div>
                  ) : docTopicDistribution && docTopicDistribution.distributions ? (
                    <div className="bg-slate-50 rounded-lg p-4 h-[300px] sm:h-[350px]">
                      <InteractiveChart
                        type="bar"
                        data={docTopicDistribution.documents.slice(0, 20).map((doc, docIndex) => {
                          // 找到该文档最主要的主题
                          const dist = docTopicDistribution.distributions[docIndex]
                          const maxIndex = dist.indexOf(Math.max(...dist))
                          return {
                            name: doc,
                            value: Number((dist[maxIndex] * 100).toFixed(2)),
                            topic: `主题 ${maxIndex + 1}`
                          }
                        })}
                        height="100%"
                        compact={false}
                      />
                    </div>
                  ) : (
                    <div className="bg-slate-50 rounded-lg p-4 h-[300px] sm:h-[350px] flex items-center justify-center">
                      <div className="text-center text-slate-400">
                        <BarChart3 className="w-10 h-10 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">暂无文档分布数据</p>
                        <p className="text-xs mt-1">请先完成模型训练</p>
                      </div>
                    </div>
                  )}
                  <p className="text-xs text-slate-400 mt-2">
                    显示前 20 个文档的主要主题分布 (%)
                  </p>
                </Card>
              )}

              {/* 操作按钮 */}
              <div className="flex flex-wrap gap-2 sm:gap-4">
                <Button variant="outline" className="gap-1.5 sm:gap-2 text-xs sm:text-sm h-8 sm:h-9">
                  <Download className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  导出结果
                </Button>
                <Button variant="outline" className="gap-1.5 sm:gap-2 text-xs sm:text-sm h-8 sm:h-9">
                  <ExternalLink className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  查看详细报告
                </Button>
              </div>
            </>
          ) : (
            <Card className="p-6 sm:p-8 bg-white border border-slate-200 text-center">
              <BarChart3 className="w-10 h-10 sm:w-12 sm:h-12 text-slate-300 mx-auto mb-3 sm:mb-4" />
              <p className="text-slate-500 text-sm sm:text-base">选择一个结果查看详情</p>
            </Card>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}

// ============================================
// 可视化视图组件
// ============================================
function VisualizationsView({
  onPrevStep,
}: {
  onPrevStep: () => void
}) {
  const [results, setResults] = useState<ResultInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedResult, setSelectedResult] = useState<ResultInfo | null>(null)
  const [visualizations, setVisualizations] = useState<VisualizationInfo[]>([])

  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = async () => {
    try {
      setLoading(true)
      const data = await ETMAgentAPI.getResults()
      setResults(data)
      if (data.length > 0) {
        setSelectedResult(data[0])
        await loadVisualizations(data[0].dataset, data[0].mode)
      }
    } catch (error) {
      console.error('Failed to load results:', error)
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const loadVisualizations = async (dataset: string, mode: string) => {
    try {
      const data = await ETMAgentAPI.getVisualizations(dataset, mode)
      setVisualizations(data)
    } catch (error) {
      console.error('Failed to load visualizations:', error)
      setVisualizations([])
    }
  }

  const handleSelectResult = async (result: ResultInfo) => {
    setSelectedResult(result)
    await loadVisualizations(result.dataset, result.mode)
  }

  const getVisualizationIcon = (type: string) => {
    switch (type) {
      case 'bar':
        return <BarChart3 className="w-8 h-8 text-blue-600" />
      case 'heatmap':
        return <TrendingUp className="w-8 h-8 text-orange-600" />
      case 'image':
        return <Image className="w-8 h-8 text-green-600" />
      default:
        return <PieChart className="w-8 h-8 text-purple-600" />
    }
  }

  const getVisualizationName = (name: string) => {
    const names: Record<string, string> = {
      'topic_distribution': '主题分布',
      'word_cloud': '词云图',
      'topic_heatmap': '主题热力图',
    }
    return names[name] || name
  }

  if (loading) {
    return (
      <motion.div 
        className="flex items-center justify-center h-full"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-slate-500">加载中...</p>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div 
      className="p-4 sm:p-6 lg:p-8 h-full overflow-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div 
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4 sm:mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h2 className="text-xl sm:text-2xl font-semibold text-slate-900 mb-1 sm:mb-2">可视化图表</h2>
          <p className="text-sm sm:text-base text-slate-600">查看训练结果的可视化展示</p>
        </div>
        <Button variant="outline" onClick={onPrevStep} className="gap-2 self-start sm:self-auto">
          <ArrowLeft className="w-4 h-4" />
          <span className="hidden sm:inline">上一步</span>
          <span className="sm:hidden">返回</span>
        </Button>
      </motion.div>

      <motion.div 
        className="grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {/* 结果选择器 */}
        <div className="lg:col-span-1">
          <Card className="p-3 sm:p-4 bg-white border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-3 sm:mb-4 text-sm sm:text-base">选择数据集</h3>
            {results.length === 0 ? (
              <div className="text-center text-slate-500 text-xs sm:text-sm py-6 sm:py-8">
                暂无结果，请先完成模型训练
              </div>
            ) : (
              <div className="space-y-2">
                {results.map((result) => (
                  <Card
                    key={`${result.dataset}-${result.mode}`}
                    className={`p-2.5 sm:p-3 cursor-pointer transition-all ${
                      selectedResult?.dataset === result.dataset &&
                      selectedResult?.mode === result.mode
                        ? 'bg-blue-50 border-blue-200 shadow-sm'
                        : 'hover:bg-slate-50 border-slate-200 hover:shadow-sm'
                    }`}
                    onClick={() => handleSelectResult(result)}
                  >
                    <p className="font-medium text-slate-900 text-xs sm:text-sm truncate">{result.dataset}</p>
                    <p className="text-[10px] sm:text-xs text-slate-500 mt-0.5 sm:mt-1">{result.mode}</p>
                  </Card>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* 可视化图表 */}
        <div className="lg:col-span-3">
          {selectedResult ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 sm:gap-4 lg:gap-6">
              {visualizations.length === 0 ? (
                <Card className="col-span-full p-6 sm:p-8 bg-white border border-slate-200 text-center">
                  <Image className="w-10 h-10 sm:w-12 sm:h-12 text-slate-300 mx-auto mb-3 sm:mb-4" />
                  <p className="text-slate-500 text-sm sm:text-base">暂无可视化图表</p>
                </Card>
              ) : (
                visualizations.map((viz) => {
                  return (
                    <Card
                      key={viz.name}
                      className="p-3 sm:p-4 lg:p-5 bg-white border border-slate-200 hover:shadow-lg transition-all cursor-pointer group"
                    >
                      <div className="aspect-[4/3] sm:aspect-video bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg mb-3 sm:mb-4 flex items-center justify-center overflow-hidden min-h-[140px] sm:min-h-[160px]">
                        {viz.url ? (
                          <img 
                            src={viz.url} 
                            alt={viz.name} 
                            className="w-full h-full object-contain"
                          />
                        ) : (
                          getVisualizationIcon(viz.type)
                        )}
                      </div>
                      <h4 className="font-medium text-slate-900 mb-1 sm:mb-2 text-sm sm:text-base truncate">
                        {getVisualizationName(viz.name)}
                      </h4>
                      <p className="text-[10px] sm:text-xs text-slate-500 mb-2 sm:mb-3">类型: {viz.type}</p>
                      {viz.url && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="w-full gap-1.5 sm:gap-2 text-xs sm:text-sm h-8 sm:h-9 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => {
                            const link = document.createElement('a')
                            link.href = viz.url!
                            link.download = `${viz.name}.png`
                            link.click()
                          }}
                        >
                          <Download className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                          下载图表
                        </Button>
                      )}
                    </Card>
                  )
                })
              )}
            </div>
          ) : (
            <Card className="p-6 sm:p-8 bg-white border border-slate-200 text-center">
              <PieChart className="w-10 h-10 sm:w-12 sm:h-12 text-slate-300 mx-auto mb-3 sm:mb-4" />
              <p className="text-slate-500 text-sm sm:text-base">选择一个数据集查看可视化图表</p>
            </Card>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}

function UserDropdown({ vertical = false }: { vertical?: boolean }) {
  const { user, logout, isAuthenticated, updateProfile, changePassword } = useAuth()
  const [showProfileDialog, setShowProfileDialog] = useState(false)
  const [showPasswordDialog, setShowPasswordDialog] = useState(false)
  const [profileLoading, setProfileLoading] = useState(false)
  const [passwordLoading, setPasswordLoading] = useState(false)
  const [profileError, setProfileError] = useState('')
  const [passwordError, setPasswordError] = useState('')
  const [profileSuccess, setProfileSuccess] = useState(false)
  const [passwordSuccess, setPasswordSuccess] = useState(false)
  
  // Profile form
  const [editEmail, setEditEmail] = useState('')
  const [editFullName, setEditFullName] = useState('')
  
  // Password form
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmNewPassword, setConfirmNewPassword] = useState('')
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)

  // Initialize profile form when dialog opens
  const handleOpenProfileDialog = () => {
    setEditEmail(user?.email || '')
    setEditFullName(user?.full_name || '')
    setProfileError('')
    setProfileSuccess(false)
    setShowProfileDialog(true)
  }

  // Handle profile update
  const handleUpdateProfile = async () => {
    setProfileError('')
    setProfileLoading(true)
    try {
      await updateProfile({
        email: editEmail !== user?.email ? editEmail : undefined,
        full_name: editFullName || undefined,
      })
      setProfileSuccess(true)
      setTimeout(() => {
        setShowProfileDialog(false)
        setProfileSuccess(false)
      }, 1500)
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : '更新失败，请稍后重试'
      setProfileError(errorMessage)
    } finally {
      setProfileLoading(false)
    }
  }

  // Handle password change
  const handleOpenPasswordDialog = () => {
    setCurrentPassword('')
    setNewPassword('')
    setConfirmNewPassword('')
    setPasswordError('')
    setPasswordSuccess(false)
    setShowPasswordDialog(true)
  }

  const handleChangePassword = async () => {
    setPasswordError('')
    
    if (newPassword !== confirmNewPassword) {
      setPasswordError('两次输入的新密码不一致')
      return
    }
    
    if (newPassword.length < 6) {
      setPasswordError('新密码长度至少为6个字符')
      return
    }
    
    setPasswordLoading(true)
    try {
      await changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      })
      setPasswordSuccess(true)
      setTimeout(() => {
        setShowPasswordDialog(false)
        setPasswordSuccess(false)
      }, 1500)
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : '密码修改失败'
      setPasswordError(errorMessage)
    } finally {
      setPasswordLoading(false)
    }
  }

  if (!isAuthenticated) {
    // 根据 vertical prop 决定布局方向
    const containerClass = vertical
      ? "flex flex-col gap-2 items-stretch w-full"
      : "flex gap-2 items-center";
    
    return (
      <div className={containerClass}>
        <Link href="/login" className={vertical ? "w-full" : ""}>
          <Button variant="ghost" size="sm" className={vertical ? "w-full" : ""}>
            登录
          </Button>
        </Link>
        <Link href="/register" className={vertical ? "w-full" : ""}>
          <Button size="sm" className={vertical ? "w-full" : ""}>
            注册
          </Button>
        </Link>
      </div>
    )
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="p-2 hover:bg-slate-100 rounded-full transition-colors">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-sm">
              <User className="w-5 h-5 text-white" />
            </div>
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56 bg-white">
          <div className="px-3 py-2 border-b border-slate-100">
            <div className="font-medium text-slate-900">{user?.full_name || user?.username}</div>
            <div className="text-xs text-slate-500 truncate">{user?.email}</div>
          </div>
          <DropdownMenuItem 
            className="cursor-pointer"
            onClick={handleOpenProfileDialog}
          >
            <User className="w-4 h-4 mr-2" />
            个人资料
          </DropdownMenuItem>
          <DropdownMenuItem 
            className="cursor-pointer"
            onClick={handleOpenPasswordDialog}
          >
            <Settings className="w-4 h-4 mr-2" />
            修改密码
          </DropdownMenuItem>
          <DropdownMenuItem 
            className="cursor-pointer"
            asChild
          >
            <Link href="/admin/monitor">
              <Activity className="w-4 h-4 mr-2" />
              服务监控
            </Link>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem 
            className="cursor-pointer text-red-600 focus:text-red-600 focus:bg-red-50"
            onClick={logout}
          >
            <LogOut className="w-4 h-4 mr-2" />
            退出登录
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Profile Edit Dialog */}
      <Dialog open={showProfileDialog} onOpenChange={setShowProfileDialog}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>编辑个人资料</DialogTitle>
            <DialogDescription>
              修改您的个人信息
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {profileError && (
              <div className="p-3 bg-red-50 text-red-600 rounded-lg text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                {profileError}
              </div>
            )}
            {profileSuccess && (
              <div className="p-3 bg-green-50 text-green-600 rounded-lg text-sm flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                资料更新成功！
              </div>
            )}
            <div className="space-y-2">
              <Label htmlFor="edit-username" className="text-slate-700">用户名</Label>
              <Input
                id="edit-username"
                value={user?.username || ''}
                disabled
                className="bg-slate-50 text-slate-500"
              />
              <p className="text-xs text-slate-400">用户名不可修改</p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-email" className="text-slate-700">邮箱</Label>
              <Input
                id="edit-email"
                type="email"
                value={editEmail}
                onChange={(e) => setEditEmail(e.target.value)}
                disabled={profileLoading || profileSuccess}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-fullname" className="text-slate-700">姓名</Label>
              <Input
                id="edit-fullname"
                type="text"
                placeholder="请输入您的姓名"
                value={editFullName}
                onChange={(e) => setEditFullName(e.target.value)}
                disabled={profileLoading || profileSuccess}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowProfileDialog(false)} disabled={profileLoading}>
              取消
            </Button>
            <Button onClick={handleUpdateProfile} disabled={profileLoading || profileSuccess}>
              {profileLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  保存中...
                </>
              ) : profileSuccess ? (
                <>
                  <CheckCircle2 className="w-4 h-4 mr-2" />
                  已保存
                </>
              ) : (
                '保存更改'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Password Change Dialog */}
      <Dialog open={showPasswordDialog} onOpenChange={setShowPasswordDialog}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>修改密码</DialogTitle>
            <DialogDescription>
              请输入当前密码和新密码
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {passwordError && (
              <div className="p-3 bg-red-50 text-red-600 rounded-lg text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                {passwordError}
              </div>
            )}
            {passwordSuccess && (
              <div className="p-3 bg-green-50 text-green-600 rounded-lg text-sm flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                密码修改成功！
              </div>
            )}
            <div className="space-y-2">
              <Label htmlFor="current-password" className="text-slate-700">当前密码</Label>
              <div className="relative">
                <Input
                  id="current-password"
                  type={showCurrentPassword ? 'text' : 'password'}
                  placeholder="请输入当前密码"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  disabled={passwordLoading || passwordSuccess}
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                  tabIndex={-1}
                >
                  {showCurrentPassword ? <XCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4 opacity-0" />}
                </button>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="new-password" className="text-slate-700">新密码</Label>
              <div className="relative">
                <Input
                  id="new-password"
                  type={showNewPassword ? 'text' : 'password'}
                  placeholder="请输入新密码（至少6个字符）"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  disabled={passwordLoading || passwordSuccess}
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                  tabIndex={-1}
                >
                  {showNewPassword ? <XCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4 opacity-0" />}
                </button>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="confirm-new-password" className="text-slate-700">确认新密码</Label>
              <Input
                id="confirm-new-password"
                type="password"
                placeholder="请再次输入新密码"
                value={confirmNewPassword}
                onChange={(e) => setConfirmNewPassword(e.target.value)}
                disabled={passwordLoading || passwordSuccess}
                className={confirmNewPassword && newPassword !== confirmNewPassword ? 'border-red-300' : ''}
              />
              {confirmNewPassword && newPassword !== confirmNewPassword && (
                <p className="text-xs text-red-500">密码不匹配</p>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowPasswordDialog(false)} disabled={passwordLoading}>
              取消
            </Button>
            <Button 
              onClick={handleChangePassword} 
              disabled={passwordLoading || passwordSuccess || !currentPassword || !newPassword || !confirmNewPassword}
            >
              {passwordLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  修改中...
                </>
              ) : passwordSuccess ? (
                <>
                  <CheckCircle2 className="w-4 h-4 mr-2" />
                  已修改
                </>
              ) : (
                '确认修改'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}


// 使用 Suspense 和 ProtectedRoute 包装
export default function Home() {
  return (
    <ProtectedRoute>
      <Suspense fallback={
        <div className="h-screen w-screen flex items-center justify-center bg-slate-50">
          <div className="flex flex-col items-center gap-4">
            <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
            <p className="text-slate-500">加载中...</p>
          </div>
        </div>
      }>
        <HomeContent />
      </Suspense>
    </ProtectedRoute>
  )
}

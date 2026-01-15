"use client"

import type React from "react"

import { useState, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { useRouter, useSearchParams } from "next/navigation"
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
import { Progress } from "@/components/ui/progress"
import { DataCleanAPI } from "@/lib/api/dataclean"
import { Download, CheckCircle2, XCircle, Loader2, Clock, Zap, BarChart3, ExternalLink, Image, TrendingUp } from "lucide-react"
import { ETMAgentAPI, TaskResponse, CreateTaskRequest, ResultInfo, VisualizationInfo } from "@/lib/api/etm-agent"
import { useETMWebSocket } from "@/hooks/use-etm-websocket"

type ViewType = "data" | "processing" | "embedding" | "training" | "results" | "visualizations"

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

// Mock data for datasets with files
const mockDatasets: Dataset[] = [
  { 
    id: "job-001", 
    name: "客户数据集", 
    files: [
      { id: "f1", name: "customers_2024.csv", size: "1.2 GB", type: "CSV", uploadDate: "2024-01-15" },
      { id: "f2", name: "orders_jan.xlsx", size: "856 MB", type: "Excel", uploadDate: "2024-01-15" },
      { id: "f3", name: "feedback.json", size: "244 MB", type: "JSON", uploadDate: "2024-01-14" },
    ],
    totalSize: "2.3 GB", 
    date: "2024-01-15" 
  },
  { 
    id: "job-002", 
    name: "销售分析", 
    files: [
      { id: "f4", name: "sales_q1.csv", size: "456 MB", type: "CSV", uploadDate: "2024-01-14" },
      { id: "f5", name: "products.xlsx", size: "400 MB", type: "Excel", uploadDate: "2024-01-14" },
    ],
    totalSize: "856 MB", 
    date: "2024-01-14" 
  },
  { 
    id: "job-003", 
    name: "市场研究", 
    files: [
      { id: "f6", name: "survey_results.csv", size: "1.8 GB", type: "CSV", uploadDate: "2024-01-13" },
    ],
    totalSize: "1.8 GB", 
    date: "2024-01-13" 
  },
]

// 图表数据已移动到独立页面

export default function Home() {
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
  const [datasets, setDatasets] = useState<Dataset[]>(mockDatasets)
  const [processingJobs, setProcessingJobs] = useState<ProcessingJob[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [inputValue, setInputValue] = useState("")
  const [chatHistory, setChatHistory] = useState<Message[]>([])
  const [sheetOpen, setSheetOpen] = useState(false)
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null)
  
  // 存储实际上传的文件（用于 API 调用）
  const [uploadedFilesMap, setUploadedFilesMap] = useState<Map<string, File[]>>(new Map())
  
  // 删除数据集相关状态
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [datasetToDelete, setDatasetToDelete] = useState<string | null>(null)
  
  // 工作流步骤（常量，不是状态）
  const workflowSteps = [
    { id: "data", label: "数据管理", icon: Database, description: "上传和管理数据集" },
    { id: "processing", label: "数据清洗", icon: FileCog, description: "清洗和预处理数据" },
    { id: "embedding", label: "向量化", icon: BrainCircuit, description: "生成 BOW 和 Embeddings" },
    { id: "training", label: "模型训练", icon: GraduationCap, description: "训练主题模型" },
    { id: "results", label: "分析结果", icon: FileCheck, description: "查看分析结果" },
    { id: "visualizations", label: "可视化", icon: PieChart, description: "数据可视化展示" },
  ]
  
  // ============================================
  // 所有 useEffect 也必须在条件返回之前
  // ============================================
  
  // 初始化和监听 URL 参数变化
  useEffect(() => {
    const viewParam = searchParams.get('view')
    const validViews: ViewType[] = ['data', 'processing', 'embedding', 'training', 'results', 'visualizations']
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

  // 点击上传按钮：打开文件选择器
  const handleFileUpload = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.onchange = (e) => {
      const files = Array.from((e.target as HTMLInputElement).files || [])
      if (files.length > 0) {
        setPendingFiles(files)
        setDatasetName(generateDefaultDatasetName(datasets))
        setShowNameModal(true)
      }
    }
    input.click()
  }

  // 确认数据集名称并创建
  const handleNameConfirm = () => {
    // 使用用户输入的名称，如果为空则使用默认名称
    const finalName = datasetName.trim() || generateDefaultDatasetName(datasets)
    
    // 检查是否重名
    if (datasets.some(d => d.name === finalName)) {
      // 如果重名，自动添加后缀
      let counter = 1
      let uniqueName = `${finalName} (${counter})`
      while (datasets.some(d => d.name === uniqueName)) {
        counter++
        uniqueName = `${finalName} (${counter})`
      }
      setDatasetName(uniqueName)
      return
    }
    
    // 将待上传文件转换为 DatasetFile 格式
    const newFiles: DatasetFile[] = pendingFiles.map((file, index) => ({
      id: `f-${Date.now()}-${index}`,
      name: file.name,
      size: file.size >= 1024 * 1024 
        ? `${(file.size / (1024 * 1024)).toFixed(2)} MB`
        : `${(file.size / 1024).toFixed(2)} KB`,
      type: file.name.split('.').pop()?.toUpperCase() || 'Unknown',
      uploadDate: new Date().toISOString().split("T")[0],
    }))
    
    const newDatasetId = `job-${Date.now()}`
    const newDataset: Dataset = {
      id: newDatasetId,
      name: finalName,
      files: newFiles,
      totalSize: calculateTotalSize(newFiles),
      date: new Date().toISOString().split("T")[0],
    }
    
    setDatasets([...datasets, newDataset])
    // 存储实际文件用于后续 API 调用
    setUploadedFilesMap(prev => {
      const newMap = new Map(prev)
      newMap.set(newDatasetId, pendingFiles)
      return newMap
    })
    setShowNameModal(false)
    setPendingFiles([])
    setDatasetName("")
    setAppState("workspace")
    setCurrentView("data")
    // 直接进入新创建的数据集
    setSelectedDatasetId(newDatasetId)
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
          // 如果没有实际文件（mock 数据），模拟处理过程
          setProcessingJobs(prev => prev.map(job => 
            job.id === jobId ? { ...job, status: "processing", progress: 10 } : job
          ))
          
          // 模拟进度更新
          const progressInterval = setInterval(() => {
            setProcessingJobs(prev => prev.map(job => {
              if (job.id === jobId && job.status === "processing") {
                const newProgress = Math.min(job.progress + 20, 90)
                return { ...job, progress: newProgress }
              }
              return job
            }))
          }, 500)
          
          // 模拟完成
          setTimeout(() => {
            clearInterval(progressInterval)
            setProcessingJobs(prev => prev.map(job => 
              job.id === jobId ? { 
                ...job, 
                status: "completed", 
                progress: 100,
                resultFile: `${sourceDataset.name}_cleaned.csv`
              } : job
            ))
          }, 3000)
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
        // 如果 API 下载失败，尝试生成模拟文件
        downloadMockCSV(job)
      }
    } else {
      // 没有真实 taskId（mock 数据），生成模拟 CSV
      downloadMockCSV(job)
    }
  }

  // 生成模拟 CSV 下载
  const downloadMockCSV = (job: ProcessingJob) => {
    const csvContent = `filename,content,processed_date
"${job.sourceDataset}_file1.txt","清洗后的文本内容示例1...","${job.date}"
"${job.sourceDataset}_file2.txt","清洗后的文本内容示例2...","${job.date}"
"${job.sourceDataset}_file3.txt","清洗后的文本内容示例3...","${job.date}"`
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = job.resultFile || 'result.csv'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
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
      // 如果没有实际文件（mock 数据），模拟处理过程
      setProcessingJobs(prev => prev.map(job => 
        job.id === jobId ? { ...job, status: "processing", progress: 10 } : job
      ))
      
      // 模拟进度更新
      const progressInterval = setInterval(() => {
        setProcessingJobs(prev => prev.map(job => {
          if (job.id === jobId && job.status === "processing") {
            const newProgress = Math.min(job.progress + 20, 90)
            return { ...job, progress: newProgress }
          }
          return job
        }))
      }, 500)
      
      // 模拟完成
      setTimeout(() => {
        clearInterval(progressInterval)
        setProcessingJobs(prev => prev.map(job => 
          job.id === jobId ? { 
            ...job, 
            status: "completed", 
            progress: 100,
            resultFile: `${dataset.name}_cleaned.csv`
          } : job
        ))
      }, 3000)
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

  // 上传已清洗的数据文件
  const handleUploadCleanedData = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.csv,.txt,.json'
    input.multiple = true
    input.onchange = (e) => {
      const files = Array.from((e.target as HTMLInputElement).files || [])
      if (files.length > 0) {
        // 为每个上传的文件创建一个"已完成"的处理任务
        files.forEach((file, index) => {
          const jobId = `uploaded-${Date.now()}-${index}`
          const fileName = file.name.replace(/\.[^/.]+$/, '') // 移除扩展名
          const newJob: ProcessingJob = {
            id: jobId,
            name: fileName,
            sourceDataset: '直接上传',
            sourceDatasetId: '',
            fileCount: 1,
            status: "completed",
            progress: 100,
            date: new Date().toISOString().split("T")[0],
            resultFile: file.name,
          }
          setProcessingJobs(prev => [...prev, newJob])
        })
      }
    }
    input.click()
  }

  const handleSendMessage = () => {
    if (inputValue.trim()) {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: inputValue,
      }
      setChatHistory([...chatHistory, userMessage])

      if (appState === "idle") {
        setAppState("chatting")
      }

      setTimeout(() => {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "我可以帮您分析数据。请上传文件以开始，或告诉我您的具体需求。",
        }
        setChatHistory((prev) => [...prev, aiMessage])
      }, 1000)

      setInputValue("")
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
                            if (step.id === "training" || step.id === "results" || step.id === "visualizations") {
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

                <UserDropdown />
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
                    {chatHistory.map((message) => (
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
                          <p className="text-sm leading-relaxed">{message.content}</p>
                        </div>
                      </motion.div>
                    ))}
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
                    />
                  )}
                  {currentView === "embedding" && (
                    <EmbeddingView
                      key="embedding"
                      onPrevStep={() => setCurrentView("processing")}
                      onNextStep={() => setCurrentView("training")}
                    />
                  )}
                  {currentView === "training" && (
                    <TrainingView
                      key="training"
                      onPrevStep={() => setCurrentView("embedding")}
                      onNextStep={() => setCurrentView("results")}
                    />
                  )}
                  {currentView === "results" && (
                    <ResultsView
                      key="results"
                      onPrevStep={() => setCurrentView("training")}
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
                <aside className="w-96 border-l border-slate-200 bg-white flex flex-col flex-shrink-0">
                  <div className="border-b border-slate-200 p-4 flex items-center justify-between">
                    <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                      <BrainCircuit className="w-5 h-5 text-blue-600" />
                      AI 助手
                    </h3>
                    <button
                      onClick={() => setChatSidebarCollapsed(true)}
                      className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                      title="收起 AI 助手"
                    >
                      <PanelRightClose className="w-5 h-5 text-slate-600" />
                    </button>
                  </div>
                  <ChatInterface
                    messages={chatHistory}
                    inputValue={inputValue}
                    onInputChange={setInputValue}
                    onSend={handleSendMessage}
                    onFileUpload={handleFileUpload}
                  />
                </aside>
              )}

              {/* 右侧工具栏 */}
              <aside className="w-16 bg-white border-l border-slate-200 flex flex-col items-center py-4 flex-shrink-0">
                <div className="flex flex-col items-center gap-2">
                  <UserDropdown />
                  
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
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleNameConfirm()
                  }
                }}
              />
              <p className="text-xs text-slate-400">
                留空将自动命名为 "{generateDefaultDatasetName(datasets)}"
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => {
              setShowNameModal(false)
              setPendingFiles([])
              setDatasetName("")
            }} className="border-slate-300">
              取消
            </Button>
            <Button 
              onClick={handleNameConfirm} 
              disabled={pendingFiles.length === 0}
              className="bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
            >
              创建数据集
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
                    <Label htmlFor={dataset.id} className="flex-1 cursor-pointer">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center flex-shrink-0">
                          <Folder className="w-5 h-5 text-blue-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-slate-900">{dataset.name}</p>
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
      <Dialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
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
      <Card className="border-2 border-dashed border-blue-500 bg-blue-50 p-16 rounded-2xl shadow-lg">
        <div className="flex flex-col items-center gap-6 text-center">
          <div className="w-16 h-16 rounded-2xl bg-blue-600 flex items-center justify-center shadow-lg">
            <Upload className="w-8 h-8 text-white" />
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

function ChatInterface({
  messages,
  inputValue,
  onInputChange,
  onSend,
  onFileUpload,
}: {
  messages: Message[]
  inputValue: string
  onInputChange: (value: string) => void
  onSend: () => void
  onFileUpload: () => void
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
                    : "bg-slate-100 text-slate-900 border border-slate-200"
                }`}
              >
                <p className="text-sm leading-relaxed">{message.content}</p>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="border-t border-slate-200 p-4 bg-white">
        <div className="flex items-end gap-2">
          <button onClick={onFileUpload} className="p-2 hover:bg-slate-100 rounded-lg transition-colors flex-shrink-0">
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
  onNextStep?: () => void
}) {
  const [isDraggingInDetail, setIsDraggingInDetail] = useState(false)
  
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
        className="p-8"
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
            <div className="bg-white border-2 border-dashed border-blue-500 rounded-2xl p-8 shadow-xl">
              <div className="flex flex-col items-center gap-4">
                <div className="w-16 h-16 rounded-2xl bg-blue-600 flex items-center justify-center">
                  <Upload className="w-8 h-8 text-white" />
                </div>
                <p className="text-blue-600 font-medium">释放以添加文件到此数据集</p>
              </div>
            </div>
          </div>
        )}

        {selectedDataset.files.length === 0 ? (
          <Card className="border-2 border-dashed border-slate-200 bg-white p-16 rounded-xl text-center">
            <div className="flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center">
                <Upload className="w-8 h-8 text-slate-400" />
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
      className="p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900 mb-2">我的数据</h2>
          <p className="text-slate-600">管理和查看您的数据集</p>
        </div>
        <Button onClick={onUpload} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
          <Upload className="w-4 h-4" />
          上传数据集
        </Button>
      </div>

      {datasets.length === 0 ? (
        <Card className="border-2 border-dashed border-slate-200 bg-white p-16 rounded-xl text-center">
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center">
              <Database className="w-8 h-8 text-slate-400" />
            </div>
            <div>
              <p className="text-slate-500 mb-2">暂无数据集</p>
              <p className="text-sm text-slate-400">点击上方按钮或在首页拖拽文件创建数据集</p>
            </div>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {datasets.map((dataset, index) => (
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
                {/* 删除按钮 */}
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteDataset(dataset.id)
                  }}
                  className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-red-500 hover:bg-red-50"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
                
                <div className="flex flex-col gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-blue-600 flex items-center justify-center">
                    <Folder className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-slate-900 text-lg mb-1">{dataset.name}</h3>
                    <p className="text-sm text-slate-500">ID: {dataset.id}</p>
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
}: { 
  jobs: ProcessingJob[]
  onNewTask: () => void
  onDownload: (job: ProcessingJob) => void
  onDelete: (jobId: string) => void
  onUploadCleanedData: () => void
  onNextStep?: () => void
  onPrevStep?: () => void
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
      className="p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div 
        className="mb-6 flex items-center justify-between"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h2 className="text-2xl font-semibold text-slate-900 mb-2">数据处理</h2>
          <p className="text-slate-600">选择数据集进行文本清洗和格式转换</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={onUploadCleanedData} variant="outline" className="gap-2">
            <Upload className="w-4 h-4" />
            上传已清洗数据
          </Button>
          <Button onClick={onNewTask} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
            <Plus className="w-4 h-4" />
            新建处理任务
          </Button>
        </div>
      </motion.div>

      {jobs.length === 0 ? (
        <Card className="border border-slate-200 bg-white p-8 rounded-xl text-center">
          <div className="flex flex-col items-center gap-4 py-8">
            <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center">
              <FileCog className="w-8 h-8 text-slate-400" />
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
}: {
  onPrevStep: () => void
  onNextStep: () => void
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
  const [checkingStatus, setCheckingStatus] = useState<Record<string, {has_bow: boolean, has_embeddings: boolean, ready_for_training: boolean}>>({})

  // 加载数据集和模型列表
  useEffect(() => {
    const loadData = async () => {
      try {
        // 加载数据集
        const datasetsRes = await ETMAgentAPI.getDatasets()
        setDatasets(datasetsRes)
        
        // 加载嵌入模型
        const modelsRes = await ETMAgentAPI.getEmbeddingModels()
        setEmbeddingModels(modelsRes.models)
        setSelectedModel(modelsRes.default)
        
        // 加载预处理任务
        const jobsRes = await ETMAgentAPI.getPreprocessingJobs()
        setPreprocessingJobs(jobsRes)
        
        // 检查每个数据集的预处理状态
        const statusChecks: Record<string, any> = {}
        for (const ds of datasetsRes) {
          try {
            const status = await ETMAgentAPI.checkPreprocessingStatus(ds.name)
            statusChecks[ds.name] = status
          } catch {
            statusChecks[ds.name] = { has_bow: false, has_embeddings: false, ready_for_training: false }
          }
        }
        setCheckingStatus(statusChecks)
      } catch (error) {
        console.error('Failed to load data:', error)
      }
    }
    loadData()
  }, [])

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
            const status = await ETMAgentAPI.checkPreprocessingStatus(job.dataset)
            setCheckingStatus(prev => ({ ...prev, [job.dataset]: status }))
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
        text_column: 'text',
        config: {
          embedding_model: selectedModel,
        }
      })
      setPreprocessingJobs(prev => [...prev, result])
    } catch (error) {
      console.error('Failed to start preprocessing:', error)
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

  return (
    <motion.div 
      className="p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div 
        className="mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">向量化处理</h2>
        <p className="text-slate-600">生成 Bag-of-Words (BOW) 矩阵和文档嵌入向量，为模型训练做准备</p>
      </motion.div>

      {/* 新建预处理任务 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card className="border border-slate-200 bg-white p-6 rounded-xl mb-6">
          <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-blue-600" />
            新建向量化任务
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 选择数据集 */}
            <div>
              <Label className="text-sm font-medium text-slate-700 mb-2 block">选择数据集</Label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
              >
                <option value="">请选择数据集...</option>
                {datasets.map(ds => (
                  <option key={ds.name} value={ds.name}>
                    {ds.name} {ds.size ? `(${ds.size} 条)` : ''}
                    {checkingStatus[ds.name]?.ready_for_training ? ' ✓ 已向量化' : ''}
                  </option>
                ))}
              </select>
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
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(job.status)}
                      <div>
                        <p className="font-medium text-slate-900">{job.dataset}</p>
                        <p className="text-sm text-slate-500">{job.message || getStatusText(job.status)}</p>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(job.status)}`}>
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
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-medium text-slate-900">{ds.name}</p>
                    {status?.ready_for_training && (
                      <CheckCircle2 className="w-5 h-5 text-green-600" />
                    )}
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="flex items-center gap-2">
                      {status?.has_bow ? (
                        <CheckCircle2 className="w-4 h-4 text-green-500" />
                      ) : (
                        <XCircle className="w-4 h-4 text-slate-300" />
                      )}
                      <span className={status?.has_bow ? 'text-green-700' : 'text-slate-400'}>BOW 矩阵</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {status?.has_embeddings ? (
                        <CheckCircle2 className="w-4 h-4 text-green-500" />
                      ) : (
                        <XCircle className="w-4 h-4 text-slate-300" />
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
            <Button 
              onClick={onNextStep} 
              disabled={readyDatasets.length === 0}
              className="bg-blue-600 hover:bg-blue-700 text-white gap-2"
            >
              下一步：模型训练
              <ArrowLeft className="w-4 h-4 rotate-180" />
            </Button>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  )
}

// ============================================
// 训练视图组件
// ============================================
function TrainingView({
  onPrevStep,
  onNextStep,
}: {
  onPrevStep: () => void
  onNextStep: () => void
}) {
  const [tasks, setTasks] = useState<TaskResponse[]>([])
  const [selectedTask, setSelectedTask] = useState<TaskResponse | null>(null)
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [messages, setMessages] = useState<Array<{ id: string; role: 'user' | 'assistant' | 'system'; content: string }>>([])

  const { lastMessage, subscribe } = useETMWebSocket()

  useEffect(() => {
    loadTasks()
  }, [])

  // 轮询更新任务状态（作为 WebSocket 的备用方案）
  useEffect(() => {
    const pollInterval = setInterval(async () => {
      // 只轮询正在处理中或等待中的任务
      const activeTasks = tasks.filter(t => t.status === 'pending' || t.status === 'processing')
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
    }, 2000) // 每2秒轮询一次

    return () => clearInterval(pollInterval)
  }, [tasks, selectedTask])

  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'step_update') {
        const systemMsg = {
          id: `sys-${Date.now()}`,
          role: 'system' as const,
          content: `**${lastMessage.step}**: ${lastMessage.message}`,
        }
        setMessages((prev) => [...prev, systemMsg])

        if (lastMessage.task_id) {
          updateTaskStatus(lastMessage.task_id)
        }
      } else if (lastMessage.type === 'task_update') {
        updateTaskStatus(lastMessage.task_id as string)
      }
    }
  }, [lastMessage])

  const loadTasks = async () => {
    try {
      const data = await ETMAgentAPI.getTasks()
      setTasks(data)
    } catch (error) {
      console.error('Failed to load tasks:', error)
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { id: `user-${Date.now()}`, role: 'user' as const, content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await ETMAgentAPI.chat(input)

      if (response.action === 'start_task' && response.data) {
        const taskRequest = response.data as unknown as CreateTaskRequest
        const task = await ETMAgentAPI.createTask(taskRequest)

        const assistantMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant' as const,
          content: `任务已创建！任务 ID: ${task.task_id}\n状态: ${task.status}\n进度: ${task.progress}%`,
        }

        setMessages((prev) => [...prev, assistantMessage])
        setTasks((prev) => [task, ...prev])
        setSelectedTask(task)
        subscribe(task.task_id)
      } else {
        const assistantMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant' as const,
          content: response.message,
        }
        setMessages((prev) => [...prev, assistantMessage])
      }
    } catch (error: unknown) {
      const errorMsg = error instanceof Error ? error.message : '处理请求失败'
      const errorMessage = { id: `error-${Date.now()}`, role: 'assistant' as const, content: `错误: ${errorMsg}` }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
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

  return (
    <motion.div 
      className="p-8 h-full overflow-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div 
        className="flex items-center justify-between mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h2 className="text-2xl font-semibold text-slate-900 mb-2">模型训练</h2>
          <p className="text-slate-600">创建和管理 ETM 主题模型训练任务</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={onPrevStep} className="gap-2">
            <ArrowLeft className="w-4 h-4" />
            上一步
          </Button>
          <Button onClick={onNextStep} className="gap-2">
            下一步
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Button>
        </div>
      </motion.div>

      {/* 快捷操作 */}
      {messages.length === 0 && (
        <Card className="p-6 bg-white border border-slate-200 mb-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center flex-shrink-0">
              <Zap className="w-6 h-6 text-blue-600" />
            </div>
            <div className="flex-1">
              <h3 className="font-medium text-slate-900 mb-2">快速开始</h3>
              <p className="text-sm text-slate-500 mb-4">选择一个示例命令开始训练</p>
              <div className="flex flex-wrap gap-2">
                {[
                  '训练 socialTwitter 数据集',
                  '使用 zero_shot 模式训练，20 个主题',
                  '训练 hatespeech，supervised 模式',
                ].map((example) => (
                  <button
                    key={example}
                    onClick={() => setInput(example)}
                    className="px-4 py-2 text-sm bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700 transition-colors"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* 任务列表 */}
      <div className="mb-6">
        <h3 className="font-medium text-slate-900 mb-4">任务列表</h3>
        {tasks.length === 0 ? (
          <Card className="p-8 bg-white border border-slate-200 text-center">
            <p className="text-slate-500">暂无任务，使用下方输入框创建新任务</p>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {tasks.map((task) => (
              <Card
                key={task.task_id}
                className={`p-4 cursor-pointer transition-all hover:shadow-md ${
                  selectedTask?.task_id === task.task_id ? 'bg-blue-50 border-blue-200' : 'bg-white border-slate-200'
                }`}
                onClick={() => setSelectedTask(task)}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-slate-900 truncate">{task.dataset}</p>
                    <p className="text-xs text-slate-500 mt-1">
                      模式: {task.mode || 'zero_shot'} · 主题数: {task.num_topics || 20}
                    </p>
                  </div>
                  {task.status !== 'completed' && task.status !== 'failed' && (
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
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">{task.progress}%</span>
                  <span className={`px-2 py-0.5 rounded ${
                    task.status === 'completed' ? 'bg-green-100 text-green-700' :
                    task.status === 'failed' ? 'bg-red-100 text-red-700' :
                    task.status === 'processing' ? 'bg-blue-100 text-blue-700' :
                    'bg-slate-100 text-slate-700'
                  }`}>
                    {task.status === 'completed' ? '已完成' :
                     task.status === 'failed' ? '失败' :
                     task.status === 'processing' ? '处理中' :
                     task.status === 'pending' ? '等待中' : task.status}
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-2">
                  {new Date(task.created_at).toLocaleString()}
                </p>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* 输入框 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="p-4 bg-white border border-slate-200">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="输入训练命令，例如：训练 socialTwitter 数据集，20 个主题..."
              className="flex-1"
              disabled={isLoading}
            />
            <Button type="submit" disabled={isLoading || !input.trim()}>
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </Button>
          </form>
        </Card>
      </motion.div>
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
    } finally {
      setLoading(false)
    }
  }

  const loadResultDetails = async (dataset: string, mode: string) => {
    try {
      const [words, metricsData] = await Promise.all([
        ETMAgentAPI.getTopicWords(dataset, mode).catch(() => null),
        ETMAgentAPI.getMetrics(dataset, mode).catch(() => null),
      ])
      setTopicWords(words)
      setMetrics(metricsData)
    } catch (error) {
      console.error('Failed to load result details:', error)
      setTopicWords(null)
      setMetrics(null)
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
      className="p-8 h-full overflow-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div 
        className="flex items-center justify-between mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h2 className="text-2xl font-semibold text-slate-900 mb-2">分析结果</h2>
          <p className="text-slate-600">查看训练完成的主题模型分析结果</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={onPrevStep} className="gap-2">
            <ArrowLeft className="w-4 h-4" />
            上一步
          </Button>
          <Button onClick={onNextStep} className="gap-2">
            下一步
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </Button>
        </div>
      </motion.div>

      <motion.div 
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {/* 结果列表 */}
        <div className="lg:col-span-1 space-y-4">
          <Card className="p-4 bg-white border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-4">结果列表</h3>
            {results.length === 0 ? (
              <div className="text-center text-slate-500 text-sm py-8">
                暂无结果，请先完成模型训练
              </div>
            ) : (
              <div className="space-y-2">
                {results.map((result) => (
                  <Card
                    key={`${result.dataset}-${result.mode}`}
                    className={`p-3 cursor-pointer transition-colors ${
                      selectedResult?.dataset === result.dataset &&
                      selectedResult?.mode === result.mode
                        ? 'bg-blue-50 border-blue-200'
                        : 'hover:bg-slate-50 border-slate-200'
                    }`}
                    onClick={() => handleSelectResult(result)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="font-medium text-slate-900 text-sm">{result.dataset}</p>
                        <p className="text-xs text-slate-500 mt-1">
                          {result.mode} · {result.num_topics} 主题
                        </p>
                        <p className="text-xs text-slate-400 mt-1">
                          {new Date(result.timestamp).toLocaleString()}
                        </p>
                      </div>
                      <BarChart3 className="w-5 h-5 text-blue-600" />
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* 结果详情 */}
        <div className="lg:col-span-2 space-y-6">
          {selectedResult ? (
            <>
              {/* 评估指标 */}
              {metrics && (
                <Card className="p-6 bg-white border border-slate-200">
                  <h3 className="font-medium text-slate-900 mb-4">评估指标</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {Object.entries(metrics).map(([key, value]) => (
                      <div key={key} className="text-center p-4 bg-slate-50 rounded-lg">
                        <p className="text-2xl font-bold text-blue-600">
                          {typeof value === 'number' ? value.toFixed(3) : String(value)}
                        </p>
                        <p className="text-sm text-slate-500 capitalize">{key}</p>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* 主题词 */}
              {topicWords && (
                <Card className="p-6 bg-white border border-slate-200">
                  <h3 className="font-medium text-slate-900 mb-4">主题词分析</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(topicWords).map(([topic, words]) => (
                      <div key={topic} className="p-4 bg-slate-50 rounded-lg">
                        <p className="font-medium text-slate-900 mb-2 capitalize">
                          {topic.replace('_', ' ')}
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {words.map((word, index) => (
                            <span key={index} className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">
                              {word}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* 操作按钮 */}
              <div className="flex gap-4">
                <Button variant="outline" className="gap-2">
                  <Download className="w-4 h-4" />
                  导出结果
                </Button>
                <Button variant="outline" className="gap-2">
                  <ExternalLink className="w-4 h-4" />
                  查看详细报告
                </Button>
              </div>
            </>
          ) : (
            <Card className="p-8 bg-white border border-slate-200 text-center">
              <BarChart3 className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-500">选择一个结果查看详情</p>
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
      className="p-8 h-full overflow-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div 
        className="flex items-center justify-between mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div>
          <h2 className="text-2xl font-semibold text-slate-900 mb-2">可视化图表</h2>
          <p className="text-slate-600">查看训练结果的可视化展示</p>
        </div>
        <Button variant="outline" onClick={onPrevStep} className="gap-2">
          <ArrowLeft className="w-4 h-4" />
          上一步
        </Button>
      </motion.div>

      <motion.div 
        className="grid grid-cols-1 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {/* 结果选择器 */}
        <div className="lg:col-span-1">
          <Card className="p-4 bg-white border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-4">选择数据集</h3>
            {results.length === 0 ? (
              <div className="text-center text-slate-500 text-sm py-8">
                暂无结果，请先完成模型训练
              </div>
            ) : (
              <div className="space-y-2">
                {results.map((result) => (
                  <Card
                    key={`${result.dataset}-${result.mode}`}
                    className={`p-3 cursor-pointer transition-colors ${
                      selectedResult?.dataset === result.dataset &&
                      selectedResult?.mode === result.mode
                        ? 'bg-blue-50 border-blue-200'
                        : 'hover:bg-slate-50 border-slate-200'
                    }`}
                    onClick={() => handleSelectResult(result)}
                  >
                    <p className="font-medium text-slate-900 text-sm">{result.dataset}</p>
                    <p className="text-xs text-slate-500 mt-1">{result.mode}</p>
                  </Card>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* 可视化图表 */}
        <div className="lg:col-span-3">
          {selectedResult ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {visualizations.length === 0 ? (
                <Card className="col-span-full p-8 bg-white border border-slate-200 text-center">
                  <Image className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                  <p className="text-slate-500">暂无可视化图表</p>
                </Card>
              ) : (
                visualizations.map((viz) => (
                  <Card
                    key={viz.name}
                    className="p-6 bg-white border border-slate-200 hover:shadow-lg transition-all cursor-pointer group"
                  >
                    <div className="aspect-video bg-slate-100 rounded-lg mb-4 flex items-center justify-center">
                      {getVisualizationIcon(viz.type)}
                    </div>
                    <h4 className="font-medium text-slate-900 mb-2">
                      {getVisualizationName(viz.name)}
                    </h4>
                    <p className="text-xs text-slate-500 mb-4">类型: {viz.type}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full gap-2 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <Download className="w-4 h-4" />
                      下载图表
                    </Button>
                  </Card>
                ))
              )}
            </div>
          ) : (
            <Card className="p-8 bg-white border border-slate-200 text-center">
              <PieChart className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-500">选择一个数据集查看可视化图表</p>
            </Card>
          )}
        </div>
      </motion.div>
    </motion.div>
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

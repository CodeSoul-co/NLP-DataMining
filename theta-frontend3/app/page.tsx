"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Upload,
  Menu,
  User,
  LayoutGrid,
  Database,
  FileCog,
  BrainCircuit,
  PieChart,
  FileText,
  List,
  Folder,
  Plus,
  Paperclip,
  Send,
  BarChart3,
  TrendingUp,
  LogOut,
  Settings,
  PanelLeftClose,
  PanelLeft,
  ArrowLeft,
  Trash2,
  X,
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
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { DataProcessingView } from "@/components/data-processing"

type ViewType = "projects" | "data" | "processing" | "analysis" | "visualization" | "report" | "tasks"

type AppState = "idle" | "chatting" | "workspace"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
}

type ProcessingJob = {
  id: string
  name: string
  sourceDataset: string
  status: "processing" | "completed"
  date: string
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

const mockChartData = [
  { name: "1月", value: 4000, sales: 2400 },
  { name: "2月", value: 3000, sales: 1398 },
  { name: "3月", value: 2000, sales: 9800 },
  { name: "4月", value: 2780, sales: 3908 },
  { name: "5月", value: 1890, sales: 4800 },
  { name: "6月", value: 2390, sales: 3800 },
]

export default function Home() {
  const [appState, setAppState] = useState<AppState>("idle")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [currentView, setCurrentView] = useState<ViewType>("data")
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
  
  // 新增状态：待上传的文件、当前查看的数据集
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null)

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
    
    const newDataset: Dataset = {
      id: `job-${Date.now()}`,
      name: finalName,
      files: newFiles,
      totalSize: calculateTotalSize(newFiles),
      date: new Date().toISOString().split("T")[0],
    }
    
    setDatasets([...datasets, newDataset])
    setShowNameModal(false)
    setPendingFiles([])
    setDatasetName("")
    setAppState("workspace")
    setCurrentView("data")
    // 直接进入新创建的数据集
    setSelectedDatasetId(newDataset.id)
  }
  
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

  const handleSourceConfirm = () => {
    if (selectedSource) {
      const sourceDataset = datasets.find((d) => d.id === selectedSource)
      if (sourceDataset) {
        const newJob: ProcessingJob = {
          id: `processed-${Date.now()}`,
          name: `${sourceDataset.name}_cleaned`,
          sourceDataset: sourceDataset.name,
          status: "processing",
          date: new Date().toISOString().split("T")[0],
        }
        setProcessingJobs([...processingJobs, newJob])

        // Simulate processing completion after 2 seconds
        setTimeout(() => {
          setProcessingJobs((prev) => prev.map((job) => (job.id === newJob.id ? { ...job, status: "completed" } : job)))
        }, 2000)
      }

      setShowSourceModal(false)
      setCurrentView("processing")
      setSelectedSource("")
    }
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

  const handleNewProcessingTask = () => {
    setShowSourceModal(true)
  }

  const isCenterChatView = currentView === "analysis" || currentView === "visualization"

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
                      <NavItem icon={LayoutGrid} label="我的项目" onClick={() => handleNavClick("projects")} />
                      <NavItem icon={Database} label="我的数据" onClick={() => handleNavClick("data")} />
                      <NavItem icon={FileCog} label="数据处理" onClick={() => handleNavClick("processing")} />
                      <NavItem icon={BrainCircuit} label="数据分析" onClick={() => handleNavClick("analysis")} />
                      <NavItem icon={PieChart} label="可视化洞察" onClick={() => handleNavClick("visualization")} />
                      <NavItem icon={FileText} label="智能报告" onClick={() => handleNavClick("report")} />
                      <NavItem icon={List} label="任务中心" onClick={() => handleNavClick("tasks")} />
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
          <motion.div
            key="workspace"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className="flex h-screen overflow-hidden"
          >
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
                <NavItem
                  icon={LayoutGrid}
                  label="我的项目"
                  active={currentView === "projects"}
                  onClick={() => setCurrentView("projects")}
                  collapsed={sidebarCollapsed}
                />
                <NavItem
                  icon={Database}
                  label="我的数据"
                  active={currentView === "data"}
                  onClick={() => setCurrentView("data")}
                  collapsed={sidebarCollapsed}
                />
                <NavItem
                  icon={FileCog}
                  label="数据处理"
                  active={currentView === "processing"}
                  onClick={() => setCurrentView("processing")}
                  collapsed={sidebarCollapsed}
                />
                <NavItem
                  icon={BrainCircuit}
                  label="数据分析"
                  active={currentView === "analysis"}
                  onClick={() => setCurrentView("analysis")}
                  collapsed={sidebarCollapsed}
                />
                <NavItem
                  icon={PieChart}
                  label="可视化洞察"
                  active={currentView === "visualization"}
                  onClick={() => setCurrentView("visualization")}
                  collapsed={sidebarCollapsed}
                />
                <NavItem
                  icon={FileText}
                  label="智能报告"
                  active={currentView === "report"}
                  onClick={() => setCurrentView("report")}
                  collapsed={sidebarCollapsed}
                />
                <NavItem
                  icon={List}
                  label="任务中心"
                  active={currentView === "tasks"}
                  onClick={() => setCurrentView("tasks")}
                  collapsed={sidebarCollapsed}
                />
              </nav>
            </motion.aside>

            <div className="flex flex-1 overflow-hidden">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="flex-1 bg-slate-50 overflow-auto"
              >
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
                    />
                  )}
                  {currentView === "processing" && (
                    <ProcessingView key="processing" jobs={processingJobs} onNewTask={handleNewProcessingTask} />
                  )}
                  {currentView === "projects" && <PlaceholderView key="projects" title="我的项目" />}
                  {currentView === "report" && <PlaceholderView key="report" title="智能报告" />}
                  {currentView === "tasks" && <PlaceholderView key="tasks" title="任务中心" />}
                  {currentView === "analysis" && <AnalysisCanvas key="analysis" />}
                  {currentView === "visualization" && <VisualizationCanvas key="visualization" />}
                </AnimatePresence>
              </motion.div>

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
                  onFileUpload={handleFileUpload}
                />
              </motion.aside>

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
          </motion.div>
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
          <DialogHeader>
            <DialogTitle className="text-slate-900">创建数据集</DialogTitle>
            <DialogDescription className="text-slate-500">
              已选择 {pendingFiles.length} 个文件，请为数据集命名
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
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
                    <div key={index} className="flex items-center justify-between text-sm py-1.5 px-2 bg-slate-50 rounded group hover:bg-slate-100">
                      <span className="text-slate-700 truncate flex-1">{file.name}</span>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className="text-slate-400">
                          {file.size >= 1024 * 1024 
                            ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
                            : `${(file.size / 1024).toFixed(1)} KB`}
                        </span>
                        <button
                          onClick={() => setPendingFiles(prev => prev.filter((_, i) => i !== index))}
                          className="text-slate-300 hover:text-red-500 transition-colors"
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
}: { 
  datasets: Dataset[]
  onUpload: () => void
  selectedDatasetId: string | null
  onSelectDataset: (id: string | null) => void
  onAddFiles: (datasetId: string, files: File[]) => void
  onRemoveFile: (datasetId: string, fileId: string) => void
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
          <Button onClick={handleAddFilesClick} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
            <Plus className="w-4 h-4" />
            添加文件
          </Button>
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
                <Button className="bg-blue-600 hover:bg-blue-700 text-white">
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
                className="border border-slate-200 bg-white hover:shadow-lg hover:border-blue-200 transition-all cursor-pointer p-6 rounded-xl"
                onClick={() => onSelectDataset(dataset.id)}
              >
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

function ProcessingView({ jobs, onNewTask }: { jobs: ProcessingJob[]; onNewTask: () => void }) {
  return (
    <div className="p-8">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900 mb-2">数据处理</h2>
          <p className="text-slate-600">创建和管理数据处理任务</p>
        </div>
        <Button onClick={onNewTask} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
          <Plus className="w-4 h-4" />
          新建处理任务
        </Button>
      </div>

      {jobs.length === 0 ? (
        <Card className="border border-slate-200 bg-white p-8 rounded-xl text-center">
          <div className="flex flex-col items-center gap-4 py-8">
            <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center">
              <FileCog className="w-8 h-8 text-slate-400" />
            </div>
            <div>
              <p className="text-slate-500 mb-2">暂无处理记录</p>
              <p className="text-sm text-slate-400">点击上方按钮创建新的数据处理任务</p>
            </div>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {jobs.map((job, index) => (
            <motion.div
              key={job.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className="border border-slate-200 bg-white hover:shadow-lg transition-all cursor-pointer p-6 rounded-xl">
                <div className="flex flex-col gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-green-600 flex items-center justify-center">
                    <FileCog className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-slate-900 text-lg mb-1">{job.name}</h3>
                    <p className="text-sm text-slate-500">源数据: {job.sourceDataset}</p>
                  </div>
                  <div className="pt-3 border-t border-slate-100 space-y-1">
                    <div className="flex items-center justify-between">
                      <p className="text-xs text-slate-600">状态:</p>
                      <span
                        className={`text-xs px-2 py-1 rounded-full ${
                          job.status === "completed" ? "bg-green-100 text-green-700" : "bg-yellow-100 text-yellow-700"
                        }`}
                      >
                        {job.status === "completed" ? "已完成" : "处理中..."}
                      </span>
                    </div>
                    <p className="text-xs text-slate-600">创建日期: {job.date}</p>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  )
}

function PlaceholderView({ title }: { title: string }) {
  return (
    <div className="p-8">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">{title}</h2>
        <p className="text-slate-600">功能开发中</p>
      </div>

      <Card className="border border-slate-200 bg-white p-16 rounded-xl text-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-20 h-20 rounded-2xl bg-slate-100 flex items-center justify-center">
            <BrainCircuit className="w-10 h-10 text-slate-400" />
          </div>
          <p className="text-slate-500">此功能即将推出</p>
        </div>
      </Card>
    </div>
  )
}

function AnalysisCanvas() {
  return (
    <div className="p-8 space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">数据分析</h2>
        <p className="text-slate-600">AI 驱动的智能分析结果</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <Card className="border border-slate-200 bg-white p-6 rounded-xl">
          <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-600" />
            销售趋势分析
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={mockChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: "white", border: "1px solid #e2e8f0", borderRadius: "8px" }} />
              <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={2} dot={{ fill: "#2563eb" }} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card className="border border-slate-200 bg-white p-6 rounded-xl">
          <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            关键指标对比
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={mockChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: "white", border: "1px solid #e2e8f0", borderRadius: "8px" }} />
              <Bar dataKey="sales" fill="#2563eb" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card className="border border-slate-200 bg-white p-6 rounded-xl col-span-2">
          <h3 className="font-semibold text-slate-900 mb-4">AI 洞察摘要</h3>
          <div className="space-y-3">
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-slate-700">
                <span className="font-medium text-blue-600">趋势发现：</span>
                销售额在 3 月份达到峰值，建议在此期间加大营销投入
              </p>
            </div>
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-slate-700">
                <span className="font-medium text-blue-600">异常检测：</span>5 月份出现明显下降，可能受季节因素影响
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}

function VisualizationCanvas() {
  return (
    <div className="p-8 space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">可视化洞察</h2>
        <p className="text-slate-600">交互式数据可视化</p>
      </div>

      <div className="grid grid-cols-1 gap-6">
        <Card className="border border-slate-200 bg-white p-6 rounded-xl">
          <h3 className="font-semibold text-slate-900 mb-4">综合数据面板</h3>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={mockChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip contentStyle={{ backgroundColor: "white", border: "1px solid #e2e8f0", borderRadius: "8px" }} />
              <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={3} />
              <Line type="monotone" dataKey="sales" stroke="#10b981" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <div className="grid grid-cols-3 gap-6">
          {[
            { label: "总销售额", value: "¥2.4M", change: "+12.5%" },
            { label: "客户数量", value: "8,234", change: "+8.2%" },
            { label: "转化率", value: "3.6%", change: "+2.1%" },
          ].map((metric) => (
            <Card key={metric.label} className="border border-slate-200 bg-white p-6 rounded-xl">
              <p className="text-sm text-slate-600 mb-2">{metric.label}</p>
              <p className="text-2xl font-bold text-slate-900 mb-1">{metric.value}</p>
              <p className="text-sm text-green-600 font-medium">{metric.change}</p>
            </Card>
          ))}
        </div>
      </div>
    </div>
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

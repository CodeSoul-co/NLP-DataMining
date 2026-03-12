"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import {
  Upload,
  Database,
  Sparkles,
  Play,
  BarChart3,
  PieChart,
  Check,
  Loader2,
  AlertCircle,
  Clock,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  FileText,
  X,
  File,
} from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"
import { ETMAgentAPI } from "@/lib/api/etm-agent"
import { AnalysisConfigPanel, type AnalysisConfig } from "./analysis-config-panel"
import { ColumnSelectPanel, type ColumnSelection } from "./column-select-panel"

// ==================== 类型定义 ====================

/** 根据项目名生成后端使用的数据集名称（上传目录名），与后端 sanitize 规则尽量一致 */
function getDatasetName(projectName: string): string {
  return projectName
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^\w\u4e00-\u9fa5-]/g, "")
    .toLowerCase() || "dataset"
}

/** 后端预处理 job 的 status 视为“进行中”的值 */
const PREPROCESSING_RUNNING_STATUSES = [
  "pending",
  "bow_generating",
  "bow_completed",
  "embedding_generating",
  "embedding_completed",
]

interface AutoPipelineProps {
  projectName: string
  /** 默认模式（用户未配置时使用，上传后会弹配置面板让用户选择） */
  mode?: "zero_shot" | "unsupervised" | "supervised"
  /** 默认主题数 */
  numTopics?: number
  /** 已有的 task_id，用于项目重新进入时恢复进度 */
  initialTaskId?: string | null
  onComplete?: (result: PipelineResult) => void
  onError?: (error: string) => void
  /** 上传完成时回调（dataset_name 来自后端），用于同步到数据库 */
  onUploadComplete?: (datasetName: string) => void
  /** 训练任务创建后回调，用于将 task_id 保存到项目数据库 */
  onTaskCreated?: (taskId: string) => void
}

interface PipelineStep {
  id: string
  name: string
  icon: React.ElementType
  status: "waiting" | "running" | "completed" | "error" | "skipped"
  progress: number
  message: string
  startTime?: Date
  endTime?: Date
}

interface PipelineResult {
  success: boolean
  taskId?: string
  dataset?: string
  metrics?: Record<string, number>
  topicWords?: Record<string, string[]>
  duration: number
}

// ==================== 初始步骤（含上传） ====================

const createInitialSteps = (): PipelineStep[] => [
  { id: "upload", name: "上传数据", icon: Upload, status: "waiting", progress: 0, message: "请上传数据文件..." },
  { id: "preprocess", name: "数据预处理", icon: Database, status: "waiting", progress: 0, message: "等待开始..." },
  { id: "embedding", name: "参数选择", icon: Sparkles, status: "waiting", progress: 0, message: "等待开始..." },
  { id: "training", name: "模型训练", icon: Play, status: "waiting", progress: 0, message: "等待开始..." },
  { id: "evaluation", name: "模型评估", icon: BarChart3, status: "waiting", progress: 0, message: "等待开始..." },
  { id: "visualization", name: "生成可视化", icon: PieChart, status: "waiting", progress: 0, message: "等待开始..." },
]

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000)
  if (seconds < 60) return `${seconds} 秒`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes} 分 ${remainingSeconds} 秒`
}

// ==================== 组件 ====================

export function AutoPipeline({
  projectName,
  mode: defaultMode = "zero_shot",
  numTopics: defaultNumTopics = 20,
  initialTaskId,
  onComplete,
  onError,
  onUploadComplete,
  onTaskCreated,
}: AutoPipelineProps) {
  /** 初始用项目名生成；上传成功后改用后端返回的 dataset_name，避免前后端 sanitize 不一致 */
  const [effectiveDatasetName, setEffectiveDatasetName] = useState<string>(() => getDatasetName(projectName))

  const [steps, setSteps] = useState<PipelineStep[]>(createInitialSteps())
  const [overallProgress, setOverallProgress] = useState(0)
  const [status, setStatus] = useState<"upload" | "column_select" | "config" | "running" | "completed" | "error">("upload")
  const [taskId, setTaskId] = useState<string | null>(null)
  const [showLogs, setShowLogs] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [startTime, setStartTime] = useState<Date | null>(null)
  const [endTime, setEndTime] = useState<Date | null>(null)
  const [result, setResult] = useState<PipelineResult | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [uploadJobId, setUploadJobId] = useState<string | null>(null)
  /** 上传完成后等待配置：弹出配置面板，用户确认后才开始分析 */
  const [showConfigPanel, setShowConfigPanel] = useState(false)
  const [pendingDatasetForConfig, setPendingDatasetForConfig] = useState<string | null>(null)
  const [pendingJobIdForConfig, setPendingJobIdForConfig] = useState<string | null>(null)
  /** CSV 上传后先选列：弹出列选择面板 */
  const [showColumnSelectPanel, setShowColumnSelectPanel] = useState(false)
  const [columnSelection, setColumnSelection] = useState<ColumnSelection | null>(null)

  const pollingRef = useRef<NodeJS.Timeout | null>(null)
  const hasUploaded = useRef(false)
  const pipelineStarted = useRef(false)
  /** 当前流程使用的数据集名（上传后由后端返回），用于结果展示 */
  const pipelineDatasetRef = useRef<string>("")

  /** 恢复已有任务：项目重新进入时，根据 initialTaskId 查询任务状态并恢复进度 */
  useEffect(() => {
    if (!initialTaskId) return
    let cancelled = false

    const restoreTask = async () => {
      try {
        const task = await ETMAgentAPI.getTask(initialTaskId)
        if (cancelled) return

        setTaskId(initialTaskId)
        pipelineStarted.current = true
        hasUploaded.current = true

        if (task.dataset) {
          setEffectiveDatasetName(task.dataset)
          pipelineDatasetRef.current = task.dataset
        }

        // 标记上传已完成
        updateStep("upload", { status: "completed", progress: 100, message: "上传完成" })

        if (task.status === "completed") {
          setStatus("completed")
          setOverallProgress(100)
          setSteps(prev => prev.map(s => ({ ...s, status: "completed" as const, progress: 100, message: "已完成" })))
          const pipelineResult: PipelineResult = {
            success: true,
            taskId: initialTaskId,
            dataset: task.dataset,
            metrics: task.metrics,
            topicWords: task.topic_words,
            duration: 0,
          }
          setResult(pipelineResult)
          addLog("✅ 训练已完成")
          return
        }

        if (task.status === "failed" || task.status === "error") {
          setStatus("error")
          addLog(`❌ ${task.error_message || "训练失败"}`)
          updateStep("training", { status: "error", message: task.error_message || "失败" })
          return
        }

        // 运行中 → 恢复轮询
        setStatus("running")
        setStartTime(task.created_at ? new Date(task.created_at) : new Date())
        setOverallProgress(task.progress || 0)
        const isDlc = task.current_step === "dlc_training"
        if (isDlc) {
          updateStep("training", {
            status: "running",
            progress: task.progress || 0,
            message: task.message || "云端训练中...",
          })
        }
        addLog(`恢复任务进度: ${task.message || task.status}`)

        pollingRef.current = setInterval(() => pollTaskStatus(initialTaskId), 2000)
      } catch (err) {
        console.error("Failed to restore task:", err)
        // 任务不存在，回退到上传界面
      }
    }

    restoreTask()

    return () => {
      cancelled = true
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialTaskId])

  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString("zh-CN")
    setLogs(prev => [...prev, `[${timestamp}] ${message}`])
  }, [])

  const updateStep = useCallback((stepId: string, updates: Partial<PipelineStep>) => {
    setSteps(prev => prev.map(step => (step.id === stepId ? { ...step, ...updates } : step)))
  }, [])

  // ---------- 上传相关 ----------
  const handleFileSelect = (files: FileList | null) => {
    if (!files) return
    setSelectedFiles(prev => [...prev, ...Array.from(files)])
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    handleFileSelect(e.dataTransfer.files)
  }

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const startPipelineAfterUpload = useCallback(
    async (
      datasetForPipeline: string,
      jobId?: string,
      config?: AnalysisConfig,
      colSel?: ColumnSelection | null
    ) => {
      if (pipelineStarted.current) return
      pipelineStarted.current = true
      pipelineDatasetRef.current = datasetForPipeline

      const mode = config?.mode ?? defaultMode
      const numTopics = config?.numTopics ?? defaultNumTopics

      setStatus("running")
      setStartTime(new Date())
      updateStep("upload", { status: "completed", progress: 100, message: "上传完成" })
      addLog(`数据集: ${datasetForPipeline}, 模式: ${mode}, 主题数: ${numTopics}`)

    updateStep("preprocess", { status: "running", progress: 10, message: "检查数据...", startTime: new Date() })

    try {
      // theta_1 流程：若有 job_id，先将上传文件落到 dataset 目录，否则预处理找不到 CSV
      if (jobId) {
        addLog("准备数据集...")
        await ETMAgentAPI.prepareDataset(jobId, datasetForPipeline)
      }
      addLog("检查预处理状态...")
      const preprocessStatus = await ETMAgentAPI.checkPreprocessingStatus(datasetForPipeline)

      if (!preprocessStatus.ready_for_training) {
        updateStep("preprocess", { status: "completed", progress: 100, message: "数据检查完成" })
        updateStep("embedding", { status: "running", progress: 0, message: "生成词袋与嵌入...", startTime: new Date() })

        let preprocessJob
        try {
          const sel = colSel ?? columnSelection
          const prepConfig: Record<string, unknown> = config ? { language: config.language } : {}
          if (sel?.textColumn) prepConfig.text_column = sel.textColumn
          preprocessJob = await ETMAgentAPI.startPreprocessing({
            dataset: datasetForPipeline,
            text_column: sel?.textColumn,
            config: Object.keys(prepConfig).length ? prepConfig : undefined,
          })
        } catch (preprocessError) {
          const msg = preprocessError instanceof Error ? preprocessError.message : String(preprocessError)
          if (msg.includes("No CSV files found")) {
            throw new Error(
              "当前数据集需要包含至少一个 CSV 文件才能进行分析。请上传包含文本列的 CSV 文件（如 text、content、cleaned_content 列），或先使用数据清洗将其他格式转为 CSV。"
            )
          }
          throw preprocessError
        }
        addLog(`预处理任务: ${preprocessJob.job_id}`)

        while (true) {
          await new Promise(r => setTimeout(r, 2000))
          const jobStatus = await ETMAgentAPI.getPreprocessingJob(preprocessJob.job_id)
          updateStep("embedding", { progress: jobStatus.progress, message: jobStatus.message || "处理中..." })
          setOverallProgress(Math.round(10 + jobStatus.progress * 0.25))
          addLog(`参数选择: ${jobStatus.progress}% - ${jobStatus.message}`)

          if (jobStatus.status === "completed") {
            updateStep("embedding", { status: "completed", progress: 100, message: "参数选择完成", endTime: new Date() })
            addLog("✅ 参数选择完成")
            break
          }
          if (jobStatus.status === "failed") {
            throw new Error(jobStatus.error_message || jobStatus.message || "参数选择失败")
          }
        }
      } else {
        addLog("数据已预处理，跳过参数选择")
        updateStep("preprocess", { status: "completed", progress: 100, message: "数据就绪" })
        updateStep("embedding", { status: "completed", progress: 100, message: "已有向量数据" })
        setOverallProgress(35)
      }

      addLog("创建训练任务...")
      updateStep("training", { status: "running", progress: 0, message: "初始化模型...", startTime: new Date() })

      const task = await ETMAgentAPI.createTask({
        dataset: datasetForPipeline,
        mode,
        num_topics: numTopics,
        vocab_size: config?.vocabSize,
        epochs: config?.epochs,
        batch_size: config?.batchSize,
        model_size: config?.modelSize,
        models: config?.models && config.models.length > 0 ? config.models.join(",") : undefined,
        job_id: jobId || uploadJobId || undefined,
      })
      setTaskId(task.task_id)
      addLog(`训练任务: ${task.task_id}`)
      onTaskCreated?.(task.task_id)

      pollingRef.current = setInterval(() => pollTaskStatus(task.task_id), 2000)
    } catch (error) {
      setStatus("error")
      setEndTime(new Date())
      const errorMessage = error instanceof Error ? error.message : "未知错误"
      addLog(`❌ ${errorMessage}`)
      setSteps(prev =>
        prev.map(step => (step.status === "running" ? { ...step, status: "error" as const, message: errorMessage } : step))
      )
      onError?.(errorMessage)
    }
  },
  [defaultMode, defaultNumTopics, columnSelection, addLog, updateStep, onError]
)

  /** 从 message 中解析进度，如 "Epoch 5/10" -> 50 */
  const parseProgressFromMessage = (message: string): number | null => {
    const epochMatch = message.match(/Epoch\s+(\d+)\s*\/\s*(\d+)/i)
    if (epochMatch) {
      const x = parseInt(epochMatch[1], 10)
      const y = parseInt(epochMatch[2], 10)
      if (y > 0) return Math.min(100, Math.round((x / y) * 100))
    }
    const pctMatch = message.match(/\(?\s*(\d+)\s*%\)?/)
    if (pctMatch) return Math.min(100, parseInt(pctMatch[1], 10))
    return null
  }

  const pollTaskStatus = useCallback(
    async (tid: string) => {
      try {
        const [task, logsData] = await Promise.all([
          ETMAgentAPI.getTask(tid),
          ETMAgentAPI.getTaskLogs(tid, 100),
        ])
        const logs = logsData?.logs ?? []

        const stepMap: Record<string, string> = {
          preprocess: "preprocess",
          preprocessing: "preprocess",
          embedding: "embedding",
          vectorizing: "embedding",
          training: "training",
          evaluation: "evaluation",
          evaluating: "evaluation",
          visualization: "visualization",
          visualizing: "visualization",
        }

        /** 每步骤只看自己的日志，独立计算进度；仅本步骤 completed 才打勾 */
        const stepStatusFromLogs: Record<string, { status: "running" | "completed"; progress: number; message: string }> = {}
        const stepOrder = ["preprocess", "embedding", "training", "evaluation", "visualization"]
        for (const stepId of stepOrder) {
          const stepLogs = logs.filter(
            (l: { step?: string }) => stepMap[l.step || ""] === stepId || l.step === stepId
          )
          const lastLog = stepLogs[stepLogs.length - 1]
          if (!lastLog) continue

          const status = lastLog.status
          const message = lastLog.message || ""
          if (status === "completed" || status === "success") {
            stepStatusFromLogs[stepId] = { status: "completed", progress: 100, message: "已完成" }
          } else {
            const progress = parseProgressFromMessage(message) ?? (stepId === "training" ? Math.min(task.progress, 100) : 0)
            stepStatusFromLogs[stepId] = {
              status: "running",
              progress: Math.min(progress, 100),
              message: message || "处理中...",
            }
          }
        }

        setSteps(prev => {
          const newSteps = [...prev]
          let anyRunning = false
          let completedCount = 0
          newSteps.forEach((step, i) => {
            if (step.id === "upload") return
            const logStatus = step.id !== "upload" ? stepStatusFromLogs[step.id] : null
            const taskStepId = task.current_step ? stepMap[task.current_step.toLowerCase()] : null
            const isCurrentByTask = step.id === taskStepId

            if (logStatus?.status === "completed") {
              newSteps[i] = { ...step, status: "completed" as const, progress: 100, message: "已完成" }
              completedCount++
            } else if (logStatus || isCurrentByTask) {
              newSteps[i] = {
                ...step,
                status: "running",
                progress: logStatus?.progress ?? (isCurrentByTask ? Math.min(task.progress, 100) : 0),
                message: logStatus?.message ?? (isCurrentByTask ? (task.message || "处理中...") : step.message),
              }
              anyRunning = true
            }
          })
          return newSteps
        })

        /** 总进度：DLC 模式直接用 task.progress，本地模式按日志步骤计算 */
        const isDlcStep = task.current_step === 'dlc_training'
        if (isDlcStep) {
          setOverallProgress(Math.min(Math.round(task.progress || 0), 100))
          // 更新 training step 的状态来反映 DLC 进度
          setSteps(prev => {
            const newSteps = [...prev]
            const trainingIdx = newSteps.findIndex(s => s.id === 'training')
            if (trainingIdx >= 0) {
              newSteps[trainingIdx] = {
                ...newSteps[trainingIdx],
                status: task.status === 'completed' ? 'completed' as const : 'running' as const,
                progress: Math.min(task.progress || 0, 100),
                message: task.message || '云端训练中...',
              }
            }
            return newSteps
          })
        } else {
          const stepIds = ["preprocess", "embedding", "training", "evaluation", "visualization"]
          let baseProgress = 0
          let currentStepProgress = 0
          for (let i = 0; i < stepIds.length; i++) {
            const stat = stepStatusFromLogs[stepIds[i]]
            if (stat?.status === "completed") {
              baseProgress = ((i + 1) / stepIds.length) * 100
            } else if (stat) {
              currentStepProgress = (stat.progress / 100) * (100 / stepIds.length)
              break
            }
          }
          setOverallProgress(Math.min(Math.round(baseProgress + currentStepProgress), 100))
        }
        const stepLabel = isDlcStep
          ? (task.dlc_status
              ? `云端训练 [${task.dlc_status}]`
              : '云端训练')
          : (task.current_step || '处理中')
        addLog(`${stepLabel}: ${task.message || ''} (${task.progress}%)`)

        if (task.status === "completed") {
          setStatus("completed")
          setEndTime(new Date())
          setSteps(prev => prev.map(step => ({ ...step, status: "completed" as const, progress: 100, message: "已完成" })))
          setOverallProgress(100)
          const pipelineResult: PipelineResult = {
            success: true,
            taskId: tid,
            dataset: pipelineDatasetRef.current || undefined,
            metrics: task.metrics,
            topicWords: task.topic_words,
            duration: startTime ? Date.now() - startTime.getTime() : 0,
          }
          setResult(pipelineResult)
          addLog("✅ 分析流程完成！")
          onComplete?.(pipelineResult)
          if (pollingRef.current) {
            clearInterval(pollingRef.current)
            pollingRef.current = null
          }
          return
        }
        if (task.status === "failed") {
          setStatus("error")
          setEndTime(new Date())
          setSteps(prev =>
            prev.map(step =>
              step.status === "running" ? { ...step, status: "error" as const, message: task.error_message || "失败" } : step
            )
          )
          addLog(`❌ ${task.error_message || "未知错误"}`)
          onError?.(task.error_message || "分析流程失败")
          if (pollingRef.current) {
            clearInterval(pollingRef.current)
            pollingRef.current = null
          }
        }
      } catch (error) {
        console.error("Poll task error:", error)
      }
    },
    [addLog, onComplete, onError, startTime]
  )

  const handleUploadSubmit = async () => {
    if (selectedFiles.length === 0) return

    const nameForUpload = effectiveDatasetName
    updateStep("upload", { status: "running", progress: 0, message: "正在上传...", startTime: new Date() })
    addLog(`开始上传 ${selectedFiles.length} 个文件，数据集名: ${nameForUpload}`)

    try {
      const uploadResult = await ETMAgentAPI.uploadDataset(selectedFiles, nameForUpload, p => {
        setUploadProgress(p)
        updateStep("upload", { progress: p, message: `上传中 ${p}%` })
      })
      const backendDatasetName = uploadResult.dataset_name
      const jobIdFromUpload = (uploadResult as any).job_id || null
      setEffectiveDatasetName(backendDatasetName)
      if (jobIdFromUpload) setUploadJobId(jobIdFromUpload)
      onUploadComplete?.(backendDatasetName)
      addLog(`✅ 上传完成，后端数据集名: ${backendDatasetName}${jobIdFromUpload ? ` (job_id: ${jobIdFromUpload})` : ''}`)
      setPendingDatasetForConfig(backendDatasetName)
      setPendingJobIdForConfig(jobIdFromUpload)
      const hasCsv = selectedFiles.some(f => f.name.toLowerCase().endsWith(".csv"))
      if (hasCsv) {
        addLog("检测到 CSV 文件，请选择文本列和清洗选项")
        setStatus("column_select")
        setShowColumnSelectPanel(true)
      } else {
        addLog("请配置分析参数后点击「开始分析」")
        setStatus("config")
        setShowConfigPanel(true)
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "上传失败"
      addLog(`❌ ${errorMessage}`)
      updateStep("upload", { status: "error", progress: 0, message: errorMessage })
      setStatus("error")
      onError?.(errorMessage)
    }
  }

  const handleRetry = () => {
    pipelineStarted.current = false
    hasUploaded.current = false
    pipelineDatasetRef.current = ""
    setEffectiveDatasetName(getDatasetName(projectName))
    setSteps(createInitialSteps())
    setOverallProgress(0)
    setStatus("upload")
    setTaskId(null)
    setLogs([])
    setStartTime(null)
    setEndTime(null)
    setResult(null)
    setSelectedFiles([])
    setUploadProgress(0)
    setUploadJobId(null)
    setShowConfigPanel(false)
    setShowColumnSelectPanel(false)
    setPendingDatasetForConfig(null)
    setPendingJobIdForConfig(null)
    setColumnSelection(null)
  }

  const handleColumnSelectConfirm = useCallback((selection: ColumnSelection) => {
    setColumnSelection(selection)
    setShowColumnSelectPanel(false)
    addLog("列选择已确认，请配置分析参数")
    setStatus("config")
    setShowConfigPanel(true)
  }, [addLog])

  const handleColumnSelectSkip = useCallback(() => {
    setShowColumnSelectPanel(false)
    addLog("跳过列选择，请配置分析参数")
    setStatus("config")
    setShowConfigPanel(true)
  }, [addLog])

  const handleConfigConfirm = useCallback(
    (config: AnalysisConfig) => {
      const dataset = pendingDatasetForConfig || effectiveDatasetName
      const jobId = pendingJobIdForConfig || uploadJobId
      setShowConfigPanel(false)
      setPendingDatasetForConfig(null)
      setPendingJobIdForConfig(null)
      if (dataset) {
        startPipelineAfterUpload(dataset, jobId, config, columnSelection)
      }
    },
    [pendingDatasetForConfig, pendingJobIdForConfig, effectiveDatasetName, uploadJobId, columnSelection, startPipelineAfterUpload]
  )

  // ---------- 顶部紧凑步骤条（单行小标签）----------
  const renderStepPill = (step: PipelineStep, index: number) => {
    const Icon = step.icon
    const isActive = step.status === "running"
    const isCompleted = step.status === "completed"
    const isError = step.status === "error"

    return (
      <div key={step.id} className="flex items-center shrink-0">
        {index > 0 && (
          <div
            className={cn(
              "w-4 h-0.5 mx-0.5 shrink-0",
              steps[index - 1].status === "completed" ? "bg-green-400" : "bg-slate-200"
            )}
          />
        )}
        <div
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all",
            isActive && "bg-blue-100 text-blue-700 ring-1 ring-blue-200",
            isCompleted && "bg-green-50 text-green-700",
            isError && "bg-red-50 text-red-700",
            step.status === "waiting" && "bg-slate-100 text-slate-500"
          )}
          title={step.message}
        >
          {isActive && step.id !== "upload" ? (
            <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin" />
          ) : isCompleted ? (
            <Check className="w-3.5 h-3.5 shrink-0 text-green-600" />
          ) : isError ? (
            <X className="w-3.5 h-3.5 shrink-0" />
          ) : (
            <Icon className="w-3.5 h-3.5 shrink-0" />
          )}
          <span className="truncate max-w-[4.5rem] sm:max-w-none">{step.name}</span>
          {(isActive || (step.id === "upload" && step.progress > 0)) && (
            <span className="text-[10px] opacity-80">{step.progress}%</span>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col p-6 lg:p-8">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h1 className="text-2xl font-bold text-slate-900">{projectName}</h1>
          <Badge
            className={cn(
              status === "upload" && "bg-amber-100 text-amber-700",
              status === "column_select" && "bg-amber-100 text-amber-700",
              status === "config" && "bg-violet-100 text-violet-700",
              status === "running" && "bg-blue-100 text-blue-700",
              status === "completed" && "bg-green-100 text-green-700",
              status === "error" && "bg-red-100 text-red-700"
            )}
          >
            {status === "upload" && "请上传数据"}
            {status === "column_select" && "请选择列"}
            {status === "config" && "请完成配置"}
            {status === "running" && "分析中..."}
            {status === "completed" && "已完成"}
            {status === "error" && "出错"}
          </Badge>
        </div>
        <p className="text-slate-500">
          {status === "upload"
            ? "上传数据后弹出配置面板，配置完成后执行：预处理 → 参数选择 → 训练 → 评估 → 可视化"
            : status === "column_select"
            ? "请选择文本列和清洗选项，确认后进入分析配置"
            : status === "config"
            ? "请在下方的配置面板中选择模型和参数，点击「开始分析」"
            : `数据集: ${effectiveDatasetName}`}
        </p>
      </div>

      {/* 顶部紧凑步骤条：单行小标签 */}
      <div className="mb-4 flex flex-wrap items-center gap-1 rounded-lg border border-slate-200 bg-slate-50/50 px-3 py-2">
        <span className="mr-2 text-xs font-medium text-slate-500 shrink-0">步骤</span>
        {steps.map((step, index) => renderStepPill(step, index))}
        {status !== "upload" && startTime && (
          <div className="ml-auto flex items-center gap-2 shrink-0 text-xs text-slate-500">
            <span className="font-semibold text-slate-700">{overallProgress}%</span>
            <Clock className="w-3.5 h-3.5" />
            {endTime
              ? formatDuration(endTime.getTime() - startTime.getTime())
              : formatDuration(Date.now() - startTime.getTime())}
          </div>
        )}
      </div>

      {/* 主内容区：上传 / 结果 / 错误 / 日志 */}
      <Card className="flex-1 flex flex-col min-h-0">
        <CardContent className="pt-6 flex-1 flex flex-col min-h-0 overflow-auto">
            {/* 列选择/配置等待阶段 */}
            {(status === "column_select" || status === "config") && (
              <div className="space-y-4 mb-4">
                <div className="p-4 bg-violet-50 rounded-xl border border-violet-100">
                  <p className="font-medium text-violet-800">上传完成</p>
                  <p className="text-sm text-violet-600 mt-1">
                    {status === "column_select"
                      ? "请在上方弹出的列选择面板中选择文本列和清洗选项。"
                      : "请在上方弹出的配置面板中选择数据语言、模型和超参数，然后点击「开始分析」。"}
                  </p>
                  {status === "column_select" && !showColumnSelectPanel && (
                    <Button
                      onClick={() => setShowColumnSelectPanel(true)}
                      variant="outline"
                      className="mt-3 border-amber-200 text-amber-700 hover:bg-amber-100"
                    >
                      打开列选择面板
                    </Button>
                  )}
                  {status === "config" && !showConfigPanel && (
                    <Button
                      onClick={() => setShowConfigPanel(true)}
                      variant="outline"
                      className="mt-3 border-violet-200 text-violet-700 hover:bg-violet-100"
                    >
                      打开配置面板
                    </Button>
                  )}
                </div>
              </div>
            )}

            {/* 上传阶段：显示上传区 */}
            {status === "upload" && (
              <div className="space-y-4 mb-4">
                <h3 className="font-semibold text-slate-900">上传数据</h3>
                <p className="text-sm text-slate-500">
                  支持 CSV、TXT、PDF、DOCX、Excel 等。上传后将以「{effectiveDatasetName}」作为数据集名称自动开始分析。
                </p>
                <p className="text-xs text-amber-600 bg-amber-50 rounded-lg px-2 py-1.5">
                  分析需至少一个 CSV 文件（含文本列如 text、content、cleaned_content）。若仅上传非 CSV 文件，请先使用数据清洗生成 CSV。
                </p>
                <div
                  onDragOver={e => {
                    e.preventDefault()
                    setIsDragging(true)
                  }}
                  onDragLeave={e => {
                    e.preventDefault()
                    setIsDragging(false)
                  }}
                  onDrop={handleDrop}
                  className={cn(
                    "border-2 border-dashed rounded-xl p-8 text-center transition-colors",
                    isDragging ? "border-blue-500 bg-blue-50" : "border-slate-200 hover:border-slate-300 bg-slate-50/50"
                  )}
                >
                  <Upload className="w-12 h-12 text-slate-400 mx-auto mb-3" />
                  <p className="text-slate-600 mb-1">
                    拖拽文件到此处，或{" "}
                    <label className="text-blue-600 cursor-pointer hover:underline">
                      点击选择
                      <input
                        type="file"
                        multiple
                        className="hidden"
                        accept=".csv,.txt,.docx,.pdf,.xlsx,.json"
                        onChange={e => handleFileSelect(e.target.files)}
                      />
                    </label>
                  </p>
                  <p className="text-xs text-slate-400">支持 CSV, TXT, DOCX, PDF, Excel, JSON</p>
                </div>
                {selectedFiles.length > 0 && (
                  <>
                    <div className="space-y-2">
                      <p className="text-sm font-medium text-slate-700">已选 {selectedFiles.length} 个文件</p>
                      <ScrollArea className="max-h-32 rounded-lg border p-2">
                        {selectedFiles.map((file, idx) => (
                          <div key={idx} className="flex items-center justify-between py-2 px-2 hover:bg-slate-50 rounded">
                            <div className="flex items-center gap-2 min-w-0">
                              <File className="w-4 h-4 text-slate-400 shrink-0" />
                              <span className="text-sm text-slate-700 truncate">{file.name}</span>
                              <span className="text-xs text-slate-400 shrink-0">{(file.size / 1024).toFixed(1)} KB</span>
                            </div>
                            <Button variant="ghost" size="sm" className="h-6 w-6 p-0 shrink-0" onClick={() => removeFile(idx)}>
                              <X className="w-4 h-4" />
                            </Button>
                          </div>
                        ))}
                      </ScrollArea>
                    </div>
                    <Button onClick={handleUploadSubmit} className="w-full" size="lg">
                      <Upload className="w-4 h-4 mr-2" />
                      上传并开始分析
                    </Button>
                  </>
                )}
              </div>
            )}

            {/* 运行中/完成：结果与日志 */}
            {status === "completed" && result && (
              <div className="space-y-4 mb-4">
                <h3 className="font-semibold text-slate-900">分析结果</h3>
                {result.metrics && (
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(result.metrics).slice(0, 4).map(([key, value]) => (
                      <div key={key} className="p-3 bg-green-50 rounded-lg">
                        <p className="text-xs text-green-600 uppercase">{key}</p>
                        <p className="text-lg font-bold text-green-700">{typeof value === "number" ? value.toFixed(4) : value}</p>
                      </div>
                    ))}
                  </div>
                )}
                {result.topicWords &&
                  Object.entries(result.topicWords).slice(0, 3).map(([topicId, words]) => (
                    <div key={topicId} className="p-3 bg-slate-50 rounded-lg">
                      <p className="text-xs text-slate-500 mb-1">主题 {parseInt(topicId) + 1}</p>
                      <p className="text-sm text-slate-700 truncate">{(words as string[]).slice(0, 6).join(", ")}</p>
                    </div>
                  ))}
              </div>
            )}

            {status === "error" && (
              <div className="mb-4">
                <div className="p-4 bg-red-50 rounded-xl border border-red-100">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-red-700">分析流程出错</p>
                      <p className="text-sm text-red-600 mt-1">请检查数据或重试</p>
                    </div>
                  </div>
                </div>
                <Button onClick={handleRetry} className="mt-3 w-full" variant="outline">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  重试
                </Button>
              </div>
            )}

            <Collapsible open={showLogs} onOpenChange={setShowLogs} className="flex-1 flex flex-col min-h-0">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full justify-between mb-2">
                  <span className="flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    执行日志
                  </span>
                  {showLogs ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="flex-1 min-h-0">
                <ScrollArea className="h-64 bg-slate-900 rounded-lg p-4">
                  <div className="font-mono text-xs text-slate-300 space-y-1">
                    {logs.length === 0 ? (
                      <p className="text-slate-500">暂无日志</p>
                    ) : (
                      logs.map((log, idx) => (
                        <div key={idx} className="whitespace-pre-wrap">
                          {log}
                        </div>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </CollapsibleContent>
            </Collapsible>
        </CardContent>
      </Card>

      <ColumnSelectPanel
        open={showColumnSelectPanel}
        onOpenChange={setShowColumnSelectPanel}
        onConfirm={handleColumnSelectConfirm}
        onSkip={handleColumnSelectSkip}
        datasetName={pendingDatasetForConfig || effectiveDatasetName}
        jobId={pendingJobIdForConfig || uploadJobId}
      />

      <AnalysisConfigPanel
        open={showConfigPanel}
        onOpenChange={setShowConfigPanel}
        onConfirm={handleConfigConfirm}
        datasetName={pendingDatasetForConfig || effectiveDatasetName}
      />
    </div>
  )
}

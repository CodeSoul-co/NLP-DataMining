"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { AppShell, type Tab } from "@/components/layout/app-shell"
import { ProjectHub, type Project } from "@/components/dashboard/project-hub"
import { NewProjectDialog, type NewProjectData } from "@/components/dashboard/new-project-dialog"
import { AutoPipeline } from "@/components/project/auto-pipeline"
import type { ChatMessage, SuggestionCard } from "@/components/chat/ai-sidebar"
import { ProtectedRoute } from "@/components/protected-route"
import { ETMAgentAPI, DatasetInfo } from "@/lib/api/etm-agent"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { OverviewTab } from "@/components/results/overview-tab"
import { TopicWordsTab } from "@/components/results/topic-words-tab"
import { MetricsTab } from "@/components/results/metrics-tab"
import { VisualizationTab } from "@/components/results/visualization-tab"
import { ExportTab } from "@/components/results/export-tab"

/** 指标展示名与方向说明：↑ 越高越好 | ↓ 越低越好 | → 越接近 0 越好 */
// Helper to generate timestamp
function getTimestamp() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
}

// Generate unique ID
function generateId() {
  return `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

// Extended project type with additional fields
interface WorkspaceProject extends Project {
  description?: string
  datasetName?: string
  mode?: "zero_shot" | "unsupervised" | "supervised"
  numTopics?: number
  pipelineStatus?: "running" | "completed" | "error"
  dbProjectId?: number  // 数据库项目 ID，用于更新/删除
  taskId?: string | null  // 关联的训练任务 ID
}

function DashboardContent() {
  const router = useRouter()
  const [tabs, setTabs] = useState<Tab[]>([
    { id: "hub", title: "项目中心", closable: false },
  ])
  const [activeTabId, setActiveTabId] = useState("hub")
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [isNewProjectDialogOpen, setIsNewProjectDialogOpen] = useState(false)
  const [projects, setProjects] = useState<WorkspaceProject[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [dynamicSuggestions, setDynamicSuggestions] = useState<SuggestionCard[]>([])
  const [projectTransitionName, setProjectTransitionName] = useState<string | null>(null)

  const transitionTimerRef = useRef<number | null>(null)

  const handleSendMessageRef = useRef<(c: string) => void>(() => {})

  useEffect(() => {
    return () => {
      if (transitionTimerRef.current) {
        window.clearTimeout(transitionTimerRef.current)
      }
    }
  }, [])

  // Load projects: 优先数据库（用户关联），再合并 datasets + tasks
  useEffect(() => {
    const loadProjects = async () => {
      try {
        const [dbProjects, datasets, tasks] = await Promise.all([
          ETMAgentAPI.getProjects(),
          ETMAgentAPI.getDatasets(),
          ETMAgentAPI.getTasks({ limit: 100 }).catch(() => []),
        ])
        const seen = new Set<string>()
        const list: WorkspaceProject[] = []

        // 预构建 dataset→task 映射：优先已完成的任务
        const taskByDataset = new Map<string, { task_id: string; status: string; pipeline_status?: string }>()
        for (const t of tasks) {
          const ds = t.dataset || (t as any).dataset_name
          if (!ds) continue
          const existing = taskByDataset.get(ds)
          if (!existing || (t.status === "completed" && existing.status !== "completed")) {
            taskByDataset.set(ds, { task_id: t.task_id, status: t.status, pipeline_status: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running" })
          }
        }

        // 1. 数据库中的项目（用户关联，跨设备同步）
        for (const p of dbProjects) {
          const key = p.dataset_name || `db-${p.id}`
          seen.add(key)
          // 如果项目没有 task_id，尝试从任务列表中匹配
          let effectiveTaskId = p.task_id ?? null
          let effectivePipelineStatus = p.pipeline_status
          if (!effectiveTaskId && p.dataset_name) {
            const matched = taskByDataset.get(p.dataset_name)
            if (matched) {
              effectiveTaskId = matched.task_id
              effectivePipelineStatus = effectivePipelineStatus || matched.pipeline_status
            }
          }
          const derivedPipelineStatus = effectivePipelineStatus === "completed" ? "completed"
            : effectivePipelineStatus === "error" ? "error"
            : effectivePipelineStatus === "running" ? "running"
            : (effectiveTaskId && effectivePipelineStatus !== "completed") ? "running"
            : (p.status === "draft" || p.status === "uploading") ? "running"
            : undefined
          list.push({
            id: `proj-db-${p.id}`,
            name: p.name,
            rows: 0,
            createdAt: p.created_at ? "已保存" : "刚刚",
            status: (derivedPipelineStatus === "completed" ? "completed" : "vectorizing") as const,
            datasetName: p.dataset_name ?? undefined,
            mode: (p.mode as any) ?? "zero_shot",
            numTopics: p.num_topics ?? 20,
            pipelineStatus: derivedPipelineStatus as any,
            dbProjectId: p.id,
            taskId: effectiveTaskId,
          })
        }

        // 2. 数据集（未在 DB 中的）
        for (const ds of datasets) {
          if (seen.has(ds.name)) continue
          seen.add(ds.name)
          list.push({
            id: `proj-${ds.name}`,
            name: ds.name,
            rows: ds.size ?? (ds as any).file_count ?? 0,
            createdAt: "已上传",
            status: "completed" as const,
            datasetName: ds.name,
          })
        }

        // 3. 任务中的数据集
        for (const t of tasks) {
          const ds = t.dataset || (t as any).dataset_name
          if (ds && !seen.has(ds)) {
            seen.add(ds)
            list.push({
              id: `proj-${ds}`,
              name: ds,
              rows: 0,
              createdAt: "已分析",
              status: (t.status === "completed" ? "completed" : "vectorizing") as const,
              datasetName: ds,
              pipelineStatus: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running",
              taskId: t.task_id || null,
            })
          }
        }

        setProjects(list)
      } catch (error) {
        console.error("Failed to load projects:", error)
      } finally {
        setIsLoading(false)
      }
    }
    loadProjects()
  }, [])

  // 刷新项目列表（与 load 相同逻辑，保留正在运行的项目）
  const refreshProjects = useCallback(async () => {
    setIsLoading(true)
    try {
      const [dbProjects, datasets, tasks] = await Promise.all([
        ETMAgentAPI.getProjects(),
        ETMAgentAPI.getDatasets(),
        ETMAgentAPI.getTasks({ limit: 100 }).catch(() => []),
      ])
      const seen = new Set<string>()
      const list: WorkspaceProject[] = []

      // 预构建 dataset→task 映射：优先已完成的任务
      const taskByDataset = new Map<string, { task_id: string; status: string; pipeline_status?: string }>()
      for (const t of tasks) {
        const ds = t.dataset || (t as any).dataset_name
        if (!ds) continue
        const existing = taskByDataset.get(ds)
        if (!existing || (t.status === "completed" && existing.status !== "completed")) {
          taskByDataset.set(ds, { task_id: t.task_id, status: t.status, pipeline_status: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running" })
        }
      }

      for (const p of dbProjects) {
        const key = p.dataset_name || `db-${p.id}`
        seen.add(key)
        // 如果项目没有 task_id，尝试从任务列表中匹配
        let effectiveTaskId = p.task_id ?? null
        let effectivePipelineStatus = p.pipeline_status
        if (!effectiveTaskId && p.dataset_name) {
          const matched = taskByDataset.get(p.dataset_name)
          if (matched) {
            effectiveTaskId = matched.task_id
            effectivePipelineStatus = effectivePipelineStatus || matched.pipeline_status
          }
        }
        const derivedPipelineStatus = effectivePipelineStatus === "completed" ? "completed"
          : effectivePipelineStatus === "error" ? "error"
          : effectivePipelineStatus === "running" ? "running"
          : (effectiveTaskId && effectivePipelineStatus !== "completed") ? "running"
          : (p.status === "draft" || p.status === "uploading") ? "running"
          : undefined
        list.push({
          id: `proj-db-${p.id}`,
          name: p.name,
          rows: 0,
          createdAt: p.created_at ? "已保存" : "刚刚",
          status: (derivedPipelineStatus === "completed" ? "completed" : "vectorizing") as const,
          datasetName: p.dataset_name ?? undefined,
          mode: (p.mode as any) ?? "zero_shot",
          numTopics: p.num_topics ?? 20,
          pipelineStatus: derivedPipelineStatus as any,
          dbProjectId: p.id,
          taskId: effectiveTaskId,
        })
      }
      for (const ds of datasets) {
        if (seen.has(ds.name)) continue
        seen.add(ds.name)
        list.push({
          id: `proj-${ds.name}`,
          name: ds.name,
          rows: ds.size ?? (ds as any).file_count ?? 0,
          createdAt: "已上传",
          status: "completed" as const,
          datasetName: ds.name,
        })
      }
      for (const t of tasks) {
        const ds = t.dataset || (t as any).dataset_name
        if (ds && !seen.has(ds)) {
          seen.add(ds)
          list.push({
            id: `proj-${ds}`,
            name: ds,
            rows: 0,
            createdAt: "已分析",
            status: (t.status === "completed" ? "completed" : "vectorizing") as const,
            datasetName: ds,
            pipelineStatus: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running",
            taskId: t.task_id || null,
          })
        }
      }
      setProjects(prev => {
        // 构建 dbProjectId → 旧项目 ID 的映射，用于迁移 temp ID
        const oldIdByDbId = new Map<number, string>()
        for (const p of prev) {
          if (p.dbProjectId) oldIdByDbId.set(p.dbProjectId, p.id)
        }

        // 迁移：如果旧列表中有 temp ID（如 new-xxx）指向同一个 dbProjectId，迁移 tab
        for (const np of list) {
          if (np.dbProjectId && oldIdByDbId.has(np.dbProjectId)) {
            const oldId = oldIdByDbId.get(np.dbProjectId)!
            if (oldId !== np.id) {
              setTabs(t => t.map(tab => tab.id === oldId ? { ...tab, id: np.id } : tab))
              setActiveTabId(a => a === oldId ? np.id : a)
            }
          }
        }

        // 保留正在运行的项目，但用新 ID 替换旧 temp ID
        const runningProjects = prev
          .filter(p => p.pipelineStatus === "running")
          .map(rp => {
            if (rp.dbProjectId) {
              const newVersion = list.find(np => np.dbProjectId === rp.dbProjectId)
              if (newVersion) return { ...rp, id: newVersion.id }
            }
            return rp
          })
        const runningIds = new Set(runningProjects.map(p => p.id))
        const runningDbIds = new Set(runningProjects.filter(p => p.dbProjectId).map(p => p.dbProjectId))
        const newProjects = list.filter(np =>
          !runningIds.has(np.id) && !(np.dbProjectId && runningDbIds.has(np.dbProjectId))
        )
        return [...runningProjects, ...newProjects]
      })
    } catch (error) {
      console.error("Failed to refresh projects:", error)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleOpenProject = (projectId: string) => {
    const existingTab = tabs.find((tab) => tab.id === projectId)
    if (existingTab) {
      setActiveTabId(projectId)
    } else {
      const project = projects.find(p => p.id === projectId)
      const projectName = project?.name || "Project"
      const newTab: Tab = {
        id: projectId,
        title: projectName,
        closable: true,
      }
      setTabs([...tabs, newTab])
      setActiveTabId(projectId)
    }
  }

  // 创建新项目：保存到数据库（需登录），并打开工作台
  const handleCreateProject = useCallback(async (data: NewProjectData) => {
    const datasetName = data.name.trim().replace(/\s+/g, "_").replace(/[^\w\u4e00-\u9fa5-]/g, "").toLowerCase() || "dataset"
    const tempProjectId = `new-${Date.now()}`
    const optimisticProject: WorkspaceProject = {
      id: tempProjectId,
      name: data.name,
      datasetName,
      mode: "zero_shot",
      numTopics: 20,
      rows: 0,
      createdAt: "刚刚",
      status: "vectorizing",
      pipelineStatus: "running",
    }

    setProjects(prev => [optimisticProject, ...prev])
    setTabs(prev => [...prev, { id: tempProjectId, title: data.name, closable: true }])
    setActiveTabId(tempProjectId)
    setProjectTransitionName(data.name)

    if (transitionTimerRef.current) {
      window.clearTimeout(transitionTimerRef.current)
    }
    transitionTimerRef.current = window.setTimeout(() => {
      setProjectTransitionName(null)
      transitionTimerRef.current = null
    }, 900)

    try {
      const created = await ETMAgentAPI.createProject({
        name: data.name,
        dataset_name: datasetName,
        mode: "zero_shot",
        num_topics: 20,
      })

      setProjects(prev => prev.map(p => {
        if (p.id !== tempProjectId) return p
        return {
          ...p,
          name: created.name,
          datasetName: created.dataset_name ?? datasetName,
          mode: (created.mode as any) ?? "zero_shot",
          numTopics: created.num_topics ?? 20,
          dbProjectId: created.id,
        }
      }))
    } catch {
      // 未登录或 API 不可用时，继续使用本地项目
    }
  }, [])

  // Pipeline 完成回调：更新本地状态，并同步到数据库（若有 dbProjectId）
  const handlePipelineComplete = useCallback(async (
    projectId: string,
    result?: { dataset?: string; taskId?: string } | null,
    dbProjectId?: number,
  ) => {
    const updates = {
      status: "completed" as const,
      pipelineStatus: "completed" as const,
      ...(result?.dataset && { datasetName: result.dataset }),
    }
    setProjects(prev => prev.map(p => (p.id === projectId ? { ...p, ...updates } : p)))

    if (dbProjectId && result) {
      try {
        await ETMAgentAPI.updateProject(dbProjectId, {
          dataset_name: result.dataset,
          status: "completed",
          pipeline_status: "completed",
          task_id: result.taskId,
        })
      } catch {
        // 忽略同步失败
      }
    }
    refreshProjects()
  }, [refreshProjects])

  const handlePipelineError = useCallback((projectId: string) => {
    setProjects(prev => prev.map(p =>
      p.id === projectId ? { ...p, status: "completed" as const, pipelineStatus: "error" } : p
    ))
  }, [])

  const handleTabChange = (tabId: string) => {
    setActiveTabId(tabId)
  }

  const handleTabClose = (tabId: string) => {
    const tab = tabs.find((t) => t.id === tabId)
    if (!tab?.closable) return

    const newTabs = tabs.filter((t) => t.id !== tabId)
    setTabs(newTabs)

    if (activeTabId === tabId) {
      setActiveTabId("hub")
    }
  }

  // 删除项目：数据库记录 + 可选删除数据集文件
  const handleDeleteProject = useCallback(async (projectId: string) => {
    const project = projects.find((p) => p.id === projectId)
    if (!project) return

    // 数据库项目：删除 DB 记录
    if (project.dbProjectId) {
      try {
        await ETMAgentAPI.deleteProject(project.dbProjectId)
      } catch (error) {
        console.error("删除项目记录失败:", error)
        return
      }
    }

    const datasetName = project.datasetName || (projectId.startsWith("proj-") && !projectId.startsWith("proj-db-") ? projectId.replace(/^proj-/, "") : null)
    if (datasetName) {
      try {
        await ETMAgentAPI.deleteDataset(datasetName)
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error)
        if (!msg.includes("404") && !msg.includes("not found")) {
          console.error("删除数据集失败:", error)
          if (!project.dbProjectId) return
        }
      }
    }

    setProjects((prev) => prev.filter((p) => p.id !== projectId))
    const newTabs = tabs.filter((t) => t.id !== projectId)
    setTabs(newTabs)
    if (activeTabId === projectId) {
      setActiveTabId("hub")
    }
  }, [projects, tabs, activeTabId])

  // 批量删除：并发执行，部分失败不阻断其他项目
  const handleBatchDelete = useCallback(async (projectIds: string[]) => {
    const results = await Promise.allSettled(
      projectIds.map(async (projectId) => {
        const project = projects.find((p) => p.id === projectId)
        if (!project) return
        if (project.dbProjectId) {
          await ETMAgentAPI.deleteProject(project.dbProjectId)
        }
        const datasetName = project.datasetName || (projectId.startsWith("proj-") && !projectId.startsWith("proj-db-") ? projectId.replace(/^proj-/, "") : null)
        if (datasetName) {
          try {
            await ETMAgentAPI.deleteDataset(datasetName)
          } catch (error) {
            const msg = error instanceof Error ? error.message : String(error)
            if (!msg.includes("404") && !msg.includes("not found")) throw error
          }
        }
      })
    )
    const deletedIds = new Set(
      projectIds.filter((_, i) => results[i].status === "fulfilled")
    )
    if (deletedIds.size === 0) return
    setProjects((prev) => prev.filter((p) => !deletedIds.has(p.id)))
    setTabs((prev) => prev.filter((t) => !deletedIds.has(t.id)))
    if (deletedIds.has(activeTabId)) setActiveTabId("hub")
  }, [projects, activeTabId])

  // Chat handlers — 使用 SSE 流式对话，fallback 到普通请求
  const handleSendMessage = useCallback(async (content: string) => {
    const userMessage: ChatMessage = {
      id: generateId(),
      role: "user",
      content,
      type: "text",
      timestamp: getTimestamp(),
    }
    setChatHistory((prev) => [...prev, userMessage])

    const aiMsgId = generateId()

    // 先添加一条空的 AI 消息，后续流式追加内容
    setChatHistory((prev) => [
      ...prev,
      { id: aiMsgId, role: "ai", content: "", type: "text", timestamp: getTimestamp() },
    ])

    try {
      let fullText = ""
      let streamed = false

      // 尝试 SSE 流式对话
      try {
        for await (const chunk of ETMAgentAPI.chatStream(content)) {
          streamed = true
          if (chunk.type === "text" && chunk.content) {
            fullText = chunk.content
            setChatHistory((prev) =>
              prev.map((m) => (m.id === aiMsgId ? { ...m, content: fullText } : m))
            )
          }
        }
      } catch {
        // SSE 不可用，fallback 到普通对话
        if (!streamed) {
          const response = await ETMAgentAPI.chat(content, {
            current_view_name: "项目中心",
            current_view: activeTabId === "hub" ? "hub" : "workspace",
            app_state: "workspace",
            session_id: "dashboard",
          })
          fullText = response.message ?? (response as { response?: string }).response ?? "（无回复）"
        }
      }

      if (fullText) {
        setChatHistory((prev) =>
          prev.map((m) => (m.id === aiMsgId ? { ...m, content: fullText } : m))
        )
      }
    } catch (error) {
      setChatHistory((prev) =>
        prev.map((m) =>
          m.id === aiMsgId
            ? {
                ...m,
                content: "无法连接 AI 服务，请确认后端已启动。",
                followUpQuestions: ["如何开始？", "支持哪些格式？"],
              }
            : m
        )
      )
    }
  }, [activeTabId])

  handleSendMessageRef.current = handleSendMessage

  // 当查看已完成项目时，拉取 interpret API 生成动态建议
  useEffect(() => {
    const project = projects.find(p => p.id === activeTabId)
    if (!project || project.pipelineStatus !== "completed" || !project.datasetName) {
      setDynamicSuggestions([])
      return
    }
    const jobId = project.datasetName
    const send = (q: string) => handleSendMessageRef.current(q)
    const fallbacks: SuggestionCard[] = [
      { title: "指标解读", description: "分析当前评估指标质量", onClick: () => send("请解读评估指标。") },
      { title: "主题解读", description: "各主题语义描述", onClick: () => send("请解读各主题含义。") },
      { title: "分析报告", description: "整体分析摘要", onClick: () => send("请生成分析报告。") },
    ]
    Promise.allSettled([
      ETMAgentAPI.interpretMetrics(jobId, "zh").catch(() => null),
      ETMAgentAPI.interpretTopics(jobId, "zh", true).catch(() => null),
      ETMAgentAPI.generateSummary(jobId, "zh").catch(() => null),
    ]).then(([m, t, s]) => {
      const cards: SuggestionCard[] = [
        m.status === "fulfilled" && (m.value as any)?.summary
          ? { title: "指标解读", description: String((m.value as any).summary).slice(0, 55) + "...", onClick: () => send("请解读评估指标。") }
          : fallbacks[0],
        t.status === "fulfilled" && (t.value as any)?.summary
          ? { title: "主题解读", description: String((t.value as any).summary).slice(0, 55) + "...", onClick: () => send("请解读各主题含义。") }
          : fallbacks[1],
        s.status === "fulfilled" && (s.value as any)?.summary
          ? { title: "分析报告", description: String((s.value as any).summary).slice(0, 55) + "...", onClick: () => send("请生成分析报告。") }
          : fallbacks[2],
      ]
      setDynamicSuggestions(cards)
    }).catch(() => setDynamicSuggestions(fallbacks))
  }, [activeTabId, projects])

  const handleDataUploaded = useCallback(async (file: File) => {
    console.log("Data uploaded:", file.name)
  }, [])

  const handleFocusChart = useCallback((chartId: string) => {
    console.log("Focus chart:", chartId)
  }, [])

  const handleClearChat = useCallback(() => {
    setChatHistory([])
  }, [])

  // 渲染内容
  const renderContent = () => {
    if (activeTabId === "hub") {
      return (
        <ProjectHub 
          onProjectSelect={handleOpenProject} 
          onNewProject={() => setIsNewProjectDialogOpen(true)}
          onDeleteProject={handleDeleteProject}
          onBatchDelete={handleBatchDelete}
          projects={projects}
          isLoading={isLoading}
        />
      )
    }

    // 查找当前项目
    const currentProject = projects.find(p => p.id === activeTabId)
    
    if (!currentProject) {
      return (
        <div className="p-8 text-center">
          <h2 className="text-xl font-semibold text-slate-900 mb-2">项目未找到</h2>
          <p className="text-slate-500">该项目可能已被删除</p>
        </div>
      )
    }

    // 新建项目 / 运行中 / 有关联任务：先上传数据，再自动执行流程。出错时也留在本页以便重试
    if (currentProject.pipelineStatus === "running" || currentProject.pipelineStatus === "error") {
      return (
        <AutoPipeline
          projectName={currentProject.name}
          mode={currentProject.mode || "zero_shot"}
          numTopics={currentProject.numTopics || 20}
          initialTaskId={currentProject.taskId}
          onComplete={(result) => handlePipelineComplete(currentProject.id, result, currentProject.dbProjectId)}
          onError={() => handlePipelineError(currentProject.id)}
          onTaskCreated={async (tid) => {
            setProjects(prev => prev.map(p =>
              p.id === currentProject.id ? { ...p, taskId: tid } : p
            ))
            if (currentProject.dbProjectId) {
              try {
                await ETMAgentAPI.updateProject(currentProject.dbProjectId, {
                  task_id: tid,
                  pipeline_status: "running",
                })
              } catch { /* skip */ }
            }
          }}
          onUploadComplete={async (datasetName) => {
            if (currentProject.dbProjectId) {
              try {
                await ETMAgentAPI.updateProject(currentProject.dbProjectId, {
                  dataset_name: datasetName,
                  status: "uploading",
                })
                setProjects(prev => prev.map(p =>
                  p.id === currentProject.id ? { ...p, datasetName } : p
                ))
              } catch { /* skip */ }
            }
          }}
        />
      )
    }

    // 已完成的项目显示结果概览
    return (
      <ProjectResultView project={currentProject} />
    )
  }

  return (
    <>
      <AppShell
        tabs={tabs}
        activeTabId={activeTabId}
        onTabChange={handleTabChange}
        onTabClose={handleTabClose}
        chatHistory={chatHistory}
        onSendMessage={handleSendMessage}
        onDataUploaded={handleDataUploaded}
        onFocusChart={handleFocusChart}
        onClearChat={handleClearChat}
        dynamicSuggestions={dynamicSuggestions}
      >
        <div className="relative min-h-[360px]">
          <div
            className={`transition-all duration-500 ${projectTransitionName ? "opacity-70 scale-[0.995]" : "opacity-100 scale-100"}`}
          >
            {renderContent()}
          </div>

          {projectTransitionName && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/70 backdrop-blur-[1px]">
              <div className="rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="h-5 w-5 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
                  <div>
                    <p className="text-sm font-medium text-slate-800">正在创建项目</p>
                    <p className="text-xs text-slate-500">{projectTransitionName} 初始化中，请稍候...</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </AppShell>
      
      <NewProjectDialog
        open={isNewProjectDialogOpen}
        onOpenChange={setIsNewProjectDialogOpen}
        onSubmit={handleCreateProject}
      />
    </>
  )
}


// 项目结果视图 — 使用统一结果 Tab 组件
function ProjectResultView({ project }: { project: WorkspaceProject }) {
  const dataset = project.datasetName || project.name
  const mode    = project.mode || "zero_shot"
  const [activeResultTab, setActiveResultTab] = useState("overview")

  return (
    <div className="p-6 lg:p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900 mb-1">{project.name}</h1>
        <p className="text-slate-500 text-sm">
          数据集: {dataset} · 模式: {mode} · 主题数: {project.numTopics || 20}
        </p>
      </div>

      <Tabs value={activeResultTab} onValueChange={setActiveResultTab}>
        <TabsList>
          <TabsTrigger value="overview">概览</TabsTrigger>
          <TabsTrigger value="topics">主题词</TabsTrigger>
          <TabsTrigger value="metrics">评估指标</TabsTrigger>
          <TabsTrigger value="viz">可视化</TabsTrigger>
          <TabsTrigger value="export">导出</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-4">
          <OverviewTab dataset={dataset} mode={mode} />
        </TabsContent>

        <TabsContent value="topics" className="mt-4">
          <TopicWordsTab dataset={dataset} mode={mode} shouldLoad={activeResultTab === "topics"} />
        </TabsContent>

        <TabsContent value="metrics" className="mt-4">
          <MetricsTab dataset={dataset} mode={mode} shouldLoad={activeResultTab === "metrics"} />
        </TabsContent>

        <TabsContent value="viz" className="mt-4">
          <VisualizationTab dataset={dataset} mode={mode} shouldLoad={activeResultTab === "viz"} />
        </TabsContent>

        <TabsContent value="export" className="mt-4">
          <ExportTab dataset={dataset} mode={mode} />
        </TabsContent>
      </Tabs>
    </div>
  )
}


// Dashboard page with auth protection
export default function DashboardPage() {
  return (
    <ProtectedRoute>
      <DashboardContent />
    </ProtectedRoute>
  )
}

"use client"

import { useState } from "react"
import Link from "next/link"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Plus, Database, Clock, Loader2, Check, AlertCircle, Sparkles, MoreVertical, Trash2, CheckSquare, ExternalLink } from "lucide-react"

type ProjectStatus = "cleaning" | "vectorizing" | "completed"

export interface Project {
  id: string
  name: string
  status?: ProjectStatus
  rows: number
  createdAt: string
  description?: string
  pipelineStatus?: "running" | "completed" | "error"
  /** 数据集名称，用于结果页跳转 */
  datasetName?: string
  /** 训练模式，用于结果页跳转 */
  mode?: string
}

interface ProjectHubProps {
  onProjectSelect: (projectId: string) => void
  onNewProject: () => void
  onDeleteProject?: (projectId: string) => void
  onBatchDelete?: (projectIds: string[]) => void
  projects?: Project[]
  isLoading?: boolean
}

function getStatusConfig(project: Project) {
  // 优先检查 pipeline 状态
  if (project.pipelineStatus === "running") {
    return {
      label: "分析中",
      className: "bg-indigo-100 text-indigo-700 border-indigo-200",
      icon: Loader2,
      animate: true,
    }
  }
  if (project.pipelineStatus === "error") {
    return {
      label: "出错",
      className: "bg-red-100 text-red-700 border-red-200",
      icon: AlertCircle,
      animate: false,
    }
  }

  // 常规状态
  switch (project.status) {
    case "completed":
      return {
        label: "已完成",
        className: "bg-emerald-100 text-emerald-700 border-emerald-200",
        icon: Check,
        animate: false,
      }
    case "cleaning":
      return {
        label: "清洗中",
        className: "bg-amber-100 text-amber-700 border-amber-200",
        icon: Loader2,
        animate: true,
      }
    case "vectorizing":
      return {
        label: "处理中",
        className: "bg-indigo-100 text-indigo-700 border-indigo-200",
        icon: Loader2,
        animate: true,
      }
    default:
      return {
        label: "就绪",
        className: "bg-slate-100 text-slate-600 border-slate-200",
        icon: Check,
        animate: false,
      }
  }
}

export function ProjectHub({ onProjectSelect, onNewProject, onDeleteProject, onBatchDelete, projects = [], isLoading }: ProjectHubProps) {
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null)
  const [batchMode, setBatchMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [batchDeleteConfirm, setBatchDeleteConfirm] = useState(false)
  const deleteTarget = deleteTargetId ? projects.find((p) => p.id === deleteTargetId) : null

  const projectList = projects.filter((p) => p.id && !p.id.startsWith("new-"))
  const selectedCount = selectedIds.size

  const handleToggleSelect = (id: string, checked: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (checked) next.add(id)
      else next.delete(id)
      return next
    })
  }

  const handleSelectAll = (checked: boolean) => {
    if (checked) setSelectedIds(new Set(projectList.map((p) => p.id)))
    else setSelectedIds(new Set())
  }

  const handleConfirmDelete = () => {
    if (deleteTargetId) {
      onDeleteProject?.(deleteTargetId)
      setDeleteTargetId(null)
    }
  }

  const handleConfirmBatchDelete = () => {
    if (selectedCount > 0 && onBatchDelete) {
      onBatchDelete(Array.from(selectedIds))
      setSelectedIds(new Set())
      setBatchDeleteConfirm(false)
      setBatchMode(false)
    }
  }

  return (
    <div className="min-h-full p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* 顶部一行：标题 + 批量管理 + 新建 */}
        <div className="flex items-center justify-between gap-4 mb-6">
          <h2 className="text-base font-semibold text-slate-700">项目列表</h2>
          <div className="flex items-center gap-2">
            {onBatchDelete && projectList.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setBatchMode((prev) => !prev)
                  if (batchMode) setSelectedIds(new Set())
                }}
                className={`h-8 px-2.5 text-sm ${batchMode ? "bg-slate-200 text-slate-800" : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"}`}
              >
                <CheckSquare className="h-3.5 w-3.5 mr-1.5" />
                批量管理
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={onNewProject}
              className="text-slate-600 hover:text-slate-900 hover:bg-slate-100 h-8 px-2.5 text-sm"
            >
              <Plus className="h-3.5 w-3.5 mr-1.5" />
              新建
            </Button>
          </div>
        </div>

        {/* 批量操作栏 */}
        {batchMode && selectedCount > 0 && onBatchDelete && (
          <div className="flex items-center justify-between gap-4 mb-4 py-2 px-3 rounded-lg bg-slate-100 border border-slate-200">
            <span className="text-sm text-slate-600">
              已选择 <strong>{selectedCount}</strong> 个项目
            </span>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => (selectedCount === projectList.length ? handleSelectAll(false) : handleSelectAll(true))} className="h-8 text-sm">
                {selectedCount === projectList.length ? "取消全选" : "全选"}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setSelectedIds(new Set())} className="h-8 text-sm">
                取消选择
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => setBatchDeleteConfirm(true)}
                className="h-8 text-sm"
              >
                <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                批量删除
              </Button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <div className="flex flex-col items-center justify-center py-16 rounded-lg border border-slate-200 bg-white min-h-[180px]">
              <div className="w-8 h-8 border-2 border-slate-300 border-t-slate-600 rounded-full animate-spin mb-3" />
              <p className="text-sm text-slate-500">加载中...</p>
            </div>
          </div>
        )}

        {/* 项目网格：无项目时只显示「新建项目」卡片，有项目时多出项目卡片，样式一致 */}
        {!isLoading && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {/* 新建项目卡片 - 与有项目时同一张卡片 */}
            <Card
              onClick={onNewProject}
              className="group border border-dashed border-slate-200 hover:border-slate-300 bg-slate-50/50 hover:bg-slate-100/50 cursor-pointer rounded-lg transition-colors"
            >
              <CardContent className="flex flex-col items-center justify-center min-h-[140px] py-6">
                <Plus className="w-8 h-8 text-slate-400 group-hover:text-slate-600 mb-2" />
                <p className="text-sm font-medium text-slate-500 group-hover:text-slate-700">新建项目</p>
                {projects.length === 0 && (
                  <p className="text-xs text-slate-400 mt-1">点击创建第一个项目</p>
                )}
              </CardContent>
            </Card>

            {/* 项目卡片列表 */}
            {projects.map((project) => {
              const statusConfig = getStatusConfig(project)
              const StatusIcon = statusConfig.icon
              const isSelected = selectedIds.has(project.id)
              const canSelect = batchMode && onBatchDelete && !project.id.startsWith("new-")

              return (
                <Card
                  key={project.id}
                  onClick={() => onProjectSelect(project.id)}
                  className={`group bg-white hover:bg-slate-50/50 cursor-pointer rounded-lg overflow-hidden border transition-colors ${
                    project.pipelineStatus === "running" 
                      ? "border-blue-200 ring-1 ring-blue-100" 
                      : project.pipelineStatus === "error"
                      ? "border-red-100"
                      : "border-slate-200 hover:border-slate-300"
                  }`}
                >
                  {/* 顶部状态条 - 细线 */}
                  <div className={`h-0.5 ${
                    project.pipelineStatus === "running" 
                      ? "bg-blue-400" 
                      : project.pipelineStatus === "error"
                      ? "bg-red-400"
                      : project.status === "completed"
                      ? "bg-emerald-400"
                      : "bg-slate-200"
                  }`} />

                  <CardContent className="p-4 relative">
                    {/* 批量选择 checkbox */}
                    {canSelect && (
                      <div className="absolute top-2 left-2 z-10" onClick={(e) => e.stopPropagation()}>
                        <Checkbox
                          checked={isSelected}
                          onCheckedChange={(c) => handleToggleSelect(project.id, !!c)}
                          className="border-slate-300 data-[state=checked]:bg-blue-600"
                        />
                      </div>
                    )}
                    {/* 项目管理：更多 -> 删除 */}
                    {onDeleteProject && (
                      <div className="absolute top-2 right-2 z-10" onClick={(e) => e.stopPropagation()}>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7 rounded text-slate-400 hover:text-slate-600 hover:bg-slate-100"
                            >
                              <MoreVertical className="h-3.5 w-3.5" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-36">
                            <DropdownMenuItem
                              variant="destructive"
                              onClick={(e) => {
                                e.preventDefault()
                                setDeleteTargetId(project.id)
                              }}
                            >
                              <Trash2 className="h-3.5 w-3.5 mr-2" />
                              删除
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    )}

                    {/* 标题和状态 */}
                    <div className={`flex items-start justify-between gap-2 mb-3 pr-7 ${canSelect ? "pl-7" : ""}`}>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-slate-800 group-hover:text-slate-900 truncate text-sm">
                          {project.name}
                        </h3>
                      </div>
                      <Badge className={`shrink-0 text-[10px] font-medium px-1.5 py-0 border ${statusConfig.className}`}>
                        <StatusIcon className={`w-2.5 h-2.5 mr-0.5 ${statusConfig.animate ? "animate-spin" : ""}`} />
                        {statusConfig.label}
                      </Badge>
                    </div>

                    {/* 信息 */}
                    <div className="flex items-center gap-3 text-xs text-slate-500">
                      <span className="flex items-center gap-1">
                        <Database className="w-3 h-3" />
                        {project.rows > 0 ? `${project.rows.toLocaleString()} 条` : "—"}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {project.createdAt}
                      </span>
                    </div>

                    {/* 运行中指示 */}
                    {project.pipelineStatus === "running" && (
                      <div className="mt-2 flex items-center gap-1.5 text-blue-600 text-xs">
                        <Sparkles className="w-3 h-3" />
                        <span>分析中...</span>
                      </div>
                    )}

                    {/* 查看结果按钮（已完成时显示） */}
                    {project.pipelineStatus === "completed" && project.datasetName && (
                      <div className="mt-3 pt-3 border-t border-slate-100" onClick={(e) => e.stopPropagation()}>
                        <Link
                          href={`/results/${encodeURIComponent(project.datasetName)}/${encodeURIComponent(project.mode ?? "zero_shot")}`}
                          className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-600 hover:text-blue-700 hover:underline"
                        >
                          <ExternalLink className="w-3 h-3" />
                          查看结果
                        </Link>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )
            })}
          </div>
        )}
      </div>

      {/* 单条删除确认弹窗 */}
      <AlertDialog open={!!deleteTargetId} onOpenChange={(open) => !open && setDeleteTargetId(null)}>
        <AlertDialogContent onClick={(e) => e.stopPropagation()}>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除项目</AlertDialogTitle>
            <AlertDialogDescription>
              {deleteTarget
                ? `删除「${deleteTarget.name}」后，其数据集与结果将一并移除，且无法恢复。确定要删除吗？`
                : "确定要删除该项目吗？"}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmDelete} className="bg-red-600 hover:bg-red-700">
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* 批量删除确认弹窗 */}
      <AlertDialog open={batchDeleteConfirm} onOpenChange={setBatchDeleteConfirm}>
        <AlertDialogContent onClick={(e) => e.stopPropagation()}>
          <AlertDialogHeader>
            <AlertDialogTitle>确认批量删除</AlertDialogTitle>
            <AlertDialogDescription>
              将删除已选的 {selectedCount} 个项目，其数据集与结果将一并移除，且无法恢复。确定要删除吗？
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmBatchDelete} className="bg-red-600 hover:bg-red-700">
              批量删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

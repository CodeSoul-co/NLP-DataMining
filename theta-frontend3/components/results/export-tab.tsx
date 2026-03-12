"use client"

import { useState } from "react"
import { ETMAgentAPI } from "@/lib/api/etm-agent"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Download, Package, Loader2 } from "lucide-react"

interface ExportTabProps {
  dataset: string
  mode: string
}

const EXPORT_OPTIONS = [
  { id: "metrics", label: "评估指标 JSON", description: "coherence、diversity 等数值指标" },
  { id: "topic_words", label: "主题词 JSON", description: "各主题的 top-K 关键词与权重" },
  { id: "visualizations", label: "图表文件 PNG/HTML", description: "词云、UMAP、雷达图等所有图表" },
]

export function ExportTab({ dataset, mode }: ExportTabProps) {
  const [selected, setSelected] = useState<Set<string>>(
    new Set(EXPORT_OPTIONS.map((o) => o.id))
  )
  const [exporting, setExporting] = useState(false)

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleExport = async () => {
    if (selected.size === 0) return
    setExporting(true)
    try {
      await ETMAgentAPI.exportResults(dataset, mode, Array.from(selected))
    } catch (e) {
      console.error("Export failed:", e)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="max-w-lg space-y-6">
      {/* 选择导出内容 */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-700">选择导出内容</h3>
        {EXPORT_OPTIONS.map((opt) => (
          <label
            key={opt.id}
            className="flex items-start gap-3 cursor-pointer group"
          >
            <Checkbox
              checked={selected.has(opt.id)}
              onCheckedChange={() => toggle(opt.id)}
              className="mt-0.5 border-slate-300 data-[state=checked]:bg-blue-600"
            />
            <div>
              <p className="text-sm font-medium text-slate-800 group-hover:text-slate-900">
                {opt.label}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">{opt.description}</p>
            </div>
          </label>
        ))}
      </div>

      {/* 导出按钮 */}
      <Button
        onClick={handleExport}
        disabled={selected.size === 0 || exporting}
        className="w-full bg-blue-600 hover:bg-blue-700"
      >
        {exporting ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            导出中...
          </>
        ) : (
          <>
            <Package className="w-4 h-4 mr-2" />
            导出 ZIP ({selected.size} 项)
          </>
        )}
      </Button>

      {/* 提示 */}
      <div className="rounded-lg bg-slate-50 border border-slate-200 px-4 py-3 text-xs text-slate-500 flex items-start gap-2">
        <Download className="w-3.5 h-3.5 mt-0.5 shrink-0" />
        <span>
          导出文件为 ZIP 压缩包，包含勾选的结果文件。下载完成后会自动弹出保存对话框。
        </span>
      </div>
    </div>
  )
}

"use client"

import { useEffect, useState } from "react"
import { ETMAgentAPI, type ResultInfo } from "@/lib/api/etm-agent"
import { Badge } from "@/components/ui/badge"
import { CheckCircle2, Clock, FileText, Hash, Layers, Loader2, AlertCircle } from "lucide-react"

interface OverviewTabProps {
  dataset: string
  mode: string
}

export function OverviewTab({ dataset, mode }: OverviewTabProps) {
  const [info, setInfo] = useState<ResultInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/results/${encodeURIComponent(dataset)}/${encodeURIComponent(mode)}`, {
      headers: {
        Authorization: `Bearer ${typeof window !== "undefined" ? localStorage.getItem("access_token") ?? "" : ""}`,
      },
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text())
        return res.json() as Promise<ResultInfo>
      })
      .then((data) => setInfo(data))
      .catch((e) => setError(e.message ?? "加载失败"))
      .finally(() => setLoading(false))
  }, [dataset, mode])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 animate-spin text-slate-400 mr-2" />
        <span className="text-slate-500">加载概览...</span>
      </div>
    )
  }

  if (error || !info) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-2 text-slate-500">
        <AlertCircle className="w-8 h-8 text-amber-400" />
        <p className="text-sm">{error ?? "暂无结果数据"}</p>
      </div>
    )
  }

  const stats = [
    { label: "数据集", value: info.dataset, icon: FileText },
    { label: "训练模式", value: info.mode, icon: Layers },
    { label: "主题数", value: info.num_topics != null ? `${info.num_topics} 个` : "—", icon: Hash },
    { label: "训练轮次", value: info.epochs_trained != null ? `${info.epochs_trained} epochs` : "—", icon: Clock },
  ]

  const flags = [
    { label: "模型权重", value: info.has_model },
    { label: "θ 矩阵", value: info.has_theta },
    { label: "β 矩阵", value: info.has_beta },
    { label: "主题词", value: info.has_topic_words },
    { label: "可视化图表", value: info.has_visualizations },
  ]

  return (
    <div className="space-y-8">
      {/* 状态 */}
      <div className="flex items-center gap-3">
        <CheckCircle2 className="w-5 h-5 text-emerald-500" />
        <span className="font-semibold text-slate-800">训练已完成</span>
        {info.timestamp && (
          <Badge variant="secondary" className="text-xs">{info.timestamp}</Badge>
        )}
      </div>

      {/* 基本统计 */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {stats.map(({ label, value, icon: Icon }) => (
          <div key={label} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <div className="flex items-center gap-2 mb-1 text-slate-500 text-xs">
              <Icon className="w-3.5 h-3.5" />
              {label}
            </div>
            <p className="font-semibold text-slate-900 text-sm">{value ?? "—"}</p>
          </div>
        ))}
      </div>

      {/* 产物状态 */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">产物检查</h3>
        <div className="flex flex-wrap gap-3">
          {flags.map(({ label, value }) => (
            <div
              key={label}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border ${
                value
                  ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                  : "bg-slate-50 border-slate-200 text-slate-400"
              }`}
            >
              {value
                ? <CheckCircle2 className="w-3 h-3" />
                : <AlertCircle className="w-3 h-3" />}
              {label}
            </div>
          ))}
        </div>
      </div>

      {/* 核心指标快览 */}
      {info.metrics && Object.keys(info.metrics).length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-slate-700 mb-3">核心指标快览</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {Object.entries(info.metrics)
              .filter(([, v]) => typeof v === "number")
              .slice(0, 8)
              .map(([key, value]) => (
                <div key={key} className="rounded-lg border border-slate-200 bg-white p-3">
                  <p className="text-xs text-slate-500 mb-1 truncate">{key}</p>
                  <p className="font-semibold text-slate-900 text-sm">
                    {typeof value === "number" ? value.toFixed(4) : String(value)}
                  </p>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}

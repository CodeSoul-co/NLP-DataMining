"use client"

import { useEffect, useState } from "react"
import { ETMAgentAPI, type MetricsResponse } from "@/lib/api/etm-agent"
import { Loader2, AlertCircle } from "lucide-react"
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from "recharts"

interface MetricsTabProps {
  dataset: string
  mode: string
  shouldLoad: boolean
}

const METRIC_LABELS: Record<string, string> = {
  topic_coherence_avg: "主题连贯性",
  topic_diversity_td: "主题多样性 TD",
  topic_diversity_irbo: "主题多样性 IRBO",
  perplexity: "困惑度(倒)",
  npmi_avg: "NPMI 均值",
  cv_avg: "CV 均值",
  topic_exclusivity: "排他性",
}

function toPercent(key: string, value: number): number {
  if (key.toLowerCase().includes("perplexity")) {
    // 困惑度越低越好，转换为 0-1 分
    return Math.max(0, Math.min(1, 1 / (1 + Math.abs(value) / 1000)))
  }
  // 大多数指标在 0~1 范围
  return Math.max(0, Math.min(1, Math.abs(value)))
}

export function MetricsTab({ dataset, mode, shouldLoad }: MetricsTabProps) {
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!shouldLoad) return
    setLoading(true)
    ETMAgentAPI.getMetrics(dataset, mode)
      .then((raw) => {
        const merged: Record<string, number> = {}
        const r = raw as MetricsResponse & { additional?: Record<string, unknown> }
        if (r.additional) {
          for (const [k, v] of Object.entries(r.additional)) {
            if (typeof v === "number") merged[k] = v
          }
        }
        for (const [k, v] of Object.entries(r)) {
          if (k !== "additional" && k !== "dataset" && k !== "mode" && k !== "timestamp" && typeof v === "number") {
            merged[k] = v
          }
        }
        setMetrics(Object.keys(merged).length ? merged : null)
      })
      .catch(() => setMetrics(null))
      .finally(() => setLoading(false))
  }, [dataset, mode, shouldLoad])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 animate-spin text-slate-400 mr-2" />
        <span className="text-slate-500">加载指标...</span>
      </div>
    )
  }

  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-2 text-slate-500">
        <AlertCircle className="w-8 h-8 text-amber-400" />
        <p className="text-sm">暂无评估指标数据</p>
      </div>
    )
  }

  const numericEntries = Object.entries(metrics).filter(([, v]) => typeof v === "number")
  const radarData = numericEntries.map(([key, value]) => ({
    metric: METRIC_LABELS[key] ?? key,
    value: toPercent(key, value),
    raw: value,
  }))

  return (
    <div className="space-y-8">
      {/* 雷达图 */}
      {radarData.length >= 3 && (
        <div className="rounded-xl border border-slate-200 bg-white p-6">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">综合评估雷达图</h3>
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarData} margin={{ top: 16, right: 32, bottom: 16, left: 32 }}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fontSize: 11, fill: "#64748b" }}
              />
              <Tooltip
                formatter={(value: number, _name: string, entry: { payload?: { raw?: number } }) => [
                  entry?.payload?.raw != null ? entry.payload.raw.toFixed(4) : value.toFixed(4),
                  "值",
                ]}
              />
              <Radar
                dataKey="value"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.18}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 数值表格 */}
      <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50">
              <th className="text-left px-4 py-2.5 font-medium text-slate-600 text-xs">指标</th>
              <th className="text-right px-4 py-2.5 font-medium text-slate-600 text-xs">数值</th>
            </tr>
          </thead>
          <tbody>
            {numericEntries.map(([key, value], idx) => (
              <tr
                key={key}
                className={`border-b border-slate-50 ${idx % 2 === 0 ? "bg-white" : "bg-slate-50/50"}`}
              >
                <td className="px-4 py-2.5 text-slate-700">
                  {METRIC_LABELS[key] ?? key}
                  {key !== (METRIC_LABELS[key] ?? key) && (
                    <span className="ml-2 text-xs text-slate-400 font-mono">{key}</span>
                  )}
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-slate-900">
                  {value.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

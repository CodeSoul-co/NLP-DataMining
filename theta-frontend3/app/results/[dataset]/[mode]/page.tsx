"use client"

import { useParams, useRouter } from "next/navigation"
import { useState } from "react"
import { ChevronLeft, LayoutDashboard, Eye, MessageSquareText, BarChart3, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { OverviewTab } from "@/components/results/overview-tab"
import { TopicWordsTab } from "@/components/results/topic-words-tab"
import { VisualizationTab } from "@/components/results/visualization-tab"
import { MetricsTab } from "@/components/results/metrics-tab"
import { ExportTab } from "@/components/results/export-tab"

type TabId = "overview" | "topics" | "visualization" | "metrics" | "export"

const TABS: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: "overview", label: "概览", icon: LayoutDashboard },
  { id: "topics", label: "主题词", icon: MessageSquareText },
  { id: "visualization", label: "可视化", icon: Eye },
  { id: "metrics", label: "评估指标", icon: BarChart3 },
  { id: "export", label: "下载", icon: Download },
]

export default function ResultPage() {
  const params = useParams()
  const router = useRouter()
  const dataset = decodeURIComponent(String(params.dataset ?? ""))
  const mode = decodeURIComponent(String(params.mode ?? ""))
  const [activeTab, setActiveTab] = useState<TabId>("overview")

  return (
    <div className="min-h-screen bg-slate-50">
      {/* 顶部导航栏 */}
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.back()}
            className="h-8 w-8 p-0 shrink-0"
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <div className="min-w-0">
            <h1 className="font-semibold text-slate-900 text-sm truncate">
              {dataset}
            </h1>
            <p className="text-xs text-slate-500">{mode}</p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.push("/dashboard")}
          className="h-8 text-xs text-slate-500 shrink-0"
        >
          返回 Dashboard
        </Button>
      </header>

      {/* Tab 导航 */}
      <div className="border-b border-slate-200 bg-white px-4 sm:px-6">
        <nav className="flex gap-1 overflow-x-auto" role="tablist">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              role="tab"
              aria-selected={activeTab === id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-1.5 px-3 py-3 text-sm whitespace-nowrap border-b-2 transition-colors ${
                activeTab === id
                  ? "border-blue-600 text-blue-600 font-medium"
                  : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab 内容 */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6">
        {activeTab === "overview" && (
          <OverviewTab dataset={dataset} mode={mode} />
        )}
        {activeTab === "topics" && (
          <TopicWordsTab dataset={dataset} mode={mode} shouldLoad={activeTab === "topics"} />
        )}
        {activeTab === "visualization" && (
          <VisualizationTab dataset={dataset} mode={mode} shouldLoad={activeTab === "visualization"} />
        )}
        {activeTab === "metrics" && (
          <MetricsTab dataset={dataset} mode={mode} shouldLoad={activeTab === "metrics"} />
        )}
        {activeTab === "export" && (
          <ExportTab dataset={dataset} mode={mode} />
        )}
      </main>
    </div>
  )
}

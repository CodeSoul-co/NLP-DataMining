"use client"

import { useState, useEffect } from "react"
import { Loader2, Image, Cloud, ChevronDown } from "lucide-react"
import { ETMAgentAPI } from "@/lib/api/etm-agent"

/** 图表文件名 → 展示名与分组 */
const VIZ_GROUP_MAP: Record<string, { displayName: string; group: string }> = {
  topic_table: { displayName: "主题词分布", group: "全局概览" },
  topic_network: { displayName: "主题关联网络", group: "全局概览" },
  doc_topic_umap: { displayName: "文档聚类分布 (UMAP)", group: "全局概览" },
  cluster_heatmap: { displayName: "主题-文档热力图", group: "全局概览" },
  topic_proportion: { displayName: "各主题占比", group: "全局概览" },
  topic_wordclouds: { displayName: "主题词云", group: "词汇分析" },
  topic_similarity: { displayName: "主题相似度矩阵", group: "词汇分析" },
  word_importance: { displayName: "词权重", group: "词汇分析" },
  topic_coherence: { displayName: "各主题NPMI连贯性", group: "评估分析" },
  topic_exclusivity: { displayName: "各主题排他性", group: "评估分析" },
  metrics: { displayName: "7指标雷达图", group: "评估分析" },
  training_loss: { displayName: "训练损失曲线", group: "训练过程" },
  pyldavis_interactive: { displayName: "LDAvis交互探索", group: "交互式" },
}

function getVizMeta(filename: string): { displayName: string; group: string } {
  const base = filename.replace(/\.[^.]+$/, "").replace(/\./g, "_")
  for (const [key, meta] of Object.entries(VIZ_GROUP_MAP)) {
    if (base.includes(key) || filename.includes(key)) return meta
  }
  return { displayName: filename, group: "其他" }
}

type OssVizItem   = { name: string; type: string; url: string; source: "oss" }
type VizItem = OssVizItem

/** 从 OSS key/path 中提取 visualization/ 之后的子路径 */
function extractVizSubpath(ossPath: string): string {
  const idx = ossPath.indexOf("visualization/")
  if (idx >= 0) return ossPath.slice(idx + "visualization/".length)
  return ossPath.split("/").pop() || ossPath
}


function VizGrid({ items, imageUrl }: { items: VizItem[]; imageUrl: (item: VizItem) => string }) {
  const groups = items.reduce<Record<string, VizItem[]>>((acc, v) => {
    const filename = v.name.split("/").pop() || v.name
    const { group } = getVizMeta(filename)
    if (!acc[group]) acc[group] = []
    acc[group].push(v)
    return acc
  }, {})

  return (
    <div className="space-y-8">
      {Object.entries(groups).map(([groupName, groupItems]) => (
        <div key={groupName}>
          <h4 className="text-sm font-semibold text-slate-700 mb-3">{groupName}</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {groupItems.map(v => {
              const filename = v.name.split("/").pop() || v.name
              const { displayName } = getVizMeta(filename)
              const isHtml = v.type === "html" || v.name.endsWith(".html")
              const url = imageUrl(v)
              return (
                <div key={v.source + ":" + v.name} className="rounded-xl border border-slate-200 bg-white overflow-hidden shadow-sm">
                  <div className="flex items-center gap-1.5 px-3 py-2 bg-slate-50">
                    <Cloud className="w-3 h-3 text-blue-400 shrink-0" />
                    <p className="text-sm font-medium text-slate-700 truncate" title={v.name}>
                      {displayName}
                    </p>
                  </div>
                  <div className="aspect-video bg-slate-100 flex items-center justify-center min-h-[200px]">
                    {isHtml ? (
                      <iframe src={url} title={displayName} className="w-full min-h-[600px] border-0" loading="lazy" />
                    ) : (
                      <img src={url} alt={displayName} className="w-full h-auto object-contain max-h-[400px]" loading="lazy" />
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

interface VisualizationTabProps {
  dataset: string
  mode: string
  /** 懒加载：只有为 true 时才请求 */
  shouldLoad: boolean
}

export function VisualizationTab({ dataset, mode, shouldLoad }: VisualizationTabProps) {
  const [ossViz, setOssViz]                 = useState<OssVizItem[]>([])
  const [ossLoading, setOssLoading]         = useState(false)

  // OSS 数据集选择器状态
  const [ossDatasets, setOssDatasets]       = useState<Array<{ name: string; chart_count: number }>>([])
  const [selectedOssDataset, setSelectedOssDataset] = useState<string>(dataset)
  const [showPicker, setShowPicker]         = useState(false)

  // 初始化时先尝试当前 dataset，如无结果则加载可用 OSS 数据集列表
  useEffect(() => {
    if (!shouldLoad || !dataset) return
    setOssLoading(true)
    setSelectedOssDataset(dataset)

    ETMAgentAPI.listOssChartFiles(dataset).then(res => {
      if (res.total > 0) {
        setOssViz(buildOssItems(res.charts))
        setShowPicker(false)
      } else {
        setOssViz([])
        setShowPicker(true)
        ETMAgentAPI.listOssDatasets().then(r => setOssDatasets(r.datasets || []))
      }
    }).finally(() => setOssLoading(false))
  }, [dataset, shouldLoad])

  // 切换 OSS 数据集时重新加载
  const handleOssDatasetChange = (name: string) => {
    setSelectedOssDataset(name)
    setOssLoading(true)
    ETMAgentAPI.listOssChartFiles(name).then(res => {
      setOssViz(buildOssItems(res.charts))
    }).catch(() => setOssViz([]))
    .finally(() => setOssLoading(false))
  }

  const buildOssItems = (charts: any[]): OssVizItem[] => {
    const seen = new Map<string, OssVizItem>()
    for (const c of charts) {
      const subpath = extractVizSubpath(c.path || c.key)
      if (!seen.has(subpath)) {
        seen.set(subpath, {
          name: subpath,
          type: c.ext === "html" ? "html" : "image",
          url: c.url,
          source: "oss" as const,
        })
      }
    }
    return [...seen.values()]
  }

  const ossImageUrl = (item: VizItem) => (item as OssVizItem).url
  const hasOss = ossViz.length > 0

  if (!shouldLoad) {
    return <p className="text-sm text-slate-500 py-8 text-center">切换到可视化 Tab 将加载图表列表</p>
  }
  if (ossLoading) {
    return (
      <div className="flex items-center justify-center py-12 gap-2">
        <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
        <span className="text-slate-500">加载图表...</span>
      </div>
    )
  }

  return (
    <div className="space-y-10">
      {/* OSS 云端图表 */}
      <section>
        <div className="flex items-center gap-3 mb-4 flex-wrap">
          <Cloud className="w-4 h-4 text-blue-500 shrink-0" />
          <h3 className="text-base font-semibold text-slate-800">云端图表 (OSS)</h3>
          {hasOss && <span className="text-xs text-slate-400">({ossViz.length} 个文件)</span>}

          {/* 数据集切换器 */}
          {(showPicker || ossDatasets.length > 0) && (
            <div className="relative ml-auto">
              <select
                className="text-sm border border-slate-300 rounded-lg px-3 py-1.5 bg-white text-slate-700 appearance-none pr-8 cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-400"
                value={selectedOssDataset}
                onChange={e => handleOssDatasetChange(e.target.value)}
              >
                {ossDatasets.length === 0 && (
                  <option value={selectedOssDataset}>{selectedOssDataset}</option>
                )}
                {ossDatasets.map(d => (
                  <option key={d.name} value={d.name}>
                    {d.name}（{d.chart_count} 张图）
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400 pointer-events-none" />
            </div>
          )}
        </div>

        {ossLoading ? (
          <div className="flex items-center gap-2 py-4 text-slate-500">
            <Loader2 className="w-4 h-4 animate-spin" /><span className="text-sm">加载云端图表...</span>
          </div>
        ) : hasOss ? (
          <VizGrid items={ossViz} imageUrl={ossImageUrl} />
        ) : (
          <div className="text-center py-8 text-slate-400 text-sm">
            <Cloud className="w-8 h-8 mx-auto mb-2 opacity-40" />
            {showPicker
              ? ossDatasets.length > 0
                ? "请从上方选择一个 OSS 数据集查看云端图表"
                : "OSS 上暂未找到可用的数据集"
              : "云端暂无图表"
            }
          </div>
        )}
      </section>
    </div>
  )
}


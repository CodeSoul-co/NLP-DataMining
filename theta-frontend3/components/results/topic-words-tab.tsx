"use client"

import { useEffect, useState } from "react"
import { ETMAgentAPI } from "@/lib/api/etm-agent"
import { Badge } from "@/components/ui/badge"
import { ChevronDown, ChevronUp, Loader2, AlertCircle } from "lucide-react"

interface TopicWordsTabProps {
  dataset: string
  mode: string
  shouldLoad: boolean
}

export function TopicWordsTab({ dataset, mode, shouldLoad }: TopicWordsTabProps) {
  const [topicWords, setTopicWords] = useState<Record<string, string[]> | null>(null)
  const [loading, setLoading] = useState(false)
  const [expandedTopics, setExpandedTopics] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (!shouldLoad) return
    setLoading(true)
    ETMAgentAPI.getTopicWords(dataset, mode, 20)
      .then((data) => setTopicWords(data))
      .catch(() => setTopicWords(null))
      .finally(() => setLoading(false))
  }, [dataset, mode, shouldLoad])

  const toggleExpand = (topic: string) => {
    setExpandedTopics((prev) => {
      const next = new Set(prev)
      if (next.has(topic)) next.delete(topic)
      else next.add(topic)
      return next
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 animate-spin text-slate-400 mr-2" />
        <span className="text-slate-500">加载主题词...</span>
      </div>
    )
  }

  if (!topicWords || Object.keys(topicWords).length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-2 text-slate-500">
        <AlertCircle className="w-8 h-8 text-amber-400" />
        <p className="text-sm">暂无主题词数据</p>
      </div>
    )
  }

  const entries = Object.entries(topicWords)

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-500 mb-4">共 {entries.length} 个主题</p>
      {entries.map(([topicKey, words], idx) => {
        const isExpanded = expandedTopics.has(topicKey)
        const displayWords = isExpanded ? words : words.slice(0, 10)
        const topicNum = topicKey.replace(/^topic_?/i, "") || String(idx)
        return (
          <div key={topicKey} className="rounded-xl border border-slate-200 bg-white overflow-hidden">
            <button
              onClick={() => toggleExpand(topicKey)}
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-50 transition-colors text-left"
            >
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs font-mono">
                  主题 {topicNum}
                </Badge>
                <span className="text-sm text-slate-600 truncate max-w-[240px]">
                  {words.slice(0, 5).join(" · ")}
                </span>
              </div>
              {isExpanded ? (
                <ChevronUp className="w-4 h-4 text-slate-400 shrink-0" />
              ) : (
                <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
              )}
            </button>
            {isExpanded && (
              <div className="px-4 pb-4 pt-1 border-t border-slate-100">
                <div className="flex flex-wrap gap-2">
                  {displayWords.map((word, wIdx) => (
                    <span
                      key={word + wIdx}
                      className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 font-medium"
                      style={{ opacity: Math.max(0.4, 1 - wIdx * 0.04) }}
                    >
                      {word}
                    </span>
                  ))}
                  {!isExpanded && words.length > 10 && (
                    <span className="text-xs text-slate-400">+{words.length - 10} 更多</span>
                  )}
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

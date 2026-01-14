"use client"

import { useEffect, useState } from "react"

const logs = [
  "$ python train.py --model etm",
  "Loading Qwen embeddings...",
  "Dataset: 10,000 documents",
  "Starting epoch 1/50...",
]

export function Terminal() {
  const [visibleLogs, setVisibleLogs] = useState<string[]>([])

  useEffect(() => {
    logs.forEach((log, i) => {
      setTimeout(() => {
        setVisibleLogs((prev) => [...prev, log])
      }, i * 800)
    })
  }, [])

  return (
    <div className="bg-black rounded-lg p-3 h-28 overflow-y-auto font-mono text-[11px]">
      {visibleLogs.map((log, i) => (
        <div key={i} className={`mb-1 ${i === 0 ? "text-white" : "text-green-400"}`}>
          {log}
        </div>
      ))}
      <div className="text-green-400 animate-pulse inline-block">█</div>
    </div>
  )
}

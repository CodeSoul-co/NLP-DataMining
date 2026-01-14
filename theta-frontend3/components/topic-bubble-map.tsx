"use client"

import { motion } from "framer-motion"

const topics = [
  { id: 1, label: "Tech", x: 30, y: 30, size: 60 },
  { id: 2, label: "Sports", x: 70, y: 25, size: 45 },
  { id: 3, label: "Politics", x: 50, y: 60, size: 55 },
  { id: 4, label: "Health", x: 20, y: 70, size: 40 },
  { id: 5, label: "Business", x: 75, y: 70, size: 50 },
]

export function TopicBubbleMap() {
  return (
    <div className="relative w-full h-48 bg-slate-950 rounded-lg overflow-hidden">
      {topics.map((topic) => (
        <motion.div
          key={topic.id}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: topic.id * 0.1 }}
          className="absolute rounded-full bg-blue-500/20 border border-blue-400/30 flex items-center justify-center cursor-pointer hover:bg-blue-500/30 transition-colors"
          style={{
            left: `${topic.x}%`,
            top: `${topic.y}%`,
            width: topic.size,
            height: topic.size,
            transform: "translate(-50%, -50%)",
          }}
        >
          <span className="text-xs font-semibold text-blue-300">{topic.label}</span>
        </motion.div>
      ))}
    </div>
  )
}

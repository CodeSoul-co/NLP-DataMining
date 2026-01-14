"use client"

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts"

const data = [
  { epoch: 1, total: 4.5, xl: 2.8, recon: 1.7 },
  { epoch: 2, total: 3.8, xl: 2.3, recon: 1.5 },
  { epoch: 3, total: 3.2, xl: 1.9, recon: 1.3 },
  { epoch: 4, total: 2.7, xl: 1.6, recon: 1.1 },
  { epoch: 5, total: 2.3, xl: 1.4, recon: 0.9 },
]

export function TrainingChart() {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <XAxis
          dataKey="epoch"
          stroke="#475569"
          tick={{ fill: "#64748b", fontSize: 11 }}
          axisLine={{ stroke: "#334155" }}
        />
        <YAxis stroke="#475569" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={{ stroke: "#334155" }} />
        <Tooltip
          contentStyle={{
            backgroundColor: "rgba(15, 23, 42, 0.95)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            borderRadius: "8px",
            backdropFilter: "blur(8px)",
          }}
          labelStyle={{ color: "#cbd5e1", fontSize: 11 }}
          itemStyle={{ fontSize: 11 }}
        />
        <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 11 }} iconSize={10} />
        <Line type="monotone" dataKey="total" stroke="#06b6d4" strokeWidth={1.5} name="Total Loss" dot={false} />
        <Line type="monotone" dataKey="xl" stroke="#ec4899" strokeWidth={1.5} name="XL Loss" dot={false} />
        <Line type="monotone" dataKey="recon" stroke="#facc15" strokeWidth={1.5} name="Recon Loss" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

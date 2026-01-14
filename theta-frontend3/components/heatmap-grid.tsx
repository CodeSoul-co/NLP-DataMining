"use client"

export function HeatmapGrid({ title }: { title: string }) {
  const size = 8
  const grid = Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => Math.random()))

  const getColor = (value: number) => {
    const intensity = Math.floor(value * 255)
    return `rgb(59, 130, ${100 + intensity})`
  }

  return (
    <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${size}, 1fr)` }}>
      {grid.flat().map((value, i) => (
        <div
          key={i}
          className="aspect-square rounded-sm"
          style={{ backgroundColor: getColor(value) }}
          title={value.toFixed(3)}
        />
      ))}
    </div>
  )
}

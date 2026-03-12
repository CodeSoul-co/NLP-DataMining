"use client"

import { useState, useEffect } from "react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Loader2 } from "lucide-react"
import { ETMAgentAPI } from "@/lib/api/etm-agent"

export interface ColumnSelection {
  textColumn: string
  metaColumns: string[]
  removeUrl: boolean
  removeHtml: boolean
  removePunctuation: boolean
  removeStopwords: boolean
  removeSpecialChars: boolean
  normalizeWhitespace: boolean
  minWordCount: number
}

interface ColumnSelectPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: (selection: ColumnSelection) => void
  onSkip?: () => void
  datasetName: string
  /** OSS 直传后文件在 data/{job_id}/，传此参数才能正确预览 */
  jobId?: string | null
}

export function ColumnSelectPanel({
  open,
  onOpenChange,
  onConfirm,
  onSkip,
  datasetName,
  jobId,
}: ColumnSelectPanelProps) {
  const [columns, setColumns] = useState<string[]>([])
  const [rows, setRows] = useState<string[][]>([])
  const [loading, setLoading] = useState(false)
  const [textColumn, setTextColumn] = useState("")
  const [metaColumns, setMetaColumns] = useState<string[]>([])
  const [removeUrl, setRemoveUrl] = useState(true)
  const [removeHtml, setRemoveHtml] = useState(true)
  const [removePunctuation, setRemovePunctuation] = useState(true)
  const [removeStopwords, setRemoveStopwords] = useState(true)
  const [removeSpecialChars, setRemoveSpecialChars] = useState(true)
  const [normalizeWhitespace, setNormalizeWhitespace] = useState(true)
  const [minWordCount, setMinWordCount] = useState(3)

  useEffect(() => {
    if (!open || !datasetName) return
    setLoading(true)
    ETMAgentAPI.getDatasetPreview(datasetName, jobId ?? undefined)
      .then(({ columns: c, rows: r }) => {
        setColumns(c)
        setRows(r)
        if (c.length > 0 && !textColumn) {
          const preferred = c.find(col =>
            ["text", "content", "cleaned_content", "body", "message"].includes(col.toLowerCase())
          )
          setTextColumn(preferred || c[0])
        }
      })
      .catch(() => setColumns([]))
      .finally(() => setLoading(false))
  }, [open, datasetName, jobId])

  const handleConfirm = () => {
    if (!textColumn) return
    onConfirm({
      textColumn,
      metaColumns,
      removeUrl,
      removeHtml,
      removePunctuation,
      removeStopwords,
      removeSpecialChars,
      normalizeWhitespace,
      minWordCount,
    })
    onOpenChange(false)
  }

  const toggleMeta = (col: string) => {
    setMetaColumns(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    )
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>列选择与清洗</DialogTitle>
          <DialogDescription>
            选择文本列和可选元数据列，配置清洗选项。数据集：{datasetName}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-12 gap-2">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>加载预览...</span>
          </div>
        ) : (
          <div className="space-y-6 py-4">
            <div>
              <Label>文本列（必选）</Label>
              <RadioGroup value={textColumn} onValueChange={setTextColumn} className="flex flex-wrap gap-3 mt-2">
                {columns.map(col => (
                  <div key={col} className="flex items-center space-x-2">
                    <RadioGroupItem value={col} id={`text-${col}`} />
                    <Label htmlFor={`text-${col}`} className="font-normal cursor-pointer">{col}</Label>
                  </div>
                ))}
              </RadioGroup>
            </div>

            {columns.length > 1 && (
              <div>
                <Label>标签/元数据列（可选，用于 STM 协变量或 DTM 时间列）</Label>
                <div className="flex flex-wrap gap-3 mt-2">
                  {columns.filter(c => c !== textColumn).map(col => (
                    <div key={col} className="flex items-center space-x-2">
                      <Checkbox
                        id={`meta-${col}`}
                        checked={metaColumns.includes(col)}
                        onCheckedChange={() => toggleMeta(col)}
                      />
                      <Label htmlFor={`meta-${col}`} className="font-normal cursor-pointer">{col}</Label>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {rows.length > 0 && (
              <div>
                <Label>前 5 行预览</Label>
                <ScrollArea className="h-32 mt-2 rounded-lg border p-2">
                  <table className="text-xs w-full">
                    <thead>
                      <tr className="border-b">
                        {columns.map(c => (
                          <th key={c} className="text-left p-2 font-medium">{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, i) => (
                        <tr key={i} className="border-b last:border-0">
                          {row.map((cell, j) => (
                            <td key={j} className="p-2 max-w-[200px] truncate" title={String(cell)}>
                              {String(cell)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </ScrollArea>
              </div>
            )}

            <div>
              <Label>清洗选项</Label>
              <div className="grid grid-cols-2 gap-2 mt-2">
                <div className="flex items-center space-x-2">
                  <Checkbox id="removeUrl" checked={removeUrl} onCheckedChange={v => setRemoveUrl(!!v)} />
                  <Label htmlFor="removeUrl" className="font-normal">删除 URL</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="removeHtml" checked={removeHtml} onCheckedChange={v => setRemoveHtml(!!v)} />
                  <Label htmlFor="removeHtml" className="font-normal">删除 HTML 标签</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="removePunc" checked={removePunctuation} onCheckedChange={v => setRemovePunctuation(!!v)} />
                  <Label htmlFor="removePunc" className="font-normal">删除标点</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="removeStop" checked={removeStopwords} onCheckedChange={v => setRemoveStopwords(!!v)} />
                  <Label htmlFor="removeStop" className="font-normal">删除停用词</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="removeSpecial" checked={removeSpecialChars} onCheckedChange={v => setRemoveSpecialChars(!!v)} />
                  <Label htmlFor="removeSpecial" className="font-normal">删除特殊字符</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox id="normSpace" checked={normalizeWhitespace} onCheckedChange={v => setNormalizeWhitespace(!!v)} />
                  <Label htmlFor="normSpace" className="font-normal">规范化空白</Label>
                </div>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <Label htmlFor="minWords" className="font-normal">最小词数</Label>
                <Input
                  id="minWords"
                  type="number"
                  min={1}
                  max={20}
                  value={minWordCount}
                  onChange={e => setMinWordCount(parseInt(e.target.value) || 3)}
                  className="w-20"
                />
              </div>
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => { onSkip?.(); onOpenChange(false); }}>跳过</Button>
          <Button onClick={handleConfirm} disabled={!textColumn} className="bg-blue-600">
            确认并继续
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

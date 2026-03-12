"use client"

import { useState } from "react"
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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Info } from "lucide-react"
import { cn } from "@/lib/utils"

// ==================== 模型配置 ====================

const LANGUAGES = [
  { value: "zh", label: "中文" },
  { value: "en", label: "English" },
  { value: "de", label: "Deutsch" },
  { value: "es", label: "Español" },
] as const

const MODEL_GROUPS = [
  {
    label: "神经",
    models: [
      { id: "theta", name: "THETA（推荐）", type: "neural" as const },
      { id: "nvdm", name: "NVDM", type: "neural" as const },
      { id: "gsm", name: "GSM", type: "neural" as const },
      { id: "prodlda", name: "ProdLDA", type: "neural" as const },
      { id: "ctm", name: "CTM", type: "neural" as const },
      { id: "etm", name: "ETM", type: "neural" as const },
      { id: "dtm", name: "DTM", type: "neural" as const },
      { id: "bertopic", name: "BERTopic", type: "neural" as const },
    ],
  },
  {
    label: "传统",
    models: [
      { id: "lda", name: "LDA", type: "traditional" as const },
      { id: "hdp", name: "HDP", type: "traditional" as const },
      { id: "stm", name: "STM", type: "traditional" as const },
      { id: "btm", name: "BTM", type: "traditional" as const },
    ],
  },
] as const

const QWEN_SIZES = [
  { value: "0.6B", label: "0.6B（默认）" },
  { value: "4B", label: "4B" },
  { value: "8B", label: "8B" },
] as const

const EMBEDDING_MODES = [
  { value: "zero_shot", label: "Zero-shot（默认）", tip: "不训练嵌入，最快" },
  { value: "unsupervised", label: "Unsupervised", tip: "无监督嵌入" },
  { value: "supervised", label: "Supervised", tip: "需额外指定标签列" },
] as const

// ==================== 类型 ====================

export interface AnalysisConfig {
  language: string
  models: string[]  // 可多选，支持 THETA 与基线模型组合
  modelSize: string
  mode: "zero_shot" | "unsupervised" | "supervised"
  numTopics: number
  epochs: number
  batchSize: number
  learningRate: number
  hiddenDim: number
  patience: number
  vocabSize: number
  klStart?: number
  klEnd?: number
  klWarmup?: number
  maxIter?: number
  nIter?: number
  dropout?: number
}

const DEFAULT_CONFIG: AnalysisConfig = {
  language: "zh",
  models: ["theta"],
  modelSize: "0.6B",
  mode: "zero_shot",
  numTopics: 20,
  epochs: 100,
  batchSize: 64,
  learningRate: 0.002,
  hiddenDim: 512,
  patience: 10,
  vocabSize: 5000,
  klStart: 0,
  klEnd: 1,
  klWarmup: 50,
}

// ==================== 组件 ====================

interface AnalysisConfigPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: (config: AnalysisConfig) => void
  datasetName: string
}

export function AnalysisConfigPanel({
  open,
  onOpenChange,
  onConfirm,
  datasetName,
}: AnalysisConfigPanelProps) {
  const [config, setConfig] = useState<AnalysisConfig>({ ...DEFAULT_CONFIG })

  const isTheta = config.models.includes("theta")
  const hasHdpOrBertopic =
    config.models.includes("hdp") || config.models.includes("bertopic")

  const handleModelToggle = (modelId: string, checked: boolean) => {
    setConfig(prev => {
      const current = new Set(prev.models)
      if (checked) current.add(modelId)
      else current.delete(modelId)
      const next = Array.from(current)
      return { ...prev, models: next.length ? next : ["theta"] }
    })
  }

  const handleConfirm = () => {
    if (config.models.length === 0) {
      setConfig(prev => ({ ...prev, models: ["theta"] }))
    }
    onConfirm(config)
    onOpenChange(false)
  }

  return (
    <TooltipProvider>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>分析配置</DialogTitle>
            <DialogDescription>
              上传完成，请配置分析参数后点击「开始分析」。数据集：{datasetName}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6 py-4">
            {/* 第一行：数据语言 */}
            <div className="space-y-2">
              <Label>数据语言</Label>
              <RadioGroup
                value={config.language}
                onValueChange={v => setConfig(prev => ({ ...prev, language: v }))}
                className="flex flex-wrap gap-4"
              >
                {LANGUAGES.map(lang => (
                  <div key={lang.value} className="flex items-center space-x-2">
                    <RadioGroupItem value={lang.value} id={`lang-${lang.value}`} />
                    <Label htmlFor={`lang-${lang.value}`} className="font-normal cursor-pointer">
                      {lang.label}
                    </Label>
                  </div>
                ))}
              </RadioGroup>
            </div>

            {/* 第二行：模型选择 */}
            <div className="space-y-3">
              <Label>模型选择</Label>
              <div className="space-y-4">
                {MODEL_GROUPS.map(group => (
                  <div key={group.label}>
                    <p className="text-xs text-slate-500 mb-2">{group.label}</p>
                    <div className="flex flex-wrap gap-3">
                      {group.models.map(m => (
                        <div key={m.id} className="flex items-center space-x-2">
                          <Checkbox
                            id={`model-${m.id}`}
                            checked={config.models.includes(m.id)}
                            onCheckedChange={checked =>
                              handleModelToggle(m.id, !!checked)
                            }
                          />
                          <Label
                            htmlFor={`model-${m.id}`}
                            className="font-normal cursor-pointer text-sm"
                          >
                            {m.name}
                          </Label>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 第三行：THETA 专属 */}
            {isTheta && (
              <div className="space-y-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
                <Label className="text-blue-700">THETA 专属</Label>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-sm">Qwen 尺寸</Label>
                    <RadioGroup
                      value={config.modelSize}
                      onValueChange={v =>
                        setConfig(prev => ({ ...prev, modelSize: v }))
                      }
                      className="flex gap-3"
                    >
                      {QWEN_SIZES.map(s => (
                        <div key={s.value} className="flex items-center space-x-2">
                          <RadioGroupItem value={s.value} id={`size-${s.value}`} />
                          <Label htmlFor={`size-${s.value}`} className="font-normal text-sm">
                            {s.label}
                          </Label>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-sm">嵌入模式</Label>
                    <RadioGroup
                      value={config.mode}
                      onValueChange={v =>
                        setConfig(prev => ({ ...prev, mode: v as typeof prev.mode }))
                      }
                      className="flex flex-wrap gap-3"
                    >
                      {EMBEDDING_MODES.map(mode => (
                        <Tooltip key={mode.value}>
                          <TooltipTrigger asChild>
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value={mode.value} id={`mode-${mode.value}`} />
                              <Label htmlFor={`mode-${mode.value}`} className="font-normal text-sm cursor-pointer">
                                {mode.label}
                              </Label>
                              <Info className="w-3.5 h-3.5 text-slate-400" />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>{mode.tip}</TooltipContent>
                        </Tooltip>
                      ))}
                    </RadioGroup>
                  </div>
                </div>
              </div>
            )}

            {/* 第四行：超参数 */}
            <div className="space-y-4">
              <Label>超参数</Label>
              <div className="grid grid-cols-2 gap-4">
                {/* 主题数 - HDP/BERTopic 时置灰 */}
                <div className={cn("space-y-2", hasHdpOrBertopic && "opacity-60")}>
                  <div className="flex justify-between">
                    <Label className="text-sm">主题数</Label>
                    <span className="text-xs text-slate-500">5–100</span>
                  </div>
                  <Slider
                    value={[config.numTopics]}
                    onValueChange={([v]) =>
                      setConfig(prev => ({ ...prev, numTopics: v }))
                    }
                    min={5}
                    max={100}
                    step={1}
                    disabled={hasHdpOrBertopic}
                  />
                  <p className="text-xs text-slate-500">{config.numTopics}</p>
                </div>

                {/* 训练轮数 - 仅神经模型 */}
                {config.models.some(m =>
                  ["theta", "nvdm", "gsm", "prodlda", "ctm", "etm", "dtm"].includes(m)
                ) && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label className="text-sm">训练轮数</Label>
                      <span className="text-xs text-slate-500">10–500</span>
                    </div>
                    <Input
                      type="number"
                      min={10}
                      max={500}
                      value={config.epochs}
                      onChange={e =>
                        setConfig(prev => ({
                          ...prev,
                          epochs: parseInt(e.target.value) || 100,
                        }))
                      }
                    />
                  </div>
                )}

                {/* 批大小 - 仅神经模型 */}
                {config.models.some(m =>
                  ["theta", "nvdm", "gsm", "prodlda", "ctm", "etm", "dtm"].includes(m)
                ) && (
                  <div className="space-y-2">
                    <Label className="text-sm">批大小</Label>
                    <Input
                      type="number"
                      min={8}
                      max={512}
                      value={config.batchSize}
                      onChange={e =>
                        setConfig(prev => ({
                          ...prev,
                          batchSize: parseInt(e.target.value) || 64,
                        }))
                      }
                    />
                  </div>
                )}

                {/* 学习率 */}
                {config.models.some(m =>
                  ["theta", "nvdm", "gsm", "prodlda", "ctm", "etm", "dtm"].includes(m)
                ) && (
                  <div className="space-y-2">
                    <Label className="text-sm">学习率</Label>
                    <Input
                      type="number"
                      min={1e-5}
                      max={0.1}
                      step={0.001}
                      value={config.learningRate}
                      onChange={e =>
                        setConfig(prev => ({
                          ...prev,
                          learningRate: parseFloat(e.target.value) || 0.002,
                        }))
                      }
                    />
                  </div>
                )}

                {/* 隐藏层维度 */}
                {config.models.some(m =>
                  ["theta", "nvdm", "gsm", "prodlda", "ctm", "etm", "dtm"].includes(m)
                ) && (
                  <div className="space-y-2">
                    <Label className="text-sm">隐藏层维度</Label>
                    <Input
                      type="number"
                      min={128}
                      max={1024}
                      value={config.hiddenDim}
                      onChange={e =>
                        setConfig(prev => ({
                          ...prev,
                          hiddenDim: parseInt(e.target.value) || 512,
                        }))
                      }
                    />
                  </div>
                )}

                {/* 词汇表大小 - 所有模型 */}
                <div className="space-y-2">
                  <Label className="text-sm">词汇表大小</Label>
                  <Input
                    type="number"
                    min={1000}
                    max={20000}
                    value={config.vocabSize}
                    onChange={e =>
                      setConfig(prev => ({
                        ...prev,
                        vocabSize: parseInt(e.target.value) || 5000,
                      }))
                    }
                  />
                </div>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              取消
            </Button>
            <Button
              onClick={handleConfirm}
              disabled={config.models.length === 0}
              className="bg-gradient-to-r from-blue-600 to-indigo-600"
            >
              开始分析
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  )
}

export type { AnalysisConfig }

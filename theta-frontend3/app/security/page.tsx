"use client"

import Link from "next/link"
import { Shield, Lock, Trash2, ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function SecurityPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-slate-50 to-blue-50/20">
      <header className="border-b border-slate-200/80 bg-white/90 backdrop-blur px-5 sm:px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 text-slate-600 hover:text-slate-900">
            <ArrowLeft className="w-4 h-4" />
            返回首页
          </Link>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-5 sm:px-6 py-12 sm:py-16">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center">
            <Shield className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900">安全白皮书</h1>
            <p className="text-slate-500 text-sm mt-0.5">加密技术与数据销毁机制</p>
          </div>
        </div>

        <div className="space-y-8 text-slate-600 leading-relaxed">
          <section>
            <h2 className="text-lg font-semibold text-slate-900 mb-3 flex items-center gap-2">
              <Lock className="w-5 h-5 text-blue-500" />
              数据加密
            </h2>
            <p>
              我们遵循严格的数据隐私协议（GDPR Compliant）。您上传的数据在传输与存储过程中均经过加密处理，仅供您当次分析使用。我们承诺不会将您的私有数据用于训练公共模型或分享给第三方。
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-slate-900 mb-3 flex items-center gap-2">
              <Trash2 className="w-5 h-5 text-blue-500" />
              数据销毁机制
            </h2>
            <p>
              分析结束后，您可随时在平台内一键销毁本次分析所涉数据。销毁后数据不可恢复，确保研究数据在您控制范围内。
            </p>
          </section>

          <p className="text-sm text-slate-500 border-t border-slate-200 pt-6">
            本文档为安全白皮书概要。详细技术说明与合规细节将随产品迭代持续更新。如有疑问请联系：duanzhenke@code-soul.com
          </p>
        </div>

        <div className="mt-10">
          <Button asChild variant="outline" className="border-slate-200">
            <Link href="/">返回首页</Link>
          </Button>
        </div>
      </main>
    </div>
  )
}

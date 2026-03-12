"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Crown, Star, Sparkles, Zap, Heart, Shield, Gem, Globe, Github, Mail, Code2 } from "lucide-react"
import Link from "next/link"

// 团队成员数据
interface TeamMember {
  id: number
  name: string
  nameEn: string
  role: string
  team: string
  group: "dev" | "design" | "research" | "product" | "ops"
  rarity: "common" | "rare" | "epic" | "legendary"
  skills: string[]
  quote?: string
  // 新增字段
  avatar?: string // 自定义头像路径
  bio?: string // 个人简介
  personality?: string // MBTI 人格
  birthYear?: number // 出生年份
  website?: string // 个人网站
  github?: string // GitHub
  email?: string // 邮箱
  idol?: string // 偶像
  motto?: string // 座右铭
  highlights?: string[] // 亮点标签
}

const TEAM_MEMBERS: TeamMember[] = [
  {
    id: 1,
    name: "段圳科",
    nameEn: "DUAN ZHENKE",
    role: "CEO & 创始人",
    team: "CORE TEAM",
    group: "dev",
    rarity: "legendary",
    skills: ["PyTorch", "LangChain", "vLLM", "Agent", "NLP"],
    avatar: "/avatars/duanshenke.jpg",
    bio: "THETA 算法工程师，应用统计在读博士生，品种为英短蓝白。平均训练百个模型消耗为一碗猪脚饭（加剁椒）。",
    motto: "Fake it until you make it",
    highlights: ["创始人", "NLP 高手", "代码爱好者"],
    github: "erwinmsmith",
  },
  {
    id: 2,
    name: "李国正",
    nameEn: "LI GUOZHENG",
    role: "Web 全栈工程师",
    team: "DEV TEAM",
    group: "dev",
    rarity: "legendary",
    skills: ["React", "Next.js", "TypeScript", "Node.js", "Python"],
    quote: "Stay hungry, stay foolish.",
    avatar: "/avatars/liguozheng.jpg",
    bio: "THETA Web 全栈工程师，以乔布斯为偶像，怀揣改变世界的初心。热爱技术，追求极致的用户体验和代码美学。",
    personality: "ENTP",
    birthYear: 2006,
    website: "liguozheng.site",
    idol: "Steve Jobs",
    motto: "Think different, code different.",
    highlights: ["00后开发者", "全栈能手", "产品思维"],
  },
  {
    id: 3,
    name: "吴凡",
    nameEn: "WU FAN",
    role: "COO",
    team: "CORE TEAM",
    group: "product",
    rarity: "epic",
    skills: ["战略规划", "商业落地", "执行力"],
    avatar: "/avatars/wufan.jpg",
    bio: "可爱但凶猛的狸花猫，在不确定中寻找商业确定性，追求懂战略更懂落地，以绝对执行力驱动战略变现。",
    highlights: ["COO", "战略执行", "商业落地"],
  },
  {
    id: 8,
    name: "潘吉群",
    nameEn: "PAN JIQUN",
    role: "算法工程师",
    team: "RESEARCH TEAM",
    group: "research",
    rarity: "rare",
    skills: ["算法研究", "模型训练"],
    avatar: "/avatars/panjiqun.png",
    bio: "算法小白，1个模型训练平均消耗100次睡眠。",
    highlights: ["算法小白", "睡眠消耗者"],
  },
  {
    id: 4,
    name: "李芳痕",
    nameEn: "LI FANGHENG",
    role: "算法工程师",
    team: "RESEARCH TEAM",
    group: "research",
    rarity: "epic",
    skills: ["NLP", "Agent", "算法调参"],
    avatar: "/avatars/lifangheng.jpg",
    bio: "THETA 算法工程师，计算机科学与技术在读本科生，橘猫（翘脚版）。",
    motto: "Fail with enthusiasm",
    highlights: ["代码爱好者", "NLP 探索者", "Agent Builder", "算法调参幸存者"],
  },
  {
    id: 5,
    name: "罗屹",
    nameEn: "LUO YI",
    role: "算法工程师",
    team: "RESEARCH TEAM",
    group: "research",
    rarity: "rare",
    skills: ["算法研究"],
    avatar: "/avatars/luoyi.jpg",
    bio: "一只特立独行的人。",
    highlights: ["特立独行"],
  },
  {
    id: 6,
    name: "栗昕",
    nameEn: "LI XIN",
    role: "算法工程师",
    team: "RESEARCH TEAM",
    group: "research",
    rarity: "epic",
    skills: ["Python", "PyTorch", "TensorFlow", "CUDA"],
    avatar: "/avatars/lixin.jpg",
    bio: "THETA 算法工程师，以简洁为美，在数据与模型中寻找规律。相信技术的力量，追求优雅的解决方案与可靠的工程实现。",
    personality: "INFJ",
    motto: "Less is more",
    highlights: ["算法研究", "工程落地"],
  },
  {
    id: 7,
    name: "丁小川",
    nameEn: "DING XIAOCHUAN",
    role: "算法工程师",
    team: "RESEARCH TEAM",
    group: "research",
    rarity: "rare",
    skills: ["算法研究", "交互探索"],
    avatar: "/avatars/dingxiaochuan.png",
    bio: "2005年生，21岁，寻找方向，坚定行走，在不断的交互中探知世界。",
    personality: "ENTJ",
    birthYear: 2005,
    highlights: ["年轻探索者", "交互学习"],
  },
]

// 根据分组获取背景渐变色
function getGroupGradient(group: TeamMember["group"]): string {
  const gradients = {
    dev: "from-blue-100 via-blue-50 to-indigo-100",
    design: "from-pink-100 via-rose-50 to-fuchsia-100",
    research: "from-emerald-100 via-teal-50 to-cyan-100",
    product: "from-amber-100 via-yellow-50 to-orange-100",
    ops: "from-slate-100 via-gray-50 to-zinc-100",
  }
  return gradients[group]
}

// 根据分组获取深色背景（用于背面）
function getGroupDarkGradient(group: TeamMember["group"]): string {
  const gradients = {
    dev: "from-blue-600 via-indigo-600 to-blue-700",
    design: "from-pink-500 via-rose-500 to-fuchsia-600",
    research: "from-emerald-600 via-teal-600 to-cyan-600",
    product: "from-amber-500 via-orange-500 to-amber-600",
    ops: "from-slate-600 via-gray-600 to-zinc-700",
  }
  return gradients[group]
}

// 根据稀有度获取边框样式
function getRarityBorder(rarity: TeamMember["rarity"]): string {
  const borders = {
    common: "border-slate-300",
    rare: "border-blue-400 shadow-blue-200/50",
    epic: "border-purple-500 shadow-purple-300/50",
    legendary: "border-amber-500 shadow-amber-300/50 ring-2 ring-amber-400/30",
  }
  return borders[rarity]
}

// 稀有度标签颜色
function getRarityBadge(rarity: TeamMember["rarity"]): { bg: string; text: string } {
  const badges = {
    common: { bg: "bg-slate-200", text: "text-slate-700" },
    rare: { bg: "bg-blue-500", text: "text-white" },
    epic: { bg: "bg-purple-600", text: "text-white" },
    legendary: { bg: "bg-gradient-to-r from-amber-500 to-orange-500", text: "text-white" },
  }
  return badges[rarity]
}

// 像素风装饰图标
function PixelDecor({ rarity, position }: { rarity: TeamMember["rarity"]; position: "tl" | "tr" | "bl" | "br" }) {
  const positionClass = {
    tl: "top-2 left-2",
    tr: "top-2 right-2",
    bl: "bottom-2 left-2",
    br: "bottom-2 right-2",
  }

  const icons = {
    legendary: <Crown className="w-4 h-4 text-amber-500 drop-shadow-sm" />,
    epic: <Gem className="w-3.5 h-3.5 text-purple-500 drop-shadow-sm" />,
    rare: <Star className="w-3 h-3 text-blue-500 drop-shadow-sm" />,
    common: <Sparkles className="w-3 h-3 text-slate-400" />,
  }

  if (position === "tr" && rarity === "legendary") {
    return (
      <div className={`absolute ${positionClass[position]} animate-bounce z-10`}>
        {icons[rarity]}
      </div>
    )
  }

  if (position === "tl" && (rarity === "epic" || rarity === "legendary")) {
    return (
      <div className={`absolute ${positionClass[position]} z-10`}>
        {rarity === "legendary" ? <Shield className="w-3.5 h-3.5 text-amber-400" /> : <Zap className="w-3 h-3 text-purple-400" />}
      </div>
    )
  }

  if (position === "br" && rarity !== "common") {
    return (
      <div className={`absolute ${positionClass[position]} opacity-60 z-10`}>
        <Heart className="w-3 h-3 text-rose-400" />
      </div>
    )
  }

  return null
}

// 团队成员卡片组件 - 翻转样式
function TeamCard({ member, index }: { member: TeamMember; index: number }) {
  const rarityBadge = getRarityBadge(member.rarity)
  const avatarSrc = member.avatar || `/avatars/cat-${member.id}.png`

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, rotateY: -10 }}
      animate={{ opacity: 1, y: 0, rotateY: 0 }}
      transition={{ duration: 0.5, delay: index * 0.08 }}
      className="team-card-flip h-[420px]"
    >
      <div className="team-card-flip-inner">
        {/* ========== 正面：头像 + 基本信息 ========== */}
        <div
          className={`team-card-front rounded-2xl border-2 ${getRarityBorder(member.rarity)} bg-white overflow-hidden shadow-lg hover:shadow-2xl transition-shadow duration-300`}
        >
          {/* 像素风装饰 */}
          <PixelDecor rarity={member.rarity} position="tl" />
          <PixelDecor rarity={member.rarity} position="tr" />
          <PixelDecor rarity={member.rarity} position="br" />

          {/* 头像区域 */}
          <div className={`aspect-square bg-gradient-to-br ${getGroupGradient(member.group)} flex items-center justify-center relative overflow-hidden`}>
            {/* 像素网格背景 */}
            <div
              className="absolute inset-0 opacity-[0.08]"
              style={{
                backgroundImage: `
                  linear-gradient(to right, currentColor 1px, transparent 1px),
                  linear-gradient(to bottom, currentColor 1px, transparent 1px)
                `,
                backgroundSize: "8px 8px",
              }}
            />

            {/* 头像 */}
            <div className="relative w-full h-full flex items-center justify-center p-4">
              <img
                src={avatarSrc}
                alt={member.name}
                className="w-full h-full object-cover rounded-xl shadow-lg"
                onError={(e) => {
                  const target = e.target as HTMLImageElement
                  target.style.display = "none"
                  target.nextElementSibling?.classList.remove("hidden")
                }}
              />
              {/* 占位符 */}
              <div className="hidden absolute inset-0 flex items-center justify-center">
                <div className="w-24 h-24 rounded-2xl bg-white/60 backdrop-blur flex items-center justify-center shadow-inner">
                  <span className="text-4xl">🐱</span>
                </div>
              </div>
            </div>

            {/* 稀有度标签 */}
            <div className={`absolute top-3 right-3 px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${rarityBadge.bg} ${rarityBadge.text} z-10`}>
              {member.rarity}
            </div>

            {/* 翻转提示 */}
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-black/40 backdrop-blur-sm rounded-full text-[10px] text-white/90 font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              悬停查看详情 ↻
            </div>
          </div>

          {/* 简要信息区域 */}
          <div className="p-4 bg-gradient-to-b from-white to-slate-50/50">
            <h3 className="font-black text-lg text-slate-900 tracking-wide uppercase mb-0.5">
              {member.nameEn}
            </h3>
            <p className="text-xs text-slate-500 mb-2">{member.name}</p>
            <p className="text-sm text-slate-600 font-medium mb-2">{member.role}</p>
            <Badge
              variant="outline"
              className="text-[10px] font-bold tracking-wider border-slate-300 text-slate-600 bg-slate-50/80"
            >
              {member.team}
            </Badge>
          </div>

          {/* 底部装饰线 */}
          <div
            className={`absolute bottom-0 left-0 right-0 h-1 ${
              member.rarity === "legendary"
                ? "bg-gradient-to-r from-amber-400 via-orange-400 to-amber-400"
                : member.rarity === "epic"
                ? "bg-gradient-to-r from-purple-400 via-fuchsia-400 to-purple-400"
                : member.rarity === "rare"
                ? "bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400"
                : "bg-slate-200"
            }`}
          />
        </div>

        {/* ========== 背面：详细信息 ========== */}
        <div
          className={`team-card-back rounded-2xl border-2 ${getRarityBorder(member.rarity)} overflow-hidden shadow-lg hover:shadow-2xl transition-shadow duration-300 bg-gradient-to-br ${getGroupDarkGradient(member.group)}`}
        >
          {/* 装饰背景 */}
          <div
            className="absolute inset-0 opacity-10"
            style={{
              backgroundImage: `
                linear-gradient(to right, white 1px, transparent 1px),
                linear-gradient(to bottom, white 1px, transparent 1px)
              `,
              backgroundSize: "12px 12px",
            }}
          />

          {/* 内容区域 */}
          <div className="relative h-full p-5 flex flex-col text-white">
            {/* 顶部：姓名和角色 */}
            <div className="mb-4">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="font-black text-xl tracking-wide uppercase">
                  {member.nameEn}
                </h3>
                {member.personality && (
                  <span className="px-2 py-0.5 bg-white/20 rounded text-[10px] font-bold">
                    {member.personality}
                  </span>
                )}
              </div>
              <p className="text-sm text-white/80">{member.role}</p>
              {member.birthYear && (
                <p className="text-xs text-white/60 mt-1">
                  {member.birthYear} 年生 · {new Date().getFullYear() - member.birthYear} 岁
                </p>
              )}
            </div>

            {/* 个人简介 */}
            {member.bio && (
              <div className="mb-4">
                <p className="text-sm text-white/90 leading-relaxed">
                  {member.bio}
                </p>
              </div>
            )}

            {/* 亮点标签 */}
            {member.highlights && member.highlights.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-4">
                {member.highlights.map((highlight) => (
                  <span
                    key={highlight}
                    className="px-2 py-0.5 bg-white/20 backdrop-blur-sm rounded-md text-[10px] font-medium"
                  >
                    {highlight}
                  </span>
                ))}
              </div>
            )}

            {/* 技能 */}
            <div className="mb-4">
              <div className="flex items-center gap-1.5 mb-2">
                <Code2 className="w-3.5 h-3.5 text-white/70" />
                <span className="text-[10px] text-white/70 uppercase tracking-wider font-bold">Skills</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {member.skills.map((skill) => (
                  <span
                    key={skill}
                    className="px-2 py-0.5 bg-black/20 rounded text-[10px] font-medium"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            {/* 引言/座右铭 */}
            {(member.quote || member.motto) && (
              <div className="mb-4 pl-3 border-l-2 border-white/30">
                <p className="text-xs text-white/80 italic">
                  "{member.motto || member.quote}"
                </p>
                {member.idol && (
                  <p className="text-[10px] text-white/50 mt-1">
                    偶像：{member.idol}
                  </p>
                )}
              </div>
            )}

            {/* 底部：联系方式 */}
            <div className="mt-auto pt-3 border-t border-white/20">
              <div className="flex items-center gap-3">
                {member.website && (
                  <a
                    href={`https://${member.website}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 text-[11px] text-white/80 hover:text-white transition-colors"
                  >
                    <Globe className="w-3.5 h-3.5" />
                    <span>{member.website}</span>
                  </a>
                )}
                {member.github && (
                  <a
                    href={`https://github.com/${member.github}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-white/80 hover:text-white transition-colors"
                  >
                    <Github className="w-3.5 h-3.5" />
                  </a>
                )}
                {member.email && (
                  <a
                    href={`mailto:${member.email}`}
                    className="flex items-center gap-1 text-white/80 hover:text-white transition-colors"
                  >
                    <Mail className="w-3.5 h-3.5" />
                  </a>
                )}
              </div>
            </div>

            {/* 稀有度装饰 */}
            <div className="absolute top-3 right-3">
              {member.rarity === "legendary" && <Crown className="w-6 h-6 text-amber-300 drop-shadow-lg" />}
              {member.rarity === "epic" && <Gem className="w-5 h-5 text-purple-300 drop-shadow-lg" />}
              {member.rarity === "rare" && <Star className="w-5 h-5 text-blue-300 drop-shadow-lg" />}
            </div>
          </div>

          {/* 底部装饰线 */}
          <div
            className={`absolute bottom-0 left-0 right-0 h-1 ${
              member.rarity === "legendary"
                ? "bg-gradient-to-r from-amber-300 via-orange-300 to-amber-300"
                : member.rarity === "epic"
                ? "bg-gradient-to-r from-purple-300 via-fuchsia-300 to-purple-300"
                : member.rarity === "rare"
                ? "bg-gradient-to-r from-blue-300 via-cyan-300 to-blue-300"
                : "bg-white/30"
            }`}
          />
        </div>
      </div>

    </motion.div>
  )
}

export default function TeamPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50">
      {/* 页面背景装饰 */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-20 -left-20 w-80 h-80 bg-blue-200/30 rounded-full blur-3xl" />
        <div className="absolute top-40 -right-20 w-96 h-96 bg-purple-200/20 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-1/3 w-72 h-72 bg-amber-200/20 rounded-full blur-3xl" />
      </div>

      {/* 顶部导航 */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200/60">
        <div className="max-w-7xl mx-auto px-5 sm:px-6 h-14 flex items-center justify-between">
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-2 text-slate-600 hover:text-slate-900">
              <ArrowLeft className="w-4 h-4" />
              返回首页
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <img src="/theta-logo.png" alt="THETA" className="h-7 w-auto" />
          </div>
        </div>
      </header>

      {/* 页面标题 */}
      <section className="max-w-7xl mx-auto px-5 sm:px-6 pt-12 pb-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-black text-slate-900 tracking-tight mb-4">
            THETA <span className="text-blue-600">TEAM</span>
          </h1>
          <p className="text-slate-600 text-lg max-w-2xl mx-auto">
            收集所有队员卡牌，解锁隐藏成就！每一位成员都是 THETA 不可或缺的一部分。
          </p>

          {/* 稀有度图例 */}
          <div className="flex flex-wrap items-center justify-center gap-4 mt-6">
            {(["common", "rare", "epic", "legendary"] as const).map((rarity) => {
              const badge = getRarityBadge(rarity)
              return (
                <div key={rarity} className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${badge.bg} ${badge.text}`}>
                    {rarity}
                  </span>
                  <span className="text-xs text-slate-500">
                    {rarity === "common" && "普通"}
                    {rarity === "rare" && "稀有"}
                    {rarity === "epic" && "史诗"}
                    {rarity === "legendary" && "传奇"}
                  </span>
                </div>
              )
            })}
          </div>

          {/* 交互提示 */}
          <p className="text-sm text-slate-400 mt-4">
            💡 将鼠标悬停在卡片上查看详细信息
          </p>
        </motion.div>
      </section>

      {/* 团队卡片网格 */}
      <section className="max-w-7xl mx-auto px-5 sm:px-6 pb-20">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {TEAM_MEMBERS.map((member, index) => (
            <TeamCard key={member.id} member={member} index={index} />
          ))}
        </div>

        {/* 底部提示 */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-12 text-center"
        >
          <p className="text-sm text-slate-400">
            🐱 更多成员卡牌正在解锁中，敬请期待！
          </p>
        </motion.div>
      </section>
    </div>
  )
}

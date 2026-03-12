---
name: THETA 前端修改清单实施计划
overview: 根据 THETA前端修改清单.docx 逐项实施前端改动，按 P0(1-4) → P1(5-7) → P2(8-9) 优先级推进。
todos:
  - id: p0-1
    content: 上传完成后弹配置面板（语言/模型/THETA专属/超参数），用户确认后开始分析
    status: completed
  - id: p0-2
    content: 进度条每步骤独立计算，按日志关键词更新对应步骤
    status: pending
  - id: p0-3
    content: 结果页 7 个指标动态渲染（读 metrics_k{K}.json）
    status: pending
  - id: p0-4
    content: 结果页添加可视化 Tab（list_visualizations 分组展示）
    status: pending
  - id: p1-5
    content: Agent 图表引用（cite/拖拽/@引用/流式/动态建议）
    status: pending
  - id: p1-6
    content: CSV 上传先选列和清洗
    status: pending
  - id: p1-7
    content: 主题卡片增强（权重/占比/点击展开）
    status: pending
  - id: p2-8
    content: 多模型对比 + 结果导出
    status: pending
isProject: false
---

# THETA 前端修改清单实施计划

根据 `THETA前端修改清单.docx` 消化并罗列的详细计划。

---

## 优先级说明

- **P0（1-4）**：先做
- **P1（5-7）**：次之  
- **P2（8-9）**：最后

---

## P0-1：上传完成后弹出配置面板

### 当前问题
- 上传完直接开始跑，参数选择传了 null，导致日志「参数选择: 100% – null」
- 用户根本没机会选模型/语言，默认 zero_shot

### 改动要点
- 上传完成后**停住**，弹出配置面板
- 用户填完后点击「开始分析」才启动
- 面板内容：
  1. **数据语言**：中文 / English / Deutsch / Español → 对应 `--language`
  2. **模型选择**：12 个模型分组展示，theta 为推荐
  3. **THETA 专属**（仅选 theta 时）：Qwen 尺寸、嵌入模式
  4. **超参数**：根据所选模型动态显示（主题数、epochs、batch_size 等）

### 后端对接
- THETA：`04_train_theta.sh` 传 `--dataset`, `--model_size`, `--mode`, `--num_topics`, `--epochs` 等
- 基线：`05_train_baseline.sh` 传 `--dataset`, `--models`, `--num_topics`, `--epochs`, `--with-viz`, `--language`

---

## P0-2：进度条步骤独立计算

### 当前问题
- 所有阶段日志算进一个总进度，training 50% 时总进度凑够就跳到结果页，训练实际未结束

### 改动要点
- 每步骤 item 只看自己那一段日志，独立计算进度
- 日志关键词 → 步骤映射：

| 日志关键词 | 更新步骤 | 进度计算 |
|------------|----------|----------|
| preprocess: (N%) | 数据预处理 | N% |
| embedding: (N%) | 嵌入 | 50 + N/2% |
| training: task_xxx | 模型训练 | 0%，步骤激活 |
| training: (N%) | 模型训练 | N% |
| evaluation/metrics | 模型评估 | 完成 ✅ |
| visualization/viz: (N%) | 生成可视化 | N% |
| 分析完成 | 全部 | 打勾，跳结果页 |

- 每个步骤只有在本步骤信号完成后才打勾，不能提前打勾

---

## P0-3：结果页 7 个指标

### 当前问题
- 只展示 3 个指标，其余 4 个丢掉

### 改动要点
- 读 `metrics_k{K}.json`，**所有字段动态渲染**
- 7 个指标及方向：
  - TOPIC_DIVERSITY_TD ↑
  - TOPIC_DIVERSITY_IRBO ↑
  - TOPIC_COHERENCE_NPMI ↑
  - TOPIC_COHERENCE_CVC_V ↑
  - TOPIC_COHERENCE_UMASS →（越接近 0 越好）
  - TOPIC_COHERENCE_AVG ↑
  - TOPIC_EXCLUSIVITY ↑
  - PPL ↓
- 每个卡片右上角加 ⓘ 悬停提示

---

## P0-4：结果页可视化 Tab

### 当前问题
- 后端生成 20+ 张图表，前端全部未展示

### 改动要点
- 训练完成后调 `list_visualizations` 获取图表列表
- 分组展示：全局概览、词汇分析、评估分析、训练过程、交互式
- 图片懒加载，切到可视化 Tab 时才请求
- pyLDAvis 用 iframe 嵌入，高度 ≥ 600px

---

## P1-5：右侧 Agent 图表引用

### 5.1 Agent → 用户（cite 图表）
- Agent 回复加 `citations` 字段，渲染引用块
- 文字 [图1][图2] 渲染成蓝色可点击角标
- 引用卡片可折叠，点击可全屏查看

### 5.2 用户 → Agent（拖拽 / @ 引用）
- 拖拽：从可视化 Tab 拖到输入框
- @：输入 @ 弹出图表选择器
- 带图消息传 `attachments` 给后端，后端调 vision API

### 5.3 流式输出（SSE）
- 对接 `/api/agent/chat/stream`

### 5.4 动态智能建议
- 调 `interpret/metrics`、`interpret/topics`、`interpret/summary` 生成卡片

---

## P1-6：上传 CSV 先选列和清洗

- 上传后调预览接口（带 `--preview`）
- 用户选文本列、标签列
- 清洗选项：删除 URL/HTML/标点/停用词等
- 确认后调 `02_clean_data.sh`，再进配置面板

---

## P1-7：主题卡片增强

- 词标签悬停显示权重
- 卡片顶部显示文档占比
- 点击展开显示 `topics/topic_N/word_importance.png`

---

## P2-8：多模型对比

- 模型选择改为多选
- 训练完成调 `08_compare_models.sh`
- 结果页加「模型对比」Tab，跨模型 7 指标对比表

---

## P2-9：结果导出

- 导出按钮：评估指标、主题词、所有图表 ZIP、模型对比报告等
